#!/usr/bin/env python3
"""
extract_crosstrans_embeddings.py
================================
Merged cross-translation extraction script (layer 18 + multilayer).

Cross-translation: for each Yarrowia lipolytica gene, the full
nucleotide sequence (promoter, exons, UTR) is kept with its real
Yarrowia nucleotides; only the intron gaps — located by the AA-guided
codon scan (same logic as prepare_yarrowia_inputs.py) — are replaced
with a back-translation of the protein's amino-acid sequence. Codons
for the intron fill are sampled by frequency from the gene's own real
exon codons (Yarrowia codon usage); the AA stream is continuous
across introns and wraps when exhausted. Output length equals the
original full-sequence length; reverse-strand genes are returned in
genomic orientation.

Part 1 (run_layer18):
  Build cross-translated sequences for Yarrowia lipolytica genes,
  extract EC numbers from UniProt "Protein names", and extract
  mean-pooled Evo2-7B layer-18 embeddings for the resulting
  nucleotide sequences.

  Input : data/raw_sequences/genes_with_promoters.csv
          data/raw_sequences/yarrowia_uniprot.csv
  Output: data/predictions_and_results/yarrowia_crosstrans_layer18_embeddings.parquet
          (Gene_Name, AA_Sequence, Nucleotide_Sequence, EC_Numbers, emb_0..emb_4095)
  Batches: data/predictions_and_results/yarrowia_crosstrans_layer18_batches/
           (records parquet + _completed_embeddings.pkl resume checkpoint)

Part 2 (run_multilayer):
  Build cross-translated sequences for Yarrowia lipolytica genes,
  extract EC numbers from UniProt "Protein names", and extract
  mean-pooled Evo2-7B activations for layers 13, 17, 18, and 20 of
  the resulting nucleotide sequences.

  Input : data/raw_sequences/genes_with_promoters.csv
          data/raw_sequences/yarrowia_uniprot.csv
  Output: data/predictions_and_results/yarrowia_crosstrans_multilayer_embeddings.parquet
          (Gene_Name, AA_Sequence, Nucleotide_Sequence, EC_Numbers,
           emb_{13|17|18|20}_{0..4095}, emb_0..emb_4095 [layer-18 alias])
  Batches: data/predictions_and_results/yarrowia_crosstrans_multilayer_batches/
           (records parquet + _completed_embeddings.pkl resume checkpoint)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/

import os
import gc
import csv
import pickle
import hashlib
from collections import OrderedDict, Counter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from definitions import (
    RAW_DIR,
    PRED_DIR,
    get_device,
    load_evo2_7b,
    rev_comp,
    CODON_TABLE,
    translate_cds,
    extract_ec_numbers,
    VALID_DNA,
)

# ============================================================
# Configuration — shared by both parts
# ============================================================
GENES_CSV = RAW_DIR / "genes_with_promoters.csv"
UNIPROT_CSV = RAW_DIR / "yarrowia_uniprot.csv"
CHECKPOINT_INTERVAL = 200
MAX_SEQ_LEN = 8192
MIN_SEQ_LEN = 30

# Dynamic batching config for Evo2
# Evo2 7B's gated FIR path builds [B, L, 3*hidden] fp32 tensors; CUDA 32-bit
# index math fails above 2^31 elements -> B*max_len hard limit = 2^31/(3*4096)
# = 174,773. Capped at 170,000 for margin.
MAX_BATCH_TOKENS = 170_000
MAX_BATCH_SEQS = 128

# ============================================================
# Configuration — layer 18 extraction
# ============================================================
LAYER18_OUTPUT_PARQUET = PRED_DIR / "yarrowia_crosstrans_layer18_embeddings.parquet"
LAYER18_BATCH_DIR = PRED_DIR / "yarrowia_crosstrans_layer18_batches"
LAYER18_RECORDS_PARQUET = os.path.join(LAYER18_BATCH_DIR, "yarrowia_crosstrans_records.parquet")
LAYER18_PROGRESS_FILE = os.path.join(LAYER18_BATCH_DIR, "_completed_embeddings.pkl")

TARGET_LAYER = 18
LAYER_NAME = f"blocks.{TARGET_LAYER}"

# ============================================================
# Configuration — multilayer extraction
# ============================================================
MULTI_OUTPUT_PARQUET = PRED_DIR / "yarrowia_crosstrans_multilayer_embeddings.parquet"
MULTI_BATCH_DIR = PRED_DIR / "yarrowia_crosstrans_multilayer_batches"
MULTI_RECORDS_PARQUET = os.path.join(MULTI_BATCH_DIR, "yarrowia_crosstrans_records.parquet")
MULTI_PROGRESS_FILE = os.path.join(MULTI_BATCH_DIR, "_completed_embeddings.pkl")

TARGET_LAYERS = [13, 17, 18, 20]
LAYER_NAMES = [f"blocks.{l}" for l in TARGET_LAYERS]


# ============================================================
# Cross-translation construction
# ============================================================
# AA -> synonymous codons (reverse of CODON_TABLE)
AA_TO_SYNONYMS = {}
for _codon, _aa in CODON_TABLE.items():
    AA_TO_SYNONYMS.setdefault(_aa, []).append(_codon)
STOP_CODONS = {"TAA", "TAG", "TGA"}


def _crosstrans_segments(seq, aa_seq, min_match=5):
    """Walk the sequence codon-by-codon against the expected AA sequence
    (same matching logic as prepare_yarrowia_inputs._try_extract_cds),
    but keep segment boundaries instead of splicing.

    Returns (segments, cds, exon_codons):
      segments    -- list of ("keep"|"fill", start, end) over seq
      cds         -- spliced exon string (for verification)
      exon_codons -- real codons read during the walk
    Returns (None, None, None) on failure.
    """
    for start in range(len(seq) - 2):
        if seq[start:start + 3] != "ATG":
            continue
        if CODON_TABLE.get(seq[start:start + 3]) != aa_seq[0]:
            continue

        segments = [("keep", 0, start)]
        exon_parts = []
        exon_codons = [seq[start:start + 3]]
        nt_pos = start + 3
        aa_pos = 1
        exon_start = start
        success = True

        while aa_pos < len(aa_seq):
            if nt_pos + 3 > len(seq):
                success = False
                break
            codon = seq[nt_pos:nt_pos + 3]
            if CODON_TABLE.get(codon, "?") == aa_seq[aa_pos]:
                aa_pos += 1
                nt_pos += 3
                exon_codons.append(codon)
            else:
                # Mismatch — intron boundary
                if nt_pos > exon_start:
                    exon_parts.append(seq[exon_start:nt_pos])
                    segments.append(("keep", exon_start, nt_pos))
                remaining_aa = len(aa_seq) - aa_pos
                required = min(min_match, remaining_aa)
                found = False
                for offset in range(1, len(seq) - nt_pos):
                    test_pos = nt_pos + offset
                    if test_pos + required * 3 > len(seq):
                        break
                    match_count = 0
                    for k in range(required):
                        c = seq[test_pos + k * 3:test_pos + k * 3 + 3]
                        if CODON_TABLE.get(c, "?") == aa_seq[aa_pos + k]:
                            match_count += 1
                        else:
                            break
                    if match_count >= required:
                        segments.append(("fill", nt_pos, test_pos))
                        nt_pos = test_pos
                        exon_start = test_pos
                        found = True
                        break
                if not found:
                    success = False
                    break

        if not success or aa_pos != len(aa_seq):
            continue

        stop = seq[nt_pos:nt_pos + 3]
        end_pos = nt_pos + 3 if stop in STOP_CODONS else nt_pos
        exon_parts.append(seq[exon_start:end_pos])
        segments.append(("keep", exon_start, end_pos))
        segments.append(("keep", end_pos, len(seq)))
        return segments, "".join(exon_parts), exon_codons
    return None, None, None


def build_crosstrans_sequence(nt_seq, aa_seq, seed_key):
    """Build a cross-translated full-length sequence for one gene.

    Keeps real nucleotides everywhere except intron gaps; each gap is
    filled in-frame with a back-translation of the protein's AA
    sequence (continuous stream, wrapping), codons sampled by
    frequency from the gene's own real exon codons. Output length
    equals the input length. Reverse-strand genes are returned in
    genomic orientation.

    Returns (sequence, verified) or (None, False).
    """
    if not aa_seq or not nt_seq or len(nt_seq) < len(aa_seq) * 3:
        return None, False

    built = None
    cds = None
    for seq in (nt_seq, rev_comp(nt_seq)):
        segments, cds, exon_codons = _crosstrans_segments(seq, aa_seq)
        if segments is not None:
            built = (segments, seq)
            break
    if built is None:
        for seq in (nt_seq, rev_comp(nt_seq)):
            segments, cds, exon_codons = _crosstrans_segments(seq, aa_seq, min_match=3)
            if segments is not None:
                built = (segments, seq)
                break
    if built is None:
        return None, False
    segments, seq = built

    usage = {}
    for c in exon_codons:
        usage.setdefault(CODON_TABLE[c], Counter())[c] += 1

    rng = np.random.default_rng(int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16))
    out = []
    aa_cursor = 0
    n_aa = len(aa_seq)
    for kind, a, b in segments:
        if kind == "keep":
            out.append(seq[a:b])
        else:
            gap = b - a
            n_codons = gap // 3
            fill = []
            for _ in range(n_codons):
                aa = aa_seq[aa_cursor % n_aa]
                aa_cursor += 1
                synonyms = AA_TO_SYNONYMS.get(aa)
                if not synonyms:
                    fill.append("GCT")
                    continue
                counts = [usage.get(aa, Counter()).get(c, 0) for c in synonyms]
                total = sum(counts)
                if total:
                    codon = rng.choice(synonyms, p=np.array(counts, dtype=float) / total)
                else:
                    codon = synonyms[0]
                fill.append(codon)
            out.append("".join(fill))
            out.append(seq[a + 3 * n_codons:b])  # leftover nt kept as-is

    result = "".join(out)
    assert len(result) == len(nt_seq), "length not preserved"
    if seq is not nt_seq:
        result = rev_comp(result)  # back to genomic orientation

    verified = translate_cds(cds) == aa_seq
    return result, verified


# ============================================================
# Part 1: Layer 18 mean-pooled embeddings
# ============================================================
def run_layer18(model, device):
    # 1. Load inputs: full genomic nt + UniProt AAs
    print(f"\nLoading genes_with_promoters.csv: {GENES_CSV}")
    genes = OrderedDict()
    with open(GENES_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            gene_name = row[1]
            nt_seq = row[6].strip()
            if gene_name and nt_seq:
                genes[gene_name] = nt_seq
    print(f"  Loaded {len(genes)} genes")

    print(f"Loading UniProt AAs: {UNIPROT_CSV}")
    uniprot = {}
    with open(UNIPROT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            ordered_locus = row[1].strip()
            orf_name = row[2].strip().replace('_', '')
            protein_name = row[4].strip()
            aa_seq = row[6].strip()
            entry = {'aa_seq': aa_seq, 'protein_name': protein_name}
            if ordered_locus:
                uniprot[ordered_locus] = entry
            if orf_name:
                uniprot[orf_name] = entry
    print(f"  Loaded {len(uniprot)} UniProt entries (by gene name)")

    # 2. Build cross-translated sequences & extract ECs
    print("\nBuilding cross-translated sequences (real exons + AA-back-translated introns)...")
    records = []
    stats = {'total': 0, 'no_uniprot': 0, 'no_aa': 0, 'built': 0, 'verified': 0, 'failed': 0}
    for gene_name, nt_seq in genes.items():
        stats['total'] += 1
        entry = uniprot.get(gene_name)
        if entry is None:
            stats['no_uniprot'] += 1
            continue
        aa_seq = entry['aa_seq']
        if not aa_seq:
            stats['no_aa'] += 1
            continue

        ct_seq, verified = build_crosstrans_sequence(nt_seq, aa_seq, seed_key=gene_name)
        if ct_seq is None:
            stats['failed'] += 1
            continue

        # Extract EC numbers
        ec_list = extract_ec_numbers(entry['protein_name'])
        ec_str = ";".join(ec_list) if ec_list else "UP"

        records.append({
            "Gene_Name": gene_name,
            "AA_Sequence": aa_seq,
            "Nucleotide_Sequence": ct_seq,
            "EC_Numbers": ec_str,
            "Length": len(ct_seq)
        })
        stats['built'] += 1
        if verified:
            stats['verified'] += 1

    df = pd.DataFrame(records)
    print(f"  Genes: {stats['total']} total, {stats['no_uniprot']} no UniProt, {stats['no_aa']} no AA, "
          f"{stats['built']} cross-translated ({stats['verified']} verified), {stats['failed']} walk failed")
    print(f"  Generated {len(df)} query sequences. Annotated with EC (not UP): {len(df[df['EC_Numbers'] != 'UP'])}")

    # Save records for audit/resume
    LAYER18_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LAYER18_RECORDS_PARQUET, index=False)
    print(f"  Records saved to: {LAYER18_RECORDS_PARQUET}")

    # 3. Extract Evo2 embeddings
    if os.path.exists(LAYER18_PROGRESS_FILE):
        with open(LAYER18_PROGRESS_FILE, "rb") as f:
            already_done = pickle.load(f)
        print(f"Resuming embedding extraction: {len(already_done)} sequences already done")
    else:
        already_done = {}  # idx -> embedding (numpy array) or None (failed/short)

    pending = []
    for idx, row in df.iterrows():
        if idx in already_done:
            continue
        seq = "".join(c for c in row["Nucleotide_Sequence"].upper() if c in VALID_DNA)
        if len(seq) < MIN_SEQ_LEN:
            already_done[idx] = None
            continue
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        pending.append((idx, seq))

    # Sort pending by length for batching efficiency
    pending.sort(key=lambda x: len(x[1]))
    print(f"  Pending extraction: {len(pending)}")

    if len(pending) > 0:
        # Register hook on blocks.18
        activations = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[LAYER_NAME] = output[0].detach()
            else:
                activations[LAYER_NAME] = output.detach()

        hook_handle = None
        for name, module in model.model.named_modules():
            if name == LAYER_NAME:
                hook_handle = module.register_forward_hook(hook_fn)
                print(f"Hook registered on: {name}")
                break
        if hook_handle is None:
            raise RuntimeError(f"Could not find module '{LAYER_NAME}'")

        # Extraction loop
        pbar = tqdm(total=len(df), initial=len(already_done), desc="Evo2 Extraction", unit="seq")

        i = 0
        processed_since_save = 0
        while i < len(pending):
            gpu_batch = []
            batch_max_len = 0
            while i < len(pending) and len(gpu_batch) < MAX_BATCH_SEQS:
                idx, seq = pending[i]
                seq_tokens = len(seq)
                new_max_len = max(batch_max_len, seq_tokens)
                if gpu_batch and (len(gpu_batch) + 1) * new_max_len > MAX_BATCH_TOKENS:
                    break
                gpu_batch.append((idx, seq))
                batch_max_len = new_max_len
                i += 1

            if not gpu_batch:
                break

            max_len = max(len(s) for _, s in gpu_batch)
            indices = [idx for idx, _ in gpu_batch]
            seqs = [s for _, s in gpu_batch]

            try:
                # Tokenize & pad
                token_ids = [[ord(c) for c in s] for s in seqs]
                padded = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
                mask = torch.zeros(len(seqs), max_len, dtype=torch.bool, device=device)
                for j, tids in enumerate(token_ids):
                    L = len(tids)
                    padded[j, :L] = torch.tensor(tids, dtype=torch.long)
                    mask[j, :L] = True

                activations.clear()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model.model(padded)

                if LAYER_NAME in activations:
                    act = activations[LAYER_NAME].float()
                    for j, idx in enumerate(indices):
                        seq_mask = mask[j]
                        seq_act = act[j, seq_mask, :]
                        emb = seq_act.mean(dim=0).cpu().numpy()
                        already_done[idx] = emb

                del padded, mask
                activations.clear()
                torch.cuda.empty_cache()
                pbar.update(len(gpu_batch))
                processed_since_save += len(gpu_batch)

            except Exception as e:
                tqdm.write(f"Batch error: {e}. Falling back to single sequence extraction...")
                for idx, seq in gpu_batch:
                    try:
                        tids = torch.tensor([[ord(c) for c in seq]], dtype=torch.long, device=device)
                        activations.clear()
                        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            _ = model.model(tids)
                        if LAYER_NAME in activations:
                            emb = activations[LAYER_NAME].float()[0].mean(dim=0).cpu().numpy()
                            already_done[idx] = emb
                        del tids
                        activations.clear()
                        torch.cuda.empty_cache()
                    except Exception as e2:
                        tqdm.write(f"Error on single seq index {idx}: {e2}")
                        already_done[idx] = None
                    pbar.update(1)
                    processed_since_save += 1

            if processed_since_save >= CHECKPOINT_INTERVAL:
                with open(LAYER18_PROGRESS_FILE, "wb") as f:
                    pickle.dump(already_done, f)
                processed_since_save = 0

        # Save final progress
        with open(LAYER18_PROGRESS_FILE, "wb") as f:
            pickle.dump(already_done, f)
        pbar.close()

        # Free Evo2 memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Compile embedding matrix and write embeddings parquet
    print("\nCompiling embedding matrix...")
    valid_indices = []
    emb_list = []
    for idx in range(len(df)):
        emb = already_done.get(idx)
        if emb is not None:
            valid_indices.append(idx)
            emb_list.append(emb)

    df_valid = df.iloc[valid_indices].copy()
    if len(df_valid) > 0:
        X = np.vstack(emb_list).astype(np.float32)
        print(f"Features matrix shape: {X.shape}")

        # Write layer 18 activations to df
        for d in range(4096):
            df_valid[f"emb_{d}"] = X[:, d]
    else:
        print("No valid embeddings extracted — nothing to write.")

    print(f"Saving embeddings to: {LAYER18_OUTPUT_PARQUET}")
    df_valid.to_parquet(LAYER18_OUTPUT_PARQUET, index=False)

    print(f"\nDone!")
    print(f"  Output: {LAYER18_OUTPUT_PARQUET}")
    print(f"  Total rows: {len(df_valid)}")
    print(f"  EC annotated: {(df_valid['EC_Numbers'] != 'UP').sum()}")
    print(f"  Uncategorized (UP): {(df_valid['EC_Numbers'] == 'UP').sum()}")


# ============================================================
# Part 2: Multilayer embeddings (layers 13, 17, 18, 20)
# ============================================================
def run_multilayer(model, device):
    # 1. Load inputs: full genomic nt + UniProt AAs
    print(f"\nLoading genes_with_promoters.csv: {GENES_CSV}")
    genes = OrderedDict()
    with open(GENES_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            gene_name = row[1]
            nt_seq = row[6].strip()
            if gene_name and nt_seq:
                genes[gene_name] = nt_seq
    print(f"  Loaded {len(genes)} genes")

    print(f"Loading UniProt AAs: {UNIPROT_CSV}")
    uniprot = {}
    with open(UNIPROT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            ordered_locus = row[1].strip()
            orf_name = row[2].strip().replace('_', '')
            protein_name = row[4].strip()
            aa_seq = row[6].strip()
            entry = {'aa_seq': aa_seq, 'protein_name': protein_name}
            if ordered_locus:
                uniprot[ordered_locus] = entry
            if orf_name:
                uniprot[orf_name] = entry
    print(f"  Loaded {len(uniprot)} UniProt entries (by gene name)")

    # 2. Build cross-translated sequences & extract ECs
    print("\nBuilding cross-translated sequences (real exons + AA-back-translated introns)...")
    records = []
    stats = {'total': 0, 'no_uniprot': 0, 'no_aa': 0, 'built': 0, 'verified': 0, 'failed': 0}
    for gene_name, nt_seq in genes.items():
        stats['total'] += 1
        entry = uniprot.get(gene_name)
        if entry is None:
            stats['no_uniprot'] += 1
            continue
        aa_seq = entry['aa_seq']
        if not aa_seq:
            stats['no_aa'] += 1
            continue

        ct_seq, verified = build_crosstrans_sequence(nt_seq, aa_seq, seed_key=gene_name)
        if ct_seq is None:
            stats['failed'] += 1
            continue

        # Extract EC numbers
        ec_list = extract_ec_numbers(entry['protein_name'])
        ec_str = ";".join(ec_list) if ec_list else "UP"

        records.append({
            "Gene_Name": gene_name,
            "AA_Sequence": aa_seq,
            "Nucleotide_Sequence": ct_seq,
            "EC_Numbers": ec_str,
            "Length": len(ct_seq)
        })
        stats['built'] += 1
        if verified:
            stats['verified'] += 1

    df = pd.DataFrame(records)
    print(f"  Genes: {stats['total']} total, {stats['no_uniprot']} no UniProt, {stats['no_aa']} no AA, "
          f"{stats['built']} cross-translated ({stats['verified']} verified), {stats['failed']} walk failed")
    print(f"  Generated {len(df)} query sequences. Annotated with EC (not UP): {len(df[df['EC_Numbers'] != 'UP'])}")

    # Save records for audit/resume
    MULTI_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MULTI_RECORDS_PARQUET, index=False)
    print(f"  Records saved to: {MULTI_RECORDS_PARQUET}")

    # 3. Extract Evo2 embeddings
    if os.path.exists(MULTI_PROGRESS_FILE):
        with open(MULTI_PROGRESS_FILE, "rb") as f:
            already_done = pickle.load(f)
        print(f"Resuming embedding extraction: {len(already_done)} sequences already done")
    else:
        already_done = {}  # idx -> dict of layer_idx -> embedding (numpy array)

    pending = []
    for idx, row in df.iterrows():
        if idx in already_done:
            continue
        seq = "".join(c for c in row["Nucleotide_Sequence"].upper() if c in VALID_DNA)
        if len(seq) < MIN_SEQ_LEN:
            already_done[idx] = None
            continue
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        pending.append((idx, seq))

    # Sort pending by length for batching efficiency
    pending.sort(key=lambda x: len(x[1]))
    print(f"  Pending extraction: {len(pending)}")

    if len(pending) > 0:
        # Register hooks on target blocks
        activations = {}
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_name] = output[0].detach()
                else:
                    activations[layer_name] = output.detach()
            return hook_fn

        hook_handles = []
        for name, module in model.model.named_modules():
            if name in LAYER_NAMES:
                h = module.register_forward_hook(make_hook(name))
                hook_handles.append(h)
        print(f"Registered {len(hook_handles)} hooks (should be {len(TARGET_LAYERS)})")

        if len(hook_handles) != len(TARGET_LAYERS):
            raise RuntimeError(f"Expected to register {len(TARGET_LAYERS)} hooks, but registered {len(hook_handles)}")

        # Extraction loop
        pbar = tqdm(total=len(df), initial=len(already_done), desc="Evo2 Multilayer Extraction", unit="seq")

        i = 0
        processed_since_save = 0
        while i < len(pending):
            gpu_batch = []
            batch_max_len = 0
            while i < len(pending) and len(gpu_batch) < MAX_BATCH_SEQS:
                idx, seq = pending[i]
                seq_tokens = len(seq)
                new_max_len = max(batch_max_len, seq_tokens)
                if gpu_batch and (len(gpu_batch) + 1) * new_max_len > MAX_BATCH_TOKENS:
                    break
                gpu_batch.append((idx, seq))
                batch_max_len = new_max_len
                i += 1

            if not gpu_batch:
                break

            max_len = max(len(s) for _, s in gpu_batch)
            indices = [idx for idx, _ in gpu_batch]
            seqs = [s for _, s in gpu_batch]

            try:
                # Tokenize & pad
                token_ids = [[ord(c) for c in s] for s in seqs]
                padded = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
                mask = torch.zeros(len(seqs), max_len, dtype=torch.bool, device=device)
                for j, tids in enumerate(token_ids):
                    L = len(tids)
                    padded[j, :L] = torch.tensor(tids, dtype=torch.long)
                    mask[j, :L] = True

                activations.clear()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model.model(padded)

                # Extract mean-pooled activations for all layers
                batch_embs = {idx: {} for idx in indices}
                for layer_idx in TARGET_LAYERS:
                    ln = f"blocks.{layer_idx}"
                    if ln in activations:
                        act = activations[ln].float()
                        for j, idx in enumerate(indices):
                            seq_mask = mask[j]
                            seq_act = act[j, seq_mask, :]
                            emb = seq_act.mean(dim=0).cpu().numpy()
                            batch_embs[idx][layer_idx] = emb

                for idx in indices:
                    already_done[idx] = batch_embs[idx]

                del padded, mask
                activations.clear()
                torch.cuda.empty_cache()
                pbar.update(len(gpu_batch))
                processed_since_save += len(gpu_batch)

            except Exception as e:
                tqdm.write(f"Batch error: {e}. Falling back to single sequence extraction...")
                for idx, seq in gpu_batch:
                    try:
                        tids = torch.tensor([[ord(c) for c in seq]], dtype=torch.long, device=device)
                        activations.clear()
                        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            _ = model.model(tids)

                        single_embs = {}
                        for layer_idx in TARGET_LAYERS:
                            ln = f"blocks.{layer_idx}"
                            if ln in activations:
                                emb = activations[ln].float()[0].mean(dim=0).cpu().numpy()
                                single_embs[layer_idx] = emb

                        already_done[idx] = single_embs
                        del tids
                        activations.clear()
                        torch.cuda.empty_cache()
                    except Exception as e2:
                        tqdm.write(f"Error on single seq index {idx}: {e2}")
                        already_done[idx] = None
                    pbar.update(1)
                    processed_since_save += 1

            if processed_since_save >= CHECKPOINT_INTERVAL:
                with open(MULTI_PROGRESS_FILE, "wb") as f:
                    pickle.dump(already_done, f)
                processed_since_save = 0

        # Save final progress
        with open(MULTI_PROGRESS_FILE, "wb") as f:
            pickle.dump(already_done, f)
        pbar.close()

        # Free Evo2 memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Compile embedding matrix and write embeddings parquet
    print("\nCompiling embedding matrix...")
    valid_indices = []
    emb_layers_list = {layer: [] for layer in TARGET_LAYERS}

    for idx in range(len(df)):
        embs_dict = already_done.get(idx)
        if embs_dict is not None and all(layer in embs_dict for layer in TARGET_LAYERS):
            valid_indices.append(idx)
            for layer in TARGET_LAYERS:
                emb_layers_list[layer].append(embs_dict[layer])

    df_valid = df.iloc[valid_indices].copy()

    if len(df_valid) > 0:
        # Construct concatenated features matrix
        X_layers = []
        for layer in TARGET_LAYERS:
            layer_matrix = np.vstack(emb_layers_list[layer]).astype(np.float32)
            X_layers.append(layer_matrix)

            # Add activations to DataFrame for this layer
            for d in range(4096):
                df_valid[f"emb_{layer}_{d}"] = layer_matrix[:, d]

        X = np.concatenate(X_layers, axis=1) # Shape: [N, 16384]
        print(f"Concatenated features matrix shape: {X.shape}")

        # Add standard emb_0..4095 aliases for layer 18
        for d in range(4096):
            df_valid[f"emb_{d}"] = df_valid[f"emb_18_{d}"]
    else:
        print("No valid embeddings extracted — nothing to write.")

    print(f"Saving embeddings to: {MULTI_OUTPUT_PARQUET}")
    df_valid.to_parquet(MULTI_OUTPUT_PARQUET, index=False)

    print(f"\nDone!")
    print(f"  Output: {MULTI_OUTPUT_PARQUET}")
    print(f"  Total rows: {len(df_valid)}")
    print(f"  EC annotated: {(df_valid['EC_Numbers'] != 'UP').sum()}")
    print(f"  Uncategorized (UP): {(df_valid['EC_Numbers'] == 'UP').sum()}")


# ============================================================
# Main Execution Flow
# ============================================================
_FORCE = "--force" in sys.argv


def main():
    device = get_device(require_cuda=True)

    if not _FORCE and os.path.exists(LAYER18_OUTPUT_PARQUET) and os.path.exists(MULTI_OUTPUT_PARQUET):
        print(f"[skip] crosstrans embeddings already extracted — skipping (pass --force to re-extract)")
        raise SystemExit(0)
    print("Loading Evo2 7B model...")
    model = load_evo2_7b()
    print("Model loaded successfully!")

    run_layer18(model, device)
    run_multilayer(model, device)


if __name__ == "__main__":
    main()
