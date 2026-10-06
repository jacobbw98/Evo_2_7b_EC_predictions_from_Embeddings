#!/usr/bin/env python3
"""
extract_cds_multilayer_embeddings.py
====================================
Extract mean-pooled Evo2-7B activations for layers 13, 17, 18, and 20 from
cross-referenced coding sequences (CDS), for downstream EC prediction with
the Flat / Hierarchical / Deep Hierarchical MLP classifiers.

Input : data/raw_sequences/ec_cds_sequences.csv
        (gene_name, cds_sequence, ec_number)
Output: data/predictions_and_results/ec_cds_multilayer_embeddings.parquet
        (Gene_Name, AA_Sequence, Nucleotide_Sequence, EC_Numbers,
         emb_{13|17|18|20}_{0..4095}, emb_0..emb_4095 [layer-18 alias])
Batches: data/predictions_and_results/ec_cds_multilayer_batches/
         (_completed_embeddings.pkl resume checkpoint)

If the output parquet already exists with all embedding columns, extraction
is skipped (resume shortcut, mirroring the original pipeline script).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    RAW_DIR, PRED_DIR,
    VALID_DNA, CODON_TABLE, translate_dna_to_aa, manual_tokenize,
    get_device, load_evo2_7b,
)

import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = RAW_DIR / "ec_cds_sequences.csv"
OUTPUT_PARQUET = PRED_DIR / "ec_cds_multilayer_embeddings.parquet"
BATCH_DIR = PRED_DIR / "ec_cds_multilayer_batches"
PROGRESS_FILE = os.path.join(BATCH_DIR, "_completed_embeddings.pkl")

TARGET_LAYERS = [13, 17, 18, 20]
LAYER_NAMES = [f"blocks.{l}" for l in TARGET_LAYERS]
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
# Main Execution Flow
# ============================================================
def main():
    # 1. Check if output parquet already exists with correct embeddings
    embeddings_loaded = False

    if os.path.exists(OUTPUT_PARQUET):
        print(f"\nOutput parquet {OUTPUT_PARQUET} already exists. Attempting to skip embedding extraction...")
        try:
            df_existing = pd.read_parquet(OUTPUT_PARQUET)
            required_cols = []
            for layer in TARGET_LAYERS:
                required_cols.extend([f"emb_{layer}_{d}" for d in range(4096)])

            if all(col in df_existing.columns for col in required_cols):
                print(f"Successfully found existing parquet with {len(df_existing)} rows. Skipping extraction.")
                embeddings_loaded = True
            else:
                print("Existing parquet does not contain all required embedding columns. Extracting...")
        except Exception as e:
            print(f"Error loading existing parquet: {e}. Extracting...")

    if not embeddings_loaded:
        # 2. Load input CDS CSV
        print(f"\nLoading input CSV: {INPUT_CSV}")
        raw_df = pd.read_csv(INPUT_CSV)
        print(f"  Loaded {len(raw_df)} rows")

        # Process records
        print("\nProcessing sequences and translating to amino acids...")
        records = []
        for idx, row in raw_df.iterrows():
            gene_name = str(row.get("gene_name", "")).strip()
            cds_seq = str(row.get("cds_sequence", "")).strip()
            ec_num = str(row.get("ec_number", "")).strip()

            if not cds_seq:
                continue

            aa_seq = translate_dna_to_aa(cds_seq)

            records.append({
                "Gene_Name": gene_name,
                "AA_Sequence": aa_seq,
                "Nucleotide_Sequence": cds_seq,
                "EC_Numbers": ec_num
            })

        df = pd.DataFrame(records)
        print(f"  Generated {len(df)} query sequences. Annotated with EC (not UP): {len(df[df['EC_Numbers'] != 'UP'])}")

        # 3. Extract Evo2 embeddings
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "rb") as f:
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

        pending.sort(key=lambda x: len(x[1]))
        print(f"  Pending extraction: {len(pending)}")

        if len(pending) > 0:
            device = get_device()

            print("Loading Evo2 7B model...")
            model = load_evo2_7b()

            # Register hook on blocks
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
            print(f"Registered {len(hook_handles)} hooks")

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
                    token_ids = [manual_tokenize(s) for s in seqs]
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
                            tids = torch.tensor([manual_tokenize(seq)], dtype=torch.long, device=device)
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
                    with open(PROGRESS_FILE, "wb") as f:
                        pickle.dump(already_done, f)
                    processed_since_save = 0

            # Save final progress
            with open(PROGRESS_FILE, "wb") as f:
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

        print(f"Saving embeddings to: {OUTPUT_PARQUET}")
        df_valid.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"  Total rows: {len(df_valid)}")

    print(f"\nDone! Embeddings available at: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
