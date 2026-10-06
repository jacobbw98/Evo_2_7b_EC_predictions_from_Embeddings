#!/usr/bin/env python3
"""
Extract Evo2 7B embeddings for Yarrowia lipolytica nucleotide sequences
(merged: layer-18 mean-pooled + multilayer extraction).

Part 1 (run_layer18):
  Extract Evo2 7B mean-pooled layer 18 activation embeddings for yarrowia
  nucleotide sequences.

  Reads yarrowia_sequences_ec.csv (Nucleotide_Sequence, EC_Numbers),
  runs each sequence through Evo2 7B, hooks into blocks.18 to capture
  activations, mean-pools across sequence positions, and saves a .parquet
  file with columns:
    - Nucleotide_Sequence: the original nucleotide sequence
    - EC_Numbers: EC classification (or "UP" for uncategorized)
    - Embedding: layer 18 mean-pooled embedding (list of floats)

  Supports resuming from checkpoints if interrupted.

Part 2 (run_multilayer):
  Extract multilayer Evo2 embeddings (layers 13, 17, 18, 20) for the
  Yarrowia lipolytica genome sequences.

  Reads:
    - data/raw_sequences/yarrowia_sequences_ec.csv
      (columns: Nucleotide_Sequence, EC_Numbers)

  Writes:
    - data/predictions_and_results/yarrowia_evo2_multilayer_embeddings.parquet
      with one row per valid input sequence and 16,384 embedding columns
      (emb_{layer}_{d} for layer in [13, 17, 18, 20], d = 0..4095) plus
      emb_0..emb_4095 aliases for layer 18.

  This parquet is the input for:
    - predict_yarrowia_ec_deep.py (Deep Hierarchical MLP, 16,384-dim input)
    - find_consumption_route_genes.py
    - plot_ec_comparison.py
    - check_mevalonate_predictions.py

  Resumable: progress is checkpointed to
    data/predictions_and_results/yarrowia_multilayer_batches/_completed_embeddings.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/

import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import transformer_engine.common.recipe

from definitions import (
    RAW_DIR,
    PRED_DIR,
    get_device,
    load_evo2_7b,
    clean_sequence,
    manual_tokenize,
    VALID_DNA,
)

# ============================================================
# Configuration — shared by both parts
# ============================================================
INPUT_CSV = RAW_DIR / "yarrowia_sequences_ec.csv"
CHECKPOINT_INTERVAL = 200  # save checkpoint every N sequences
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
LAYER18_OUTPUT_PARQUET = PRED_DIR / "yarrowia_evo2_layer18_embeddings.parquet"
LAYER18_BATCH_DIR = PRED_DIR / "yarrowia_embedding_batches_layer_18"
LAYER18_PROGRESS_FILE = os.path.join(LAYER18_BATCH_DIR, "_completed_indices.pkl")

TARGET_LAYER = 18
LAYER_NAME = f"blocks.{TARGET_LAYER}"

# ============================================================
# Configuration — multilayer extraction
# ============================================================
MULTI_OUTPUT_PARQUET = PRED_DIR / "yarrowia_evo2_multilayer_embeddings.parquet"
MULTI_BATCH_DIR = PRED_DIR / "yarrowia_multilayer_batches"
MULTI_RECORDS_PARQUET = os.path.join(MULTI_BATCH_DIR, "yarrowia_multilayer_records.parquet")
MULTI_PROGRESS_FILE = os.path.join(MULTI_BATCH_DIR, "_completed_embeddings.pkl")

TARGET_LAYERS = [13, 17, 18, 20]
LAYER_NAMES = [f"blocks.{l}" for l in TARGET_LAYERS]


# ============================================================
# Part 1: Layer 18 mean-pooled embeddings
# ============================================================
def run_layer18(model, device):
    # ============================================================
    # Hook to capture layer 18 activations
    # ============================================================
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations[LAYER_NAME] = output[0].detach()
        else:
            activations[LAYER_NAME] = output.detach()

    # ============================================================
    # Load input data
    # ============================================================
    print(f"\nLoading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Total sequences: {len(df)}")
    print(f"  Seq length: min={df['Nucleotide_Sequence'].str.len().min()}, "
          f"max={df['Nucleotide_Sequence'].str.len().max()}, "
          f"mean={df['Nucleotide_Sequence'].str.len().mean():.0f}")

    ec_counts = df['EC_Numbers'].value_counts()
    n_with_ec = (df['EC_Numbers'] != 'UP').sum()
    print(f"  With EC numbers: {n_with_ec}")
    print(f"  Uncategorized (UP): {(df['EC_Numbers'] == 'UP').sum()}")

    # ============================================================
    # Prepare sequences
    # ============================================================
    LAYER18_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Load progress
    if os.path.exists(LAYER18_PROGRESS_FILE):
        with open(LAYER18_PROGRESS_FILE, "rb") as f:
            already_done = pickle.load(f)
        print(f"Resuming: {len(already_done)} sequences already completed")
    else:
        already_done = {}  # idx -> embedding (numpy array)

    # Build pending list: (original_idx, cleaned_sequence)
    pending = []
    skipped_done = 0
    skipped_bad = 0

    for idx, row in df.iterrows():
        if idx in already_done:
            skipped_done += 1
            continue
        seq = clean_sequence(str(row["Nucleotide_Sequence"]))
        if len(seq) < MIN_SEQ_LEN:
            skipped_bad += 1
            already_done[idx] = None  # Mark as done but no embedding
            continue
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        pending.append((idx, seq))

    # Sort by sequence length for efficient batching
    pending.sort(key=lambda x: len(x[1]))

    print(f"\n  Pending extraction: {len(pending)}")
    print(f"  Already done: {skipped_done}")
    print(f"  Skipped (too short): {skipped_bad}")

    if len(pending) == 0:
        print("All sequences already extracted!")
    else:
        # ============================================================
        # Register hook on blocks.18
        # ============================================================
        hook_handle = None
        for name, module in model.model.named_modules():
            if name == LAYER_NAME:
                hook_handle = module.register_forward_hook(hook_fn)
                print(f"\n  Hook registered on: {name}")
                break

        if hook_handle is None:
            raise RuntimeError(f"Could not find module '{LAYER_NAME}' in model")

        # ============================================================
        # Extract embeddings with dynamic batching
        # ============================================================
        pbar = tqdm(total=len(df), initial=skipped_done, desc="Extracting",
                    unit="seq", dynamic_ncols=True)

        processed_since_save = 0
        i = 0

        while i < len(pending):
            # Build a GPU batch
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
                # Tokenize and pad
                token_ids = [manual_tokenize(s) for s in seqs]
                padded = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
                mask = torch.zeros(len(seqs), max_len, dtype=torch.bool, device=device)
                for j, tids in enumerate(token_ids):
                    L = len(tids)
                    padded[j, :L] = torch.tensor(tids, dtype=torch.long)
                    mask[j, :L] = True

                # Forward pass
                activations.clear()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model.model(padded)

                # Extract mean-pooled embeddings
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
                tqdm.write(f"  Batch error ({len(gpu_batch)} seqs, max_len={max_len}): {e}")
                # Fallback: process one-by-one
                for idx, seq in gpu_batch:
                    try:
                        tids = torch.tensor([manual_tokenize(seq)], dtype=torch.long, device=device)
                        activations.clear()
                        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            _ = model.model(tids)

                        if LAYER_NAME in activations:
                            act = activations[LAYER_NAME].float()
                            emb = act[0].mean(dim=0).cpu().numpy()
                            already_done[idx] = emb

                        del tids
                        activations.clear()
                        torch.cuda.empty_cache()
                    except Exception as e2:
                        tqdm.write(f"    Single seq error (idx={idx}): {e2}")
                        already_done[idx] = None

                    pbar.update(1)
                    processed_since_save += 1

            # Periodic checkpoint
            if processed_since_save >= CHECKPOINT_INTERVAL:
                with open(LAYER18_PROGRESS_FILE, "wb") as f:
                    pickle.dump(already_done, f)
                tqdm.write(f"  Checkpoint saved: {len(already_done)} sequences done")
                processed_since_save = 0
                gc.collect()

        # Final checkpoint
        with open(LAYER18_PROGRESS_FILE, "wb") as f:
            pickle.dump(already_done, f)

        # Remove hook
        hook_handle.remove()
        pbar.close()

        print(f"\nExtraction complete: {len(already_done)} sequences processed")

    # ============================================================
    # Assemble final parquet
    # ============================================================
    print(f"\nAssembling final parquet file...")

    rows = []
    emb_dim = None
    missing = 0

    for idx, row in df.iterrows():
        emb = already_done.get(idx)
        if emb is None:
            missing += 1
            continue

        if emb_dim is None:
            emb_dim = len(emb)

        row_data = {
            "Nucleotide_Sequence": row["Nucleotide_Sequence"],
            "EC_Numbers": row["EC_Numbers"],
        }
        # Store each embedding dimension as a separate column
        for d in range(emb_dim):
            row_data[f"emb_{d}"] = float(emb[d])

        rows.append(row_data)

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(LAYER18_OUTPUT_PARQUET, index=False)

    print(f"\nDone!")
    print(f"  Output: {LAYER18_OUTPUT_PARQUET}")
    print(f"  Total rows: {len(result_df)}")
    print(f"  Embedding dimensions: {emb_dim}")
    print(f"  Skipped (no embedding): {missing}")
    print(f"  EC annotated: {(result_df['EC_Numbers'] != 'UP').sum()}")
    print(f"  Uncategorized (UP): {(result_df['EC_Numbers'] == 'UP').sum()}")


# ============================================================
# Part 2: Multilayer embeddings (layers 13, 17, 18, 20)
# ============================================================
def run_multilayer(model, device):
    # 1. Load input CSV
    print(f"\nLoading input CSV: {INPUT_CSV}")
    raw_df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(raw_df)} rows")

    # 2. Build records (direct nucleotide sequences — no back-translation)
    print("\nBuilding query records...")
    records = []
    for idx, row in raw_df.iterrows():
        seq = str(row.get("Nucleotide_Sequence", "")).strip()
        if not seq:
            continue
        ec_str = str(row.get("EC_Numbers", "UP")).strip() or "UP"
        records.append({
            "Nucleotide_Sequence": seq,
            "EC_Numbers": ec_str,
            "Length": len(seq),
        })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df)} query sequences. "
          f"Annotated with EC (not UP): {len(df[df['EC_Numbers'] != 'UP'])}")

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
        pbar = tqdm(total=len(df), initial=len(already_done), desc="Evo2 Yarrowia Multilayer Extraction", unit="seq")

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
        # Add per-layer embedding columns
        for layer in TARGET_LAYERS:
            layer_matrix = np.vstack(emb_layers_list[layer]).astype(np.float32)
            for d in range(4096):
                df_valid[f"emb_{layer}_{d}"] = layer_matrix[:, d]

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
        print(f"[skip] yarrowia embeddings already extracted — skipping (pass --force to re-extract)")
        raise SystemExit(0)
    print("Loading Evo2 7B model...")
    model = load_evo2_7b()
    print("Model loaded successfully!")

    run_layer18(model, device)
    run_multilayer(model, device)


if __name__ == "__main__":
    main()
