#!/usr/bin/env python3
"""
extract_benchmark2_embeddings.py
================================
Extract MEAN-TOKEN-POOLED Evo2-7B embeddings for benchmark_2 Train.parquet
and Valid.parquet:

  run_layer18    — layer 18 (blocks.18, full block output) mean-pooled;
                   batches in train/valid_embedding_batches, merged into
                   Train/Valid_with_embeddings.parquet (column emb_mean_blocks_18).
  run_multilayer — layers 13, 17, 20 extracted in a SINGLE pass (all 3 hooks
                   fire during one forward pass); batches saved to separate
                   train/valid_embedding_batches_layer_{L} directories for
                   compatibility with the training script.

Optimizations applied:
  - torch.inference_mode() instead of torch.no_grad() (less overhead)
  - bfloat16 autocast for ~2x throughput on H200 Tensor Cores
  - expandable_segments to reduce CUDA memory fragmentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    BENCH2_DIR,
    clean_sequence, manual_tokenize, merge_embeddings,
    get_device, load_evo2_7b,
)

import os
import gc
import time
import glob
import pickle
import pandas as pd
import torch
from tqdm import tqdm


def run_layer18(model, device):
    # ============================================================
    # Configuration
    # ============================================================
    # Only extract layer 18 block output
    TARGET_LAYER = 18
    layer_names = [f"blocks.{TARGET_LAYER}"]

    # Files to process
    FILES = [
        {"input": BENCH2_DIR / "Train.parquet",
         "output": BENCH2_DIR / "Train_with_embeddings.parquet",
         "batch_dir": BENCH2_DIR / "train_embedding_batches"},
        {"input": BENCH2_DIR / "Valid.parquet",
         "output": BENCH2_DIR / "Valid_with_embeddings.parquet",
         "batch_dir": BENCH2_DIR / "valid_embedding_batches"},
    ]

    # Larger batch size for checkpoint saves since dataset is ~50x bigger than benchmark_4
    BATCH_SIZE = 500

    MAX_SEQ_LEN = 8192
    MIN_SEQ_LEN = 30

    print(f"Layer hooks: {layer_names}")
    print(f"Files to process: {len(FILES)}")

    # ============================================================
    # Process each file
    # ============================================================
    for file_info in FILES:
        input_path = file_info["input"]
        output_path = file_info["output"]
        batch_dir = file_info["batch_dir"]
        progress_file = os.path.join(batch_dir, "_completed_keys.pkl")
        batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)} → mean-token layer {TARGET_LAYER} embeddings")
        print(f"{'='*70}")

        # Read the raw parquet for sequence data
        df_raw = pd.read_parquet(input_path)
        print(f"Loaded {len(df_raw)} rows, {df_raw['EC'].nunique()} unique ECs")
        print(f"Seq length: min={df_raw['Sequence'].str.len().min()}, "
              f"max={df_raw['Sequence'].str.len().max()}, mean={df_raw['Sequence'].str.len().mean():.0f}")

        df_raw["row_key"] = df_raw["AC"]

        # Resume support
        already_done = set()
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "rb") as f:
                    already_done = pickle.load(f)
                print(f"Resuming: {len(already_done)} already processed.")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"WARNING: progress file error ({e}), rebuilding...")
                for bf in sorted(glob.glob(os.path.join(batch_dir, "batch_*.pkl"))):
                    try:
                        with open(bf, "rb") as f:
                            already_done.update(pickle.load(f).keys())
                    except Exception:
                        pass
                print(f"  Rebuilt: {len(already_done)} keys.")

        existing = sorted(glob.glob(os.path.join(batch_dir, "batch_*.pkl")))
        next_batch = len(existing)

        # ----------------------------------------------------------
        # Collect and prepare all pending sequences
        # ----------------------------------------------------------
        pending = []  # list of (row_key, cleaned_seq)
        bad_seq = 0
        skipped = 0
        for idx, row in df_raw.iterrows():
            row_key = row["row_key"]
            if row_key in already_done:
                skipped += 1
                continue
            seq = clean_sequence(str(row["Sequence"]))
            if len(seq) < MIN_SEQ_LEN:
                bad_seq += 1
                already_done.add(row_key)
                continue
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
            pending.append((row_key, seq))

        # Sort by length so batches have minimal padding waste
        pending.sort(key=lambda x: len(x[1]))

        remaining = len(pending)
        print(f"Extracting {remaining} remaining sequences ({skipped} skipped, {bad_seq} bad)...",
              flush=True)

        # Dynamic batch sizing — tuned for H200 with Evo2 7B (~76GB model weight)
        # B*max_len hard limit = 2^31/(3*4096) = 174,773 (Evo2 gated FIR fp32);
        # capped at 170,000 for margin.
        TARGET_VRAM_TOKENS = 170_000
        MAX_BATCH_SEQS = 128

        current_batch = {}
        processed = 0
        errors = 0
        t0 = time.time()

        i = 0
        pbar = tqdm(total=remaining, desc=os.path.basename(input_path),
                    miniters=1, dynamic_ncols=True)
        while i < len(pending):
            # Build a GPU batch: pack sequences until we hit the token/seq budget
            gpu_batch_keys = []
            gpu_batch_seqs = []
            batch_max_len = 0

            while i < len(pending):
                key, seq = pending[i]
                seq_len = len(seq)
                new_max_len = max(batch_max_len, seq_len)
                new_token_count = new_max_len * (len(gpu_batch_seqs) + 1)
                if gpu_batch_seqs and (new_token_count > TARGET_VRAM_TOKENS
                                       or len(gpu_batch_seqs) >= MAX_BATCH_SEQS):
                    break
                gpu_batch_keys.append(key)
                gpu_batch_seqs.append(seq)
                batch_max_len = new_max_len
                i += 1

            # Tokenize and pad the batch
            actual_lengths = [len(s) for s in gpu_batch_seqs]
            padded = [manual_tokenize(s) + [0] * (batch_max_len - len(s))
                      for s in gpu_batch_seqs]

            try:
                input_ids = torch.tensor(padded, dtype=torch.long).to(device)

                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    result = model(input_ids, return_embeddings=True, layer_names=layer_names)
                    outputs = result[1]

                # Extract MEAN-token embedding for each sequence in the batch
                for b_idx in range(len(gpu_batch_keys)):
                    seq_len = actual_lengths[b_idx]
                    row_embeddings = {}
                    for layer in layer_names:
                        emb = outputs[layer][b_idx].float()
                        # Mean pool across all valid (non-padding) token positions
                        pooled = emb[:seq_len, :].mean(dim=0).cpu().numpy()
                        row_embeddings[layer] = pooled
                    current_batch[gpu_batch_keys[b_idx]] = row_embeddings
                    already_done.add(gpu_batch_keys[b_idx])
                    processed += 1

                pbar.update(len(gpu_batch_keys))
                del input_ids, result, outputs

            except Exception as e:
                print(f"\n  Batch error ({len(gpu_batch_seqs)} seqs, "
                      f"max_len={batch_max_len}): {e}", flush=True)
                # Fall back to processing this batch one-by-one
                for b_idx in range(len(gpu_batch_keys)):
                    key = gpu_batch_keys[b_idx]
                    seq = gpu_batch_seqs[b_idx]
                    try:
                        single_ids = torch.tensor([manual_tokenize(seq)]).to(device)
                        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            res = model(single_ids, return_embeddings=True,
                                        layer_names=layer_names)
                            outs = res[1]
                        row_embeddings = {}
                        for layer in layer_names:
                            emb = outs[layer][0].float()
                            # Mean pool across all tokens
                            pooled = emb.mean(dim=0).cpu().numpy()
                            row_embeddings[layer] = pooled
                        current_batch[key] = row_embeddings
                        already_done.add(key)
                        processed += 1
                        del single_ids, res, outs
                    except Exception as e2:
                        print(f"    Fallback error {key}: {e2}", flush=True)
                        already_done.add(key)
                        errors += 1
                    pbar.update(1)  # Update per-sequence during fallback

            # Clear GPU cache periodically
            torch.cuda.empty_cache()

            # Save checkpoint batch to disk
            if len(current_batch) >= BATCH_SIZE:
                bp = os.path.join(batch_dir, f"batch_{next_batch:05d}.pkl")
                with open(bp, "wb") as f:
                    pickle.dump(current_batch, f)
                with open(progress_file, "wb") as f:
                    pickle.dump(already_done, f)
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                tqdm.write(f"  Checkpoint {next_batch} ({len(current_batch)} seqs) | "
                           f"Total: {processed}/{remaining} | {rate:.2f} seq/s | "
                           f"GPU batch: {len(gpu_batch_seqs)} seqs × {batch_max_len} tok")
                next_batch += 1
                current_batch = {}
                gc.collect()

        pbar.close()

        if current_batch:
            bp = os.path.join(batch_dir, f"batch_{next_batch:05d}.pkl")
            with open(bp, "wb") as f:
                pickle.dump(current_batch, f)
            with open(progress_file, "wb") as f:
                pickle.dump(already_done, f)
            print(f"\n  Final batch {next_batch} ({len(current_batch)} seqs)")

        print(f"\nExtraction done: {processed} processed, {skipped} skipped, "
              f"{bad_seq} bad, {errors} errors")

        # ============================================================
        # Merge batches into output parquet (memory-safe; see helper)
        # ============================================================
        del df_raw
        gc.collect()
        col_layer_map = [(f"emb_mean_{layer.replace('.', '_')}", layer)
                         for layer in layer_names]
        merge_embeddings(input_path, output_path, batch_dir, col_layer_map)

    print(f"\n{'='*70}")
    print("ALL DONE — Mean-token layer 18 embeddings appended to both parquet files!")
    print(f"  New column: emb_mean_blocks_18")
    print(f"  Each embedding: 4096-dim, mean-token pooled")
    print(f"{'='*70}")


def run_multilayer(model, device):
    # Extract ALL 3 layers in a SINGLE forward pass
    TARGET_LAYERS = [13, 17, 20]
    layer_names = [f"blocks.{l}" for l in TARGET_LAYERS]

    FILES = [
        {"input": BENCH2_DIR / "Train.parquet",
         "batch_dir_prefix": BENCH2_DIR / "train_embedding_batches_layer_"},
        {"input": BENCH2_DIR / "Valid.parquet",
         "batch_dir_prefix": BENCH2_DIR / "valid_embedding_batches_layer_"},
    ]

    BATCH_SIZE = 500
    MAX_SEQ_LEN = 8192
    MIN_SEQ_LEN = 30

    print(f"Extracting ALL layers in single pass: {layer_names}")
    print(f"Files to process: {len(FILES)}")

    # ============================================================
    # Hook factory — captures activations from ALL layers at once
    # ============================================================
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook_fn

    # ============================================================
    # Process each file (single pass extracts all 3 layers)
    # ============================================================
    for file_info in FILES:
        input_path = file_info["input"]
        batch_dir_prefix = file_info["batch_dir_prefix"]

        # Create batch dirs for all layers
        batch_dirs = {}
        for layer_idx in TARGET_LAYERS:
            bd = f"{batch_dir_prefix}{layer_idx}"
            Path(bd).mkdir(parents=True, exist_ok=True)
            batch_dirs[layer_idx] = bd

        # Use a shared progress file (all layers extracted together)
        progress_file = os.path.join(batch_dirs[TARGET_LAYERS[0]], "_completed_keys_multilayer.pkl")

        fname = os.path.basename(input_path)
        print(f"\n{'='*70}")
        print(f"Processing: {fname} → layers {TARGET_LAYERS} (single pass)")
        print(f"{'='*70}")

        df_raw = pd.read_parquet(input_path)
        print(f"Loaded {len(df_raw)} rows, {df_raw['EC'].nunique()} unique ECs")
        print(f"Seq length: min={df_raw['Sequence'].str.len().min()}, "
              f"max={df_raw['Sequence'].str.len().max()}, "
              f"mean={df_raw['Sequence'].str.len().mean():.0f}")

        df_raw["row_key"] = df_raw["AC"]

        # Load progress
        if os.path.exists(progress_file):
            with open(progress_file, "rb") as f:
                already_done = pickle.load(f)
            print(f"Resuming: {len(already_done)} already completed")
        else:
            already_done = set()

        # Build pending list
        pending = []
        skipped = 0
        bad_seq = 0
        for _, row in df_raw.iterrows():
            row_key = row["row_key"]
            if row_key in already_done:
                skipped += 1
                continue
            seq = clean_sequence(str(row["Sequence"]))
            if len(seq) < MIN_SEQ_LEN:
                bad_seq += 1
                already_done.add(row_key)
                continue
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
            pending.append((row_key, seq))

        pending.sort(key=lambda x: len(x[1]))

        remaining = len(pending)
        print(f"Extracting {remaining} remaining sequences ({skipped} skipped, {bad_seq} bad)...",
              flush=True)

        if remaining == 0:
            print(f"  All done for {fname}!")
            del df_raw
            gc.collect()
            continue

        # Dynamic batch sizing
        # Evo2 7B's gated FIR path builds [B, L, 3*hidden] fp32 tensors; CUDA 32-bit
        # index math fails above 2^31 elements -> B*max_len hard limit = 2^31/(3*4096)
        # = 174,773. Capped at 170,000 for margin.
        MAX_BATCH_TOKENS = 170_000
        MAX_BATCH_SEQS = 128

        # Per-layer batch buffers
        current_batches = {l: {} for l in TARGET_LAYERS}
        batch_counts = {
            l: len(glob.glob(os.path.join(batch_dirs[l], "batch_*.pkl")))
            for l in TARGET_LAYERS
        }

        pbar = tqdm(total=len(df_raw), initial=skipped, desc=f"{fname}",
                    unit="seq", dynamic_ncols=True)

        # Register hooks for ALL target layers at once
        hook_handles = []
        for name, module in model.model.named_modules():
            if name in layer_names:
                h = module.register_forward_hook(make_hook(name))
                hook_handles.append(h)
        print(f"  Registered {len(hook_handles)} hooks (should be {len(TARGET_LAYERS)})")

        i = 0
        while i < len(pending):
            # Build a GPU batch
            gpu_batch = []
            batch_max_len = 0
            while i < len(pending) and len(gpu_batch) < MAX_BATCH_SEQS:
                key, seq = pending[i]
                seq_tokens = len(seq)
                new_max_len = max(batch_max_len, seq_tokens)
                if gpu_batch and (len(gpu_batch) + 1) * new_max_len > MAX_BATCH_TOKENS:
                    break
                gpu_batch.append((key, seq))
                batch_max_len = new_max_len
                i += 1

            if not gpu_batch:
                break

            max_len = max(len(s) for _, s in gpu_batch)
            keys = [k for k, _ in gpu_batch]
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

                # Single forward pass — ALL hooks fire, capturing all 3 layers
                activations.clear()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _ = model.model(padded)

                # Extract mean-pooled embeddings for ALL layers at once
                for layer_idx in TARGET_LAYERS:
                    ln = f"blocks.{layer_idx}"
                    if ln in activations:
                        act = activations[ln].float()
                        for j, key in enumerate(keys):
                            seq_mask = mask[j]
                            seq_act = act[j, seq_mask, :]
                            emb = seq_act.mean(dim=0).cpu().numpy()
                            current_batches[layer_idx][key] = {ln: emb}

                # Mark all as done
                for key in keys:
                    already_done.add(key)

                del padded, mask
                activations.clear()
                torch.cuda.empty_cache()

                pbar.update(len(gpu_batch))

            except Exception as e:
                tqdm.write(f"  Batch error ({len(gpu_batch)} seqs, max_len={max_len}): {e}")
                # Fallback: process one-by-one
                for key, seq in gpu_batch:
                    try:
                        tids = torch.tensor([manual_tokenize(seq)], dtype=torch.long, device=device)
                        activations.clear()
                        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            _ = model.model(tids)

                        for layer_idx in TARGET_LAYERS:
                            ln = f"blocks.{layer_idx}"
                            if ln in activations:
                                act = activations[ln].float()
                                emb = act[0].mean(dim=0).cpu().numpy()
                                current_batches[layer_idx][key] = {ln: emb}

                        already_done.add(key)
                        del tids
                        activations.clear()
                        torch.cuda.empty_cache()
                    except Exception as e2:
                        tqdm.write(f"    Single seq error ({key}): {e2}")
                        already_done.add(key)

                    pbar.update(1)

            # Save batch checkpoints for all layers
            min_batch_len = min(len(current_batches[l]) for l in TARGET_LAYERS)
            if min_batch_len >= BATCH_SIZE:
                for layer_idx in TARGET_LAYERS:
                    batch_counts[layer_idx] += 1
                    bp = os.path.join(batch_dirs[layer_idx],
                                      f"batch_{batch_counts[layer_idx]:06d}.pkl")
                    with open(bp, "wb") as f:
                        pickle.dump(current_batches[layer_idx], f)
                    current_batches[layer_idx] = {}

                with open(progress_file, "wb") as f:
                    pickle.dump(already_done, f)

                tqdm.write(f"  Saved batch (all layers), total done: {len(already_done)}")
                gc.collect()

        # Save remaining for all layers
        for layer_idx in TARGET_LAYERS:
            if current_batches[layer_idx]:
                batch_counts[layer_idx] += 1
                bp = os.path.join(batch_dirs[layer_idx],
                                  f"batch_{batch_counts[layer_idx]:06d}.pkl")
                with open(bp, "wb") as f:
                    pickle.dump(current_batches[layer_idx], f)

        with open(progress_file, "wb") as f:
            pickle.dump(already_done, f)

        # Remove hooks
        for h in hook_handles:
            h.remove()
        hook_handles.clear()

        pbar.close()
        print(f"Done: {fname} — {len(already_done)} sequences × {len(TARGET_LAYERS)} layers extracted")
        del df_raw
        gc.collect()

    print(f"\n{'='*70}")
    print("ALL LAYERS EXTRACTED!")
    print(f"{'='*70}")
    for l in TARGET_LAYERS:
        print(f"  Layer {l}: train_embedding_batches_layer_{l}/ , valid_embedding_batches_layer_{l}/")
    print(f"  Layer 18: train_embedding_batches/ , valid_embedding_batches/ (already existed)")


_FORCE = "--force" in sys.argv


def main():
    device = get_device(require_cuda=True)
    if not _FORCE and os.path.exists(BENCH2_DIR / "Train_with_embeddings.parquet") and os.path.exists(BENCH2_DIR / "Valid_with_embeddings.parquet"):
        print(f"[skip] benchmark_2 embeddings already extracted — skipping (pass --force to re-extract)")
        raise SystemExit(0)
    print("Loading Evo2 7B model...")
    model = load_evo2_7b()
    print("Model loaded successfully!")

    run_layer18(model, device)
    run_multilayer(model, device)


if __name__ == "__main__":
    main()
