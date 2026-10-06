#!/usr/bin/env python3
"""
extract_benchmark4_embeddings.py
================================
Extract Evo2-7B embeddings for benchmark_4 Train.parquet and Valid.parquet
in three sequential steps (one model load):

  run_with_embeddings — layers 9, 24, 26; per layer: block (full block
     output), mlp (MLP output, pre-residual) and post_norm (post-norm,
     pre-MLP), pooled via LAST-TOKEN (final non-padding position).
     Appends emb_blocks_{L}, emb_blocks_{L}_mlp, emb_blocks_{L}_post_norm
     columns to Train/Valid_with_embeddings.parquet.
  run_all_layers — full-block output of ALL 32 layers (blocks.0 ..
     blocks.31), last-token pooled, saved to
     Train/Valid_all_layers_embeddings.parquet (columns emb_blocks_0 ..
     emb_blocks_31).
  run_mean_token — full-block output of ALL 32 layers, MEAN-pooled across
     all valid (non-padding) token positions, appended to the all-layers
     parquets (columns emb_mean_blocks_0 .. emb_mean_blocks_31, preserving
     last-token cols).

Optimizations applied:
  - torch.inference_mode() instead of torch.no_grad() (less overhead)
  - bfloat16 autocast for ~2x throughput on H200 Tensor Cores
  - expandable_segments to reduce CUDA memory fragmentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    BENCH4_DIR,
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


def run_with_embeddings(model, device):
    # ============================================================
    # Configuration
    # ============================================================
    # Layers of interest; per layer we capture 3 sub-embeddings:
    #   block      -> output of blocks.{L}            (full block output)
    #   mlp        -> output of blocks.{L}.mlp        (pre-residual)
    #   post_norm  -> output of blocks.{L}.post_norm  (pre-MLP)
    TARGET_LAYERS = [9, 24, 26]
    layer_names = []
    for L in TARGET_LAYERS:
        layer_names += [f"blocks.{L}", f"blocks.{L}.mlp", f"blocks.{L}.post_norm"]

    # Files to process
    FILES = [
        {"input": str(BENCH4_DIR / "Train.parquet"),
         "output": str(BENCH4_DIR / "Train_with_embeddings.parquet"),
         "batch_dir": str(BENCH4_DIR / "train_embedding_batches_b4")},
        {"input": str(BENCH4_DIR / "Valid.parquet"),
         "output": str(BENCH4_DIR / "Valid_with_embeddings.parquet"),
         "batch_dir": str(BENCH4_DIR / "valid_embedding_batches_b4")},
    ]

    BATCH_SIZE = 100

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
        Path(batch_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)} → last-token layers {TARGET_LAYERS} embeddings (block/mlp/post_norm)")
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
        # Evo2's gated FIR path builds [B, L, 3*hidden] fp32 tensors; CUDA 32-bit
        # index math fails above 2^31 elements -> B*max_len hard limit = 2^31/(3*4096)
        # = 174,773. Capped at 170,000 for margin.
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

                # Extract LAST-TOKEN embedding for each sequence in the batch
                for b_idx in range(len(gpu_batch_keys)):
                    seq_len = actual_lengths[b_idx]
                    row_embeddings = {}
                    for layer in layer_names:
                        emb = outputs[layer][b_idx].float()
                        # Last valid (non-padding) token position
                        pooled = emb[seq_len - 1, :].cpu().numpy()
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
                            # Last token of the (unpadded single) sequence
                            pooled = emb[len(seq) - 1, :].cpu().numpy()
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
        col_layer_map = []
        for L in TARGET_LAYERS:
            col_layer_map += [
                (f"emb_blocks_{L}", f"blocks.{L}"),
                (f"emb_blocks_{L}_mlp", f"blocks.{L}.mlp"),
                (f"emb_blocks_{L}_post_norm", f"blocks.{L}.post_norm"),
            ]
        merge_embeddings(input_path, output_path, batch_dir, col_layer_map)

    print(f"\n{'='*70}")
    print("ALL DONE — last-token embeddings for layers 9/24/26 (block/mlp/post_norm) appended to both parquet files!")
    for L in TARGET_LAYERS:
        print(f"  New columns: emb_blocks_{L}, emb_blocks_{L}_mlp, emb_blocks_{L}_post_norm")
    print(f"  Each embedding: 4096-dim, last-token pooled")
    print(f"{'='*70}")


def run_all_layers(model, device):
    # ============================================================
    # Configuration
    # ============================================================
    # Full-block output of every transformer layer
    TARGET_LAYERS = list(range(32))
    layer_names = [f"blocks.{i}" for i in TARGET_LAYERS]

    # Files to process
    FILES = [
        {"input": str(BENCH4_DIR / "Train.parquet"),
         "output": str(BENCH4_DIR / "Train_all_layers_embeddings.parquet"),
         "batch_dir": str(BENCH4_DIR / "train_embedding_batches_all_layers")},
        {"input": str(BENCH4_DIR / "Valid.parquet"),
         "output": str(BENCH4_DIR / "Valid_all_layers_embeddings.parquet"),
         "batch_dir": str(BENCH4_DIR / "valid_embedding_batches_all_layers")},
    ]

    BATCH_SIZE = 100

    MAX_SEQ_LEN = 8192
    MIN_SEQ_LEN = 30

    print(f"Layer hooks: {len(layer_names)} (blocks.0 .. blocks.31)")
    print(f"Files to process: {len(FILES)}")

    # ============================================================
    # Process each file
    # ============================================================
    for file_info in FILES:
        input_path = file_info["input"]
        output_path = file_info["output"]
        batch_dir = file_info["batch_dir"]
        progress_file = os.path.join(batch_dir, "_completed_keys.pkl")
        Path(batch_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)} → last-token all-32-layers embeddings")
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

                # Extract LAST-TOKEN embedding for each sequence in the batch
                for b_idx in range(len(gpu_batch_keys)):
                    seq_len = actual_lengths[b_idx]
                    row_embeddings = {}
                    for layer in layer_names:
                        emb = outputs[layer][b_idx].float()
                        # Last valid (non-padding) token position
                        pooled = emb[seq_len - 1, :].cpu().numpy()
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
                            # Last token of the (unpadded single) sequence
                            pooled = emb[len(seq) - 1, :].cpu().numpy()
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
        col_layer_map = [(f"emb_{layer.replace('.', '_')}", layer)
                         for layer in layer_names]
        merge_embeddings(input_path, output_path, batch_dir, col_layer_map)

    print(f"\n{'='*70}")
    print("ALL DONE — last-token embeddings for all 32 layers appended to both parquet files!")
    print(f"  New columns: emb_blocks_0 .. emb_blocks_31")
    print(f"  Each embedding: 4096-dim, last-token pooled")
    print(f"{'='*70}")


def run_mean_token(model, device):
    # ============================================================
    # Configuration
    # ============================================================
    # Full-block output of every transformer layer, mean-token pooled
    TARGET_LAYERS = list(range(32))
    layer_names = [f"blocks.{i}" for i in TARGET_LAYERS]

    # Files to process — mean columns are appended to the all-layers parquets
    # (fall back to the raw parquets if the all-layers files don't exist yet)
    FILES = [
        {"input": str(BENCH4_DIR / "Train.parquet"),
         "all_layers": str(BENCH4_DIR / "Train_all_layers_embeddings.parquet"),
         "output": str(BENCH4_DIR / "Train_all_layers_embeddings.parquet"),
         "batch_dir": str(BENCH4_DIR / "train_mean_token_batches")},
        {"input": str(BENCH4_DIR / "Valid.parquet"),
         "all_layers": str(BENCH4_DIR / "Valid_all_layers_embeddings.parquet"),
         "output": str(BENCH4_DIR / "Valid_all_layers_embeddings.parquet"),
         "batch_dir": str(BENCH4_DIR / "valid_mean_token_batches")},
    ]

    BATCH_SIZE = 100

    MAX_SEQ_LEN = 8192
    MIN_SEQ_LEN = 30

    print(f"Layer hooks: {len(layer_names)} (blocks.0 .. blocks.31)")
    print(f"Files to process: {len(FILES)}")

    # ============================================================
    # Process each file
    # ============================================================
    for file_info in FILES:
        input_path = file_info["input"]
        output_path = file_info["output"]
        batch_dir = file_info["batch_dir"]
        progress_file = os.path.join(batch_dir, "_completed_keys.pkl")
        Path(batch_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)} → mean-token all-32-layers embeddings")
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
                            # Mean pool across all tokens of the single sequence
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
        # Merge batches and append mean columns to the all-layers parquet
        # (memory-safe; see helper)
        # ============================================================
        base = (file_info["all_layers"]
                if os.path.exists(file_info["all_layers"]) else input_path)
        print(f"Merge base: {os.path.basename(str(base))}")
        del df_raw
        gc.collect()
        col_layer_map = [(f"emb_mean_{layer.replace('.', '_')}", layer)
                         for layer in layer_names]
        merge_embeddings(input_path, output_path, batch_dir, col_layer_map,
                         base_path=base)

    print(f"\n{'='*70}")
    print("ALL DONE — mean-token embeddings for all 32 layers appended to both parquet files!")
    print(f"  New columns: emb_mean_blocks_0 .. emb_mean_blocks_31")
    print(f"  Each embedding: 4096-dim, mean-token pooled")
    print(f"{'='*70}")


_FORCE = "--force" in sys.argv


def main():
    device = get_device(require_cuda=True)
    if not _FORCE and os.path.exists(BENCH4_DIR / "Train_with_embeddings.parquet") and os.path.exists(BENCH4_DIR / "Valid_with_embeddings.parquet") and os.path.exists(BENCH4_DIR / "Train_all_layers_embeddings.parquet") and os.path.exists(BENCH4_DIR / "Valid_all_layers_embeddings.parquet"):
        print(f"[skip] benchmark_4 embeddings already extracted — skipping (pass --force to re-extract)")
        raise SystemExit(0)
    print("Loading Evo2 7B model...")
    model = load_evo2_7b()
    print("Model loaded successfully!")

    run_with_embeddings(model, device)
    run_all_layers(model, device)
    run_mean_token(model, device)


if __name__ == "__main__":
    main()
