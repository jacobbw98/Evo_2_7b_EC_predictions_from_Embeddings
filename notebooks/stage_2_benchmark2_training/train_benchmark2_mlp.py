#!/usr/bin/env python3
"""
Train benchmark-2 layer-18 MLP classifiers (merged script).

Two experiments sharing the same inputs
(benchmark_2/Train.parquet + train_embedding_batches/,
 benchmark_2/Valid.parquet + valid_embedding_batches/):

  --experiment mlp          
      Wide MLP: 4096 -> 2048 (ReLU, BN, Dropout 0.3) -> 1024 (ReLU, BN,
      Dropout 0.3) -> num_classes
      AdamW (lr=3e-4, wd=0.01) + Linear warmup (10 epochs) + CosineAnnealingLR,
      label smoothing (0.1), gradient clipping (max_norm=1.0), OOM-resilient
      batch sizing (starts at 2048, halves on OOM, recovers after success).
      Output (models/benchmark_2/mlp_results/):
        mlp_layer_18.pt, label_encoder.pkl, results_summary.json/pkl,
        loss_curves.npz, loss_curves.png, accuracy_by_level.png

  --experiment hierarchical 
      Hierarchical Multi-Task MLP:
        Shared Backbone: 4096 -> 2048 -> 1024
        4 Cascaded Heads:
          Head1 (EC1):  1024 -> 256 -> 7 classes
          Head2 (EC2):  1024+7 -> 256 -> 74 classes
          Head3 (EC3):  1024+74 -> 256 -> 270 classes
          Head4 (EC4):  1024+270 -> 512 -> 4640 classes
      Multi-task loss (all 4 EC levels), cascade of previous-level logits,
      focal loss per level, OOM-resilient batch sizing.
      Output (models/benchmark_2/hierarchical_mlp_results/):
        hierarchical_mlp.pt, label_encoders.pkl, results_summary.json/pkl,
        history.npz, loss_curves.png, accuracy_by_level.png

NOTE: Embeddings are loaded from batch pickle files rather than parquet
because PyArrow's int32 list offsets overflow at 1.16M rows x 4096 floats.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    BENCH2_DIR, MODELS_DIR, get_device,
    EC_MLP, HierarchicalEC_MLP, FocalLoss,
    load_embeddings_from_batches, ec_at_level,
)

import os
import gc
import time
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

# ============================================================
# Paths & constants
# ============================================================
TRAIN_PARQUET = BENCH2_DIR / "Train.parquet"
VALID_PARQUET = BENCH2_DIR / "Valid.parquet"
TRAIN_BATCH_DIR = BENCH2_DIR / "train_embedding_batches"
VALID_BATCH_DIR = BENCH2_DIR / "valid_embedding_batches"
OUTPUT_DIR = MODELS_DIR / "benchmark_2" / "mlp_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HIER_OUTPUT_DIR = MODELS_DIR / "benchmark_2" / "hierarchical_mlp_results"
HIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_FORCE = "--force" in sys.argv

LAYER_KEY = "blocks.18"
RANDOM_STATE = 42

MLP_EPOCHS = 200
MLP_BATCH_SIZE = 2048     # Large batch for smoother gradients (4,640 classes need it)
MLP_LR = 3e-4             # Reduced from 1e-3 for stability with large output layer
MLP_WEIGHT_DECAY = 0.01   # AdamW-style weight decay (paper uses 0.01)
LABEL_SMOOTHING = 0.1     # Prevents overconfidence, improves generalization
GRAD_CLIP_NORM = 1.0      # Prevent exploding gradients with 4,640-class output
WARMUP_EPOCHS = 10        # Linear warmup before cosine annealing
FOCAL_GAMMA = 2.0         # Focal loss gamma — higher = more focus on hard examples


# ============================================================
# Experiment 1: Wide MLP
# ============================================================
def run_mlp(device):
    if not _FORCE and os.path.exists(os.path.join(OUTPUT_DIR, "mlp_layer_18.pt")) \
            and os.path.exists(os.path.join(OUTPUT_DIR, "results_summary.json")):
        print("[skip] flat MLP already trained (mlp_results) — skipping (pass --force to retrain)")
        return
    # ============================================================
    # Load data — labels from parquet, embeddings from batch pickles
    # ============================================================
    print(f"\n{'='*70}")
    print("Loading data...")
    print(f"{'='*70}")

    # Load labels
    print("Loading Train labels...", flush=True)
    df_train = pd.read_parquet(TRAIN_PARQUET, columns=["AC", "EC"])
    print(f"  {len(df_train)} rows, {df_train['EC'].nunique()} ECs", flush=True)

    print("Loading Valid labels...", flush=True)
    df_valid = pd.read_parquet(VALID_PARQUET, columns=["AC", "EC"])
    print(f"  {len(df_valid)} rows, {df_valid['EC'].nunique()} ECs", flush=True)

    # Load embeddings from batch pickle files (bypasses PyArrow overflow)
    print("\nLoading embeddings from batch pickles...", flush=True)
    train_emb_dict = load_embeddings_from_batches(TRAIN_BATCH_DIR, LAYER_KEY)
    valid_emb_dict = load_embeddings_from_batches(VALID_BATCH_DIR, LAYER_KEY)

    # ============================================================
    # Encode labels
    # ============================================================
    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    n_classes = len(le.classes_)
    print(f"Total classes: {n_classes}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # ============================================================
    # Build feature matrices — join labels with embeddings by AC key
    # ============================================================
    print("\nBuilding feature matrices...", flush=True)
    t0 = time.time()

    # Train: filter to rows with embeddings and build aligned arrays
    train_mask = df_train["AC"].isin(train_emb_dict)
    train_acs = df_train.loc[train_mask, "AC"].values
    ec_train_str = df_train.loc[train_mask, "EC"].values
    y_train = le.transform(ec_train_str)
    X_train = np.array([train_emb_dict[ac] for ac in train_acs], dtype=np.float32)

    # Valid: same
    valid_mask = df_valid["AC"].isin(valid_emb_dict)
    valid_acs = df_valid.loc[valid_mask, "AC"].values
    ec_valid_str = df_valid.loc[valid_mask, "EC"].values
    y_valid = le.transform(ec_valid_str)
    X_valid = np.array([valid_emb_dict[ac] for ac in valid_acs], dtype=np.float32)

    input_dim = X_train.shape[1]
    print(f"  Train embeddings matched: {train_mask.sum()}/{len(df_train)}")
    print(f"  Valid embeddings matched: {valid_mask.sum()}/{len(df_valid)}")
    print(f"  X_train: {X_train.shape}  X_valid: {X_valid.shape}  "
          f"input_dim: {input_dim}  ({time.time()-t0:.1f}s)", flush=True)

    # Free memory
    del df_train, df_valid, train_emb_dict, valid_emb_dict
    gc.collect()

    # ============================================================
    # Compute balanced class weights
    # ============================================================
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weights_tensor = torch.tensor(cw, dtype=torch.float32)

    # ============================================================
    # Data — full matrices resident on the GPU.
    # A DataLoader over GPU tensors does ~4k Python-level index ops per
    # batch (~18 ms/step, measured); direct perm-sliced HBM gathers are
    # ~2 ms/step. 19 GB train + 9.5 GB valid fits the 143 GB H200.
    # ============================================================
    X_tr_t = torch.from_numpy(X_train).float().to(device)
    y_tr_t = torch.from_numpy(y_train).long().to(device)
    X_va_t = torch.from_numpy(X_valid).float().to(device)
    y_va_t = torch.from_numpy(y_valid).long().to(device)
    n_train = len(y_tr_t)
    n_valid = len(y_va_t)
    del X_train, X_valid  # (y arrays kept: ~18 MB, used for len() in results)
    gc.collect()
    # Start with requested batch size; will be reduced on OOM and recovered
    current_batch_size = MLP_BATCH_SIZE

    # ============================================================
    # Train MLP — full run, no early stopping
    # Training recipe:
    #   - Wide MLP (4096 → 2048 → 1024 → num_classes)
    #   - AdamW optimizer (lr=3e-4, weight_decay=0.01)
    #   - Linear warmup (10 epochs) + CosineAnnealingLR
    #   - Label smoothing (0.1) for regularization
    #   - Gradient clipping (max_norm=1.0)
    #   - OOM-resilient batch sizing
    #   - Best model saved by validation accuracy
    # ============================================================
    print(f"\n{'='*70}")
    print(f"Training MLP: {input_dim} → 2048 → 1024 → {n_classes}")
    print(f"  Epochs: {MLP_EPOCHS} (no early stopping)")
    print(f"  Optimizer: AdamW (lr={MLP_LR}, wd={MLP_WEIGHT_DECAY})")
    print(f"  Scheduler: Linear warmup ({WARMUP_EPOCHS} epochs) + CosineAnnealingLR")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Gradient clipping: max_norm={GRAD_CLIP_NORM}")
    print(f"  Batch size: {MLP_BATCH_SIZE} (with OOM fallback)")
    print(f"{'='*70}", flush=True)

    model = EC_MLP(input_dim, n_classes, dropout=0.3).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Label smoothing + class weights for imbalanced EC distribution
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor.to(device),
        label_smoothing=LABEL_SMOOTHING
    )

    # AdamW with proper weight decay (not L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    # Cosine annealing after warmup: smooth decay from lr to eta_min
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MLP_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )

    # Linear warmup: ramp from 0 to MLP_LR over WARMUP_EPOCHS
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )

    # Chain them: warmup first, then cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    train_losses, valid_losses, valid_accs = [], [], []
    oom_reduced = False  # Track if we reduced batch size due to OOM

    t0 = time.time()
    for epoch in range(1, MLP_EPOCHS + 1):
        # --- Train (GPU-resident data; perm-sliced batches, no DataLoader) ---
        model.train()
        train_loss = 0.0
        train_samples = 0
        perm = torch.randperm(n_train, device=device)
        n_batches = (n_train + current_batch_size - 1) // current_batch_size

        try:
            for bi in range(n_batches):
                start = bi * current_batch_size
                end = min(start + current_batch_size, n_train)
                idx = perm[start:end]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                # Gradient clipping to prevent exploding gradients with 4,640 classes
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_samples += xb.size(0)

        except torch.cuda.OutOfMemoryError:
            # OOM: reduce batch size and retry this epoch
            torch.cuda.empty_cache()
            new_bs = max(256, current_batch_size // 2)
            print(f"\n  ⚠ OOM at epoch {epoch} with batch_size={current_batch_size}, "
                  f"reducing to {new_bs}", flush=True)
            current_batch_size = new_bs
            oom_reduced = True
            optimizer.state.clear()
            model.train()
            train_loss = 0.0
            train_samples = 0
            n_batches = (n_train + current_batch_size - 1) // current_batch_size
            for bi in range(n_batches):
                start = bi * current_batch_size
                end = min(start + current_batch_size, n_train)
                idx = perm[start:end]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_samples += xb.size(0)

        train_loss /= train_samples if train_samples > 0 else 1

        # After a successful epoch with reduced batch, try to recover original size
        if oom_reduced and epoch > 1 and current_batch_size < MLP_BATCH_SIZE:
            try:
                # Test if original batch size fits now (optimizer state may have settled)
                idx_test = torch.arange(MLP_BATCH_SIZE, device=device)
                with torch.no_grad():
                    _ = model(X_tr_t[idx_test])
                del idx_test
                torch.cuda.empty_cache()
                # If we get here, the original batch size fits
                current_batch_size = MLP_BATCH_SIZE
                oom_reduced = False
                print(f"  ✓ Recovered batch_size to {MLP_BATCH_SIZE}", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pass  # Stay at reduced batch size

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        n_val_batches = (n_valid + current_batch_size - 1) // current_batch_size
        with torch.no_grad():
            for bi in range(n_val_batches):
                start = bi * current_batch_size
                end = min(start + current_batch_size, n_valid)
                xb, yb = X_va_t[start:end], y_va_t[start:end]
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_loss /= n_valid
        val_acc = correct / total

        # Step the scheduler
        scheduler.step()

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{MLP_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"bs={current_batch_size}", flush=True)

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.1f}s")
    print(f"  Best epoch: {best_epoch} (val_acc={best_val_acc:.4f})", flush=True)

    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, os.path.join(OUTPUT_DIR, "mlp_layer_18.pt"))

    # Save loss curves
    np.savez(os.path.join(OUTPUT_DIR, "loss_curves.npz"),
             train_loss=np.array(train_losses), valid_loss=np.array(valid_losses),
             valid_acc=np.array(valid_accs))

    # ============================================================
    # Predict on validation set
    # ============================================================
    print("\nPredicting on validation set...", flush=True)
    all_preds = []
    with torch.no_grad():
        n_val_batches = (n_valid + MLP_BATCH_SIZE - 1) // MLP_BATCH_SIZE
        for bi in range(n_val_batches):
            start = bi * MLP_BATCH_SIZE
            all_preds.append(model(X_va_t[start:start + MLP_BATCH_SIZE]).argmax(dim=1).cpu().numpy())
    y_pred_valid = np.concatenate(all_preds)
    ec_pred_valid = le.inverse_transform(y_pred_valid)

    # ============================================================
    # Predict on training set
    # ============================================================
    print("Predicting on training set...", flush=True)
    all_preds_train = []
    with torch.no_grad():
        n_train_batches = (n_train + MLP_BATCH_SIZE - 1) // MLP_BATCH_SIZE
        for bi in range(n_train_batches):
            start = bi * MLP_BATCH_SIZE
            all_preds_train.append(model(X_tr_t[start:start + MLP_BATCH_SIZE]).argmax(dim=1).cpu().numpy())
    y_pred_train = np.concatenate(all_preds_train)
    ec_pred_train = le.inverse_transform(y_pred_train)

    # ============================================================
    # Evaluate at all EC hierarchy levels
    # ============================================================
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")

    results = {
        "train_time": train_time,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "n_classes": n_classes,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "input_dim": int(input_dim),
    }

    for level in [1, 2, 3, 4]:
        true_v = [ec_at_level(ec, level) for ec in ec_valid_str]
        pred_v = [ec_at_level(ec, level) for ec in ec_pred_valid]
        acc = accuracy_score(true_v, pred_v)
        f1_mac = f1_score(true_v, pred_v, average="macro", zero_division=0)
        f1_wt = f1_score(true_v, pred_v, average="weighted", zero_division=0)

        results[f"level_{level}_accuracy"] = acc
        results[f"level_{level}_f1_macro"] = f1_mac
        results[f"level_{level}_f1_weighted"] = f1_wt

        print(f"  Level {level}: Accuracy={acc:.4f}  F1_macro={f1_mac:.4f}  F1_weighted={f1_wt:.4f}")

    # Train accuracy (full EC)
    true_t = [ec_at_level(ec, 4) for ec in ec_train_str]
    pred_t = [ec_at_level(ec, 4) for ec in ec_pred_train]
    results["train_accuracy"] = accuracy_score(true_t, pred_t)
    print(f"\n  Train Full EC Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Valid Full EC Accuracy: {results['level_4_accuracy']:.4f}")

    # ============================================================
    # Save results
    # ============================================================
    # JSON-serializable version
    results_json = {}
    for k, v in results.items():
        if isinstance(v, (float, np.floating)):
            results_json[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            results_json[k] = int(v)
        else:
            results_json[k] = v

    with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "results_summary.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {OUTPUT_DIR}")

    # ============================================================
    # Plots
    # ============================================================

    # --- Loss + accuracy curves ---
    fig, ax1 = plt.subplots(figsize=(14, 5))
    epochs_range = np.arange(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, label="Train Loss", color="#2196F3", lw=2, alpha=0.8)
    ax1.plot(epochs_range, valid_losses, label="Valid Loss", color="#E91E63", lw=2, alpha=0.8)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # Validation accuracy on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, valid_accs, label="Valid Accuracy", color="#4CAF50", lw=2, alpha=0.7, linestyle="-")
    ax2.set_ylabel("Validation Accuracy", fontsize=12, color="#4CAF50")
    ax2.tick_params(axis="y", labelcolor="#4CAF50")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")

    ax1.set_title("Wide MLP Training — Loss & Accuracy (Layer 18, Mean Token, Warmup+Cosine)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: loss_curves.png")

    # --- Accuracy by EC level ---
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [1, 2, 3, 4]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    accuracies = [results[f"level_{lv}_accuracy"] for lv in levels]
    f1_macros = [results[f"level_{lv}_f1_macro"] for lv in levels]
    f1_weights = [results[f"level_{lv}_f1_weighted"] for lv in levels]

    x = np.arange(len(levels))
    width = 0.25
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x, f1_macros, width, label="F1 Macro", color=colors, alpha=0.55)
    bars3 = ax.bar(x + width, f1_weights, width, label="F1 Weighted", color=colors, alpha=0.35,
                   edgecolor=colors, linewidth=2)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lv}" for lv in levels], fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Wide MLP EC Prediction — Layer 18 Mean Token Embeddings\n(Benchmark 2)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_by_level.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: accuracy_by_level.png")

    # ============================================================
    # Final summary
    # ============================================================
    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    print(f"Model:    Wide MLP (4096 → 2048 → 1024 → {n_classes})")
    print(f"Feature:  Evo2 7B Layer 18 Mean-Token-Pooled (4096-dim)")
    print(f"Train:    {len(y_train)} samples, Accuracy={results['train_accuracy']:.4f}")
    print(f"Valid:    {len(y_valid)} samples")
    for lv in [1, 2, 3, 4]:
        print(f"  Level {lv}: Acc={results[f'level_{lv}_accuracy']:.4f} "
              f"F1m={results[f'level_{lv}_f1_macro']:.4f} "
              f"F1w={results[f'level_{lv}_f1_weighted']:.4f}")
    print(f"Best epoch: {best_epoch} | Training time: {train_time:.1f}s")
    print(f"Output: {OUTPUT_DIR}")


# ============================================================
# Experiment 2: Hierarchical Multi-Task MLP
# ============================================================
def run_hierarchical(device):
    if not _FORCE and os.path.exists(os.path.join(HIER_OUTPUT_DIR, "hierarchical_mlp.pt")) \
            and os.path.exists(os.path.join(HIER_OUTPUT_DIR, "results_summary.json")):
        print("[skip] hierarchical MLP already trained — skipping (pass --force to retrain)")
        return
    # ============================================================
    # Load data
    # ============================================================
    print(f"\n{'='*70}")
    print("Loading data...")
    print(f"{'='*70}")

    print("Loading Train labels...", flush=True)
    df_train = pd.read_parquet(TRAIN_PARQUET, columns=["AC", "EC"])
    print(f"  {len(df_train)} rows, {df_train['EC'].nunique()} ECs", flush=True)

    print("Loading Valid labels...", flush=True)
    df_valid = pd.read_parquet(VALID_PARQUET, columns=["AC", "EC"])
    print(f"  {len(df_valid)} rows, {df_valid['EC'].nunique()} ECs", flush=True)

    print("\nLoading embeddings from batch pickles...", flush=True)
    train_emb_dict = load_embeddings_from_batches(TRAIN_BATCH_DIR, LAYER_KEY)
    valid_emb_dict = load_embeddings_from_batches(VALID_BATCH_DIR, LAYER_KEY)

    # ============================================================
    # Build hierarchical labels for all 4 EC levels
    # ============================================================
    print("\nBuilding hierarchical labels...", flush=True)

    # Create label encoders for each EC level
    label_encoders = {}
    for level in [1, 2, 3, 4]:
        le = LabelEncoder()
        train_labels = df_train["EC"].apply(lambda x: ec_at_level(x, level)).values
        valid_labels = df_valid["EC"].apply(lambda x: ec_at_level(x, level)).values
        le.fit(np.concatenate([train_labels, valid_labels]))
        label_encoders[level] = le
        print(f"  Level {level}: {len(le.classes_)} classes")

    # Save label encoders
    with open(os.path.join(HIER_OUTPUT_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)

    # ============================================================
    # Build feature matrices
    # ============================================================
    print("\nBuilding feature matrices...", flush=True)
    t0 = time.time()

    # Train
    train_mask = df_train["AC"].isin(train_emb_dict)
    train_acs = df_train.loc[train_mask, "AC"].values
    ec_train_str = df_train.loc[train_mask, "EC"].values
    X_train = np.array([train_emb_dict[ac] for ac in train_acs], dtype=np.float32)

    # Encode labels at all 4 levels
    y_train = {}
    for level in [1, 2, 3, 4]:
        labels = [ec_at_level(ec, level) for ec in ec_train_str]
        y_train[level] = label_encoders[level].transform(labels)

    # Valid
    valid_mask = df_valid["AC"].isin(valid_emb_dict)
    valid_acs = df_valid.loc[valid_mask, "AC"].values
    ec_valid_str = df_valid.loc[valid_mask, "EC"].values
    X_valid = np.array([valid_emb_dict[ac] for ac in valid_acs], dtype=np.float32)

    y_valid = {}
    for level in [1, 2, 3, 4]:
        labels = [ec_at_level(ec, level) for ec in ec_valid_str]
        y_valid[level] = label_encoders[level].transform(labels)

    input_dim = X_train.shape[1]
    n_classes = {level: len(label_encoders[level].classes_) for level in [1, 2, 3, 4]}
    print(f"  X_train: {X_train.shape}  X_valid: {X_valid.shape}")
    print(f"  Classes: EC1={n_classes[1]}, EC2={n_classes[2]}, EC3={n_classes[3]}, EC4={n_classes[4]}")
    print(f"  ({time.time()-t0:.1f}s)", flush=True)

    del df_train, df_valid, train_emb_dict, valid_emb_dict
    gc.collect()

    # ============================================================
    # Data — GPU-resident (multi-label: X, y1, y2, y3, y4);
    # perm-sliced batches instead of DataLoader (see wide-MLP section)
    # ============================================================
    X_tr_t = torch.from_numpy(X_train).float().to(device)
    y_tr_t = {lv: torch.from_numpy(y_train[lv]).long().to(device) for lv in [1, 2, 3, 4]}
    X_va_t = torch.from_numpy(X_valid).float().to(device)
    y_va_t = {lv: torch.from_numpy(y_valid[lv]).long().to(device) for lv in [1, 2, 3, 4]}
    n_train = len(y_tr_t[1])
    n_valid = len(y_va_t[1])
    del X_train, X_valid  # (y dicts kept: small, used for len() in results)
    gc.collect()

    current_batch_size = MLP_BATCH_SIZE

    # ============================================================
    # Build model and losses
    # ============================================================
    print(f"\n{'='*70}")
    print(f"Training Hierarchical MLP")
    print(f"  Backbone: {input_dim} → 2048 → 1024")
    print(f"  Head1 (EC1): 1024 → 256 → {n_classes[1]}")
    print(f"  Head2 (EC2): 1024+{n_classes[1]} → 256 → {n_classes[2]}")
    print(f"  Head3 (EC3): 1024+{n_classes[2]} → 256 → {n_classes[3]}")
    print(f"  Head4 (EC4): 1024+{n_classes[3]} → 512 → {n_classes[4]}")
    print(f"  Epochs: {MLP_EPOCHS}  |  Focal gamma: {FOCAL_GAMMA}")
    print(f"  LR: {MLP_LR}  |  Batch: {MLP_BATCH_SIZE}  |  Warmup: {WARMUP_EPOCHS}")
    print(f"{'='*70}", flush=True)

    model = HierarchicalEC_MLP(input_dim, n_classes[1], n_classes[2],
                                n_classes[3], n_classes[4], dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}", flush=True)

    # Focal loss for each level (no class weights needed — focal loss handles imbalance)
    criteria = {
        level: FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=0.1).to(device)
        for level in [1, 2, 3, 4]
    }

    # Loss weights: higher weight for finer-grained levels (where we need the most help)
    loss_weights = {1: 0.5, 2: 1.0, 3: 1.5, 4: 3.0}

    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MLP_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS])

    # ============================================================
    # Training loop
    # ============================================================
    best_val_acc4 = 0.0
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [],
               "val_acc1": [], "val_acc2": [], "val_acc3": [], "val_acc4": []}
    oom_reduced = False

    def train_one_epoch(bs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        perm = torch.randperm(n_train, device=device)
        n_batches = (n_train + bs - 1) // bs
        for bi in range(n_batches):
            start = bi * bs
            end = min(start + bs, n_train)
            idx = perm[start:end]
            xb = X_tr_t[idx]
            targets = {lv: y_tr_t[lv][idx] for lv in [1, 2, 3, 4]}
            optimizer.zero_grad()
            l1, l2, l3, l4 = model(xb)
            loss = (loss_weights[1] * criteria[1](l1, targets[1]) +
                    loss_weights[2] * criteria[2](l2, targets[2]) +
                    loss_weights[3] * criteria[3](l3, targets[3]) +
                    loss_weights[4] * criteria[4](l4, targets[4]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        return total_loss / max(n_samples, 1)

    t0 = time.time()
    for epoch in range(1, MLP_EPOCHS + 1):
        # --- Train with OOM handling ---
        try:
            train_loss = train_one_epoch(current_batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_bs = max(256, current_batch_size // 2)
            print(f"\n  ⚠ OOM at epoch {epoch}, reducing batch {current_batch_size}→{new_bs}", flush=True)
            current_batch_size = new_bs
            oom_reduced = True
            optimizer.state.clear()
            train_loss = train_one_epoch(current_batch_size)

        # OOM recovery
        if oom_reduced and current_batch_size < MLP_BATCH_SIZE and epoch > 1:
            try:
                idx_test = torch.arange(MLP_BATCH_SIZE, device=device)
                with torch.no_grad():
                    model(X_tr_t[idx_test])
                del idx_test
                torch.cuda.empty_cache()
                current_batch_size = MLP_BATCH_SIZE
                oom_reduced = False
                print(f"  ✓ Recovered batch_size to {MLP_BATCH_SIZE}", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = {1: 0, 2: 0, 3: 0, 4: 0}
        total = 0
        n_val_batches = (n_valid + current_batch_size - 1) // current_batch_size
        with torch.no_grad():
            for bi in range(n_val_batches):
                start = bi * current_batch_size
                end = min(start + current_batch_size, n_valid)
                xb = X_va_t[start:end]
                targets = {lv: y_va_t[lv][start:end] for lv in [1, 2, 3, 4]}
                l1, l2, l3, l4 = model(xb)
                logits = {1: l1, 2: l2, 3: l3, 4: l4}
                loss = sum(loss_weights[lv] * criteria[lv](logits[lv], targets[lv]) for lv in [1,2,3,4])
                val_loss += loss.item() * xb.size(0)
                for lv in [1, 2, 3, 4]:
                    correct[lv] += (logits[lv].argmax(1) == targets[lv]).sum().item()
                total += xb.size(0)

        val_loss /= n_valid
        val_accs = {lv: correct[lv] / total for lv in [1, 2, 3, 4]}

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for lv in [1, 2, 3, 4]:
            history[f"val_acc{lv}"].append(val_accs[lv])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{MLP_EPOCHS}  loss={train_loss:.4f}/{val_loss:.4f}  "
                  f"EC1={val_accs[1]:.4f} EC2={val_accs[2]:.4f} "
                  f"EC3={val_accs[3]:.4f} EC4={val_accs[4]:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  bs={current_batch_size}", flush=True)

        if val_accs[4] > best_val_acc4:
            best_val_acc4 = val_accs[4]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.1f}s")
    print(f"  Best epoch: {best_epoch} (EC4 val_acc={best_val_acc4:.4f})", flush=True)

    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, os.path.join(HIER_OUTPUT_DIR, "hierarchical_mlp.pt"))

    # ============================================================
    # Final evaluation
    # ============================================================
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")

    # Predict on validation
    all_preds = {1: [], 2: [], 3: [], 4: []}
    n_val_batches = (n_valid + MLP_BATCH_SIZE - 1) // MLP_BATCH_SIZE
    with torch.no_grad():
        for bi in range(n_val_batches):
            start = bi * MLP_BATCH_SIZE
            end = min(start + MLP_BATCH_SIZE, n_valid)
            l1, l2, l3, l4 = model(X_va_t[start:end])
            for lv, logits in zip([1,2,3,4], [l1,l2,l3,l4]):
                all_preds[lv].append(logits.argmax(1).cpu().numpy())

    results = {"train_time": train_time, "best_epoch": best_epoch, "n_train": len(y_train[4]),
               "n_valid": len(y_valid[4]), "input_dim": input_dim, "model_params": n_params}

    for level in [1, 2, 3, 4]:
        preds = np.concatenate(all_preds[level])
        pred_labels = label_encoders[level].inverse_transform(preds)
        true_labels = [ec_at_level(ec, level) for ec in ec_valid_str]

        acc = accuracy_score(true_labels, pred_labels)
        f1_mac = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        f1_wt = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

        results[f"level_{level}_accuracy"] = acc
        results[f"level_{level}_f1_macro"] = f1_mac
        results[f"level_{level}_f1_weighted"] = f1_wt
        print(f"  Level {level}: Accuracy={acc:.4f}  F1_macro={f1_mac:.4f}  F1_weighted={f1_wt:.4f}")

    # ============================================================
    # Save results
    # ============================================================
    results_json = {}
    for k, v in results.items():
        if isinstance(v, (float, np.floating)):
            results_json[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            results_json[k] = int(v)
        else:
            results_json[k] = v

    with open(os.path.join(HIER_OUTPUT_DIR, "results_summary.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    with open(os.path.join(HIER_OUTPUT_DIR, "results_summary.pkl"), "wb") as f:
        pickle.dump(results, f)
    np.savez(os.path.join(HIER_OUTPUT_DIR, "history.npz"), **{k: np.array(v) for k, v in history.items()})

    print(f"\nResults saved to: {HIER_OUTPUT_DIR}")

    # ============================================================
    # Plots
    # ============================================================
    # Loss curves
    fig, ax1 = plt.subplots(figsize=(14, 5))
    epochs_range = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#2196F3", lw=2, alpha=0.8)
    ax1.plot(epochs_range, history["val_loss"], label="Valid Loss", color="#E91E63", lw=2, alpha=0.8)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best ({best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    colors_acc = ["#81C784", "#4CAF50", "#2E7D32", "#1B5E20"]
    for i, lv in enumerate([1, 2, 3, 4]):
        ax2.plot(epochs_range, history[f"val_acc{lv}"], label=f"EC{lv} Acc",
                 color=colors_acc[i], lw=1.5, alpha=0.8)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax1.set_title("Hierarchical MLP — Loss & EC-Level Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(HIER_OUTPUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: loss_curves.png")

    # Accuracy by level
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [1, 2, 3, 4]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    accuracies = [results[f"level_{lv}_accuracy"] for lv in levels]
    f1_macros = [results[f"level_{lv}_f1_macro"] for lv in levels]
    f1_weights = [results[f"level_{lv}_f1_weighted"] for lv in levels]

    x = np.arange(len(levels))
    width = 0.25
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x, f1_macros, width, label="F1 Macro", color=colors, alpha=0.55)
    bars3 = ax.bar(x + width, f1_weights, width, label="F1 Weighted", color=colors, alpha=0.35,
                   edgecolor=colors, linewidth=2)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"EC Level {lv}" for lv in levels], fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Hierarchical MLP — EC Prediction by Level\n(Evo2 Layer 18 Mean Token, Benchmark 2)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HIER_OUTPUT_DIR, "accuracy_by_level.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: accuracy_by_level.png")

    # ============================================================
    # Final summary
    # ============================================================
    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    print(f"Model:    Hierarchical MLP ({n_params:,} params)")
    print(f"Feature:  Evo2 7B Layer 18 Mean-Token-Pooled (4096-dim)")
    print(f"Train:    {len(y_train[4])} samples")
    print(f"Valid:    {len(y_valid[4])} samples")
    for lv in [1, 2, 3, 4]:
        print(f"  EC{lv}: Acc={results[f'level_{lv}_accuracy']:.4f} "
              f"F1m={results[f'level_{lv}_f1_macro']:.4f} "
              f"F1w={results[f'level_{lv}_f1_weighted']:.4f}")
    print(f"Best epoch: {best_epoch} | Training time: {train_time:.1f}s")
    print(f"Output: {HIER_OUTPUT_DIR}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train benchmark-2 layer-18 MLP classifiers (wide MLP and/or hierarchical MLP).")
    parser.add_argument("--experiment", choices=["mlp", "hierarchical", "both"], default="both",
                        help="Which experiment to run (default: both)")
    args = parser.parse_args()

    device = get_device(require_cuda=True)

    if args.experiment in ("mlp", "both"):
        run_mlp(device)
    if args.experiment in ("hierarchical", "both"):
        run_hierarchical(device)


if __name__ == "__main__":
    main()
