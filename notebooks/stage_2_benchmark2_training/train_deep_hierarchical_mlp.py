#!/usr/bin/env python3
"""
Deep Hierarchical MLP for EC prediction from Evo2 multi-layer embeddings.

Features:
  - Deeper backbone: 3 hidden layers with residual connections
  - Multi-layer embeddings: concatenates layers 13, 17, 18, 20 (16,384-dim)
    Falls back to layer 18 only (4,096-dim) if other layers not yet extracted
  - Mixup augmentation: interpolates training samples for regularization
  - Cosine classifier for EC4: normalizes weights & features for better large-class perf
  - 50 epochs

Architecture (scales with input dim):
  Single layer:  4096 → 2048(+res) → 1024(+res) → 512 → 4 cascaded heads
  Multi layer:  16384 → 4096(+res) → 2048(+res) → 1024 → 4 cascaded heads
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    BENCH2_DIR, MODELS_DIR, get_device,
    FocalLoss, ResBlock, CosineClassifier, DeepHierarchicalEC_MLP,
    load_b2_layer_dicts, ec_at_level,
)

import os, gc, time, pickle, json
import ctypes
# glibc malloc: with many arenas, freed 64 KB heap chunks (numpy arrays below
# the mmap threshold) are not returned to the OS — RSS climbs to the
# historical high-water mark (~202 GB > 200 GiB job cgroup → OOM kill).
# One arena + malloc_trim after frees keeps RSS near live data (~110 GB).
os.environ.setdefault("MALLOC_ARENA_MAX", "1")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Paths & constants
# ============================================================
TRAIN_PARQUET = BENCH2_DIR / "Train.parquet"
VALID_PARQUET = BENCH2_DIR / "Valid.parquet"
OUTPUT_DIR = MODELS_DIR / "benchmark_2" / "deep_hierarchical_mlp_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_FORCE = "--force" in sys.argv
if not _FORCE and (OUTPUT_DIR / "model.pt").exists() and (OUTPUT_DIR / "results_summary.json").exists():
    print(f"[skip] {OUTPUT_DIR} already trained — skipping (pass --force to retrain)")
    raise SystemExit(0)

DEVICE = get_device(require_cuda=True)

MLP_EPOCHS = 50
MLP_BATCH_SIZE = 4096     # Large batch — balanced for GPU memory with model + optimizer overhead
MLP_LR = 1e-3             # Scale LR up with batch size (linear scaling rule)
MLP_WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 5
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2  # Mixup interpolation strength

# Layer configuration: try multi-layer, fall back to single
MULTI_LAYERS = [13, 17, 18, 20]  # Best from benchmark_4
SINGLE_LAYER = "blocks.18"


def _malloc_trim():
    """Ask glibc to return freed heap memory to the OS (RSS accounting)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_unique_features(df, layer_dicts, layers):
    """Build deduplicated feature matrix + row index (memory-safe).

    Train/Valid contain ~2.3x duplicate ACs. Storing one 16384-dim row per
    unique AC (33.5 GB train / 16.7 GB valid) instead of one row per sample
    (75.8 / 68.5 GB) keeps peak RSS under the 200 GiB job cgroup. Every
    original row stays a sample: row_idx maps each row (duplicates included)
    to its unique-AC position, so training/eval data is mathematically
    identical to the per-row layout.

    Returns (X_unique, row_idx, mask): X_unique (n_unique, dim) float32;
    row_idx (n_kept,) int64; mask (n_rows,) bool (AC present in ALL layers).
    """
    dim = len(layers) * 4096

    # Keys present in ALL available layers (same restriction as the
    # former isin(concat_dict) mask)
    common_keys = set(layer_dicts[layers[0]].keys())
    for l in layers[1:]:
        common_keys &= set(layer_dicts[l].keys())
    print(f"  Common keys across all layers: {len(common_keys)}")

    # Unique ACs in first-occurrence order
    uniq = [ac for ac in df["AC"].unique() if ac in common_keys]
    n_uniq = len(uniq)
    print(f"  Building X_unique: {n_uniq} unique ACs x {dim} dim ...", flush=True)
    X = np.empty((n_uniq, dim), dtype=np.float32)
    for j, l in enumerate(layers):
        ld = layer_dicts[l]
        X[:, j * 4096:(j + 1) * 4096] = np.stack([ld[ac] for ac in uniq])

    # Row -> unique-AC position (-1 when the AC is missing)
    pos = {ac: i for i, ac in enumerate(uniq)}
    full_idx = df["AC"].map(pos).to_numpy(dtype=np.int64, na_value=-1)
    mask = full_idx >= 0
    row_idx = full_idx[mask]
    print(f"  Kept {int(mask.sum())}/{len(df)} rows "
          f"({n_uniq} unique ACs, "
          f"{1 - n_uniq / max(mask.sum(), 1):.1%} dedup)", flush=True)
    return X, row_idx, mask


def mixup_data(x, targets_list, alpha=0.2):
    """Mixup: interpolate pairs of training samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    targets_a = targets_list
    targets_b = [t[index] for t in targets_list]
    return mixed_x, targets_a, targets_b, lam


# ============================================================
# Load data
# ============================================================
print(f"\n{'='*70}")
print("Loading data...")
print(f"{'='*70}")

df_train = pd.read_parquet(TRAIN_PARQUET, columns=["AC", "EC"])
print(f"Train: {len(df_train)} rows, {df_train['EC'].nunique()} ECs")
df_valid = pd.read_parquet(VALID_PARQUET, columns=["AC", "EC"])
print(f"Valid: {len(df_valid)} rows, {df_valid['EC'].nunique()} ECs")

# Load embeddings (multi-layer if available, single-layer fallback).
# Memory-safe ordering under the 200 GiB job cgroup: build X_train from the
# train dict, free it, THEN load the valid dict and build X_valid.
# Peak stays ~165 GB (both X matrices + one dict) instead of ~200 GB+
# (both dicts + both X matrices) which was OOM-killed by the memcg.
print("\nLoading train embeddings...")
train_layer_dicts, layers_used = load_b2_layer_dicts("train")
emb_dim = len(layers_used) * 4096
print(f"\nUsing layers: {layers_used} → {emb_dim}-dim input")

X_train, train_row_idx, train_mask = build_unique_features(
    df_train, train_layer_dicts, layers_used)
del train_layer_dicts
gc.collect()
_malloc_trim()

# ============================================================
# Build hierarchical labels
# ============================================================
label_encoders = {}
for level in [1, 2, 3, 4]:
    le = LabelEncoder()
    tl = df_train["EC"].apply(lambda x: ec_at_level(x, level)).values
    vl = df_valid["EC"].apply(lambda x: ec_at_level(x, level)).values
    le.fit(np.concatenate([tl, vl]))
    label_encoders[level] = le
    print(f"  Level {level}: {len(le.classes_)} classes")

with open(os.path.join(OUTPUT_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

# Build label arrays (kept rows: AC present in ALL loaded layers)
ec_train_str = df_train.loc[train_mask, "EC"].values

y_train = {}
for level in [1, 2, 3, 4]:
    y_train[level] = label_encoders[level].transform(
        [ec_at_level(ec, level) for ec in ec_train_str])

print(f"\nLoading valid embeddings...")
valid_layer_dicts, _ = load_b2_layer_dicts("valid")
X_valid, valid_row_idx, valid_mask = build_unique_features(
    df_valid, valid_layer_dicts, layers_used)
del valid_layer_dicts
gc.collect()
_malloc_trim()
ec_valid_str = df_valid.loc[valid_mask, "EC"].values

y_valid = {}
for level in [1, 2, 3, 4]:
    y_valid[level] = label_encoders[level].transform(
        [ec_at_level(ec, level) for ec in ec_valid_str])

input_dim = X_train.shape[1]
n_classes = {lv: len(label_encoders[lv].classes_) for lv in [1, 2, 3, 4]}
print(f"\nX_train: {X_train.shape}  X_valid: {X_valid.shape}  (unique-AC storage)")
print(f"Samples: train={len(train_row_idx)}  valid={len(valid_row_idx)}")
print(f"Classes: EC1={n_classes[1]}, EC2={n_classes[2]}, EC3={n_classes[3]}, EC4={n_classes[4]}")

del df_train, df_valid
gc.collect()

# ============================================================
# Data tensors live on GPU — per-batch CPU gather + H2D transfer was
# profiled at ~26 ms/step (dominant epoch cost); HBM gathers are free.
# 33.5 GB train + valid fits the 143 GB H200 alongside the model.
# ============================================================
print("\nMoving data tensors to GPU (one-time copy; per-step CPU gather of 64 MB batches was ~26 ms/step)...", flush=True)
train_X_gpu = torch.from_numpy(X_train).to(DEVICE)
train_idx_gpu = torch.from_numpy(train_row_idx).to(DEVICE)
train_y1_gpu = torch.from_numpy(y_train[1]).to(DEVICE)
train_y2_gpu = torch.from_numpy(y_train[2]).to(DEVICE)
train_y3_gpu = torch.from_numpy(y_train[3]).to(DEVICE)
train_y4_gpu = torch.from_numpy(y_train[4]).to(DEVICE)

valid_X_gpu = torch.from_numpy(X_valid).to(DEVICE)
valid_idx_gpu = torch.from_numpy(valid_row_idx).to(DEVICE)
valid_y1_gpu = torch.from_numpy(y_valid[1]).to(DEVICE)
valid_y2_gpu = torch.from_numpy(y_valid[2]).to(DEVICE)
valid_y3_gpu = torch.from_numpy(y_valid[3]).to(DEVICE)
valid_y4_gpu = torch.from_numpy(y_valid[4]).to(DEVICE)

# Free numpy arrays
del X_train, X_valid, train_row_idx, valid_row_idx, y_train, y_valid
gc.collect()

n_train = len(train_idx_gpu)
n_valid = len(valid_idx_gpu)

print(f"  Train: {n_train} samples, Valid: {n_valid} samples")
if torch.cuda.is_available():
    vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    print(f"  GPU memory available for model/optimizer: {vram_free:.1f} GB", flush=True)

current_bs = MLP_BATCH_SIZE

# ============================================================
# Model
# ============================================================
print(f"\n{'='*70}")
print("Training Deep Hierarchical MLP")
h1 = max(2048, input_dim // 2)
h2 = max(1024, h1 // 2)
h3 = max(512, h2 // 2)
print(f"  Input: {input_dim}-dim (layers {layers_used})")
print(f"  Backbone: {input_dim} → {h1}(+res) → {h2}(+res) → {h3}")
print(f"  Heads: EC1({n_classes[1]}) EC2({n_classes[2]}) EC3({n_classes[3]}) EC4({n_classes[4]},cosine)")
print(f"  Epochs: {MLP_EPOCHS}  |  Mixup α={MIXUP_ALPHA}  |  Focal γ={FOCAL_GAMMA}")
print(f"{'='*70}", flush=True)

model = DeepHierarchicalEC_MLP(input_dim, n_classes[1], n_classes[2],
                                n_classes[3], n_classes[4], dropout=0.3).to(DEVICE)

# Compile model for kernel fusion (PyTorch 2.0+) — significant speedup
try:
    model = torch.compile(model)
    print(f"  ✓ torch.compile() enabled — first epoch will be slow (compiling)", flush=True)
except Exception as e:
    print(f"  torch.compile() not available: {e}", flush=True)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

criteria = {lv: FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=0.1) for lv in [1,2,3,4]}
loss_weights = {1: 0.5, 2: 1.0, 3: 1.5, 4: 3.0}

optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MLP_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
warmup_sched = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_EPOCHS)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

# ============================================================
# Training
# ============================================================
best_val_acc4 = 0.0
best_epoch = 0
best_state = None
history = {"train_loss": [], "val_loss": [],
           "val_acc1": [], "val_acc2": [], "val_acc3": [], "val_acc4": []}

t0 = time.time()
for epoch in range(1, MLP_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    n_samples = 0

    # Shuffle indices on GPU (perm + index gathers all stay in HBM)
    perm = torch.randperm(n_train, device=DEVICE)
    n_batches = (n_train + current_bs - 1) // current_bs

    try:
        for bi in range(n_batches):
            start = bi * current_bs
            end = min(start + current_bs, n_train)
            idx = perm[start:end]

            # Gather batch on GPU — no per-step CPU→GPU transfer.
            # idx = original-row indices (label tensors are in original row
            # order); train_idx_gpu maps them onto the dedup'd unique-AC X rows.
            # (Labels must NOT be indexed by the mapped X rows — that pairs
            # each unique embedding with the wrong row's label.)
            xb = train_X_gpu[train_idx_gpu[idx]]
            targets = [train_y1_gpu[idx], train_y2_gpu[idx], train_y3_gpu[idx], train_y4_gpu[idx]]

            # Mixup augmentation
            if MIXUP_ALPHA > 0 and epoch > WARMUP_EPOCHS:
                mixed_x, targets_a, targets_b, lam = mixup_data(xb, targets, MIXUP_ALPHA)
                l1, l2, l3, l4 = model(mixed_x)
                logits_list = [l1, l2, l3, l4]
                loss = 0
                for lv_idx, lv in enumerate([1, 2, 3, 4]):
                    loss_a = criteria[lv](logits_list[lv_idx], targets_a[lv_idx])
                    loss_b = criteria[lv](logits_list[lv_idx], targets_b[lv_idx])
                    loss += loss_weights[lv] * (lam * loss_a + (1 - lam) * loss_b)
            else:
                l1, l2, l3, l4 = model(xb)
                loss = sum(loss_weights[lv] * criteria[lv](logits, t)
                          for lv, logits, t in zip([1,2,3,4], [l1,l2,l3,l4], targets))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        new_bs = max(256, current_bs // 2)
        print(f"\n  ⚠ OOM at epoch {epoch}, reducing batch {current_bs}→{new_bs}", flush=True)
        current_bs = new_bs
        # Reset optimizer state — OOM during optimizer.step() can leave
        # partially-initialized state dicts (has 'exp_avg' but not 'exp_avg_sq'),
        # which causes KeyError on the next step.
        optimizer.state.clear()
        continue

    train_loss /= max(n_samples, 1)

    # Validate — direct GPU tensor slicing, no DataLoader
    model.eval()
    val_loss = 0.0
    correct = {1: 0, 2: 0, 3: 0, 4: 0}
    total = 0
    n_val_batches = (n_valid + current_bs - 1) // current_bs

    with torch.no_grad():
        for bi in range(n_val_batches):
            start = bi * current_bs
            end = min(start + current_bs, n_valid)
            xb = valid_X_gpu[valid_idx_gpu[start:end]]
            targets = [valid_y1_gpu[start:end], valid_y2_gpu[start:end],
                       valid_y3_gpu[start:end], valid_y4_gpu[start:end]]
            l1, l2, l3, l4 = model(xb)
            logits = [l1, l2, l3, l4]
            loss = sum(loss_weights[lv] * criteria[lv](logits[i], targets[i])
                      for i, lv in enumerate([1,2,3,4]))
            val_loss += loss.item() * xb.size(0)
            for i, lv in enumerate([1, 2, 3, 4]):
                correct[lv] += (logits[i].argmax(1) == targets[i]).sum().item()
            total += xb.size(0)

    val_loss /= n_valid
    val_accs = {lv: correct[lv] / total for lv in [1, 2, 3, 4]}
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    for lv in [1,2,3,4]:
        history[f"val_acc{lv}"].append(val_accs[lv])

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{MLP_EPOCHS}  loss={train_loss:.4f}/{val_loss:.4f}  "
              f"EC1={val_accs[1]:.4f} EC2={val_accs[2]:.4f} "
              f"EC3={val_accs[3]:.4f} EC4={val_accs[4]:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

    if val_accs[4] > best_val_acc4:
        best_val_acc4 = val_accs[4]
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

train_time = time.time() - t0
print(f"\n  Done in {train_time:.1f}s — Best epoch {best_epoch} (EC4={best_val_acc4:.4f})")

model.load_state_dict(best_state)
model.eval()
torch.save(best_state, os.path.join(OUTPUT_DIR, "model.pt"))

# ============================================================
# Evaluation
# ============================================================
print(f"\n{'='*70}")
print("EVALUATION")
print(f"{'='*70}")

all_preds = {1: [], 2: [], 3: [], 4: []}
with torch.no_grad():
    n_val_batches = (n_valid + current_bs - 1) // current_bs
    for bi in range(n_val_batches):
        start = bi * current_bs
        end = min(start + current_bs, n_valid)
        xb = valid_X_gpu[valid_idx_gpu[start:end]]
        l1, l2, l3, l4 = model(xb)
        for lv, logits in zip([1,2,3,4], [l1,l2,l3,l4]):
            all_preds[lv].append(logits.argmax(1).cpu().numpy())

results = {"train_time": train_time, "best_epoch": best_epoch,
           "layers_used": layers_used, "input_dim": input_dim,
           "n_train": n_train, "n_valid": n_valid,
           "mixup_alpha": MIXUP_ALPHA, "focal_gamma": FOCAL_GAMMA}

for level in [1, 2, 3, 4]:
    preds = np.concatenate(all_preds[level])
    pred_labels = label_encoders[level].inverse_transform(preds)
    true_labels = [ec_at_level(ec, level) for ec in ec_valid_str]
    acc = accuracy_score(true_labels, pred_labels)
    f1m = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1w = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    results[f"level_{level}_accuracy"] = acc
    results[f"level_{level}_f1_macro"] = f1m
    results[f"level_{level}_f1_weighted"] = f1w
    print(f"  EC{level}: Acc={acc:.4f}  F1m={f1m:.4f}  F1w={f1w:.4f}")

# Save
results_json = {k: (float(v) if isinstance(v, (float, np.floating)) else
                     int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in results.items()}
with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as f:
    json.dump(results_json, f, indent=2)
np.savez(os.path.join(OUTPUT_DIR, "history.npz"), **{k: np.array(v) for k, v in history.items()})

# Plot
fig, ax1 = plt.subplots(figsize=(14, 5))
ep = np.arange(1, len(history["train_loss"]) + 1)
ax1.plot(ep, history["train_loss"], label="Train Loss", color="#2196F3", lw=2)
ax1.plot(ep, history["val_loss"], label="Valid Loss", color="#E91E63", lw=2)
ax1.axvline(x=best_epoch, color="gray", ls="--", alpha=0.5, label=f"Best ({best_epoch})")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
for lv, c in zip([1,2,3,4], ["#81C784","#4CAF50","#2E7D32","#1B5E20"]):
    ax2.plot(ep, history[f"val_acc{lv}"], label=f"EC{lv}", color=c, lw=1.5)
ax2.set_ylabel("Accuracy")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, fontsize=9, loc="center right")
ax1.set_title(f"Deep Hierarchical MLP — Layers {layers_used}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nResults saved to {OUTPUT_DIR}")
print(f"Layers: {layers_used} | Input dim: {input_dim} | Best EC4: {best_val_acc4:.4f}")

