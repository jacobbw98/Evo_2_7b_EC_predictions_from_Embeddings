#!/usr/bin/env python3
"""
Evaluate all 3 trained MLP models with error bars by sequence length group.

Loads the already-trained models from:
  - mlp_results/mlp_layer_18.pt
  - hierarchical_mlp_results/hierarchical_mlp.pt
  - deep_hierarchical_mlp_results/model.pt

Splits validation set into 3 equal-sized groups by sequence length:
  Short, Medium, Long

Generates accuracy_by_level.png with error bars for each model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    BENCH2_DIR, MODELS_DIR, get_device,
    EC_MLP, HierarchicalEC_MLP, DeepHierarchicalEC_MLP, ResBlock, CosineClassifier,
    load_embeddings_from_batches, load_b2_layer_dicts, ec_at_level,
)

import os, gc, pickle, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_PARQUET = BENCH2_DIR / "Valid.parquet"
VALID_BATCH_DIR = BENCH2_DIR / "valid_embedding_batches"
DEVICE = get_device(require_cuda=True)
assert DEVICE.type == "cuda", "CUDA not available — this script requires a GPU"
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
BATCH_SIZE = 4096

# ============================================================
# Utility functions
# ============================================================
def load_multilayer_valid():
    """Load and concatenate multi-layer embeddings for validation set."""
    layer_dicts, available = load_b2_layer_dicts("valid")
    if len(available) == 1:
        return layer_dicts[18], 4096, [18]
    common = set(layer_dicts[available[0]].keys())
    for l in available[1:]:
        common &= set(layer_dicts[l].keys())
    concat = {k: np.concatenate([layer_dicts[l][k] for l in available]) for k in common}
    return concat, len(available) * 4096, available

# ============================================================
# Load validation labels + sequence lengths
# ============================================================
print(f"\n{'='*70}")
print("Loading validation data...")
print(f"{'='*70}")

df_valid = pd.read_parquet(VALID_PARQUET, columns=["AC", "EC", "Sequence"])
print(f"  {len(df_valid)} rows")

# Compute sequence lengths and assign groups
seq_lens = df_valid["Sequence"].str.len().values
sorted_indices = np.argsort(seq_lens)
n = len(sorted_indices)
t1, t2 = n // 3, 2 * n // 3

# Map each row to a group (0=short, 1=medium, 2=long)
group_assignments = np.empty(n, dtype=np.int32)
group_assignments[sorted_indices[:t1]] = 0
group_assignments[sorted_indices[t1:t2]] = 1
group_assignments[sorted_indices[t2:]] = 2

group_names = ["Short", "Medium", "Long"]
sorted_lens = seq_lens[sorted_indices]
print(f"  Short:  n={t1}, max_len={sorted_lens[t1-1]}")
print(f"  Medium: n={t2-t1}, max_len={sorted_lens[t2-1]}")
print(f"  Long:   n={n-t2}, max_len={sorted_lens[-1]}")

# Drop Sequence column to save memory
del seq_lens, sorted_lens
df_valid.drop(columns=["Sequence"], inplace=True)
gc.collect()

# ============================================================
# Load embeddings
# ============================================================
print("\nLoading single-layer (18) embeddings...")
single_emb_dict = load_embeddings_from_batches(VALID_BATCH_DIR, "blocks.18")
print(f"  {len(single_emb_dict)} embeddings")

print("\nLoading multi-layer embeddings...")
multi_emb_dict, multi_dim, layers_used = load_multilayer_valid()
print(f"  {len(multi_emb_dict)} embeddings, {multi_dim}-dim, layers={layers_used}")

# ============================================================
# Build aligned arrays
# ============================================================
print("\nBuilding feature matrices...")

# Single-layer (used by MLP and Hierarchical MLP)
valid_mask_single = df_valid["AC"].isin(single_emb_dict)
valid_acs_single = df_valid.loc[valid_mask_single, "AC"].values
ec_valid_str_single = df_valid.loc[valid_mask_single, "EC"].values
X_valid_single = np.array([single_emb_dict[ac] for ac in valid_acs_single], dtype=np.float32)
groups_single = group_assignments[valid_mask_single.values]
print(f"  Single-layer: {X_valid_single.shape}")

# Multi-layer (used by Deep Hierarchical MLP)
valid_mask_multi = df_valid["AC"].isin(multi_emb_dict)
valid_acs_multi = df_valid.loc[valid_mask_multi, "AC"].values
ec_valid_str_multi = df_valid.loc[valid_mask_multi, "EC"].values
X_valid_multi = np.array([multi_emb_dict[ac] for ac in valid_acs_multi], dtype=np.float32)
groups_multi = group_assignments[valid_mask_multi.values]
print(f"  Multi-layer:  {X_valid_multi.shape}")

del single_emb_dict, multi_emb_dict, df_valid, group_assignments
gc.collect()

# ============================================================
# Helper: predict in batches
# ============================================================
def predict_batched(model, X_np, is_hierarchical=False):
    """Run inference, return predictions. For hierarchical models returns dict of level->preds."""
    model.eval()
    X_t = torch.from_numpy(X_np).float()
    n = X_t.shape[0]
    if is_hierarchical:
        all_preds = {1: [], 2: [], 3: [], 4: []}
    else:
        all_preds = []

    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            xb = X_t[i:i+BATCH_SIZE].to(DEVICE)
            if is_hierarchical:
                l1, l2, l3, l4 = model(xb)
                for lv, logits in zip([1,2,3,4], [l1,l2,l3,l4]):
                    all_preds[lv].append(logits.argmax(1).cpu().numpy())
            else:
                all_preds.append(model(xb).argmax(1).cpu().numpy())

    if is_hierarchical:
        return {lv: np.concatenate(v) for lv, v in all_preds.items()}
    return np.concatenate(all_preds)

# ============================================================
# Helper: compute metrics per group
# ============================================================
def compute_metrics_by_group(ec_true_str, ec_pred_str_or_dict, groups, levels, is_hierarchical=False):
    """
    Returns dict: {level: {"overall": {acc,f1m,f1w}, "groups": [{acc,f1m,f1w}, ...]}}
    For non-hierarchical: ec_pred_str_or_dict is an array of predicted EC strings.
    For hierarchical: it's a dict {level: array_of_predicted_EC_strings}.
    """
    results = {}
    for level in levels:
        true_lv = np.array([ec_at_level(ec, level) for ec in ec_true_str])
        if is_hierarchical:
            pred_lv = np.array([ec_at_level(ec, level) for ec in ec_pred_str_or_dict[level]])
        else:
            pred_lv = np.array([ec_at_level(ec, level) for ec in ec_pred_str_or_dict])

        # Overall
        acc = accuracy_score(true_lv, pred_lv)
        f1m = f1_score(true_lv, pred_lv, average="macro", zero_division=0)
        f1w = f1_score(true_lv, pred_lv, average="weighted", zero_division=0)

        # Per group
        group_metrics = []
        for g in range(3):
            mask = groups == g
            if mask.sum() == 0:
                group_metrics.append({"acc": 0, "f1m": 0, "f1w": 0})
                continue
            ga = accuracy_score(true_lv[mask], pred_lv[mask])
            gf1m = f1_score(true_lv[mask], pred_lv[mask], average="macro", zero_division=0)
            gf1w = f1_score(true_lv[mask], pred_lv[mask], average="weighted", zero_division=0)
            group_metrics.append({"acc": ga, "f1m": gf1m, "f1w": gf1w})

        results[level] = {"overall": {"acc": acc, "f1m": f1m, "f1w": f1w},
                          "groups": group_metrics}
    return results

# ============================================================
# Helper: plot accuracy by level with error bars
# ============================================================
def plot_accuracy_by_level(metrics, title, output_path):
    """Plot bar chart with error bars (sample SD across Short/Medium/Long groups)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    levels = [1, 2, 3, 4]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    x = np.arange(len(levels))
    width = 0.25

    for metric_idx, (metric_key, metric_label) in enumerate(
            [("acc", "Accuracy"), ("f1m", "F1 Macro"), ("f1w", "F1 Weighted")]):
        vals = []
        sds = []
        for lv in levels:
            group_vals = [metrics[lv]["groups"][g][metric_key] for g in range(3)]
            vals.append(metrics[lv]["overall"][metric_key])
            sds.append(float(np.std(group_vals, ddof=1)) if len(group_vals) >= 2 else 0.0)

        offset = (metric_idx - 1) * width
        alpha = [0.85, 0.55, 0.35][metric_idx]
        bars = ax.bar(x + offset, vals, width, label=metric_label,
                      color=colors, alpha=alpha,
                      edgecolor=colors if metric_idx == 2 else None,
                      linewidth=2 if metric_idx == 2 else 0)

        # Error bars = sample SD (ddof=1) across Short/Medium/Long group values
        yerr_vals = sds

        ax.errorbar(x + offset, vals, yerr=yerr_vals,
                    fmt='none', ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"EC Level {lv}" for lv in levels], fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)


    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================
# 1. Flat MLP
# ============================================================
print(f"\n{'='*70}")
print("Evaluating: Flat MLP (layer 18)")
print(f"{'='*70}")

mlp_dir = MODELS_DIR / "benchmark_2" / "mlp_results"
with open(os.path.join(mlp_dir, "label_encoder.pkl"), "rb") as f:
    le_flat = pickle.load(f)

n_classes_flat = len(le_flat.classes_)
model_flat = EC_MLP(4096, n_classes_flat).to(DEVICE)
model_flat.load_state_dict(torch.load(os.path.join(mlp_dir, "mlp_layer_18.pt"),
                                       map_location=DEVICE, weights_only=True))
model_flat.eval()
print(f"  Loaded model: {n_classes_flat} classes")

preds_flat = predict_batched(model_flat, X_valid_single, is_hierarchical=False)
ec_pred_flat = le_flat.inverse_transform(preds_flat)

metrics_flat = compute_metrics_by_group(ec_valid_str_single, ec_pred_flat,
                                         groups_single, [1, 2, 3, 4])

for lv in [1, 2, 3, 4]:
    m = metrics_flat[lv]
    g = m["groups"]
    print(f"  EC{lv}: Acc={m['overall']['acc']:.4f}  "
          f"[S={g[0]['acc']:.4f} M={g[1]['acc']:.4f} L={g[2]['acc']:.4f}]")

plot_accuracy_by_level(
    metrics_flat,
    "Wide MLP EC Prediction — Layer 18 Mean Token Embeddings (Benchmark 2)",
    os.path.join(mlp_dir, "accuracy_by_level.png"))

del model_flat, preds_flat, ec_pred_flat
torch.cuda.empty_cache(); gc.collect()

# ============================================================
# 2. Hierarchical MLP
# ============================================================
print(f"\n{'='*70}")
print("Evaluating: Hierarchical MLP (layer 18)")
print(f"{'='*70}")

hier_dir = MODELS_DIR / "benchmark_2" / "hierarchical_mlp_results"
with open(os.path.join(hier_dir, "label_encoders.pkl"), "rb") as f:
    le_hier = pickle.load(f)

n_cls_hier = {lv: len(le_hier[lv].classes_) for lv in [1, 2, 3, 4]}
model_hier = HierarchicalEC_MLP(4096, n_cls_hier[1], n_cls_hier[2],
                                 n_cls_hier[3], n_cls_hier[4]).to(DEVICE)
model_hier.load_state_dict(torch.load(os.path.join(hier_dir, "hierarchical_mlp.pt"),
                                       map_location=DEVICE, weights_only=True))
model_hier.eval()
print(f"  Loaded model: EC classes = {n_cls_hier}")

preds_hier = predict_batched(model_hier, X_valid_single, is_hierarchical=True)
ec_pred_hier = {}
for lv in [1, 2, 3, 4]:
    ec_pred_hier[lv] = le_hier[lv].inverse_transform(preds_hier[lv])

metrics_hier = compute_metrics_by_group(ec_valid_str_single, ec_pred_hier,
                                         groups_single, [1, 2, 3, 4],
                                         is_hierarchical=True)

for lv in [1, 2, 3, 4]:
    m = metrics_hier[lv]
    g = m["groups"]
    print(f"  EC{lv}: Acc={m['overall']['acc']:.4f}  "
          f"[S={g[0]['acc']:.4f} M={g[1]['acc']:.4f} L={g[2]['acc']:.4f}]")

plot_accuracy_by_level(
    metrics_hier,
    "Hierarchical MLP — EC Prediction by Level (Evo2 Layer 18)",
    os.path.join(hier_dir, "accuracy_by_level.png"))

del model_hier, preds_hier, ec_pred_hier
torch.cuda.empty_cache(); gc.collect()

# Free single-layer data before loading multi-layer model
del X_valid_single
gc.collect()

# ============================================================
# 3. Deep Hierarchical MLP (train_deep_hierarchical_mlp.py)
# ============================================================
print(f"\n{'='*70}")
print(f"Evaluating: Deep Hierarchical MLP (layers {layers_used})")
print(f"{'='*70}")

deep_dir = MODELS_DIR / "benchmark_2" / "deep_hierarchical_mlp_results"
with open(os.path.join(deep_dir, "label_encoders.pkl"), "rb") as f:
    le_deep = pickle.load(f)

n_cls_deep = {lv: len(le_deep[lv].classes_) for lv in [1, 2, 3, 4]}
model_deep = DeepHierarchicalEC_MLP(multi_dim, n_cls_deep[1], n_cls_deep[2],
                                     n_cls_deep[3], n_cls_deep[4]).to(DEVICE)
state = torch.load(os.path.join(deep_dir, "model.pt"), map_location=DEVICE, weights_only=True)
# Handle torch.compile prefix
clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model_deep.load_state_dict(clean_state)
model_deep.eval()
print(f"  Loaded model: {multi_dim}-dim input, EC classes = {n_cls_deep}")

preds_deep = predict_batched(model_deep, X_valid_multi, is_hierarchical=True)
ec_pred_deep = {}
for lv in [1, 2, 3, 4]:
    ec_pred_deep[lv] = le_deep[lv].inverse_transform(preds_deep[lv])

metrics_deep = compute_metrics_by_group(ec_valid_str_multi, ec_pred_deep,
                                         groups_multi, [1, 2, 3, 4],
                                         is_hierarchical=True)

for lv in [1, 2, 3, 4]:
    m = metrics_deep[lv]
    g = m["groups"]
    print(f"  EC{lv}: Acc={m['overall']['acc']:.4f}  "
          f"[S={g[0]['acc']:.4f} M={g[1]['acc']:.4f} L={g[2]['acc']:.4f}]")

plot_accuracy_by_level(
    metrics_deep,
    f"Deep Hierarchical MLP — Layers {layers_used}",
    os.path.join(deep_dir, "accuracy_by_level.png"))

# ============================================================
# Save all per-group metrics
# ============================================================
def metrics_to_serializable(metrics):
    out = {}
    for lv, data in metrics.items():
        out[f"level_{lv}"] = {
            "overall": {k: float(v) for k, v in data["overall"].items()},
            "groups": {group_names[g]: {k: float(v) for k, v in data["groups"][g].items()}
                       for g in range(3)}
        }
    return out

all_results = {
    "flat_mlp": metrics_to_serializable(metrics_flat),
    "hierarchical_mlp": metrics_to_serializable(metrics_hier),
    "deep_hierarchical_mlp": metrics_to_serializable(metrics_deep),
    "group_info": {
        "method": "sorted by sequence length, split into equal thirds",
        "groups": group_names,
    }
}

out_path = MODELS_DIR / "benchmark_2" / "error_bar_metrics.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*70}")
print("DONE!")
print(f"{'='*70}")
print(f"Per-group metrics saved to: {out_path}")
print(f"Updated figures:")
print(f"  {mlp_dir}/accuracy_by_level.png")
print(f"  {hier_dir}/accuracy_by_level.png")
print(f"  {deep_dir}/accuracy_by_level.png")
