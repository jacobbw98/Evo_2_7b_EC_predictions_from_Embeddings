#!/usr/bin/env python3
"""
Two additional figures for the Evo2 EC-prediction pipeline (saved to figures/):

1. f1_by_layer_mlp_mean_token.png
   Identical style to accuracy_by_layer_mlp_mean_token.png (grouped bar
   chart of EC levels 1-4 across all 32 Evo2 layers, error bars = SD across
   the Short/Medium/Long sequence-length tercile groups), but bar values are
   the *macro F1* scores from
   models/benchmark_4/mlp_mean_per_layer/results_summary.json
   (per-group f1m from models/benchmark_4/error_bar_metrics.json).

2. deep_hierarchical_mlp_yarrowia_confusion_l1.png
   Level-1 EC confusion matrix (true rows x top-1 predicted columns) for the
   DeepHierarchicalEC_MLP trained on benchmark_2, evaluated on the Yarrowia
   whole-gene set. Reads the existing prediction file
   data/predictions_and_results/yarrowia_evo2_multilayer_embeddings_predicted.parquet
   (no re-inference). Unannotated rows (EC_Numbers == 'UP') are excluded;
   multi-EC rows use the first EC number.

Usage: python plot_f1_and_confusion.py   (matplotlib, pandas/pyarrow; no install)
"""
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import DATA_DIR, FIGURES_DIR, MODELS_DIR

import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH4_MODELS_DIR = MODELS_DIR / "benchmark_4"
LAYERS = list(range(32))

# ============================================================
# Figure 1: macro F1 by layer - MLP Mean Token
# (mirrors accuracy_by_layer_mlp_mean_token.png exactly)
# ============================================================
with open(BENCH4_MODELS_DIR / "mlp_mean_per_layer" / "results_summary.json") as f:
    results = json.load(f)

with open(BENCH4_MODELS_DIR / "error_bar_metrics.json") as f:
    eb_data = json.load(f)

CONFIG = "MLP Mean Token"


def group_f1_sd(layer, level):
    """Sample SD of Short/Medium/Long group macro F1 at one layer/level."""
    try:
        g = eb_data[CONFIG][str(layer)][f"level_{level}"]["groups"]
        vals = [g[k]["f1m"] for k in ("Short", "Medium", "Long") if k in g]
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    except (KeyError, TypeError):
        return 0.0


level_colors = {"1": "#2196F3", "2": "#4CAF50", "3": "#FF9800", "4": "#E91E63"}

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(32)
width = 0.2

for i, (level, color) in enumerate(level_colors.items()):
    vals = []
    errs = []
    for l in LAYERS:
        r = results.get(str(l), {})
        vals.append(r.get(f"level_{level}_f1_macro", 0.0))
        errs.append(group_f1_sd(l, level))
    ax.bar(x + i * width, vals, width, label=f"Level {level}", color=color, alpha=0.85,
           yerr=errs, ecolor="black", capsize=1.2)

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_xlabel("Evo2 Layer")
ax.set_ylabel("F1 Score (macro)")
ax.set_ylim(0, 1.0)
ax.set_title("MLP Mean Token - EC Prediction F1 Score by Layer", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

f1_path = FIGURES_DIR / "benchmark_4" / "f1_by_layer_mlp_mean_token.png"
(FIGURES_DIR / "benchmark_4").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "confusion_matrixs").mkdir(parents=True, exist_ok=True)
plt.savefig(f1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {f1_path}")

# ============================================================
# Figure 2: DeepHierarchicalEC_MLP - Yarrowia L1 confusion matrix
# ============================================================
PARQUET_PATH = (DATA_DIR / "predictions_and_results"
                / "yarrowia_evo2_multilayer_embeddings_predicted.parquet")
table = pq.read_table(
    PARQUET_PATH,
    columns=["EC_Numbers", "Predicted_EC_L1_top1"],
).to_pandas()

df = table[table["EC_Numbers"] != "UP"].copy()
df["true_l1"] = df["EC_Numbers"].map(lambda e: str(e).split(";")[0].strip().split(".")[0])
df["pred_l1"] = df["Predicted_EC_L1_top1"].astype(str)

classes = [str(c) for c in range(1, 8)]
idx = {c: i for i, c in enumerate(classes)}
cm = np.zeros((7, 7), dtype=int)
for t, p in zip(df["true_l1"], df["pred_l1"]):
    if t in idx and p in idx:
        cm[idx[t], idx[p]] += 1
assert cm.sum() == len(df), "rows dropped from confusion matrix"

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=classes, yticklabels=classes)
ax.set_xlabel("Predicted 1st digit"); ax.set_ylabel("True 1st digit")
ax.set_title(f"DH-MLP (Yarrowia) \u2014 1st-digit EC confusion matrix")
fig.tight_layout()
cm_path = FIGURES_DIR / "confusion_matrixs" / "deep_hierarchical_mlp_yarrowia_confusion_l1.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_path}")

n_correct = int((df["true_l1"] == df["pred_l1"]).sum())
print(f"Top-1 L1 accuracy on annotated Yarrowia: {n_correct / len(df):.4f} ({n_correct}/{len(df)})")
print("Done!")
