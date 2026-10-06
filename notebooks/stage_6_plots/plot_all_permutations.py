#!/usr/bin/env python3
"""
Plot Layer Accuracy Comparisons across Benchmark 4 Permutations.

Generates 4 accuracy-by-layer plots (one per model x pooling permutation),
each with error bars = SD across the Short/Medium/Long sequence-length group
accuracies (from models/benchmark_4/error_bar_metrics.json),
plus a summary comparison table.

Reads results from (written by the per-layer training scripts):
  - models/benchmark_4/xgboost_per_layer/results_summary.json      (XGBoost x Last Token)
  - models/benchmark_4/xgboost_mean_per_layer/results_summary.json  (XGBoost x Mean Token)
  - models/benchmark_4/mlp_last_per_layer/results_summary.json       (MLP x Last Token)
  - models/benchmark_4/mlp_mean_per_layer/results_summary.json       (MLP x Mean Token)

Outputs (to figures/benchmark_4/):
  - 4 accuracy-by-layer bar charts (same style)
  - 1 combined overlay plot (non-RF models)
  - 1 Random Forest plot (RF layer/type curves + combined model)
  - 1 summary comparison table (printed + saved as CSV and PNG)
  (If RF models exist, the RF plot and table include the Random Forest
   results from train_benchmark4_rf.py: 3 layer/type curves at layers
   9/24/26 + the 36864-feature combined model.)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Project paths — shared via definitions.py
# (script lives at <root>/notebooks/<stage>/, one level under notebooks/)
# ============================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import FIGURES_DIR, MODELS_DIR

BENCH4_MODELS_DIR = MODELS_DIR / "benchmark_4"
OUTPUT_DIR = FIGURES_DIR / "benchmark_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYERS = list(range(32))

# Load results from all 4 permutations
CONFIGS = {
    "XGBoost Last Token": {
        "dir": BENCH4_MODELS_DIR / "xgboost_per_layer",
        "json": "results_summary.json",
    },
    "XGBoost Mean Token": {
        "dir": BENCH4_MODELS_DIR / "xgboost_mean_per_layer",
        "json": "results_summary.json",
    },
    "MLP Last Token": {
        "dir": BENCH4_MODELS_DIR / "mlp_last_per_layer",
        "json": "results_summary.json",
    },
    "MLP Mean Token": {
        "dir": BENCH4_MODELS_DIR / "mlp_mean_per_layer",
        "json": "results_summary.json",
    },
}

all_data = {}
for name, cfg in CONFIGS.items():
    json_path = os.path.join(cfg["dir"], cfg["json"])
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_data[name] = json.load(f)
        print(f"Loaded: {name} ({len(all_data[name])} layers)")
    else:
        print(f"WARNING: Not found — {json_path}")

if not all_data:
    raise SystemExit("ERROR: No results found. Run the per-layer training scripts first.")

# Load per-layer, per-level group accuracies (Short/Medium/Long) for error bars
# (written earlier in the pipeline by stage_5 evaluate_benchmark4_error_bars.py)
eb_path = os.path.join(BENCH4_MODELS_DIR, "error_bar_metrics.json")
eb_data = {}
if os.path.exists(eb_path):
    with open(eb_path) as f:
        eb_data = json.load(f)
    print(f"Loaded error-bar metrics: {eb_path} ({len(eb_data)} configs)")
else:
    print(f"WARNING: not found — {eb_path}; plots will have no error bars")

# Load RF results (train_benchmark4_rf.py; skipped when not trained yet)
rf_summary = {}
_rf_sum_path = os.path.join(BENCH4_MODELS_DIR, "rf_benchmark4", "results_summary.json")
if os.path.exists(_rf_sum_path):
    with open(_rf_sum_path) as f:
        rf_summary = json.load(f)
    print(f"Loaded: Random Forest (9 layer/type models)")
else:
    print(f"WARNING: not found — {_rf_sum_path}; RF figure will be omitted")

rf_combined_results = None
_rf_comb_path = os.path.join(BENCH4_MODELS_DIR, "rf_combined_results", "results.pkl")
if os.path.exists(_rf_comb_path):
    import pickle
    with open(_rf_comb_path, "rb") as f:
        rf_combined_results = pickle.load(f)
    print("Loaded: RF Combined (36864 features)")
else:
    print(f"WARNING: not found — {_rf_comb_path}; RF figure will omit RF Combined")

RF_TYPES = [("RF Block", "block"), ("RF MLP", "mlp"), ("RF Post Norm", "post_norm")]

def group_sd(config, layer, level):
    """Sample SD of Short/Medium/Long group accuracy at one layer/level."""
    try:
        g = eb_data[config][str(layer)][f"level_{level}"]["groups"]
        vals = [g[k]["acc"] for k in ("Short", "Medium", "Long") if k in g]
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    except (KeyError, TypeError):
        return 0.0

def group_range(config, layer, level):
    """(min, max) of Short/Medium/Long group accuracy at one layer/level (error band)."""
    try:
        g = eb_data[config][str(layer)][f"level_{level}"]["groups"]
        vals = [g[k]["acc"] for k in ("Short", "Medium", "Long") if k in g]
        return (min(vals), max(vals)) if vals else None
    except (KeyError, TypeError):
        return None

# Plot 1–4: Individual accuracy-by-layer bar charts
level_colors = {"1": "#2196F3", "2": "#4CAF50", "3": "#FF9800", "4": "#E91E63"}

for name, results in all_data.items():
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(32)
    width = 0.2

    for i, (level, color) in enumerate(level_colors.items()):
        vals = []
        errs = []
        for l in LAYERS:
            r = results.get(str(l), {})
            vals.append(r.get(f"level_{level}_accuracy", 0.0))
            errs.append(group_sd(name, l, level))
        ax.bar(x + i * width, vals, width, label=f"Level {level}", color=color, alpha=0.85,
               yerr=errs, ecolor="black", capsize=1.2)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([str(l) for l in LAYERS])
    ax.set_xlabel("Evo2 Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{name} — EC Prediction Accuracy by Layer", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    fig_path = os.path.join(OUTPUT_DIR, f"accuracy_by_layer_{safe_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

# Plot 5: Combined comparison — Full EC accuracy across layers
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(32)

model_colors = {
    "XGBoost Last Token": "#2196F3",
    "XGBoost Mean Token": "#00BCD4",
    "MLP Last Token":     "#E91E63",
    "MLP Mean Token":     "#FF9800",
}
markers = {
    "XGBoost Last Token": "o",
    "XGBoost Mean Token": "s",
    "MLP Last Token":     "^",
    "MLP Mean Token":     "D",
}

for name, results in all_data.items():
    vals = []
    los = []
    his = []
    for l in LAYERS:
        r = results.get(str(l), {})
        vals.append(r.get("level_4_accuracy", 0.0))
        rng = group_range(name, l, 4)
        if rng is None:
            los.append(vals[-1])
            his.append(vals[-1])
        else:
            los.append(rng[0])
            his.append(rng[1])
    color = model_colors.get(name, "gray")
    ax.plot(x, vals, label=name, color=color,
            marker=markers.get(name, "o"), linewidth=2, markersize=5, alpha=0.85)
    ax.fill_between(x, los, his, alpha=0.15, color=color)

ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_xlabel("Evo2 Layer", fontsize=12)
ax.set_ylabel("Full EC Level 4 Accuracy", fontsize=12)
ax.set_ylim(0, None)
ax.set_title("Full EC Prediction Accuracy — All Models Compared", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "combined_accuracy_comparison_b4.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")
# Plot 6: Random Forest only — Full EC accuracy (skipped when RF not trained)
if rf_summary or rf_combined_results is not None:
    fig, ax = plt.subplots(figsize=(16, 7))
    _rf_styles = {
        "RF Block":     ("#9C27B0", "^"),
        "RF MLP":       ("#795548", "s"),
        "RF Post Norm": ("#607D8B", "D"),
    }
    for type_name, t in RF_TYPES:
        keys = [L for L in (9, 24, 26) if f"layer_{L}_{t}" in rf_summary]
        if not keys:
            continue
        color, marker = _rf_styles.get(type_name, ("#9C27B0", "^"))
        vl = [rf_summary[f"layer_{L}_{t}"].get("level_4_accuracy", 0.0) for L in keys]
        ax.plot(keys, vl, label=type_name, color=color, marker=marker,
                linestyle="--", linewidth=1.8, markersize=7, alpha=0.9)
        los, his = [], []
        for i, L in enumerate(keys):
            rng = group_range(type_name, L, 4)
            if rng is None:
                los.append(vl[i])
                his.append(vl[i])
            else:
                los.append(rng[0])
                his.append(rng[1])
        ax.fill_between(keys, los, his, alpha=0.15, color=color)
    if rf_combined_results is not None:
        ax.scatter([17.5], [rf_combined_results["level_results"][4]["accuracy"]],
                   marker="*", s=320, color="#FF5722", zorder=5,
                   label="RF Combined (L9+24+26)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in LAYERS])
    ax.set_xlabel("Evo2 Layer", fontsize=12)
    ax.set_ylabel("Full EC Level 4 Accuracy", fontsize=12)
    ax.set_ylim(0, None)
    ax.set_title("Full EC Prediction Accuracy — Random Forest", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "rf_accuracy_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
else:
    print("WARNING: no RF results — skipping RF figure (run train_benchmark4_rf.py first)")

# Summary comparison table
print()
print("=" * 80)
print("SUMMARY COMPARISON TABLE")
print("=" * 80)
table_rows = []
for name, results in all_data.items():
    best_layer = max(LAYERS, key=lambda l: results.get(str(l), {}).get("level_4_accuracy", 0.0))
    best_r = results[str(best_layer)]
    valid_acc = best_r.get("level_4_accuracy", 0.0)
    train_acc = best_r.get("train_accuracy", None)
    if train_acc is None:
        train_acc_str = "N/A"
    else:
        train_acc_str = f"{train_acc:.4f}"
    table_rows.append({
        "Model": name,
        "Best Layer": best_layer,
        "Train EC Accuracy": train_acc_str,
        "Valid EC Accuracy": f"{valid_acc:.4f}",
    })
for type_name, t in RF_TYPES:
    keys = [L for L in (9, 24, 26) if f"layer_{L}_{t}" in rf_summary]
    if not keys:
        continue
    best_L = max(keys, key=lambda L: rf_summary[f"layer_{L}_{t}"].get("level_4_accuracy", 0.0))
    br = rf_summary[f"layer_{best_L}_{t}"]
    table_rows.append({
        "Model": type_name, "Best Layer": best_L,
        "Train EC Accuracy": f"{br.get('train_accuracy', float('nan')):.4f}",
        "Valid EC Accuracy": f"{br.get('level_4_accuracy', 0.0):.4f}",
    })
if rf_combined_results is not None:
    table_rows.append({
        "Model": "RF Combined (L9+24+26)", "Best Layer": "9+24+26",
        "Train EC Accuracy": f"{rf_combined_results.get('train_accuracy', float('nan')):.4f}",
        "Valid EC Accuracy": f"{rf_combined_results['level_results'][4]['accuracy']:.4f}",
    })

header = f"{'Model':<25} {'Best Layer':<12} {'Train EC Acc':<15} {'Valid EC Acc':<15}"
print(header)
print("-" * len(header))
for row in table_rows:
    print(f"{row['Model']:<25} {row['Best Layer']:<12} {row['Train EC Accuracy']:<15} {row['Valid EC Accuracy']:<15}")

df_table = pd.DataFrame(table_rows)
if len(df_table) == 0:
    print("WARNING: df_table is empty. No summary table generated.")
else:
    csv_path = os.path.join(OUTPUT_DIR, "summary_comparison.csv")
    df_table.to_csv(csv_path, index=False)
    print(f"Table saved: {csv_path}")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    for j in range(len(df_table.columns)):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(df_table) + 1):
        for j in range(len(df_table.columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f0f0f0")
    ax.set_title("Model Comparison — Best Layer Performance", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "summary_comparison_table.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Table image saved: {fig_path}")

print(f"All comparison plots saved to: {OUTPUT_DIR}")
print("Done!")
