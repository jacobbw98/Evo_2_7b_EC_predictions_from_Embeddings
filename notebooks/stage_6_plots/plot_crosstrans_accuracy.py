#!/usr/bin/env python3
"""
plot_crosstrans_accuracy.py
==========================
Re-plot the cross-translation accuracy-by-level bar graphs for the
H-MLP (layer-18) and DH-MLP (multilayer) arms directly from the
existing predicted parquets -- no model inference required.

Reads : data/predictions_and_results/yarrowia_crosstrans_layer18_predicted.parquet
        data/predictions_and_results/yarrowia_crosstrans_multilayer_predicted.parquet
Plots : figures/crosstrans_layer18_accuracy_by_level.png
        figures/crosstrans_multilayer_accuracy_by_level.png

Mirrors the plotting code in
notebooks/stage_3_prediction_eval/evaluate_crosstrans_{layer18,multilayer}.py
(same layout, palette, error bars) so the figures stay in sync.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import PRED_DIR, FIGURES_DIR, ec_at_level, length_tercile_groups

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared palette -- must stay in sync with evaluate_crosstrans_{layer18,multilayer}.py
C_TOP1 = "#457B9D"  # Classic blue-grey
C_TOP2 = "#E9C46A"  # Warm gold
C_TOP3 = "#E76F51"  # Coral

FIGURES = [
    (PRED_DIR / "yarrowia_crosstrans_layer18_predicted.parquet",
     FIGURES_DIR / "yarrowia" / "crosstrans_layer18_accuracy_by_level.png",
     "EC Prediction Accuracy by Level H-MLP (Cross-Translated Sequences)"),
    (PRED_DIR / "yarrowia_crosstrans_multilayer_predicted.parquet",
     FIGURES_DIR / "yarrowia" / "crosstrans_multilayer_accuracy_by_level.png",
     "EC Prediction Accuracy by Level DH-MLP (Cross-Translated Sequences)"),
]

LEVELS = [1, 2, 3, 4]
SERIES = ("top1", "top2", "top3")
COLORS = {"top1": C_TOP1, "top2": C_TOP2, "top3": C_TOP3}


def level_accuracies(df):
    """Per-level top-1/2/3 accuracy on annotated genes + tercile-group means."""
    annotated = df[df["EC_Numbers"] != "UP"].copy()
    gpos = length_tercile_groups(annotated["Length"].values)
    out = {}
    for level in LEVELS:
        true_lv = np.array([ec_at_level(e, level) for e in annotated["EC_Numbers"]])
        hit1 = true_lv == annotated[f"Predicted_EC_L{level}_top1"].to_numpy()
        hit2 = hit1 | (true_lv == annotated[f"Predicted_EC_L{level}_top2"].to_numpy())
        hit3 = hit2 | (true_lv == annotated[f"Predicted_EC_L{level}_top3"].to_numpy())
        out[level] = {
            "top1": float(hit1.mean()),
            "top2": float(hit2.mean()),
            "top3": float(hit3.mean()),
            "groups": [
                tuple(float(h[gpos == gi].mean()) if (gpos == gi).sum() else 0.0
                      for h in (hit1, hit2, hit3))
                for gi in range(3)
            ],
        }
    return out, len(annotated)


def plot_accuracies(acc, out_path, title):
    levels = [f"Level {l}" for l in LEVELS]
    x = np.arange(len(levels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, k in enumerate(SERIES):
        vals = [acc[l][k] for l in LEVELS]
        errs = [float(np.std([acc[l]["groups"][gi][i] for gi in range(3)], ddof=1)) for l in LEVELS]
        plt.bar(x + (i - 1) * width, vals, width, label=f"Top-{i + 1} Accuracy",
                color=COLORS[k], edgecolor='grey', alpha=0.95,
                yerr=errs, ecolor='black', capsize=3)

    plt.xlabel("EC Hierarchy Level", fontweight='bold', fontsize=12)
    plt.ylabel("Accuracy", fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14, pad=15)
    plt.xticks(x, levels, fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')

    # Add values on top of bars
    for i in range(len(levels)):
        for j, k in enumerate(SERIES):
            plt.text(i + (j - 1) * width, acc[LEVELS[i]][k] + 0.01,
                     f"{acc[LEVELS[i]][k]*100:.1f}%", ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"  saved: {out_path}")
    plt.close()


def main():
    (FIGURES_DIR / "yarrowia").mkdir(parents=True, exist_ok=True)
    cols = ["EC_Numbers", "Length"] + [
        f"Predicted_EC_L{l}_{k}" for l in LEVELS for k in SERIES]
    for parquet, out_png, title in FIGURES:
        print(f"\n{title}")
        df = pd.read_parquet(parquet, columns=cols)
        acc, n_annot = level_accuracies(df)
        print(f"  annotated genes: {n_annot}")
        for level in LEVELS:
            print(f"  Level {level}: Top-1 {acc[level]['top1']:.4f}  "
                  f"Top-2 {acc[level]['top2']:.4f}  Top-3 {acc[level]['top3']:.4f}")
        plot_accuracies(acc, out_png, title)


if __name__ == "__main__":
    main()
