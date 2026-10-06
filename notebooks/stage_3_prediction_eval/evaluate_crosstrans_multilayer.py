#!/usr/bin/env python3
"""
evaluate_crosstrans_multilayer.py
================================
Load concatenated multi-layer (13, 17, 18, 20) Evo2 embeddings of
cross-translated Yarrowia lipolytica UniProt sequences, run the benchmark_2
deep hierarchical MLP classifier (top-3 predictions per EC hierarchy level),
evaluate top-1/2/3 accuracy on the EC-annotated subset, and plot accuracy
by level.

Input : data/predictions_and_results/yarrowia_crosstrans_multilayer_embeddings.parquet
        (produced by extract_crosstrans_embeddings.py)
Models: models/benchmark_2/deep_hierarchical_mlp_results/{model.pt,
        label_encoders.pkl}
Output: data/predictions_and_results/yarrowia_crosstrans_multilayer_predicted.parquet
Plot  : figures/crosstrans_multilayer_accuracy_by_level.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    PRED_DIR, MODELS_DIR, FIGURES_DIR, get_device,
    DeepHierarchicalEC_MLP, ResBlock, CosineClassifier, ec_at_level, length_tercile_groups,
)

import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
EMBEDDINGS_PARQUET = PRED_DIR / "yarrowia_crosstrans_multilayer_embeddings.parquet"
MODEL_DIR = MODELS_DIR / "benchmark_2" / "deep_hierarchical_mlp_results"
MODEL_WEIGHTS_PATH = MODEL_DIR / "model.pt"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

OUTPUT_PARQUET = PRED_DIR / "yarrowia_crosstrans_multilayer_predicted.parquet"
ACCURACY_PLOT_PATH = FIGURES_DIR / "yarrowia" / "crosstrans_multilayer_accuracy_by_level.png"

TARGET_LAYERS = [13, 17, 18, 20]
INPUT_DIM = 4096 * len(TARGET_LAYERS)  # 16384
INFERENCE_BATCH_SIZE = 4096

# ============================================================
# Main Execution Flow
# ============================================================
def main():
    device = get_device()
    print(f"Device: {device}")

    # 1. Load embeddings parquet
    print(f"\nLoading embeddings from: {EMBEDDINGS_PARQUET}")
    if not os.path.exists(EMBEDDINGS_PARQUET):
        raise FileNotFoundError(
            f"Embeddings parquet not found at {EMBEDDINGS_PARQUET}. "
            "Run extract_crosstrans_embeddings.py first.")
    df_valid = pd.read_parquet(EMBEDDINGS_PARQUET)
    print(f"  Loaded {len(df_valid)} rows")

    required_cols = []
    for layer in TARGET_LAYERS:
        required_cols.extend([f"emb_{layer}_{d}" for d in range(4096)])
    if not all(col in df_valid.columns for col in required_cols):
        raise ValueError("Embeddings parquet is missing emb_{layer}_{d} columns for layers 13/17/18/20.")

    X_layers = []
    for layer in TARGET_LAYERS:
        cols = [f"emb_{layer}_{d}" for d in range(4096)]
        X_layers.append(df_valid[cols].values.astype(np.float32))
    X = np.concatenate(X_layers, axis=1)  # Shape: [N, 16384]
    print(f"Concatenated features matrix shape: {X.shape}")

    # 2. Load label encoders
    print(f"Loading label encoders from: {LABEL_ENCODERS_PATH}")
    if not os.path.exists(LABEL_ENCODERS_PATH):
        raise FileNotFoundError(f"Label encoders pkl not found at {LABEL_ENCODERS_PATH}")
    with open(LABEL_ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    n_classes = {level: len(label_encoders[level].classes_) for level in [1, 2, 3, 4]}
    print(f"Classes: L1={n_classes[1]}, L2={n_classes[2]}, L3={n_classes[3]}, L4={n_classes[4]}")

    # 3. Prediction using Deep Hierarchical MLP
    print("\nRunning Deep Hierarchical MLP classifier...")
    model_eval = DeepHierarchicalEC_MLP(
        input_dim=INPUT_DIM,
        n_ec1=n_classes[1],
        n_ec2=n_classes[2],
        n_ec3=n_classes[3],
        n_ec4=n_classes[4],
        dropout=0.0
    ).to(device)

    print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_eval.load_state_dict(clean_state_dict)
    model_eval.eval()

    level_preds = {
        level: {
            "top1_indices": [], "top2_indices": [], "top3_indices": [],
            "top1_probs": [], "top2_probs": [], "top3_probs": []
        }
        for level in [1, 2, 3, 4]
    }

    X_tensor = torch.from_numpy(X)
    with torch.no_grad():
        for i in range(0, len(X_tensor), INFERENCE_BATCH_SIZE):
            xb = X_tensor[i:i+INFERENCE_BATCH_SIZE].to(device)
            logits_list = model_eval(xb)

            for level_idx, logits in enumerate(logits_list, start=1):
                probs = torch.softmax(logits, dim=1)
                batch_probs, batch_indices = torch.topk(probs, k=3, dim=1)

                batch_indices = batch_indices.cpu().numpy()
                batch_probs = batch_probs.cpu().numpy()

                level_preds[level_idx]["top1_indices"].append(batch_indices[:, 0])
                level_preds[level_idx]["top2_indices"].append(batch_indices[:, 1])
                level_preds[level_idx]["top3_indices"].append(batch_indices[:, 2])

                level_preds[level_idx]["top1_probs"].append(batch_probs[:, 0])
                level_preds[level_idx]["top2_probs"].append(batch_probs[:, 1])
                level_preds[level_idx]["top3_probs"].append(batch_probs[:, 2])

    # Decode predictions
    decoded_preds = {}
    probs_results = {}
    for level in [1, 2, 3, 4]:
        t1_idx = np.concatenate(level_preds[level]["top1_indices"])
        t2_idx = np.concatenate(level_preds[level]["top2_indices"])
        t3_idx = np.concatenate(level_preds[level]["top3_indices"])

        t1_prob = np.concatenate(level_preds[level]["top1_probs"])
        t2_prob = np.concatenate(level_preds[level]["top2_probs"])
        t3_prob = np.concatenate(level_preds[level]["top3_probs"])

        decoded_preds[level] = {
            "top1": label_encoders[level].inverse_transform(t1_idx),
            "top2": label_encoders[level].inverse_transform(t2_idx),
            "top3": label_encoders[level].inverse_transform(t3_idx),
        }
        probs_results[level] = {
            "top1": t1_prob,
            "top2": t2_prob,
            "top3": t3_prob,
        }

    # Add columns to DataFrame
    df_valid["Predicted_EC"] = decoded_preds[4]["top1"]
    for level in [1, 2, 3, 4]:
        df_valid[f"Predicted_EC_L{level}_top1"] = decoded_preds[level]["top1"]
        df_valid[f"Predicted_EC_L{level}_top2"] = decoded_preds[level]["top2"]
        df_valid[f"Predicted_EC_L{level}_top3"] = decoded_preds[level]["top3"]
        df_valid[f"Prob_L{level}_top1"] = probs_results[level]["top1"]
        df_valid[f"Prob_L{level}_top2"] = probs_results[level]["top2"]
        df_valid[f"Prob_L{level}_top3"] = probs_results[level]["top3"]

    (FIGURES_DIR / "yarrowia").mkdir(parents=True, exist_ok=True)
    print(f"Saving final predictions to: {OUTPUT_PARQUET}")
    df_valid.to_parquet(OUTPUT_PARQUET, index=False)

    del model_eval
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Evaluate and plot accuracy
    annotated = df_valid[df_valid["EC_Numbers"] != "UP"].copy()
    print(f"\nEvaluating performance on {len(annotated)} annotated genes:")

    level_accuracies = {}

    # Sequence-length groups (terciles) for error bars
    gpos = length_tercile_groups(annotated["Length"].values)

    for level in [1, 2, 3, 4]:
        true_lv = np.array([ec_at_level(e, level) for e in annotated["EC_Numbers"]])
        pred_lv1 = np.array(decoded_preds[level]["top1"])[annotated.index]
        pred_lv2 = np.array(decoded_preds[level]["top2"])[annotated.index]
        pred_lv3 = np.array(decoded_preds[level]["top3"])[annotated.index]

        hit1 = true_lv == pred_lv1
        hit2 = hit1 | (true_lv == pred_lv2)
        hit3 = hit2 | (true_lv == pred_lv3)
        acc1 = hit1.mean()
        acc2 = hit2.mean()
        acc3 = hit3.mean()

        level_accuracies[level] = {
            "top1": acc1, "top2": acc2, "top3": acc3,
            "groups": [
                tuple(float(h[gpos == gi].mean()) if (gpos == gi).sum() else 0.0
                      for h in (hit1, hit2, hit3))
                for gi in range(3)
            ],
        }

        print(f"  Level {level}:")
        print(f"    Top-1 Accuracy: {acc1:.4f}")
        print(f"    Top-2 Accuracy: {acc2:.4f}")
        print(f"    Top-3 Accuracy: {acc3:.4f}")

    # Generate bar graph
    levels = ["Level 1", "Level 2", "Level 3", "Level 4"]
    top1_vals = [level_accuracies[l]["top1"] for l in [1, 2, 3, 4]]
    top2_vals = [level_accuracies[l]["top2"] for l in [1, 2, 3, 4]]
    top3_vals = [level_accuracies[l]["top3"] for l in [1, 2, 3, 4]]
    top1_errs = [float(np.std([level_accuracies[l]["groups"][gi][0] for gi in range(3)], ddof=1)) for l in [1, 2, 3, 4]]
    top2_errs = [float(np.std([level_accuracies[l]["groups"][gi][1] for gi in range(3)], ddof=1)) for l in [1, 2, 3, 4]]
    top3_errs = [float(np.std([level_accuracies[l]["groups"][gi][2] for gi in range(3)], ddof=1)) for l in [1, 2, 3, 4]]

    x = np.arange(len(levels))
    width = 0.25

    plt.figure(figsize=(10, 6))

    # Modern color palette
    c_top1 = "#457B9D"  # Classic blue-grey
    c_top2 = "#E9C46A"  # Warm gold
    c_top3 = "#E76F51"  # Coral

    plt.bar(x - width, top1_vals, width, label="Top-1 Accuracy", color=c_top1, edgecolor='grey', alpha=0.95,
            yerr=top1_errs, ecolor='black', capsize=3)
    plt.bar(x, top2_vals, width, label="Top-2 Accuracy", color=c_top2, edgecolor='grey', alpha=0.95,
            yerr=top2_errs, ecolor='black', capsize=3)
    plt.bar(x + width, top3_vals, width, label="Top-3 Accuracy", color=c_top3, edgecolor='grey', alpha=0.95,
            yerr=top3_errs, ecolor='black', capsize=3)

    plt.xlabel("EC Hierarchy Level", fontweight='bold', fontsize=12)
    plt.ylabel("Accuracy", fontweight='bold', fontsize=12)
    plt.title("EC Prediction Accuracy by Level DH-MLP (Cross-Translated Sequences)", fontweight='bold', fontsize=14, pad=15)
    plt.xticks(x, levels, fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')

    # Add values on top of bars
    for i in range(len(levels)):
        plt.text(i - width, top1_vals[i] + 0.01, f"{top1_vals[i]*100:.1f}%", ha='center', va='bottom', fontsize=9, color='black')
        plt.text(i, top2_vals[i] + 0.01, f"{top2_vals[i]*100:.1f}%", ha='center', va='bottom', fontsize=9, color='black')
        plt.text(i + width, top3_vals[i] + 0.01, f"{top3_vals[i]*100:.1f}%", ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=300)
    print(f"\nAccuracy plot saved to: {ACCURACY_PLOT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
