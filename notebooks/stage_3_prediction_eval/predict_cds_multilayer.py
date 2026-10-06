#!/usr/bin/env python3
"""
predict_cds_multilayer.py
=========================
Load multi-layer (13, 17, 18, 20) Evo2 embeddings of cross-referenced
coding sequences, run predictions with three benchmark_2 classifiers:

  1. Deep Hierarchical MLP (16384-dim concatenation of all 4 layers)
  2. Hierarchical MLP      (layer-18 slice, 4096-dim)
  3. Flat MLP              (layer-18 slice, 4096-dim, full EC number)

...evaluate top-1/2/3 accuracy on the EC-annotated subset, and plot
accuracy by hierarchy level for each model.

Input : data/predictions_and_results/ec_cds_multilayer_embeddings.parquet
        (produced by extract_cds_multilayer_embeddings.py)
Models: models/benchmark_2/deep_hierarchical_mlp_results/{model.pt, label_encoders.pkl}
        models/benchmark_2/hierarchical_mlp_results/{hierarchical_mlp.pt, label_encoders.pkl}
        models/benchmark_2/mlp_results/{mlp_layer_18.pt, label_encoder.pkl}
Output: data/predictions_and_results/ec_cds_predictions.parquet
Plots : figures/accuracy_by_level_deep_hierarchical.png
        figures/accuracy_by_level_hierarchical.png
        figures/accuracy_by_level_mlp.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    PRED_DIR, MODELS_DIR, FIGURES_DIR, get_device,
    EC_MLP, HierarchicalEC_MLP, DeepHierarchicalEC_MLP, ResBlock, CosineClassifier, ec_at_level, length_tercile_groups,
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
EMBEDDINGS_PARQUET = PRED_DIR / "ec_cds_multilayer_embeddings.parquet"

MODEL_DIR = MODELS_DIR / "benchmark_2" / "deep_hierarchical_mlp_results"
MODEL_WEIGHTS_PATH = MODEL_DIR / "model.pt"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

HIER_DIR = MODELS_DIR / "benchmark_2" / "hierarchical_mlp_results"
HIER_WEIGHTS_PATH = HIER_DIR / "hierarchical_mlp.pt"
HIER_LABEL_ENCODERS_PATH = HIER_DIR / "label_encoders.pkl"

MLP_DIR = MODELS_DIR / "benchmark_2" / "mlp_results"
MLP_WEIGHTS_PATH = MLP_DIR / "mlp_layer_18.pt"
MLP_LABEL_ENCODERS_PATH = MLP_DIR / "label_encoder.pkl"

OUTPUT_PARQUET = PRED_DIR / "ec_cds_predictions.parquet"

ACCURACY_PLOT_PATH_DEEP = FIGURES_DIR / "benchmark_2" / "accuracy_by_level_deep_hierarchical.png"
ACCURACY_PLOT_PATH_HIER = FIGURES_DIR / "benchmark_2" / "accuracy_by_level_hierarchical.png"
ACCURACY_PLOT_PATH_MLP = FIGURES_DIR / "benchmark_2" / "accuracy_by_level_mlp.png"

TARGET_LAYERS = [13, 17, 18, 20]
INFERENCE_BATCH_SIZE = 4096

# ============================================================
# Plotting Helper
# ============================================================
def plot_accuracy_figures(model_name, level_accuracies, output_plot_path):
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
    plt.title(f"EC Prediction Accuracy by Level\n({model_name})", fontweight='bold', fontsize=14, pad=15)
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
    plt.savefig(output_plot_path, dpi=300)
    print(f"Accuracy plot saved to: {output_plot_path}")
    plt.close()



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
            "Run extract_cds_multilayer_embeddings.py first.")
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
    print("Loading label encoders...")
    with open(LABEL_ENCODERS_PATH, "rb") as f:
        encoders_deep = pickle.load(f)
    n_classes_deep = {level: len(encoders_deep[level].classes_) for level in [1, 2, 3, 4]}

    with open(HIER_LABEL_ENCODERS_PATH, "rb") as f:
        encoders_hier = pickle.load(f)
    n_classes_hier = {level: len(encoders_hier[level].classes_) for level in [1, 2, 3, 4]}

    with open(MLP_LABEL_ENCODERS_PATH, "rb") as f:
        le_flat = pickle.load(f)
    n_classes_flat = len(le_flat.classes_)

    # ============================================================
    # 3. Model Predictions & Evaluations
    # ============================================================
    print(f"\n{'='*70}")
    print("Evaluating models on coding sequences...")
    print(f"{'='*70}")

    # --- MODEL 1: Deep Hierarchical MLP ---
    print("\n--- Model 1: Deep Hierarchical MLP ---")
    model_deep = DeepHierarchicalEC_MLP(
        input_dim=16384,
        n_ec1=n_classes_deep[1],
        n_ec2=n_classes_deep[2],
        n_ec3=n_classes_deep[3],
        n_ec4=n_classes_deep[4],
        dropout=0.0
    ).to(device)

    print(f"Loading weights: {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_deep.load_state_dict(clean_state_dict)
    model_deep.eval()

    level_preds_deep = {
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
            logits_list = model_deep(xb)

            for level_idx, logits in enumerate(logits_list, start=1):
                probs = torch.softmax(logits, dim=1)
                batch_probs, batch_indices = torch.topk(probs, k=3, dim=1)

                batch_indices = batch_indices.cpu().numpy()
                batch_probs = batch_probs.cpu().numpy()

                level_preds_deep[level_idx]["top1_indices"].append(batch_indices[:, 0])
                level_preds_deep[level_idx]["top2_indices"].append(batch_indices[:, 1])
                level_preds_deep[level_idx]["top3_indices"].append(batch_indices[:, 2])

                level_preds_deep[level_idx]["top1_probs"].append(batch_probs[:, 0])
                level_preds_deep[level_idx]["top2_probs"].append(batch_probs[:, 1])
                level_preds_deep[level_idx]["top3_probs"].append(batch_probs[:, 2])

    decoded_deep = {}
    probs_deep = {}
    for level in [1, 2, 3, 4]:
        t1_idx = np.concatenate(level_preds_deep[level]["top1_indices"])
        t2_idx = np.concatenate(level_preds_deep[level]["top2_indices"])
        t3_idx = np.concatenate(level_preds_deep[level]["top3_indices"])
        t1_prob = np.concatenate(level_preds_deep[level]["top1_probs"])
        t2_prob = np.concatenate(level_preds_deep[level]["top2_probs"])
        t3_prob = np.concatenate(level_preds_deep[level]["top3_probs"])

        decoded_deep[level] = {
            "top1": encoders_deep[level].inverse_transform(t1_idx),
            "top2": encoders_deep[level].inverse_transform(t2_idx),
            "top3": encoders_deep[level].inverse_transform(t3_idx),
        }
        probs_deep[level] = {
            "top1": t1_prob,
            "top2": t2_prob,
            "top3": t3_prob,
        }

    df_valid["Predicted_EC_DeepHier"] = decoded_deep[4]["top1"]
    for level in [1, 2, 3, 4]:
        df_valid[f"Predicted_EC_DeepHier_L{level}_top1"] = decoded_deep[level]["top1"]
        df_valid[f"Predicted_EC_DeepHier_L{level}_top2"] = decoded_deep[level]["top2"]
        df_valid[f"Predicted_EC_DeepHier_L{level}_top3"] = decoded_deep[level]["top3"]
        df_valid[f"Prob_DeepHier_L{level}_top1"] = probs_deep[level]["top1"]
        df_valid[f"Prob_DeepHier_L{level}_top2"] = probs_deep[level]["top2"]
        df_valid[f"Prob_DeepHier_L{level}_top3"] = probs_deep[level]["top3"]

    del model_deep
    torch.cuda.empty_cache()

    # --- MODEL 2: Hierarchical MLP ---
    print("\n--- Model 2: Hierarchical MLP ---")
    model_hier = HierarchicalEC_MLP(
        input_dim=4096,
        n_ec1=n_classes_hier[1],
        n_ec2=n_classes_hier[2],
        n_ec3=n_classes_hier[3],
        n_ec4=n_classes_hier[4],
        dropout=0.0
    ).to(device)

    print(f"Loading weights: {HIER_WEIGHTS_PATH}")
    state_dict = torch.load(HIER_WEIGHTS_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_hier.load_state_dict(clean_state_dict)
    model_hier.eval()

    level_preds_hier = {
        level: {
            "top1_indices": [], "top2_indices": [], "top3_indices": [],
            "top1_probs": [], "top2_probs": [], "top3_probs": []
        }
        for level in [1, 2, 3, 4]
    }

    X_l18_tensor = torch.from_numpy(X[:, 8192:12288])
    with torch.no_grad():
        for i in range(0, len(X_l18_tensor), INFERENCE_BATCH_SIZE):
            xb = X_l18_tensor[i:i+INFERENCE_BATCH_SIZE].to(device)
            logits_list = model_hier(xb)

            for level_idx, logits in enumerate(logits_list, start=1):
                probs = torch.softmax(logits, dim=1)
                batch_probs, batch_indices = torch.topk(probs, k=3, dim=1)

                batch_indices = batch_indices.cpu().numpy()
                batch_probs = batch_probs.cpu().numpy()

                level_preds_hier[level_idx]["top1_indices"].append(batch_indices[:, 0])
                level_preds_hier[level_idx]["top2_indices"].append(batch_indices[:, 1])
                level_preds_hier[level_idx]["top3_indices"].append(batch_indices[:, 2])

                level_preds_hier[level_idx]["top1_probs"].append(batch_probs[:, 0])
                level_preds_hier[level_idx]["top2_probs"].append(batch_probs[:, 1])
                level_preds_hier[level_idx]["top3_probs"].append(batch_probs[:, 2])

    decoded_hier = {}
    probs_hier = {}
    for level in [1, 2, 3, 4]:
        t1_idx = np.concatenate(level_preds_hier[level]["top1_indices"])
        t2_idx = np.concatenate(level_preds_hier[level]["top2_indices"])
        t3_idx = np.concatenate(level_preds_hier[level]["top3_indices"])
        t1_prob = np.concatenate(level_preds_hier[level]["top1_probs"])
        t2_prob = np.concatenate(level_preds_hier[level]["top2_probs"])
        t3_prob = np.concatenate(level_preds_hier[level]["top3_probs"])

        decoded_hier[level] = {
            "top1": encoders_hier[level].inverse_transform(t1_idx),
            "top2": encoders_hier[level].inverse_transform(t2_idx),
            "top3": encoders_hier[level].inverse_transform(t3_idx),
        }
        probs_hier[level] = {
            "top1": t1_prob,
            "top2": t2_prob,
            "top3": t3_prob,
        }

    df_valid["Predicted_EC_Hier"] = decoded_hier[4]["top1"]
    for level in [1, 2, 3, 4]:
        df_valid[f"Predicted_EC_Hier_L{level}_top1"] = decoded_hier[level]["top1"]
        df_valid[f"Predicted_EC_Hier_L{level}_top2"] = decoded_hier[level]["top2"]
        df_valid[f"Predicted_EC_Hier_L{level}_top3"] = decoded_hier[level]["top3"]
        df_valid[f"Prob_Hier_L{level}_top1"] = probs_hier[level]["top1"]
        df_valid[f"Prob_Hier_L{level}_top2"] = probs_hier[level]["top2"]
        df_valid[f"Prob_Hier_L{level}_top3"] = probs_hier[level]["top3"]

    del model_hier
    torch.cuda.empty_cache()

    # --- MODEL 3: Flat MLP ---
    print("\n--- Model 3: Flat MLP ---")
    model_flat = EC_MLP(
        input_dim=4096,
        num_classes=n_classes_flat,
        dropout=0.0
    ).to(device)

    print(f"Loading weights: {MLP_WEIGHTS_PATH}")
    state_dict = torch.load(MLP_WEIGHTS_PATH, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_flat.load_state_dict(clean_state_dict)
    model_flat.eval()

    flat_preds = {
        "top1_indices": [], "top2_indices": [], "top3_indices": [],
        "top1_probs": [], "top2_probs": [], "top3_probs": []
    }

    with torch.no_grad():
        for i in range(0, len(X_l18_tensor), INFERENCE_BATCH_SIZE):
            xb = X_l18_tensor[i:i+INFERENCE_BATCH_SIZE].to(device)
            logits = model_flat(xb)
            probs = torch.softmax(logits, dim=1)
            batch_probs, batch_indices = torch.topk(probs, k=3, dim=1)

            batch_indices = batch_indices.cpu().numpy()
            batch_probs = batch_probs.cpu().numpy()

            flat_preds["top1_indices"].append(batch_indices[:, 0])
            flat_preds["top2_indices"].append(batch_indices[:, 1])
            flat_preds["top3_indices"].append(batch_indices[:, 2])

            flat_preds["top1_probs"].append(batch_probs[:, 0])
            flat_preds["top2_probs"].append(batch_probs[:, 1])
            flat_preds["top3_probs"].append(batch_probs[:, 2])

    t1_idx = np.concatenate(flat_preds["top1_indices"])
    t2_idx = np.concatenate(flat_preds["top2_indices"])
    t3_idx = np.concatenate(flat_preds["top3_indices"])
    t1_prob = np.concatenate(flat_preds["top1_probs"])
    t2_prob = np.concatenate(flat_preds["top2_probs"])
    t3_prob = np.concatenate(flat_preds["top3_probs"])

    decoded_flat_l4 = {
        "top1": le_flat.inverse_transform(t1_idx),
        "top2": le_flat.inverse_transform(t2_idx),
        "top3": le_flat.inverse_transform(t3_idx),
    }

    df_valid["Predicted_EC_Flat"] = decoded_flat_l4["top1"]
    for k in ["top1", "top2", "top3"]:
        df_valid[f"Predicted_EC_Flat_L4_{k}"] = decoded_flat_l4[k]
    df_valid["Prob_Flat_L4_top1"] = t1_prob
    df_valid["Prob_Flat_L4_top2"] = t2_prob
    df_valid["Prob_Flat_L4_top3"] = t3_prob

    del model_flat
    torch.cuda.empty_cache()

    # Save all updated predictions to parquet
    (FIGURES_DIR / "benchmark_2").mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final predictions to: {OUTPUT_PARQUET}")
    df_valid.to_parquet(OUTPUT_PARQUET, index=False)

    # ============================================================
    # 4. Evaluation and Plotting
    # ============================================================
    annotated = df_valid[df_valid["EC_Numbers"] != "UP"].copy()
    print(f"\nEvaluating performance on {len(annotated)} annotated genes:")

    # Sequence-length groups (terciles) for error bars
    gpos = length_tercile_groups(annotated["AA_Sequence"].str.len().values)

    def evaluate_predictions(annotated_df, decoded_dict, groups, is_flat=False):
        level_accuracies = {}
        for level in [1, 2, 3, 4]:
            true_lv = np.array([ec_at_level(e, level) for e in annotated_df["EC_Numbers"]])

            if is_flat:
                # Truncate level 4 prediction string to target level
                pred_lv1 = np.array([ec_at_level(e, level) for e in decoded_dict["top1"]])[annotated_df.index]
                pred_lv2 = np.array([ec_at_level(e, level) for e in decoded_dict["top2"]])[annotated_df.index]
                pred_lv3 = np.array([ec_at_level(e, level) for e in decoded_dict["top3"]])[annotated_df.index]
            else:
                pred_lv1 = np.array(decoded_dict[level]["top1"])[annotated_df.index]
                pred_lv2 = np.array(decoded_dict[level]["top2"])[annotated_df.index]
                pred_lv3 = np.array(decoded_dict[level]["top3"])[annotated_df.index]

            hit1 = true_lv == pred_lv1
            hit2 = hit1 | (true_lv == pred_lv2)
            hit3 = hit2 | (true_lv == pred_lv3)
            acc1 = hit1.mean()
            acc2 = hit2.mean()
            acc3 = hit3.mean()

            level_accuracies[level] = {
                "top1": acc1, "top2": acc2, "top3": acc3,
                "groups": [
                    tuple(float(h[groups == gi].mean()) if (groups == gi).sum() else 0.0
                          for h in (hit1, hit2, hit3))
                    for gi in range(3)
                ],
            }

        return level_accuracies

    print("\n--- Deep Hierarchical MLP Results ---")
    acc_deep = evaluate_predictions(annotated, decoded_deep, groups=gpos, is_flat=False)
    for level in [1, 2, 3, 4]:
        print(f"  Level {level}: Top-1={acc_deep[level]['top1']:.4f}, Top-2={acc_deep[level]['top2']:.4f}, Top-3={acc_deep[level]['top3']:.4f}")
    plot_accuracy_figures("DH-MLP", acc_deep, ACCURACY_PLOT_PATH_DEEP)

    print("\n--- Hierarchical MLP Results ---")
    acc_hier = evaluate_predictions(annotated, decoded_hier, groups=gpos, is_flat=False)
    for level in [1, 2, 3, 4]:
        print(f"  Level {level}: Top-1={acc_hier[level]['top1']:.4f}, Top-2={acc_hier[level]['top2']:.4f}, Top-3={acc_hier[level]['top3']:.4f}")
    plot_accuracy_figures("H-MLP", acc_hier, ACCURACY_PLOT_PATH_HIER)

    print("\n--- Flat MLP Results ---")
    acc_flat = evaluate_predictions(annotated, decoded_flat_l4, groups=gpos, is_flat=True)
    for level in [1, 2, 3, 4]:
        print(f"  Level {level}: Top-1={acc_flat[level]['top1']:.4f}, Top-2={acc_flat[level]['top2']:.4f}, Top-3={acc_flat[level]['top3']:.4f}")
    plot_accuracy_figures("Flat MLP", acc_flat, ACCURACY_PLOT_PATH_MLP)


if __name__ == "__main__":
    main()
