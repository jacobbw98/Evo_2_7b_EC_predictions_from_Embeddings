#!/usr/bin/env python3
"""
Predict EC numbers for Yarrowia sequences using the trained Hierarchical MLP model.

Loads:
  - yarrowia_evo2_layer18_embeddings.parquet (input embeddings)
  - benchmark_2/hierarchical_mlp_results/label_encoders.pkl (class encoders)
  - benchmark_2/hierarchical_mlp_results/hierarchical_mlp.pt (model weights)

Outputs:
  - Overwrites and saves a parquet file with added top-3 predictions for all EC levels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    DATA_DIR, MODELS_DIR, PRED_DIR, get_device,
    HierarchicalEC_MLP, ec_at_level,
)

import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
PARQUET_PATH = PRED_DIR / "yarrowia_evo2_layer18_embeddings.parquet"
MODEL_DIR = MODELS_DIR / "benchmark_2" / "hierarchical_mlp_results"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
MODEL_WEIGHTS_PATH = MODEL_DIR / "hierarchical_mlp.pt"
OUTPUT_PARQUET_PATH = PRED_DIR / "yarrowia_evo2_layer18_embeddings_predicted.parquet"

DEVICE = get_device()
BATCH_SIZE = 4096


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    # 1. Load label encoders
    print(f"Loading label encoders from: {LABEL_ENCODERS_PATH}")
    if not os.path.exists(LABEL_ENCODERS_PATH):
        raise FileNotFoundError(f"Label encoders pkl not found at {LABEL_ENCODERS_PATH}")
    
    with open(LABEL_ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    
    n_classes = {level: len(label_encoders[level].classes_) for level in [1, 2, 3, 4]}
    print(f"Loaded encoders for {len(label_encoders)} levels:")
    for lv, n_cls in n_classes.items():
        print(f"  Level {lv}: {n_cls} classes")

    # 2. Load the validation/prediction embeddings
    print(f"\nLoading embeddings parquet from: {PARQUET_PATH}")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet file not found at {PARQUET_PATH}")
    
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Extract embedding columns (emb_0 to emb_4095)
    emb_cols = [f"emb_{i}" for i in range(4096)]
    # Verify columns exist
    missing_cols = [c for c in emb_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Parquet is missing {len(missing_cols)} embedding columns! (e.g. {missing_cols[:5]})")

    X = df[emb_cols].values.astype(np.float32)
    print(f"Extracted feature matrix shape: {X.shape}")

    # 3. Instantiate and load model
    print(f"\nInitializing model...")
    model = HierarchicalEC_MLP(
        input_dim=4096, 
        n_ec1=n_classes[1], 
        n_ec2=n_classes[2], 
        n_ec3=n_classes[3], 
        n_ec4=n_classes[4], 
        dropout=0.0
    ).to(DEVICE)
    
    print(f"Loading model weights from: {MODEL_WEIGHTS_PATH}")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 4. Predict
    print(f"\nPredicting EC numbers in batches of {BATCH_SIZE}...")
    
    # Store predictions for each level separately
    level_preds = {
        level: {
            "top1_indices": [], "top2_indices": [], "top3_indices": [],
            "top1_probs": [], "top2_probs": [], "top3_probs": []
        }
        for level in [1, 2, 3, 4]
    }
    
    X_tensor = torch.from_numpy(X)
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), BATCH_SIZE), desc="Inference"):
            xb = X_tensor[i:i+BATCH_SIZE].to(DEVICE)
            logits_list = model(xb) # returns (l1, l2, l3, l4)
            
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
    
    # Concatenate and decode predictions for each level
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

    # 5. Add to DataFrame
    df["Predicted_EC"] = decoded_preds[4]["top1"]  # For backward compatibility
    
    # Save top-3 columns for all levels
    for level in [1, 2, 3, 4]:
        df[f"Predicted_EC_L{level}_top1"] = decoded_preds[level]["top1"]
        df[f"Predicted_EC_L{level}_top2"] = decoded_preds[level]["top2"]
        df[f"Predicted_EC_L{level}_top3"] = decoded_preds[level]["top3"]
        df[f"Prob_L{level}_top1"] = probs_results[level]["top1"]
        df[f"Prob_L{level}_top2"] = probs_results[level]["top2"]
        df[f"Prob_L{level}_top3"] = probs_results[level]["top3"]

    # Save output to both overwrite input (if wanted) and write a separate file
    print(f"\nSaving predictions to: {OUTPUT_PARQUET_PATH}")
    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    
    # Overwriting the original parquet as well to ensure it is in yarrowia_evo2_layer18_embeddings.parquet
    print(f"Overwriting original parquet: {PARQUET_PATH}")
    df.to_parquet(PARQUET_PATH, index=False)

    print(f"\nDone! Prediction complete.")
    
    # Show first few rows for Level 4
    preview = df[["EC_Numbers", "Predicted_EC_L4_top1", "Prob_L4_top1", "Predicted_EC_L4_top2", "Prob_L4_top2", "Predicted_EC_L4_top3", "Prob_L4_top3"]].head(15)
    print("\nPreview of actual vs predicted top-3 Level 4 EC numbers and probabilities:")
    print(preview.to_string())

    # Calculate stats (only where EC_Numbers is not "UP")
    annotated = df[df["EC_Numbers"] != "UP"]
    if len(annotated) > 0:
        print("\nAccuracy and Top-k Performance by EC Level (excluding 'UP'):")
        for level in [1, 2, 3, 4]:
            true_lv = np.array([ec_at_level(e, level) for e in annotated["EC_Numbers"]])
            pred_lv1 = np.array(decoded_preds[level]["top1"])[annotated.index]
            pred_lv2 = np.array(decoded_preds[level]["top2"])[annotated.index]
            pred_lv3 = np.array(decoded_preds[level]["top3"])[annotated.index]
            
            matches1 = (true_lv == pred_lv1).sum()
            matches2 = ((true_lv == pred_lv1) | (true_lv == pred_lv2)).sum()
            matches3 = ((true_lv == pred_lv1) | (true_lv == pred_lv2) | (true_lv == pred_lv3)).sum()
            
            acc1 = matches1 / len(annotated)
            acc2 = matches2 / len(annotated)
            acc3 = matches3 / len(annotated)
            
            print(f"  Level {level}:")
            print(f"    Top-1 Accuracy: {acc1:.4f} ({matches1}/{len(annotated)})")
            print(f"    Top-2 Accuracy: {acc2:.4f} ({matches2}/{len(annotated)})")
            print(f"    Top-3 Accuracy: {acc3:.4f} ({matches3}/{len(annotated)})")


if __name__ == "__main__":
    main()
