#!/usr/bin/env python3
"""
Train the benchmark_4 Random Forest models (CPU-only, no torch device).

Both experiments read Train/Valid_with_embeddings.parquet:
  - rf_benchmark4: 9 RF models, one per layer/type; models/benchmark_4/rf_benchmark4/
  - rf_combined: single RF on all 9 concatenated feature columns;
      models/benchmark_4/rf_combined_results/

Usage:
  python train_benchmark4_rf.py
  python train_benchmark4_rf.py --experiment rf_benchmark4
  (--experiment choices: rf_benchmark4, rf_combined; default: run both)

=== rf_benchmark4: Train 9 Random Forest models to predict EC number.
One model per (layer, type) combination:
  Layers: 9, 24, 26
  Types: block, mlp, post_norm
Train on Train_with_embeddings.parquet, test on Valid_with_embeddings.parquet.

Outputs (models/benchmark_4/rf_benchmark4/):
  rf_layer_{L}_{type}.pkl   per-(layer,type) models
  results_summary.json/.pkl per-model metric summary
  label_encoder.pkl         full-EC label encoder
  rf_benchmark4_comparison.png  level-4 accuracy bar chart

=== rf_combined: Train a single Random Forest on the CONCATENATED
embeddings of all 9 benchmark_4 feature columns (layers 9/24/26 x
block/mlp/post_norm = 9 x 4096 = 36864 features) to predict full EC number.

Train on Train_with_embeddings.parquet, test on Valid_with_embeddings.parquet.

Outputs (models/benchmark_4/rf_combined_results/):
  rf_combined_model.pkl     the trained Random Forest
  results.pkl               level_results, per_ec, train_time, n_train,
                            n_valid, n_features, n_classes
  label_encoder.pkl         full-EC label encoder
  confusion_1st_digit.png   1st-digit EC confusion matrix
  hierarchy_level_results.png  accuracy/F1 by EC level
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import BENCH4_DIR, MODELS_DIR, ec_at_level

import os
import glob, gc, time, pickle, json

_FORCE = "--force" in sys.argv
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PQ  = str(BENCH4_DIR / "Train_with_embeddings.parquet")
VALID_PQ  = str(BENCH4_DIR / "Valid_with_embeddings.parquet")

def run_rf_benchmark4():
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "rf_benchmark4")
    if not _FORCE and os.path.exists(os.path.join(MODEL_DIR, "results_summary.json")) \
            and glob.glob(os.path.join(MODEL_DIR, "rf_*.pkl")):
        print("[skip] rf_benchmark4 already trained — skipping (pass --force to retrain)")
        return
    LAYERS = [9, 24, 26]
    TYPES  = ["block", "mlp", "post_norm"]
    N_ESTIMATORS = 300

    def col_name(L, t):
        return f"emb_blocks_{L}" if t == "block" else f"emb_blocks_{L}_{t}"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==== Load data & encode labels ====
    print("="*70, flush=True)
    print("Random Forest benchmark_4 (9 x layer/type models)", flush=True)
    print("="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ)
    df_valid = pd.read_parquet(VALID_PQ)
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    n_cls = len(le.classes_)
    ec_tr = df_train["EC"].values
    ec_va = df_valid["EC"].values
    print(f"Classes: {n_cls}", flush=True)

    results = {}
    labels = []
    accs = []
    for L in LAYERS:
        for t in TYPES:
            col = col_name(L, t)
            key = f"layer_{L}_{t}"
            print()
            print(f"--- {key} ({col}) ---", flush=True)
            t0 = time.time()
            tr_mask = df_train[col].notna().values
            va_mask = df_valid[col].notna().values
            X_tr = np.array(df_train.loc[tr_mask, col].tolist(), dtype=np.float32)
            X_va = np.array(df_valid.loc[va_mask, col].tolist(), dtype=np.float32)
            y_tr = le.transform(ec_tr[tr_mask])
            y_va = le.transform(ec_va[va_mask])
            ec_tr_l = ec_tr[tr_mask]
            ec_va_l = ec_va[va_mask]
            print(f"  X_tr:{X_tr.shape} X_va:{X_va.shape} ({time.time()-t0:.1f}s)", flush=True)

            mname = f"rf_layer_{L}_{t}.pkl"
            mpath = os.path.join(MODEL_DIR, mname)
            train_time = 0.0
            if os.path.exists(mpath):
                print(f"  Found existing model: {mpath} — loading", flush=True)
                mdl = pickle.load(open(mpath, "rb"))
            else:
                t0 = time.time()
                mdl = RandomForestClassifier(
                    n_estimators=N_ESTIMATORS, n_jobs=-1,
                    class_weight="balanced", random_state=42)
                mdl.fit(X_tr, y_tr)
                train_time = time.time() - t0
                pickle.dump(mdl, open(mpath, "wb"))
                print(f"  Trained in {train_time:.0f}s — saved {mpath}", flush=True)

            yp_va = mdl.predict(X_va)
            yp_tr = mdl.predict(X_tr)
            ec_pv = le.inverse_transform(yp_va)
            ec_pt = le.inverse_transform(yp_tr)

            lr = {"train_time": train_time}
            for lv in [1, 2, 3, 4]:
                tv = [ec_at_level(e, lv) for e in ec_va_l]
                pv = [ec_at_level(e, lv) for e in ec_pv]
                lr[f"level_{lv}_accuracy"] = accuracy_score(tv, pv)
                lr[f"level_{lv}_f1_macro"] = f1_score(tv, pv, average="macro", zero_division=0)
                lr[f"level_{lv}_f1_weighted"] = f1_score(tv, pv, average="weighted", zero_division=0)
            tt = [ec_at_level(e, 4) for e in ec_tr_l]
            pt = [ec_at_level(e, 4) for e in ec_pt]
            lr["train_accuracy"] = accuracy_score(tt, pt)
            results[key] = lr
            print(f"  Valid={lr['level_4_accuracy']:.4f} Train={lr['train_accuracy']:.4f}", flush=True)
            labels.append(key)
            accs.append(lr["level_4_accuracy"])
            del X_tr, X_va, mdl; gc.collect()

    # ==== Save results ====
    rj = {k: {kk: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
              for kk, v in r.items()} for k, r in results.items()}
    with open(os.path.join(MODEL_DIR, "results_summary.json"), "w") as f:
        json.dump(rj, f, indent=2)
    with open(os.path.join(MODEL_DIR, "results_summary.pkl"), "wb") as f:
        pickle.dump(results, f)
    pickle.dump(le, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))

    # ==== Comparison plot ====
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([l.replace("_", "\n", 1) for l in labels], accs, color="steelblue")
    ax.set_ylabel("Accuracy (full EC)")
    ax.set_title("Random Forest benchmark_4 — level-4 accuracy by layer/type")
    for i, v in enumerate(accs):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "rf_benchmark4_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll results and models saved to: {MODEL_DIR}")
    print("DONE!", flush=True)

def run_rf_combined():
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "rf_combined_results")
    if not _FORCE and os.path.exists(os.path.join(MODEL_DIR, "results_summary.json")) \
            and glob.glob(os.path.join(MODEL_DIR, "rf_*.pkl")):
        print("[skip] rf_combined already trained — skipping (pass --force to retrain)")
        return
    LAYERS = [9, 24, 26]
    SUFFIX = {"block": "", "mlp": "_mlp", "post_norm": "_post_norm"}
    COLS = [f"emb_blocks_{L}{SUFFIX[t]}" for L in LAYERS for t in ("block", "mlp", "post_norm")]
    N_ESTIMATORS = 300

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==== Load data & build concatenated features ====
    print("="*70, flush=True)
    print(f"Random Forest combined ({len(COLS)} x 4096 = {len(COLS)*4096} features)", flush=True)
    print("="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ)
    df_valid = pd.read_parquet(VALID_PQ)
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    n_cls = len(le.classes_)
    ec_tr = df_train["EC"].values
    ec_va = df_valid["EC"].values

    tr_mask = df_train[COLS[0]].notna().values
    va_mask = df_valid[COLS[0]].notna().values
    X_tr = np.concatenate([np.array(df_train.loc[tr_mask, c].tolist(), dtype=np.float32) for c in COLS], axis=1)
    X_va = np.concatenate([np.array(df_valid.loc[va_mask, c].tolist(), dtype=np.float32) for c in COLS], axis=1)
    y_tr = le.transform(ec_tr[tr_mask])
    y_va = le.transform(ec_va[va_mask])
    ec_tr_l = ec_tr[tr_mask]
    ec_va_l = ec_va[va_mask]
    print(f"X_tr:{X_tr.shape} X_va:{X_va.shape} | classes:{n_cls}", flush=True)
    del df_train, df_valid; gc.collect()

    # ==== Train (or load) the combined RF ====
    mpath = os.path.join(MODEL_DIR, "rf_combined_model.pkl")
    if os.path.exists(mpath):
        print(f"Found existing model: {mpath} — loading", flush=True)
        mdl = pickle.load(open(mpath, "rb"))
        train_time = 0.0
    else:
        t0 = time.time()
        mdl = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, n_jobs=-1,
            class_weight="balanced", random_state=42)
        mdl.fit(X_tr, y_tr)
        train_time = time.time() - t0
        pickle.dump(mdl, open(mpath, "wb"))
        print(f"Trained in {train_time:.0f}s — saved {mpath}", flush=True)

    yp_va = mdl.predict(X_va)
    yp_tr = mdl.predict(X_tr)
    ec_pv = le.inverse_transform(yp_va)
    ec_pt = le.inverse_transform(yp_tr)

    # ==== Results ====
    level_results = {}
    for lv in [1, 2, 3, 4]:
        tv = [ec_at_level(e, lv) for e in ec_va_l]
        pv = [ec_at_level(e, lv) for e in ec_pv]
        level_results[lv] = {
            "accuracy": accuracy_score(tv, pv),
            "f1_macro": f1_score(tv, pv, average="macro", zero_division=0),
            "f1_weighted": f1_score(tv, pv, average="weighted", zero_division=0),
        }
        print(f"Level {lv}: acc={level_results[lv]['accuracy']:.4f} "
              f"f1_macro={level_results[lv]['f1_macro']:.4f} "
              f"f1_weighted={level_results[lv]['f1_weighted']:.4f}", flush=True)

    per_ec = {ec: v for ec, v in classification_report(
        ec_va_l, ec_pv, output_dict=True, zero_division=0).items() if ec != "accuracy"}

    results = {
        "level_results": level_results,
        "per_ec": per_ec,
        "train_time": train_time,
        "n_train": int(len(y_tr)),
        "n_valid": int(len(y_va)),
        "n_features": int(X_tr.shape[1]),
        "n_classes": int(n_cls),
        "train_accuracy": accuracy_score(ec_tr_l, ec_pt),
    }
    pickle.dump(results, open(os.path.join(MODEL_DIR, "results.pkl"), "wb"))
    pickle.dump(le, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))

    # ==== Plots ====
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))
    y1 = [ec_at_level(e, 1) for e in ec_va_l]
    p1 = [ec_at_level(e, 1) for e in ec_pv]
    cm = confusion_matrix(y1, p1, labels=[str(c) for c in range(1, 8)])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[str(c) for c in range(1, 8)], yticklabels=[str(c) for c in range(1, 8)])
    ax.set_xlabel("Predicted 1st digit"); ax.set_ylabel("True 1st digit")
    ax.set_title("RF combined — 1st-digit EC confusion matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_1st_digit.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = list(range(1, 5))
    ax.bar([f"Level {lv}" for lv in levels],
           [level_results[lv]["accuracy"] for lv in levels],
           color="steelblue", label="Accuracy")
    ax.bar([f"Level {lv}" for lv in levels],
           [level_results[lv]["f1_macro"] for lv in levels],
           color="darkorange", label="F1 macro")
    ax.bar([f"Level {lv}" for lv in levels],
           [level_results[lv]["f1_weighted"] for lv in levels],
           color="green", label="F1 weighted")
    ax.set_ylabel("Score"); ax.set_title("RF combined — EC hierarchy level results")
    ax.legend()
    for i, lv in enumerate(levels):
        ax.text(i, level_results[lv]["accuracy"] + 0.01,
                f"{level_results[lv]['accuracy']:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "hierarchy_level_results.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll results and model saved to: {MODEL_DIR}")
    print(f"Val full-EC acc = {level_results[4]['accuracy']:.4f} | train {results['train_accuracy']:.4f} | {train_time:.0f}s", flush=True)
    print("DONE!", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Train benchmark_4 Random Forest models (CPU-only).")
    parser.add_argument("--experiment",
                        choices=["rf_benchmark4", "rf_combined"],
                        default=None,
                        help="Run a single experiment (default: run both).")
    args = parser.parse_args()

    if args.experiment is None:
        run_rf_benchmark4()
        run_rf_combined()
    elif args.experiment == "rf_benchmark4":
        run_rf_benchmark4()
    else:
        run_rf_combined()

if __name__ == "__main__":
    main()
