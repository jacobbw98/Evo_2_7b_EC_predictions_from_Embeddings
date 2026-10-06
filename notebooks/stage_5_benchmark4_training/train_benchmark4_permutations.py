#!/usr/bin/env python3
"""
Train the benchmark_4 per-layer permutation models: XGBoost and MLP,
last-token and mean-token, plus the 10000-epoch layer-17 MLP run.

All experiments read Train/Valid_all_layers_embeddings.parquet:

Usage:
  python train_benchmark4_permutations.py
  python train_benchmark4_permutations.py --experiment xgb_block
  (--experiment choices: xgb_block, xgb_mean, mlp_last, mlp_mean, mlp_layer17;
   default: run all five in that order)

XGBoost experiments are CPU-only; MLP experiments require CUDA
(get_device(require_cuda=True) is created inside each MLP run function).

=== xgb_block: Train XGBoost EC classifiers for ALL 32 Evo2 layers using
LAST-TOKEN (full block output) embeddings from benchmark_4.

For each layer 0..31:
  - Feature: emb_blocks_{L} (4096-dim last-token pooled) from
    Train_all_layers_embeddings.parquet / Valid_all_layers_embeddings.parquet
  - Model: XGBoost multi:softmax (saved as xgb_layer_{L:02d}.json)
  - If a previously saved model exists it is loaded and evaluated;
    otherwise it is trained with early stopping on the validation set.
  - Metrics: level 1-4 accuracy + macro/weighted F1 (valid and train)

Outputs (models/benchmark_4/xgboost_per_layer/):
  - xgb_layer_{L:02d}.json        per-layer models
  - results_summary.json / .pkl   per-layer metric summary
  - label_encoder.pkl             full-EC label encoder

=== xgb_mean: same as xgb_block but with MEAN-TOKEN embeddings
(emb_mean_blocks_{L}, xgb_mean_layer_{L:02d}.json,
models/benchmark_4/xgboost_mean_per_layer/).

=== mlp_last: Train MLP EC classifiers for ALL 32 Evo2 layers using
LAST-TOKEN (full block output) embeddings from benchmark_4.

For each layer 0..31:
  - Feature: emb_blocks_{L} (4096-dim last-token pooled) from
    Train_all_layers_embeddings.parquet / Valid_all_layers_embeddings.parquet
  - Model: 4096 -> 1024 -> 512 -> n_classes MLP, Adam (1e-3, wd 1e-4),
    ReduceLROnPlateau, early stopping (patience 15), max 1000 epochs
    (saved as mlp_last_layer_{L:02d}.pt)
  - If a previously saved model exists it is loaded and evaluated.
  - Metrics: level 1-4 accuracy + macro/weighted F1 (valid and train)

Outputs (models/benchmark_4/mlp_last_per_layer/):
  - mlp_last_layer_{L:02d}.pt    per-layer models
  - results_summary.json / .pkl  per-layer metric summary
  - label_encoder.pkl            full-EC label encoder

=== mlp_mean: same as mlp_last but with MEAN-TOKEN embeddings
(emb_mean_blocks_{L}, mlp_mean_layer_{L:02d}.pt,
models/benchmark_4/mlp_mean_per_layer/).

=== mlp_layer17: Train a 2-hidden-layer MLP on Evo2 layer 17 embeddings
(4096-dim) to predict full EC number. 10000 epochs with early stopping.

Architecture: 4096 -> 1024 (ReLU, Dropout 0.3) -> 512 (ReLU, Dropout 0.3) -> num_classes

Feature: emb_blocks_17 from Train/Valid_all_layers_embeddings.parquet
Outputs (models/benchmark_4/mlp_layer17_10000ep/):
  mlp_layer17.pt, label_encoder.pkl, results_summary.json/.pkl,
  loss_curve.png, loss_curves.npz, accuracy_by_level.png,
  f1_by_level.png, confusion_1st_digit.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import BENCH4_DIR, MODELS_DIR, get_device, EC_MLP_Compact, ec_at_level, gpu_batches

import os, gc, time, pickle, json, glob

_FORCE = "--force" in sys.argv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PQ = str(BENCH4_DIR / "Train_all_layers_embeddings.parquet")
VALID_PQ = str(BENCH4_DIR / "Valid_all_layers_embeddings.parquet")
LAYERS = list(range(32))

def predict_xgb_robust(mpath, X_tr, y_tr, X_va, y_va, n_cls, le):
    try:
        mdl = xgb.XGBClassifier()
        mdl.load_model(mpath)
        yp_va = mdl.predict(X_va)
        yp_tr = mdl.predict(X_tr)
        return le.inverse_transform(yp_va), le.inverse_transform(yp_tr)
    except Exception as e1:
        try:
            bst = xgb.Booster()
            bst.load_model(mpath)
            dval = xgb.DMatrix(X_va)
            dtr = xgb.DMatrix(X_tr)
            preds_va = bst.predict(dval)
            preds_tr = bst.predict(dtr)
            if preds_va.ndim > 1:
                yp_va = preds_va.argmax(axis=1)
                yp_tr = preds_tr.argmax(axis=1)
            else:
                yp_va = np.round(preds_va).astype(int)
                yp_tr = np.round(preds_tr).astype(int)
            return le.inverse_transform(yp_va), le.inverse_transform(yp_tr)
        except Exception as e2:
            print(f"  Warning: Failed to load model {mpath} ({e1}; {e2}). Retraining...", flush=True)
            cc = np.bincount(y_tr, minlength=n_cls)
            sw = np.where(cc>0, len(y_tr)/(n_cls*cc), 0)[y_tr]
            mdl = xgb.XGBClassifier(
                objective="multi:softmax", eval_metric="mlogloss",
                tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
                max_depth=10, max_bin=1024, learning_rate=0.1, n_estimators=1000,
                subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, random_state=42,
                verbosity=1, num_class=n_cls, early_stopping_rounds=50)
            mdl.fit(X_tr, y_tr, eval_set=[(X_va,y_va)], sample_weight=sw, verbose=100)
            mdl.save_model(mpath)
            yp_va = mdl.predict(X_va)
            yp_tr = mdl.predict(X_tr)
            return le.inverse_transform(yp_va), le.inverse_transform(yp_tr)

def run_xgb_block():
    BASE = str(BENCH4_DIR)
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "xgboost_per_layer")

    # ==== Load data & encode labels ====
    print("\n" + "="*70 + "\nLoading data for training...\n" + "="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ)
    df_valid = pd.read_parquet(VALID_PQ)
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    y_tr_enc = le.transform(df_train["EC"].values)
    y_va_enc = le.transform(df_valid["EC"].values)
    ec_tr_str = df_train["EC"].values
    ec_va_str = df_valid["EC"].values
    n_cls = len(le.classes_)
    print(f"Classes: {n_cls}", flush=True)

    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr_enc)

    # ==== Train/evaluate all 32 layers (XGBoost, last-token) ====
    os.makedirs(MODEL_DIR, exist_ok=True)
    summary_path = os.path.join(MODEL_DIR, "results_summary.json")

    if os.path.exists(summary_path):
        print()
        print("=" * 70)
        print(f"XGBoost Last Token — Loading cached results summary from {summary_path}", flush=True)
        print("=" * 70)
        with open(summary_path) as f:
            res_dict = json.load(f)
        results = {int(k): v for k, v in res_dict.items()}
    else:
        results = {}
        print()
        print("=" * 70)
        print(f"XGBoost Last Token — Evaluating/Training all 32 layers", flush=True)
        print("=" * 70)

        for li in LAYERS:
            col = f"emb_blocks_{li}"
            print()
            print(f"--- Layer {li} ({col}) ---", flush=True)
            t0 = time.time()
            tr_mask = df_train[col].notna(); va_mask = df_valid[col].notna()
            X_tr = np.array(df_train.loc[tr_mask,col].tolist(), dtype=np.float32)
            X_va = np.array(df_valid.loc[va_mask,col].tolist(), dtype=np.float32)
            y_tr = y_tr_enc[tr_mask.values]; y_va = y_va_enc[va_mask.values]
            ec_tr_l = ec_tr_str[tr_mask.values]; ec_va_l = ec_va_str[va_mask.values]
            print(f"  X_tr:{X_tr.shape} X_va:{X_va.shape} ({time.time()-t0:.1f}s)", flush=True)

            mname = f"xgb_layer_{li:02d}.json"
            mpath = os.path.join(MODEL_DIR, mname)
            train_time = 0.0
            ec_pv, ec_pt = predict_xgb_robust(mpath, X_tr, y_tr, X_va, y_va, n_cls, le)

            lr = {"train_time": train_time}
            for lv in [1,2,3,4]:
                tv=[ec_at_level(e,lv) for e in ec_va_l]; pv=[ec_at_level(e,lv) for e in ec_pv]
                lr[f"level_{lv}_accuracy"] = accuracy_score(tv,pv)
                lr[f"level_{lv}_f1_macro"] = f1_score(tv,pv,average="macro",zero_division=0)
                lr[f"level_{lv}_f1_weighted"] = f1_score(tv,pv,average="weighted",zero_division=0)
            tt=[ec_at_level(e,4) for e in ec_tr_l]; pt=[ec_at_level(e,4) for e in ec_pt]
            lr["train_accuracy"] = accuracy_score(tt,pt)
            results[li] = lr
            print(f"  Valid={lr['level_4_accuracy']:.4f} Train={lr['train_accuracy']:.4f} ({train_time:.0f}s)", flush=True)
            del X_tr, X_va, y_tr, y_va; gc.collect()

    rj = {str(l):{k:(float(v) if isinstance(v,(float,np.floating)) else int(v) if isinstance(v,(int,np.integer)) else v) for k,v in r.items()} for l,r in results.items()}
    with open(os.path.join(MODEL_DIR,"results_summary.json"),"w") as f: json.dump(rj,f,indent=2)
    with open(os.path.join(MODEL_DIR,"results_summary.pkl"),"wb") as f: pickle.dump(results,f)
    pickle.dump(le, open(os.path.join(MODEL_DIR,"label_encoder.pkl"),"wb"))

    print(f"\nAll results and models saved to: {MODEL_DIR}")
    print("DONE!", flush=True)

def run_xgb_mean():
    BASE = str(BENCH4_DIR)
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "xgboost_mean_per_layer")

    # ==== Load data & encode labels ====
    print("\n" + "="*70 + "\nLoading data for training...\n" + "="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ)
    df_valid = pd.read_parquet(VALID_PQ)
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    y_tr_enc = le.transform(df_train["EC"].values)
    y_va_enc = le.transform(df_valid["EC"].values)
    ec_tr_str = df_train["EC"].values
    ec_va_str = df_valid["EC"].values
    n_cls = len(le.classes_)
    print(f"Classes: {n_cls}", flush=True)

    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr_enc)

    # ==== Train/evaluate all 32 layers (XGBoost, mean-token) ====
    os.makedirs(MODEL_DIR, exist_ok=True)
    summary_path = os.path.join(MODEL_DIR, "results_summary.json")

    if os.path.exists(summary_path):
        print()
        print("=" * 70)
        print(f"XGBoost Mean Token — Loading cached results summary from {summary_path}", flush=True)
        print("=" * 70)
        with open(summary_path) as f:
            res_dict = json.load(f)
        results = {int(k): v for k, v in res_dict.items()}
    else:
        results = {}
        print()
        print("=" * 70)
        print(f"XGBoost Mean Token — Evaluating/Training all 32 layers", flush=True)
        print("=" * 70)

        for li in LAYERS:
            col = f"emb_mean_blocks_{li}"
            print()
            print(f"--- Layer {li} ({col}) ---", flush=True)
            t0 = time.time()
            tr_mask = df_train[col].notna(); va_mask = df_valid[col].notna()
            X_tr = np.array(df_train.loc[tr_mask,col].tolist(), dtype=np.float32)
            X_va = np.array(df_valid.loc[va_mask,col].tolist(), dtype=np.float32)
            y_tr = y_tr_enc[tr_mask.values]; y_va = y_va_enc[va_mask.values]
            ec_tr_l = ec_tr_str[tr_mask.values]; ec_va_l = ec_va_str[va_mask.values]
            print(f"  X_tr:{X_tr.shape} X_va:{X_va.shape} ({time.time()-t0:.1f}s)", flush=True)

            mname = f"xgb_mean_layer_{li:02d}.json"
            mpath = os.path.join(MODEL_DIR, mname)
            train_time = 0.0
            ec_pv, ec_pt = predict_xgb_robust(mpath, X_tr, y_tr, X_va, y_va, n_cls, le)

            lr = {"train_time": train_time}
            for lv in [1,2,3,4]:
                tv=[ec_at_level(e,lv) for e in ec_va_l]; pv=[ec_at_level(e,lv) for e in ec_pv]
                lr[f"level_{lv}_accuracy"] = accuracy_score(tv,pv)
                lr[f"level_{lv}_f1_macro"] = f1_score(tv,pv,average="macro",zero_division=0)
                lr[f"level_{lv}_f1_weighted"] = f1_score(tv,pv,average="weighted",zero_division=0)
            tt=[ec_at_level(e,4) for e in ec_tr_l]; pt=[ec_at_level(e,4) for e in ec_pt]
            lr["train_accuracy"] = accuracy_score(tt,pt)
            results[li] = lr
            print(f"  Valid={lr['level_4_accuracy']:.4f} Train={lr['train_accuracy']:.4f} ({train_time:.0f}s)", flush=True)
            del X_tr, X_va, y_tr, y_va; gc.collect()

    rj = {str(l):{k:(float(v) if isinstance(v,(float,np.floating)) else int(v) if isinstance(v,(int,np.integer)) else v) for k,v in r.items()} for l,r in results.items()}
    with open(os.path.join(MODEL_DIR,"results_summary.json"),"w") as f: json.dump(rj,f,indent=2)
    with open(os.path.join(MODEL_DIR,"results_summary.pkl"),"wb") as f: pickle.dump(results,f)
    pickle.dump(le, open(os.path.join(MODEL_DIR,"label_encoder.pkl"),"wb"))

    print(f"\nAll results and models saved to: {MODEL_DIR}")
    print("DONE!", flush=True)

def run_mlp_last():
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "mlp_last_per_layer")
    device = get_device(require_cuda=True)

    # ==== Load data & encode labels ====
    print("\n" + "="*70 + "\nLoading data for training...\n" + "="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ)
    df_valid = pd.read_parquet(VALID_PQ)
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    y_tr_enc = le.transform(df_train["EC"].values)
    y_va_enc = le.transform(df_valid["EC"].values)
    ec_tr_str = df_train["EC"].values
    ec_va_str = df_valid["EC"].values
    n_cls = len(le.classes_)
    print(f"Classes: {n_cls}", flush=True)

    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr_enc)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    # ==== Train/evaluate all 32 layers (MLP, last-token) ====
    os.makedirs(MODEL_DIR, exist_ok=True)
    summary_path = os.path.join(MODEL_DIR, "results_summary.json")

    if os.path.exists(summary_path):
        print()
        print("=" * 70)
        print(f"MLP Last Token — Loading cached results summary from {summary_path}", flush=True)
        print("=" * 70)
        with open(summary_path) as f:
            res_dict = json.load(f)
        results = {int(k): v for k, v in res_dict.items()}
    else:
        results = {}
        print()
        print("=" * 70)
        print(f"MLP Last Token — Evaluating/Training all 32 layers", flush=True)
        print("=" * 70)

        for li in LAYERS:
            col = f"emb_blocks_{li}"
            print()
            print(f"--- Layer {li} ({col}) ---", flush=True)
            t0 = time.time()
            tr_mask = df_train[col].notna(); va_mask = df_valid[col].notna()
            X_tr = np.array(df_train.loc[tr_mask,col].tolist(), dtype=np.float32)
            X_va = np.array(df_valid.loc[va_mask,col].tolist(), dtype=np.float32)
            y_tr = y_tr_enc[tr_mask.values]; y_va = y_va_enc[va_mask.values]
            ec_tr_l = ec_tr_str[tr_mask.values]; ec_va_l = ec_va_str[va_mask.values]
            print(f"  X_tr:{X_tr.shape} X_va:{X_va.shape} ({time.time()-t0:.1f}s)", flush=True)

            mname = f"mlp_last_layer_{li:02d}.pt"
            mpath = os.path.join(MODEL_DIR, mname)
            Xt = torch.from_numpy(X_tr).float().to(device)
            yt = torch.from_numpy(y_tr).long().to(device)
            Xv = torch.from_numpy(X_va).float().to(device)
            yv = torch.from_numpy(y_va).long().to(device)

            if os.path.exists(mpath):
                print(f"  Found existing model: {mpath} — loading state dict", flush=True)
                mdl = EC_MLP_Compact(X_tr.shape[1], n_cls).to(device)
                mdl.load_state_dict(torch.load(mpath, map_location=device))
                mdl.eval()
                train_time = 0.0
                best_ep = best_tl = final_tl = best_vl = None
            else:
                mdl = EC_MLP_Compact(X_tr.shape[1], n_cls).to(device)
                crit = nn.CrossEntropyLoss(weight=cw_tensor.to(device))
                opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
                best_vl, best_ep, best_st, pat = float("inf"), 0, None, 0
                best_tl, final_tl = None, None
                t0 = time.time()
                for ep in range(1, 1001):
                    mdl.train(); tl_sum=0
                    for xb,yb in gpu_batches(Xt, yt, 256, True):
                        opt.zero_grad(); loss=crit(mdl(xb),yb); loss.backward(); opt.step()
                        tl_sum+=loss.item()*xb.size(0)
                    mdl.eval(); vl_sum=0
                    with torch.no_grad():
                        for xb,yb in gpu_batches(Xv, yv, 256, False):
                            vl_sum+=crit(mdl(xb),yb).item()*xb.size(0)
                    vl_avg=vl_sum/len(yv); tl_avg=tl_sum/len(yt); final_tl=tl_avg; sched.step(vl_avg)
                    if ep%50==0 or ep==1:
                        print(f"  Ep {ep:4d} train={tl_sum/len(yt):.4f} val={vl_avg:.4f}", flush=True)
                    if vl_avg<best_vl:
                        best_vl=vl_avg; best_ep=ep; best_tl=tl_avg; pat=0
                        best_st={k:v.cpu().clone() for k,v in mdl.state_dict().items()}
                    else:
                        pat+=1
                        if pat>=15:
                            print(f"  Early stop ep {ep} (best:{best_ep})", flush=True); break
                train_time = time.time()-t0
                mdl.load_state_dict(best_st); mdl.eval()
                torch.save(best_st, os.path.join(MODEL_DIR, mname))
                torch.cuda.empty_cache()

            preds=[]
            with torch.no_grad():
                for xb,_ in gpu_batches(Xv, yv, 256, False): preds.append(mdl(xb).argmax(1).cpu().numpy())
            ec_pv = le.inverse_transform(np.concatenate(preds))
            preds_t=[]
            with torch.no_grad():
                for xb,_ in gpu_batches(Xt, yt, 256, False): preds_t.append(mdl(xb).argmax(1).cpu().numpy())
            ec_pt = le.inverse_transform(np.concatenate(preds_t))
            del mdl, Xt, yt, Xv, yv; torch.cuda.empty_cache()

            lr = {"train_time": train_time, "best_epoch": best_ep,
                  "train_loss_final": final_tl, "train_loss_at_best": best_tl,
                  "valid_mlogloss_at_best": best_vl}
            for lv in [1,2,3,4]:
                tv=[ec_at_level(e,lv) for e in ec_va_l]; pv=[ec_at_level(e,lv) for e in ec_pv]
                lr[f"level_{lv}_accuracy"] = accuracy_score(tv,pv)
                lr[f"level_{lv}_f1_macro"] = f1_score(tv,pv,average="macro",zero_division=0)
                lr[f"level_{lv}_f1_weighted"] = f1_score(tv,pv,average="weighted",zero_division=0)
            tt=[ec_at_level(e,4) for e in ec_tr_l]; pt=[ec_at_level(e,4) for e in ec_pt]
            lr["train_accuracy"] = accuracy_score(tt,pt)
            results[li] = lr
            print(f"  Valid={lr['level_4_accuracy']:.4f} Train={lr['train_accuracy']:.4f} ({train_time:.0f}s)", flush=True)
            del X_tr, X_va, y_tr, y_va; gc.collect()

    rj = {str(l):{k:(float(v) if isinstance(v,(float,np.floating)) else int(v) if isinstance(v,(int,np.integer)) else v) for k,v in r.items()} for l,r in results.items()}
    with open(os.path.join(MODEL_DIR,"results_summary.json"),"w") as f: json.dump(rj,f,indent=2)
    with open(os.path.join(MODEL_DIR,"results_summary.pkl"),"wb") as f: pickle.dump(results,f)
    pickle.dump(le, open(os.path.join(MODEL_DIR,"label_encoder.pkl"),"wb"))

    print(f"\nAll results and models saved to: {MODEL_DIR}")
    print("DONE!", flush=True)

def run_mlp_mean():
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "mlp_mean_per_layer")
    device = get_device(require_cuda=True)

    # ==== Load data & encode labels ====
    print("\n" + "="*70 + "\nLoading data for training...\n" + "="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ)
    df_valid = pd.read_parquet(VALID_PQ)
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    y_tr_enc = le.transform(df_train["EC"].values)
    y_va_enc = le.transform(df_valid["EC"].values)
    ec_tr_str = df_train["EC"].values
    ec_va_str = df_valid["EC"].values
    n_cls = len(le.classes_)
    print(f"Classes: {n_cls}", flush=True)

    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr_enc)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    # ==== Train/evaluate all 32 layers (MLP, mean-token) ====
    os.makedirs(MODEL_DIR, exist_ok=True)
    summary_path = os.path.join(MODEL_DIR, "results_summary.json")

    if os.path.exists(summary_path):
        print()
        print("=" * 70)
        print(f"MLP Mean Token — Loading cached results summary from {summary_path}", flush=True)
        print("=" * 70)
        with open(summary_path) as f:
            res_dict = json.load(f)
        results = {int(k): v for k, v in res_dict.items()}
    else:
        results = {}
        print()
        print("=" * 70)
        print(f"MLP Mean Token — Evaluating/Training all 32 layers", flush=True)
        print("=" * 70)

        for li in LAYERS:
            col = f"emb_mean_blocks_{li}"
            print()
            print(f"--- Layer {li} ({col}) ---", flush=True)
            t0 = time.time()
            tr_mask = df_train[col].notna(); va_mask = df_valid[col].notna()
            X_tr = np.array(df_train.loc[tr_mask,col].tolist(), dtype=np.float32)
            X_va = np.array(df_valid.loc[va_mask,col].tolist(), dtype=np.float32)
            y_tr = y_tr_enc[tr_mask.values]; y_va = y_va_enc[va_mask.values]
            ec_tr_l = ec_tr_str[tr_mask.values]; ec_va_l = ec_va_str[va_mask.values]
            print(f"  X_tr:{X_tr.shape} X_va:{X_va.shape} ({time.time()-t0:.1f}s)", flush=True)

            mname = f"mlp_mean_layer_{li:02d}.pt"
            mpath = os.path.join(MODEL_DIR, mname)
            Xt = torch.from_numpy(X_tr).float().to(device)
            yt = torch.from_numpy(y_tr).long().to(device)
            Xv = torch.from_numpy(X_va).float().to(device)
            yv = torch.from_numpy(y_va).long().to(device)

            if os.path.exists(mpath):
                print(f"  Found existing model: {mpath} — loading state dict", flush=True)
                mdl = EC_MLP_Compact(X_tr.shape[1], n_cls).to(device)
                mdl.load_state_dict(torch.load(mpath, map_location=device))
                mdl.eval()
                train_time = 0.0
                best_ep = best_tl = final_tl = best_vl = None
            else:
                mdl = EC_MLP_Compact(X_tr.shape[1], n_cls).to(device)
                crit = nn.CrossEntropyLoss(weight=cw_tensor.to(device))
                opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
                best_vl, best_ep, best_st, pat = float("inf"), 0, None, 0
                best_tl, final_tl = None, None
                t0 = time.time()
                for ep in range(1, 1001):
                    mdl.train(); tl_sum=0
                    for xb,yb in gpu_batches(Xt, yt, 256, True):
                        opt.zero_grad(); loss=crit(mdl(xb),yb); loss.backward(); opt.step()
                        tl_sum+=loss.item()*xb.size(0)
                    mdl.eval(); vl_sum=0
                    with torch.no_grad():
                        for xb,yb in gpu_batches(Xv, yv, 256, False):
                            vl_sum+=crit(mdl(xb),yb).item()*xb.size(0)
                    vl_avg=vl_sum/len(yv); tl_avg=tl_sum/len(yt); final_tl=tl_avg; sched.step(vl_avg)
                    if ep%50==0 or ep==1:
                        print(f"  Ep {ep:4d} train={tl_sum/len(yt):.4f} val={vl_avg:.4f}", flush=True)
                    if vl_avg<best_vl:
                        best_vl=vl_avg; best_ep=ep; best_tl=tl_avg; pat=0
                        best_st={k:v.cpu().clone() for k,v in mdl.state_dict().items()}
                    else:
                        pat+=1
                        if pat>=15:
                            print(f"  Early stop ep {ep} (best:{best_ep})", flush=True); break
                train_time = time.time()-t0
                mdl.load_state_dict(best_st); mdl.eval()
                torch.save(best_st, os.path.join(MODEL_DIR, mname))
                torch.cuda.empty_cache()

            preds=[]
            with torch.no_grad():
                for xb,_ in gpu_batches(Xv, yv, 256, False): preds.append(mdl(xb).argmax(1).cpu().numpy())
            ec_pv = le.inverse_transform(np.concatenate(preds))
            preds_t=[]
            with torch.no_grad():
                for xb,_ in gpu_batches(Xt, yt, 256, False): preds_t.append(mdl(xb).argmax(1).cpu().numpy())
            ec_pt = le.inverse_transform(np.concatenate(preds_t))
            del mdl, Xt, yt, Xv, yv; torch.cuda.empty_cache()

            lr = {"train_time": train_time, "best_epoch": best_ep,
                  "train_loss_final": final_tl, "train_loss_at_best": best_tl,
                  "valid_mlogloss_at_best": best_vl}
            for lv in [1,2,3,4]:
                tv=[ec_at_level(e,lv) for e in ec_va_l]; pv=[ec_at_level(e,lv) for e in ec_pv]
                lr[f"level_{lv}_accuracy"] = accuracy_score(tv,pv)
                lr[f"level_{lv}_f1_macro"] = f1_score(tv,pv,average="macro",zero_division=0)
                lr[f"level_{lv}_f1_weighted"] = f1_score(tv,pv,average="weighted",zero_division=0)
            tt=[ec_at_level(e,4) for e in ec_tr_l]; pt=[ec_at_level(e,4) for e in ec_pt]
            lr["train_accuracy"] = accuracy_score(tt,pt)
            results[li] = lr
            print(f"  Valid={lr['level_4_accuracy']:.4f} Train={lr['train_accuracy']:.4f} ({train_time:.0f}s)", flush=True)
            del X_tr, X_va, y_tr, y_va; gc.collect()

    rj = {str(l):{k:(float(v) if isinstance(v,(float,np.floating)) else int(v) if isinstance(v,(int,np.integer)) else v) for k,v in r.items()} for l,r in results.items()}
    with open(os.path.join(MODEL_DIR,"results_summary.json"),"w") as f: json.dump(rj,f,indent=2)
    with open(os.path.join(MODEL_DIR,"results_summary.pkl"),"wb") as f: pickle.dump(results,f)
    pickle.dump(le, open(os.path.join(MODEL_DIR,"label_encoder.pkl"),"wb"))

    print(f"\nAll results and models saved to: {MODEL_DIR}")
    print("DONE!", flush=True)

def run_mlp_layer17():
    LAYER = 17
    COL   = f"emb_blocks_{LAYER}"
    MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "mlp_layer17_10000ep")
    if not _FORCE and os.path.exists(os.path.join(MODEL_DIR, "mlp_layer17.pt")) \
            and os.path.exists(os.path.join(MODEL_DIR, "results_summary.json")):
        print("[skip] mlp_layer17_10000ep already trained — skipping (pass --force to retrain)")
        return
    MAX_EPOCHS = 10000
    PATIENCE   = 15
    BATCH_SIZE = 256

    device = get_device(require_cuda=True)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==== Load data & encode labels ====
    print("="*70, flush=True)
    print(f"MLP Layer {LAYER} ({COL}) — {MAX_EPOCHS}-epoch run", flush=True)
    print("="*70, flush=True)
    df_train = pd.read_parquet(TRAIN_PQ, columns=[COL, "EC"])
    df_valid = pd.read_parquet(VALID_PQ, columns=[COL, "EC"])
    print(f"Train: {len(df_train)} rows | Valid: {len(df_valid)} rows", flush=True)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train["EC"].values, df_valid["EC"].values]))
    n_cls = len(le.classes_)
    y_tr = le.transform(df_train["EC"].values)
    y_va = le.transform(df_valid["EC"].values)
    ec_tr = df_train["EC"].values
    ec_va = df_valid["EC"].values

    tr_mask = df_train[COL].notna().values
    va_mask = df_valid[COL].notna().values
    X_tr = np.array(df_train.loc[tr_mask, COL].tolist(), dtype=np.float32)
    X_va = np.array(df_valid.loc[va_mask, COL].tolist(), dtype=np.float32)
    y_tr = y_tr[tr_mask]; y_va = y_va[va_mask]
    ec_tr = ec_tr[tr_mask]; ec_va = ec_va[va_mask]
    print(f"X_tr:{X_tr.shape} X_va:{X_va.shape} | classes:{n_cls}", flush=True)
    del df_train, df_valid; gc.collect()

    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr).long().to(device)
    Xv = torch.from_numpy(X_va).float().to(device)
    yv = torch.from_numpy(y_va).long().to(device)

    mdl = EC_MLP_Compact(X_tr.shape[1], n_cls).to(device)
    crit = nn.CrossEntropyLoss(weight=cw_tensor.to(device))
    opt  = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    # ==== Training loop ====
    best_vl, best_ep, best_st, pat = float("inf"), 0, None, 0
    train_losses, val_losses = [], []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        mdl.train(); tl_sum = 0
        for xb, yb in gpu_batches(Xt, yt, BATCH_SIZE, True):
            opt.zero_grad(); loss = crit(mdl(xb), yb); loss.backward(); opt.step()
            tl_sum += loss.item() * xb.size(0)
        mdl.eval(); vl_sum = 0
        with torch.no_grad():
            for xb, yb in gpu_batches(Xv, yv, BATCH_SIZE, False):
                vl_sum += crit(mdl(xb), yb).item() * xb.size(0)
        tl_avg, vl_avg = tl_sum / len(yt), vl_sum / len(yv)
        train_losses.append(tl_avg); val_losses.append(vl_avg)
        sched.step(vl_avg)
        if ep % 100 == 0 or ep == 1:
            print(f"  Ep {ep:5d} train={tl_avg:.4f} val={vl_avg:.4f} (best_ep={best_ep})", flush=True)
        if vl_avg < best_vl:
            best_vl, best_ep, pat = vl_avg, ep, 0
            best_st = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"  Early stop at ep {ep} (best: {best_ep})", flush=True)
                break
    train_time = time.time() - t0
    total_epochs_run = len(train_losses)
    mdl.load_state_dict(best_st); mdl.eval()

    # ==== Save model, losses, encoder ====
    torch.save(best_st, os.path.join(MODEL_DIR, "mlp_layer17.pt"))
    np.savez(os.path.join(MODEL_DIR, "loss_curves.npz"),
             epochs=np.arange(1, total_epochs_run + 1),
             train_loss=np.array(train_losses), val_loss=np.array(val_losses))
    pickle.dump(le, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))

    # ==== Evaluate ====
    def predict_ec(model, X, y):
        out = []
        with torch.no_grad():
            for xb, _ in gpu_batches(X, y, BATCH_SIZE, False):
                out.append(model(xb).argmax(1).cpu().numpy())
        return le.inverse_transform(np.concatenate(out))

    ec_pv = predict_ec(mdl, Xv, yv)
    ec_pt = predict_ec(mdl, Xt, yt)

    results = {"train_time": train_time, "best_epoch": best_ep,
               "best_val_loss": best_vl, "input_dim": X_tr.shape[1],
               "total_epochs_run": total_epochs_run}
    for lv in [1, 2, 3, 4]:
        tv = [ec_at_level(e, lv) for e in ec_va]
        pv = [ec_at_level(e, lv) for e in ec_pv]
        results[f"level_{lv}_accuracy"] = accuracy_score(tv, pv)
        results[f"level_{lv}_f1_macro"] = f1_score(tv, pv, average="macro", zero_division=0)
        results[f"level_{lv}_f1_weighted"] = f1_score(tv, pv, average="weighted", zero_division=0)

    rj = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
          for k, v in results.items()}
    with open(os.path.join(MODEL_DIR, "results_summary.json"), "w") as f:
        json.dump(rj, f, indent=1)
    pickle.dump(results, open(os.path.join(MODEL_DIR, "results_summary.pkl"), "wb"))

    # ==== Plots ====
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, total_epochs_run + 1), train_losses, label="Train loss")
    ax.plot(range(1, total_epochs_run + 1), val_losses, label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"MLP Layer {LAYER} loss curves (best epoch {best_ep})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(MODEL_DIR, "loss_curve.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    levels = [1, 2, 3, 4]
    accs  = [results[f"level_{lv}_accuracy"] for lv in levels]
    f1m   = [results[f"level_{lv}_f1_macro"] for lv in levels]
    f1w   = [results[f"level_{lv}_f1_weighted"] for lv in levels]
    axes[0].bar([f"Level {lv}" for lv in levels], accs, color="steelblue")
    axes[0].axhline(0, color="k", lw=0.5); axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"MLP Layer {LAYER} accuracy by EC level")
    for i, v in enumerate(accs): axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    axes[1].bar([f"Level {lv}" for lv in levels], f1m, color="darkorange", label="macro")
    axes[1].bar([f"Level {lv}" for lv in levels], f1w, color="green", label="weighted", alpha=0.7)
    axes[1].set_ylabel("F1"); axes[1].set_title(f"MLP Layer {LAYER} F1 by EC level"); axes[1].legend()
    for i, v in enumerate(f1m): axes[1].text(i - 0.2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(MODEL_DIR, "accuracy_by_level.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    y1 = [ec_at_level(e, 1) for e in ec_va]
    p1 = [ec_at_level(e, 1) for e in ec_pv]
    cm = confusion_matrix(y1, p1, labels=[str(c) for c in range(1, 8)])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[str(c) for c in range(1, 8)], yticklabels=[str(c) for c in range(1, 8)])
    ax.set_xlabel("Predicted 1st digit"); ax.set_ylabel("True 1st digit")
    ax.set_title(f"MLP Layer {LAYER} — 1st-digit EC confusion matrix")
    fig.tight_layout(); fig.savefig(os.path.join(MODEL_DIR, "confusion_1st_digit.png"), dpi=150); plt.close(fig)

    print(f"\nSaved model + artifacts to {MODEL_DIR}", flush=True)
    print(f"Val level-4 acc = {results['level_4_accuracy']:.4f} | best epoch {best_ep}/{total_epochs_run} | {train_time:.0f}s", flush=True)
    print("DONE!", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Train benchmark_4 per-layer permutation models "
                    "(xgb_block, xgb_mean, mlp_last, mlp_mean, mlp_layer17).")
    parser.add_argument("--experiment",
                        choices=["xgb_block", "xgb_mean", "mlp_last", "mlp_mean", "mlp_layer17"],
                        default=None,
                        help="Run a single experiment (default: run all five in order).")
    args = parser.parse_args()

    if args.experiment is None:
        run_xgb_block()
        run_xgb_mean()
        run_mlp_last()
        run_mlp_mean()
        run_mlp_layer17()
    elif args.experiment == "xgb_block":
        run_xgb_block()
    elif args.experiment == "xgb_mean":
        run_xgb_mean()
    elif args.experiment == "mlp_last":
        run_mlp_last()
    elif args.experiment == "mlp_mean":
        run_mlp_mean()
    else:
        run_mlp_layer17()

if __name__ == "__main__":
    main()
