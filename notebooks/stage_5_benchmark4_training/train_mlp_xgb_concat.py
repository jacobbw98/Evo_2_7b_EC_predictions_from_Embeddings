#!/usr/bin/env python3
"""
Train MLP and XGBoost models on Evo2 layer embeddings for EC number prediction.

Experiments:
  1. MLP on layer 17 alone                (4096 features)
  2. XGBoost on concatenated layers [17, 12, 16]  (12288 features)
  3. MLP     on concatenated layers [17, 12, 16]  (12288 features)
  4. XGBoost on concatenated layers [17, 2, 30]   (12288 features)
  5. MLP     on concatenated layers [17, 2, 30]   (12288 features)

MLP architecture: input → 1024 (ReLU, Dropout 0.3) → 512 (ReLU, Dropout 0.3) → num_classes
All models evaluated at each EC hierarchy level (1st–4th digit).

Feature columns: emb_blocks_{L} from Train/Valid_all_layers_embeddings.parquet
Outputs (models/benchmark_4/mlp_xgb_concat_results/):
  mlp_mlp_layer17.pt, mlp_mlp_concat_17_12_16.pt, mlp_mlp_concat_17_2_30.pt,
  xgb_xgb_concat_17_12_16.json, xgb_xgb_concat_17_2_30.json,
  results_summary.json / .pkl, label_encoder.pkl,
  accuracy_comparison.png, f1_comparison.png,
  confusion_1st_digit_mlp_concat_17_12_16.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import BENCH4_DIR, MODELS_DIR, get_device, EC_MLP_Compact, ec_at_level, gpu_batches

import os, gc, time, pickle, json
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

# Configuration
TRAIN_PQ  = str(BENCH4_DIR / "Train_all_layers_embeddings.parquet")
VALID_PQ  = str(BENCH4_DIR / "Valid_all_layers_embeddings.parquet")
MODEL_DIR = str(MODELS_DIR / "benchmark_4" / "mlp_xgb_concat_results")

EXPERIMENTS = [
    # (name, model_type, layers, description)
    ("mlp_layer17",            "mlp",     [17],          "MLP on Layer 17 alone"),
    ("xgb_concat_17_12_16",    "xgboost", [17, 12, 16],  "XGBoost on concatenated layers [17, 12, 16]"),
    ("mlp_concat_17_12_16",    "mlp",     [17, 12, 16],  "MLP on concatenated layers [17, 12, 16]"),
    ("xgb_concat_17_2_30",     "xgboost", [17, 2, 30],   "XGBoost on concatenated layers [17, 2, 30]"),
    ("mlp_concat_17_2_30",     "mlp",     [17, 2, 30],   "MLP on concatenated layers [17, 2, 30]"),
]
MAX_EPOCHS = 1000
PATIENCE   = 15
BATCH_SIZE = 256

DEVICE = get_device(require_cuda=True)

def model_file(name, model_type):
    return os.path.join(MODEL_DIR, f"{'mlp' if model_type=='mlp' else 'xgb'}_{name}.{'pt' if model_type=='mlp' else 'json'}")

_FORCE = "--force" in sys.argv
if not _FORCE and os.path.exists(os.path.join(MODEL_DIR, "results_summary.json")) \
        and all(os.path.exists(model_file(n, t)) for n, t, _, _ in EXPERIMENTS):
    print("[skip] mlp_xgb_concat_results already trained — skipping (pass --force to retrain)")
    raise SystemExit(0)

# ==== Load data & encode labels ====
print("="*70, flush=True)
print("MLP / XGBoost concatenation experiments (5 models)", flush=True)
print("="*70, flush=True)
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

os.makedirs(MODEL_DIR, exist_ok=True)
results = {}

def fit_mlp(X_tr, y_tr, X_va, y_va, mpath):
    Xt = torch.from_numpy(X_tr).float().to(DEVICE)
    yt = torch.from_numpy(y_tr).long().to(DEVICE)
    Xv = torch.from_numpy(X_va).float().to(DEVICE)
    yv = torch.from_numpy(y_va).long().to(DEVICE)
    mdl = EC_MLP_Compact(X_tr.shape[1], n_cls).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw_tensor.to(DEVICE))
    opt  = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    best_vl, best_ep, best_st, pat = float("inf"), 0, None, 0
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
        vl_avg = vl_sum / len(yv)
        sched.step(vl_avg)
        if ep % 50 == 0 or ep == 1:
            print(f"    Ep {ep:4d} train={tl_sum/len(yt):.4f} val={vl_avg:.4f}", flush=True)
        if vl_avg < best_vl:
            best_vl, best_ep, pat = vl_avg, ep, 0
            best_st = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"    Early stop ep {ep} (best:{best_ep})", flush=True)
                break
    train_time = time.time() - t0
    mdl.load_state_dict(best_st); mdl.eval()
    torch.save(best_st, mpath)
    return mdl, train_time, best_ep, best_vl

def fit_xgb(X_tr, y_tr, X_va, y_va, mpath):
    cc = np.bincount(y_tr, minlength=n_cls)
    sw = np.where(cc > 0, len(y_tr) / (n_cls * cc), 0)[y_tr]
    mdl = xgb.XGBClassifier(
        objective="multi:softmax", eval_metric="mlogloss",
        tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
        max_depth=10, max_bin=1024, learning_rate=0.1, n_estimators=1000,
        subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, random_state=42,
        verbosity=1, num_class=n_cls, early_stopping_rounds=50)
    t0 = time.time()
    mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=sw, verbose=100)
    train_time = time.time() - t0
    mdl.save_model(mpath)
    best_iteration = int(mdl.best_iteration) if hasattr(mdl, "best_iteration") and mdl.best_iteration is not None else 1000
    return mdl, train_time, best_iteration, None

# ==== Run experiments ====
for name, model_type, layers, description in EXPERIMENTS:
    cols = [f"emb_blocks_{L}" for L in layers]
    print()
    print(f"--- {name} | {description} | input_dim={len(cols)*4096} ---", flush=True)
    t0 = time.time()
    tr_mask = df_train[cols[0]].notna().values
    va_mask = df_valid[cols[0]].notna().values
    X_tr = np.concatenate([np.array(df_train.loc[tr_mask, c].tolist(), dtype=np.float32) for c in cols], axis=1)
    X_va = np.concatenate([np.array(df_valid.loc[va_mask, c].tolist(), dtype=np.float32) for c in cols], axis=1)
    y_tr = y_tr_enc[tr_mask]; y_va = y_va_enc[va_mask]
    ec_tr_l = ec_tr_str[tr_mask]; ec_va_l = ec_va_str[va_mask]
    print(f"  X_tr:{X_tr.shape} X_va:{X_va.shape} ({time.time()-t0:.1f}s)", flush=True)

    mpath = model_file(name, model_type)
    if model_type == "mlp":
        mdl, train_time, best_ep, best_vl = fit_mlp(X_tr, y_tr, X_va, y_va, mpath)
    else:
        mdl, train_time, best_it, _ = fit_xgb(X_tr, y_tr, X_va, y_va, mpath)

    preds = []
    if model_type == "mlp":
        Xv_t = torch.from_numpy(X_va).float().to(DEVICE)
        yv_t = torch.from_numpy(y_va).long().to(DEVICE)
        with torch.no_grad():
            for xb, _ in gpu_batches(Xv_t, yv_t, BATCH_SIZE, False):
                preds.append(mdl(xb).argmax(1).cpu().numpy())
        del Xv_t, yv_t; torch.cuda.empty_cache()
    else:
        preds = [mdl.predict(X_va)]
    ec_pv = le.inverse_transform(np.concatenate(preds))

    lr = {"train_time": train_time,
          "layers": layers, "model_type": model_type,
          "description": description, "input_dim": int(X_tr.shape[1])}
    if model_type == "mlp":
        lr["best_epoch"] = int(best_ep)
        lr["best_val_loss"] = float(best_vl)
    else:
        lr["best_iteration"] = int(best_it)
    for lv in [1, 2, 3, 4]:
        tv = [ec_at_level(e, lv) for e in ec_va_l]
        pv = [ec_at_level(e, lv) for e in ec_pv]
        lr[f"level_{lv}_accuracy"] = accuracy_score(tv, pv)
        lr[f"level_{lv}_f1_macro"] = f1_score(tv, pv, average="macro", zero_division=0)
        lr[f"level_{lv}_f1_weighted"] = f1_score(tv, pv, average="weighted", zero_division=0)
    results[name] = lr
    print(f"  Val={lr['level_4_accuracy']:.4f} ({train_time:.0f}s)", flush=True)
    del X_tr, X_va, y_tr, y_va, mdl; gc.collect()
    if model_type == "mlp":
        torch.cuda.empty_cache()

# ==== Save results ====
rj = {k: {kk: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
          for kk, v in r.items()} for k, r in results.items()}
with open(os.path.join(MODEL_DIR, "results_summary.json"), "w") as f:
    json.dump(rj, f, indent=1)
with open(os.path.join(MODEL_DIR, "results_summary.pkl"), "wb") as f:
    pickle.dump(results, f)
pickle.dump(le, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))

# ==== Comparison plots ====
sns.set_style("whitegrid")
names = list(results.keys())
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(names))
for lv, c in [(1, "#1f77b4"), (2, "#ff7f0e"), (3, "#2ca02c"), (4, "#d62728")]:
    ax.bar(x + (lv - 2.5) * 0.2, [results[n][f"level_{lv}_accuracy"] for n in names],
           width=0.2, label=f"Level {lv}", color=c)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_concat", "\nconcat") for n in names], fontsize=8)
ax.set_ylabel("Accuracy"); ax.set_title("MLP/XGBoost concat experiments — accuracy by EC level")
ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "accuracy_comparison.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))
for lv, c in [(1, "#1f77b4"), (2, "#ff7f0e"), (3, "#2ca02c"), (4, "#d62728")]:
    ax.bar(x + (lv - 2.5) * 0.2, [results[n][f"level_{lv}_f1_macro"] for n in names],
           width=0.2, label=f"Level {lv}", color=c)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_concat", "\nconcat") for n in names], fontsize=8)
ax.set_ylabel("F1 (macro)"); ax.set_title("MLP/XGBoost concat experiments — F1 macro by EC level")
ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "f1_comparison.png"), dpi=150)
plt.close(fig)

# Confusion matrix for the 17-12-16 MLP concatenation
print()
print("Computing 1st-digit confusion matrix for mlp_concat_17_12_16 ...", flush=True)
mdl = EC_MLP_Compact(12288, n_cls).to(DEVICE)
mdl.load_state_dict(torch.load(model_file("mlp_concat_17_12_16", "mlp"), map_location=DEVICE))
mdl.eval()
cols = [f"emb_blocks_{L}" for L in [17, 12, 16]]
va_mask = df_valid[cols[0]].notna().values
X_va = np.concatenate([np.array(df_valid.loc[va_mask, c].tolist(), dtype=np.float32) for c in cols], axis=1)
ec_va_l = ec_va_str[va_mask]
preds = []
Xv_t = torch.from_numpy(X_va).float().to(DEVICE)
yv_t = torch.from_numpy(y_va_enc[va_mask]).long().to(DEVICE)
with torch.no_grad():
    for xb, _ in gpu_batches(Xv_t, yv_t, BATCH_SIZE, False):
        preds.append(mdl(xb).argmax(1).cpu().numpy())
ec_pv = le.inverse_transform(np.concatenate(preds))
fig, ax = plt.subplots(figsize=(8, 7))
y1 = [ec_at_level(e, 1) for e in ec_va_l]
p1 = [ec_at_level(e, 1) for e in ec_pv]
cm = confusion_matrix(y1, p1, labels=[str(c) for c in range(1, 8)])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[str(c) for c in range(1, 8)], yticklabels=[str(c) for c in range(1, 8)])
ax.set_xlabel("Predicted 1st digit"); ax.set_ylabel("True 1st digit")
ax.set_title("MLP concat [17,12,16] — 1st-digit EC confusion matrix")
fig.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "confusion_1st_digit_mlp_concat_17_12_16.png"), dpi=150)
plt.close(fig)

print(f"\nAll results and models saved to: {MODEL_DIR}")
print("DONE!", flush=True)
