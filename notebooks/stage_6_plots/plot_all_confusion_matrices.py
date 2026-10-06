#!/usr/bin/env python3
"""
Plot 1st-digit EC confusion matrices for every trained benchmark_4 model.

Runs validation-set inference (18,420 rows) with every saved checkpoint and
saves one house-style confusion matrix per model to figures/confusion_matrixs/.

Families (features come from the same parquets the trainers used; the same
per-column notna masks are applied, so prediction sets match training):
  xgboost_per_layer/       xgb_layer_{L:02d}.json       emb_blocks_{L}          (Valid_all_layers_embeddings.parquet)
  xgboost_mean_per_layer/  xgb_mean_layer_{L:02d}.json  emb_mean_blocks_{L}     (Valid_all_layers_embeddings.parquet)
  mlp_last_per_layer/      mlp_last_layer_{L:02d}.pt    emb_blocks_{L}          (Valid_all_layers_embeddings.parquet)
  mlp_mean_per_layer/      mlp_mean_layer_{L:02d}.pt    emb_mean_blocks_{L}     (Valid_all_layers_embeddings.parquet)
  mlp_layer17_10000ep/     mlp_layer17.pt               emb_blocks_17          (Valid_all_layers_embeddings.parquet)
  mlp_xgb_concat_results/  3 MLP + 2 XGB concat models  concat emb_blocks_{L}   (Valid_all_layers_embeddings.parquet)
  rf_benchmark4/           9 rf_layer_{L}_{t}.pkl       emb_blocks_{L}{_mlp|_post_norm} (Valid_with_embeddings.parquet)
  rf_combined_results/     rf_combined_model.pkl        9-column concat         (Valid_with_embeddings.parquet)

House style (matches existing confusion_1st_digit.png artifacts):
  7x7 count matrix, Blues heatmap, "Predicted/True 1st digit" axes, dpi=150.

Usage:
  python plot_all_confusion_matrices.py            # all 144 models
  python plot_all_confusion_matrices.py --only rf  # only the 10 RF models
  python plot_all_confusion_matrices.py --only fast  # the 134 XGB/MLP models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import BENCH4_DIR, MODELS_DIR, PROJECT_ROOT, EC_MLP_Compact, ec_at_level

import os, gc, time, pickle, json, glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

VALID_ALL_PQ = str(BENCH4_DIR / "Valid_all_layers_embeddings.parquet")
VALID_RF_PQ  = str(BENCH4_DIR / "Valid_with_embeddings.parquet")
OUT_DIR = str(PROJECT_ROOT / "figures" / "confusion_matrixs")

LAYERS = list(range(32))
RF_LAYERS = [9, 24, 26]
RF_TYPES  = ["block", "mlp", "post_norm"]
RF_SUFFIX = {"block": "", "mlp": "_mlp", "post_norm": "_post_norm"}
CONCAT_EXPS = [
    ("mlp_concat_17_12_16", "mlp", [17, 12, 16]),
    ("mlp_concat_17_2_30",  "mlp", [17, 2, 30]),
    ("xgb_concat_17_12_16", "xgboost", [17, 12, 16]),
    ("xgb_concat_17_2_30",  "xgboost", [17, 2, 30]),
]
DIGITS = [str(c) for c in range(1, 8)]

results = []

def plot_cm(y1, p1, title, out_path):
    cm = confusion_matrix(y1, p1, labels=DIGITS)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=DIGITS, yticklabels=DIGITS)
    ax.set_xlabel("Predicted 1st digit"); ax.set_ylabel("True 1st digit")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def level_accs(ec_true, ec_pred):
    a = {}
    for lv in [1, 4]:
        a[lv] = accuracy_score([ec_at_level(e, lv) for e in ec_true],
                               [ec_at_level(e, lv) for e in ec_pred])
    return a

def record(family, model_name, title, n_valid, accs):
    results.append({"family": family, "model": model_name, "n_valid": n_valid,
                    "level_1_acc": round(accs[1], 4), "level_4_acc": round(accs[4], 4)})
    print(f"  {model_name}: n={n_valid} L1={accs[1]:.4f} L4={accs[4]:.4f}", flush=True)

def load_le(model_dir):
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        return pickle.load(f)

def mlp_predict(mdl, X, batch=512):
    out = []
    mdl.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        for i in range(0, len(Xt), batch):
            out.append(mdl(Xt[i:i+batch]).argmax(1).numpy())
    return np.concatenate(out)

def xgb_predict(mpath, X):
    mdl = xgb.XGBClassifier()
    mdl.load_model(mpath)
    return mdl.predict(X)

def run_one(family, model_name, title, le, X, ec_va, mpath, kind, out_dir, dim=None):
    """kind: 'xgb' | 'mlp'"""
    t0 = time.time()
    if kind == "xgb":
        y_idx = xgb_predict(mpath, X)
    elif kind == "mlp":
        mdl = EC_MLP_Compact(dim, len(le.classes_))
        mdl.load_state_dict(torch.load(mpath, map_location="cpu", weights_only=True))
        last = mdl.net[-1]
        assert last.out_features == len(le.classes_), \
            f"{model_name}: head {last.out_features} != {len(le.classes_)} classes"
        y_idx = mlp_predict(mdl, X)
        del mdl
    ec_pv = le.inverse_transform(y_idx)
    y1 = [ec_at_level(e, 1) for e in ec_va]
    p1 = [ec_at_level(e, 1) for e in ec_pv]
    plot_cm(y1, p1, title, os.path.join(out_dir, f"{model_name}.png"))
    accs = level_accs(ec_va, ec_pv)
    record(family, model_name, title, len(ec_va), accs)
    print(f"    -> {model_name}.png ({time.time()-t0:.1f}s)", flush=True)
    return accs

# ===================== data loaders =====================
def _load_cols(path, emb_cols):
    """Load EC + list<float> embedding columns column-by-column via pyarrow.

    Avoids the pandas object-dtype blowup (a Python float per element across
    64 cols x 18k rows x 4096 d would be ~150 GB); peaks at final arrays plus
    one column in flight. Per-column notna rows (None lists) become NaN rows
    and are masked, matching the trainers' notna masks.
    """
    print(f"Loading {path} (EC + {len(emb_cols)} emb cols, columnar) ...", flush=True)
    t0 = time.time()
    pf = pq.ParquetFile(path)
    t = pf.read(columns=["EC"])
    ec = np.asarray(t.column("EC").to_pylist())
    n = len(ec)
    mask = np.ones(n, dtype=bool)
    data = {}
    for i, c in enumerate(emb_cols):
        pl = pf.read(columns=[c]).column(0).to_pylist()
        try:
            arr = np.array(pl, dtype=np.float32)
        except (TypeError, ValueError):
            arr = np.full((n, 4096), np.nan, dtype=np.float32)
            for r, v in enumerate(pl):
                if v is not None:
                    arr[r] = v
            mask &= ~np.isnan(arr).any(axis=1)
        data[c] = arr
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(emb_cols)} cols ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return ec, data, mask

def load_all_layers():
    """EC + all 64 embedding columns from Valid_all_layers_embeddings.parquet."""
    emb_cols = [f"emb_blocks_{L}" for L in LAYERS] + [f"emb_mean_blocks_{L}" for L in LAYERS]
    return _load_cols(VALID_ALL_PQ, emb_cols)

def load_rf():
    """EC + 9 RF feature columns from Valid_with_embeddings.parquet."""
    cols = [f"emb_blocks_{L}{RF_SUFFIX[t]}" for L in RF_LAYERS for t in RF_TYPES]
    return _load_cols(VALID_RF_PQ, cols)


# ===================== families =====================
def run_fast():
    out_dir = OUT_DIR
    MODEL4 = str(MODELS_DIR / "benchmark_4")
    ec, data, mask = load_all_layers()

    # --- per-layer sweeps ---
    for family, xkind, feat, pat, le_path in [
        ("xgboost_per_layer", "xgb", lambda L: f"emb_blocks_{L}",
         "xgb_layer_{L:02d}.json", os.path.join(MODEL4, "xgboost_per_layer", "label_encoder.pkl")),
        ("xgboost_mean_per_layer", "xgb", lambda L: f"emb_mean_blocks_{L}",
         "xgb_mean_layer_{L:02d}.json", os.path.join(MODEL4, "xgboost_mean_per_layer", "label_encoder.pkl")),
        ("mlp_last_per_layer", "mlp", lambda L: f"emb_blocks_{L}",
         "mlp_last_layer_{L:02d}.pt", os.path.join(MODEL4, "mlp_last_per_layer", "label_encoder.pkl")),
        ("mlp_mean_per_layer", "mlp", lambda L: f"emb_mean_blocks_{L}",
         "mlp_mean_layer_{L:02d}.pt", os.path.join(MODEL4, "mlp_mean_per_layer", "label_encoder.pkl")),
    ]:
        print(f"\n=== {family} ===", flush=True)
        le = load_le(os.path.dirname(le_path))
        for L in LAYERS:
            col = feat(L)
            X, ec_va = data[col][mask], ec[mask]
            mpath = os.path.join(os.path.dirname(le_path), pat.format(L=L))
            if not os.path.exists(mpath):
                print(f"  MISSING {mpath}", flush=True); continue
            kindtag = "XGBoost" if xkind == "xgb" else "MLP"
            tok = "Last Token" if col.startswith("emb_blocks_") else "Mean Token"
            title = f"{kindtag} Layer {L:02d} ({tok}) — 1st-digit EC confusion matrix"
            run_one(family, Path(mpath).stem, title, le, X, ec_va, mpath,
                    xkind, out_dir, dim=4096)

    # --- mlp_layer17 10000-epoch ---
    print(f"\n=== mlp_layer17_10000ep ===", flush=True)
    mdir = os.path.join(MODEL4, "mlp_layer17_10000ep")
    le = load_le(mdir)
    X, ec_va = data["emb_blocks_17"][mask], ec[mask]
    run_one("mlp_layer17_10000ep", "mlp_layer17",
            "MLP Layer 17 — 1st-digit EC confusion matrix",
            le, X, ec_va, os.path.join(mdir, "mlp_layer17.pt"), "mlp", out_dir, dim=4096)

    # --- concat experiments ---
    print(f"\n=== mlp_xgb_concat_results ===", flush=True)
    cdir = os.path.join(MODEL4, "mlp_xgb_concat_results")
    le = load_le(cdir)
    m = mask
    # standalone layer-17 MLP trained in the concat experiments
    run_one("mlp_xgb_concat", "mlp_mlp_layer17",
            "MLP Layer 17 (concat experiments) — 1st-digit EC confusion matrix",
            le, data["emb_blocks_17"][m], ec[m],
            os.path.join(cdir, "mlp_mlp_layer17.pt"), "mlp", out_dir, dim=4096)
    for name, mtype, layers in CONCAT_EXPS:
        cols = [f"emb_blocks_{L}" for L in layers]
        X = np.concatenate([data[c][m] for c in cols], axis=1)
        kind = "mlp" if mtype == "mlp" else "xgb"
        mpath = os.path.join(cdir, f"{'mlp' if mtype=='mlp' else 'xgb'}_{name}.{'pt' if mtype=='mlp' else 'json'}")
        if not os.path.exists(mpath):
            print(f"  MISSING {mpath}", flush=True); continue
        kindtag = "MLP" if mtype == "mlp" else "XGBoost"
        title = f"{kindtag} concat {layers} — 1st-digit EC confusion matrix"
        run_one("mlp_xgb_concat", Path(mpath).stem, title, le, X, ec[m],
                mpath, kind, out_dir, dim=len(cols) * 4096)
    del data
    gc.collect()

def run_rf():
    out_dir = OUT_DIR
    MODEL4 = str(MODELS_DIR / "benchmark_4")
    ec, data, mask = load_rf()

    print(f"\n=== rf_benchmark4 (9 models) ===", flush=True)
    rdir = os.path.join(MODEL4, "rf_benchmark4")
    le = load_le(rdir)
    for L in RF_LAYERS:
        for t in RF_TYPES:
            col = f"emb_blocks_{L}{RF_SUFFIX[t]}"
            m = mask
            mpath = os.path.join(rdir, f"rf_layer_{L}_{t}.pkl")
            if not os.path.exists(mpath):
                print(f"  MISSING {mpath}", flush=True); continue
            print(f"  loading {os.path.basename(mpath)} ...", flush=True)
            t0 = time.time()
            mdl = pickle.load(open(mpath, "rb"))
            print(f"    loaded in {time.time()-t0:.0f}s", flush=True)
            t0 = time.time()
            y_idx = mdl.predict(data[col][m])
            ec_pv = le.inverse_transform(y_idx)
            y1 = [ec_at_level(e, 1) for e in ec[m]]
            p1 = [ec_at_level(e, 1) for e in ec_pv]
            plot_cm(y1, p1, f"RF Layer {L} ({t}) — 1st-digit EC confusion matrix",
                    os.path.join(out_dir, f"rf_layer_{L}_{t}.png"))
            accs = level_accs(ec[m], ec_pv)
            record("rf_benchmark4", f"rf_layer_{L}_{t}", "", len(ec[m]), accs)
            print(f"    -> rf_layer_{L}_{t}.png (predict {time.time()-t0:.0f}s)", flush=True)
            del mdl
            gc.collect()

    print(f"\n=== rf_combined ===", flush=True)
    cdir = os.path.join(MODEL4, "rf_combined_results")
    le = load_le(cdir)
    m = mask
    cols = [f"emb_blocks_{L}{RF_SUFFIX[t]}" for L in RF_LAYERS for t in RF_TYPES]
    X = np.concatenate([data[c][m] for c in cols], axis=1)
    print(f"  loading rf_combined_model.pkl ...", flush=True)
    t0 = time.time()
    mdl = pickle.load(open(os.path.join(cdir, "rf_combined_model.pkl"), "rb"))
    print(f"    loaded in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    y_idx = mdl.predict(X)
    ec_pv = le.inverse_transform(y_idx)
    y1 = [ec_at_level(e, 1) for e in ec[m]]
    p1 = [ec_at_level(e, 1) for e in ec_pv]
    plot_cm(y1, p1, "RF Combined (layers 9+24+26, block+mlp+post_norm) — 1st-digit EC confusion matrix",
            os.path.join(out_dir, "rf_combined_model.png"))
    accs = level_accs(ec[m], ec_pv)
    record("rf_combined", "rf_combined_model", "", len(ec[m]), accs)
    print(f"    -> rf_combined_model.png (predict {time.time()-t0:.0f}s)", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "fast", "rf"], default="all")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    if args.only in ("all", "fast"):
        run_fast()
    if args.only in ("all", "rf"):
        run_rf()
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "summary.csv")
    if os.path.exists(out_csv):  # merge with prior partial runs (e.g. fast then rf)
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["family", "model"], keep="last")
    df.to_csv(out_csv, index=False)
    print(f"\nDONE: {len(df)} models written this run; {len(df)} total in summary -> {OUT_DIR} ({(time.time()-t_start)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
