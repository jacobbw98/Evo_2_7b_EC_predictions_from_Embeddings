#!/usr/bin/env python3
"""
Generate accuracy-by-layer plots with error bars for all 4 benchmark_4 permutations.

Loads trained models from:
  - xgboost_per_layer/        (XGBoost × Last Token)
  - xgboost_mean_per_layer/   (XGBoost × Mean Token)
  - mlp_last_per_layer/       (MLP × Last Token)
  - mlp_mean_per_layer/       (MLP × Mean Token)

Splits validation set into 3 equal groups by sequence length (Short/Medium/Long),
computes per-group accuracy, and adds error bars to all plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import BENCH4_DIR, MODELS_DIR, FIGURES_DIR, EC_MLP_Compact, ec_at_level

import os, gc, json, pickle
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

VALID_PQ = BENCH4_DIR / "Valid_all_layers_embeddings.parquet"
VALID_SEQ_PQ = BENCH4_DIR / "Valid.parquet"  # Has Sequence column
OUTPUT_DIR = MODELS_DIR / "benchmark_4"
(FIGURES_DIR / "benchmark_2").mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "benchmark_4").mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = list(range(32))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEVICE.type == "cuda", "CUDA not available — this script requires a GPU"
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")

# ============================================================
# Load validation data + sequence length groups
# ============================================================
print(f"\n{'='*70}")
print("Loading validation data...")
print(f"{'='*70}")

# Get sequence lengths from the original parquet (smaller, faster)
df_seq = pd.read_parquet(VALID_SEQ_PQ, columns=["AC", "Sequence"])
seq_lens = df_seq["Sequence"].str.len().values
sorted_indices = np.argsort(seq_lens)
n = len(sorted_indices)
t1, t2 = n // 3, 2 * n // 3

# Map each row to a group
group_map = np.empty(n, dtype=np.int32)
group_map[sorted_indices[:t1]] = 0
group_map[sorted_indices[t1:t2]] = 1
group_map[sorted_indices[t2:]] = 2

sorted_lens = seq_lens[sorted_indices]
print(f"  Short:  n={t1}, max_len={sorted_lens[t1-1]}")
print(f"  Medium: n={t2-t1}, max_len={sorted_lens[t2-1]}")
print(f"  Long:   n={n-t2}, max_len={sorted_lens[-1]}")

# Build AC->group_index mapping (assumes same row order as embeddings parquet)
ac_order = df_seq["AC"].values
del df_seq, seq_lens, sorted_lens
gc.collect()

# Load the full embeddings parquet
print("Loading Valid embeddings parquet...", flush=True)
df_valid = pd.read_parquet(VALID_PQ)
print(f"  {len(df_valid)} rows, {len(df_valid.columns)} columns")

# Encode labels
le = LabelEncoder()
# Need train labels too for fitting
df_train_ec = pd.read_parquet(
    BENCH4_DIR / "Train_all_layers_embeddings.parquet", columns=["EC"])
le.fit(np.concatenate([df_train_ec["EC"].values, df_valid["EC"].values]))
del df_train_ec
n_cls = len(le.classes_)
print(f"  {n_cls} classes")

y_va_enc = le.transform(df_valid["EC"].values)
ec_va_str = df_valid["EC"].values

# ============================================================
# Configs
# ============================================================
CONFIGS = {
    "XGBoost Last Token": {
        "dir": MODELS_DIR / "benchmark_4" / "xgboost_per_layer",
        "col_prefix": "emb_blocks",
        "model_type": "xgb",
        "model_pattern": "xgb_layer_{li:02d}.json",
    },
    "XGBoost Mean Token": {
        "dir": MODELS_DIR / "benchmark_4" / "xgboost_mean_per_layer",
        "col_prefix": "emb_mean_blocks",
        "model_type": "xgb",
        "model_pattern": "xgb_mean_layer_{li:02d}.json",
    },
    "MLP Last Token": {
        "dir": MODELS_DIR / "benchmark_4" / "mlp_last_per_layer",
        "col_prefix": "emb_blocks",
        "model_type": "mlp",
        "model_pattern": "mlp_last_layer_{li:02d}.pt",
    },
    "MLP Mean Token": {
        "dir": MODELS_DIR / "benchmark_4" / "mlp_mean_per_layer",
        "col_prefix": "emb_mean_blocks",
        "model_type": "mlp",
        "model_pattern": "mlp_mean_layer_{li:02d}.pt",
    },
}

# ============================================================
# Evaluate all configs, all layers, per group
# ============================================================
all_metrics = {}  # config_name -> {layer -> {level -> {overall, groups}}}

for cfg_name, cfg in CONFIGS.items():
    print(f"\n{'='*70}")
    print(f"Evaluating: {cfg_name}")
    print(f"{'='*70}")

    cfg_metrics = {}
    for li in tqdm(LAYERS, desc=cfg_name):
        col = f"{cfg['col_prefix']}_{li}"
        va_mask = df_valid[col].notna()
        X_va = np.array(df_valid.loc[va_mask, col].tolist(), dtype=np.float32)
        y_va = y_va_enc[va_mask.values]
        ec_va_l = ec_va_str[va_mask.values]
        groups = group_map[va_mask.values]

        # Load model and predict
        if cfg["model_type"] == "xgb":
            mpath = os.path.join(cfg["dir"], cfg["model_pattern"].format(li=li))
            mdl = xgb.XGBClassifier(device='cuda')
            mdl.load_model(mpath)
            yp_va = mdl.predict(X_va)
            ec_pv = le.inverse_transform(yp_va)
            del mdl
        else:  # MLP
            mpath = os.path.join(cfg["dir"], cfg["model_pattern"].format(li=li))
            mdl = EC_MLP_Compact(X_va.shape[1], n_cls).to(DEVICE)
            mdl.load_state_dict(torch.load(mpath, map_location=DEVICE, weights_only=True))
            mdl.eval()
            preds = []
            X_t = torch.from_numpy(X_va).float()
            with torch.no_grad():
                for i in range(0, len(X_t), 4096):
                    preds.append(mdl(X_t[i:i+4096].to(DEVICE)).argmax(1).cpu().numpy())
            ec_pv = le.inverse_transform(np.concatenate(preds))
            del mdl, X_t, preds
            torch.cuda.empty_cache()

        # Compute metrics per level, per group
        layer_metrics = {}
        for lv in [1, 2, 3, 4]:
            true_lv = np.array([ec_at_level(e, lv) for e in ec_va_l])
            pred_lv = np.array([ec_at_level(e, lv) for e in ec_pv])
            acc = accuracy_score(true_lv, pred_lv)
            f1m = f1_score(true_lv, pred_lv, average="macro", zero_division=0)
            f1w = f1_score(true_lv, pred_lv, average="weighted", zero_division=0)
            grp_metrics = []
            for g in range(3):
                mask = groups == g
                if mask.sum() == 0:
                    grp_metrics.append({"acc": 0, "f1m": 0, "f1w": 0})
                else:
                    grp_metrics.append({
                        "acc": accuracy_score(true_lv[mask], pred_lv[mask]),
                        "f1m": f1_score(true_lv[mask], pred_lv[mask], average="macro", zero_division=0),
                        "f1w": f1_score(true_lv[mask], pred_lv[mask], average="weighted", zero_division=0),
                    })
            layer_metrics[lv] = {"overall": {"acc": acc, "f1m": f1m, "f1w": f1w},
                                  "groups": grp_metrics}
        cfg_metrics[li] = layer_metrics
        del X_va, y_va, ec_va_l, groups
        gc.collect()

    all_metrics[cfg_name] = cfg_metrics
    # Print best layer
    best_l = max(LAYERS, key=lambda l: cfg_metrics[l][4]["overall"]["acc"])
    print(f"  Best layer: {best_l} (EC4 acc={cfg_metrics[best_l][4]['overall']['acc']:.4f})")

# ============================================================
# Evaluate Random Forest models (if trained) — same group split.
# RF models come from train_benchmark4_rf.py (CPU, run separately);
# skipped gracefully when not trained yet.
# ============================================================
rf_metrics = {}             # "RF <Type>" -> {layer: layer_metrics}
rf_combined_metrics = None  # layer_metrics for the 9-column concatenated RF

_RF_DIR      = MODELS_DIR / "benchmark_4" / "rf_benchmark4"
_RF_COMB_DIR = MODELS_DIR / "benchmark_4" / "rf_combined_results"
if (_RF_DIR / "rf_layer_9_block.pkl").exists() or (_RF_COMB_DIR / "rf_combined_model.pkl").exists():
    print(f"\n{'='*70}")
    print("Evaluating Random Forest models...")
    print(f"{'='*70}")
    import pickle as _pkl

    def _load_le():
        for d in (_RF_DIR, _RF_COMB_DIR):
            p = d / "label_encoder.pkl"
            if p.exists():
                return _pkl.load(open(p, "rb"))
        return None
    rf_le = _load_le()

    df_rf = pd.read_parquet(BENCH4_DIR / "Valid_with_embeddings.parquet")
    assert len(df_rf) == len(group_map), "RF parquet row count != Valid.parquet"
    assert (df_rf["EC"].values == ec_va_str).all(), "RF parquet EC order != all-layers parquet"

    def _layer_metrics_from(pred_ec, true_ec, groups):
        lm = {}
        for lv in [1, 2, 3, 4]:
            true_lv = np.array([ec_at_level(e, lv) for e in true_ec])
            pred_lv = np.array([ec_at_level(e, lv) for e in pred_ec])
            grp = []
            for g in range(3):
                m = groups == g
                if m.sum() == 0:
                    grp.append({"acc": 0, "f1m": 0, "f1w": 0})
                else:
                    grp.append({"acc": accuracy_score(true_lv[m], pred_lv[m]),
                                "f1m": f1_score(true_lv[m], pred_lv[m], average="macro", zero_division=0),
                                "f1w": f1_score(true_lv[m], pred_lv[m], average="weighted", zero_division=0)})
            lm[lv] = {"overall": {"acc": accuracy_score(true_lv, pred_lv),
                                  "f1m": f1_score(true_lv, pred_lv, average="macro", zero_division=0),
                                  "f1w": f1_score(true_lv, pred_lv, average="weighted", zero_division=0)},
                      "groups": grp}
        return lm

    if rf_le is not None:
        _RF_TYPES = {"RF Block": "block", "RF MLP": "mlp", "RF Post Norm": "post_norm"}
        for type_name, t in _RF_TYPES.items():
            per_layer = {}
            for L in (9, 24, 26):
                mpath = _RF_DIR / f"rf_layer_{L}_{t}.pkl"
                if not mpath.exists():
                    continue
                mdl = _pkl.load(open(mpath, "rb"))
                col = f"emb_blocks_{L}" if t == "block" else f"emb_blocks_{L}_{t}"
                va_mask = df_rf[col].notna().values
                X_va = np.array(df_rf.loc[va_mask, col].tolist(), dtype=np.float32)
                ec_pv = rf_le.inverse_transform(mdl.predict(X_va))
                per_layer[L] = _layer_metrics_from(ec_pv, ec_va_str[va_mask], group_map[va_mask])
                del mdl, X_va
            if per_layer:
                rf_metrics[type_name] = per_layer
                best_l = max(per_layer, key=lambda l: per_layer[l][4]["overall"]["acc"])
                print(f"  {type_name}: best layer {best_l} (EC4 acc={per_layer[best_l][4]['overall']['acc']:.4f})")
            gc.collect()

    _comb_model = _RF_COMB_DIR / "rf_combined_model.pkl"
    if _comb_model.exists() and rf_le is not None:
        print("  RF Combined (9 x 4096 = 36864 features)...", flush=True)
        mdl = _pkl.load(open(_comb_model, "rb"))
        _sfx = {"block": "", "mlp": "_mlp", "post_norm": "_post_norm"}
        cols = [f"emb_blocks_{L}{_sfx[t]}" for L in (9, 24, 26) for t in ("block", "mlp", "post_norm")]
        va_mask = df_rf[cols[0]].notna().values
        X_va = np.concatenate([np.array(df_rf.loc[va_mask, c].tolist(), dtype=np.float32) for c in cols], axis=1)
        ec_pv = rf_le.inverse_transform(mdl.predict(X_va))
        rf_combined_metrics = _layer_metrics_from(ec_pv, ec_va_str[va_mask], group_map[va_mask])
        print(f"  RF Combined: EC4 acc={rf_combined_metrics[4]['overall']['acc']:.4f}")
        del mdl, X_va
    del df_rf
    gc.collect()
else:
    print("\nRF models not found — skipping Random Forest evaluation (run train_benchmark4_rf.py first).")

# ============================================================
# Plot 1-4: Individual accuracy-by-layer bar charts WITH error bars
# ============================================================
print(f"\n{'='*70}")
print("Generating plots...")
print(f"{'='*70}")

level_colors = {"1": "#2196F3", "2": "#4CAF50", "3": "#FF9800", "4": "#E91E63"}

for cfg_name, cfg_metrics in all_metrics.items():
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(32)
    width = 0.2

    for i, (level_str, color) in enumerate(level_colors.items()):
        lv = int(level_str)
        vals = []
        gvals_list = []
        for l in LAYERS:
            overall = cfg_metrics[l][lv]["overall"]["acc"]
            gvals = [cfg_metrics[l][lv]["groups"][g]["acc"] for g in range(3)]
            vals.append(overall)
            gvals_list.append(gvals)

        yerr_vals = [float(np.std(gvals_list[j], ddof=1)) if len(gvals_list[j]) >= 2 else 0.0 for j in range(len(LAYERS))]

        ax.bar(x + i * width, vals, width, label=f"Level {lv}", color=color, alpha=0.85)
        ax.errorbar(x + i * width, vals, yerr=yerr_vals,
                    fmt='none', ecolor='black', elinewidth=1.2, capsize=2, capthick=1.0)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([str(l) for l in LAYERS])
    ax.set_xlabel("Evo2 Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{cfg_name} — EC Prediction Accuracy by Layer",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe = cfg_name.lower().replace(" ", "_")
    fig_path = FIGURES_DIR / "benchmark_4" / f"accuracy_by_layer_{safe}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

# ============================================================
# Plot 5: Combined comparison — Full EC accuracy with error bands
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(32)

model_colors = {
    "XGBoost Last Token": "#2196F3", "XGBoost Mean Token": "#00BCD4",
    "MLP Last Token": "#E91E63", "MLP Mean Token": "#FF9800",
}
markers = {
    "XGBoost Last Token": "o", "XGBoost Mean Token": "s",
    "MLP Last Token": "^", "MLP Mean Token": "D",
}

for cfg_name, cfg_metrics in all_metrics.items():
    vals = [cfg_metrics[l][4]["overall"]["acc"] for l in LAYERS]
    lo = [min(cfg_metrics[l][4]["groups"][g]["acc"] for g in range(3)) for l in LAYERS]
    hi = [max(cfg_metrics[l][4]["groups"][g]["acc"] for g in range(3)) for l in LAYERS]
    color = model_colors.get(cfg_name, "gray")
    ax.plot(x, vals, label=cfg_name, color=color,
            marker=markers.get(cfg_name, "o"), lw=2, ms=5, alpha=0.85)
    ax.fill_between(x, lo, hi, alpha=0.15, color=color)

ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_xlabel("Evo2 Layer", fontsize=12)
ax.set_ylabel("Full EC Level 4 Accuracy", fontsize=12)
ax.set_ylim(0, None)
ax.set_title("Full EC Prediction Accuracy — All Models Compared",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_path = FIGURES_DIR / "benchmark_2" / "combined_accuracy_comparison_b2.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ============================================================
# Plot 6: Random Forest only — Full EC accuracy with error bands
# ============================================================
if rf_metrics or rf_combined_metrics is not None:
    fig, ax = plt.subplots(figsize=(16, 7))
    _rf_styles = {
        "RF Block":     ("#9C27B0", "^"),
        "RF MLP":       ("#795548", "s"),
        "RF Post Norm": ("#607D8B", "D"),
    }
    for type_name, per_layer in rf_metrics.items():
        color, marker = _rf_styles.get(type_name, ("#9C27B0", "^"))
        ls_layers = sorted(per_layer)
        vl = [per_layer[l][4]["overall"]["acc"] for l in ls_layers]
        ax.plot(ls_layers, vl, label=type_name, color=color, marker=marker,
                linestyle="--", linewidth=1.8, markersize=7, alpha=0.9)
        lo = [min(per_layer[l][4]["groups"][g]["acc"] for g in range(3)) for l in ls_layers]
        hi = [max(per_layer[l][4]["groups"][g]["acc"] for g in range(3)) for l in ls_layers]
        ax.fill_between(ls_layers, lo, hi, alpha=0.15, color=color)
    if rf_combined_metrics is not None:
        ax.scatter([17.5], [rf_combined_metrics[4]["overall"]["acc"]],
                   marker="*", s=320, color="#FF5722", zorder=5,
                   label="RF Combined (L9+24+26)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in LAYERS])
    ax.set_xlabel("Evo2 Layer", fontsize=12)
    ax.set_ylabel("Full EC Level 4 Accuracy", fontsize=12)
    ax.set_ylim(0, None)
    ax.set_title("Full EC Prediction Accuracy — Random Forest",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "benchmark_4" / "rf_accuracy_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================
# Summary table
# ============================================================
print(f"\n{'='*80}")
print("SUMMARY COMPARISON TABLE")
print(f"{'='*80}")

table_rows = []
for cfg_name, cfg_metrics in all_metrics.items():
    best_l = max(LAYERS, key=lambda l: cfg_metrics[l][4]["overall"]["acc"])
    acc = cfg_metrics[best_l][4]["overall"]["acc"]
    gvals = [cfg_metrics[best_l][4]["groups"][g]["acc"] for g in range(3)]
    table_rows.append({
        "Model": cfg_name, "Best Layer": best_l,
        "Valid EC Acc": f"{acc:.4f}",
        "Short Acc": f"{gvals[0]:.4f}", "Medium Acc": f"{gvals[1]:.4f}",
        "Long Acc": f"{gvals[2]:.4f}",
    })
for type_name, per_layer in rf_metrics.items():
    best_l = max(per_layer, key=lambda l: per_layer[l][4]["overall"]["acc"])
    gvals = [per_layer[best_l][4]["groups"][g]["acc"] for g in range(3)]
    table_rows.append({
        "Model": type_name, "Best Layer": best_l,
        "Valid EC Acc": f"{per_layer[best_l][4]['overall']['acc']:.4f}",
        "Short Acc": f"{gvals[0]:.4f}", "Medium Acc": f"{gvals[1]:.4f}",
        "Long Acc": f"{gvals[2]:.4f}",
    })
if rf_combined_metrics is not None:
    gvals = [rf_combined_metrics[4]["groups"][g]["acc"] for g in range(3)]
    table_rows.append({
        "Model": "RF Combined (L9+24+26)", "Best Layer": "9+24+26",
        "Valid EC Acc": f"{rf_combined_metrics[4]['overall']['acc']:.4f}",
        "Short Acc": f"{gvals[0]:.4f}", "Medium Acc": f"{gvals[1]:.4f}",
        "Long Acc": f"{gvals[2]:.4f}",
    })

hdr = f"{'Model':<25} {'Best Layer':<12} {'Valid EC':<12} {'Short':<10} {'Medium':<10} {'Long':<10}"
print(hdr)
print("-" * len(hdr))
for r in table_rows:
    print(f"{r['Model']:<25} {r['Best Layer']:<12} {r['Valid EC Acc']:<12} "
          f"{r['Short Acc']:<10} {r['Medium Acc']:<10} {r['Long Acc']:<10}")

df_t = pd.DataFrame(table_rows)
df_t.to_csv(os.path.join(OUTPUT_DIR, "summary_comparison.csv"), index=False)

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")
tbl = ax.table(cellText=df_t.values, colLabels=df_t.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.8)
for j in range(len(df_t.columns)):
    tbl[0, j].set_facecolor("#2196F3")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(df_t) + 1):
    for j in range(len(df_t.columns)):
        if i % 2 == 0:
            tbl[i, j].set_facecolor("#f0f0f0")
ax.set_title("Model Comparison — Best Layer Performance (with Seq Length Groups)",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "benchmark_4" / "summary_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# Save all per-group metrics
# ============================================================
group_names = ["Short", "Medium", "Long"]
save_metrics = {}
for cfg_name, cfg_metrics in all_metrics.items():
    save_metrics[cfg_name] = {}
    for l in LAYERS:
        save_metrics[cfg_name][str(l)] = {}
        for lv in [1, 2, 3, 4]:
            m = cfg_metrics[l][lv]
            save_metrics[cfg_name][str(l)][f"level_{lv}"] = {
                "overall": {k: float(v) for k, v in m["overall"].items()},
                "groups": {group_names[g]: {k: float(v) for k, v in m["groups"][g].items()}
                           for g in range(3)}
            }
for type_name, per_layer in rf_metrics.items():
    save_metrics[type_name] = {}
    for l in per_layer:
        save_metrics[type_name][str(l)] = {}
        for lv in [1, 2, 3, 4]:
            m = per_layer[l][lv]
            save_metrics[type_name][str(l)][f"level_{lv}"] = {
                "overall": {k: float(v) for k, v in m["overall"].items()},
                "groups": {group_names[g]: {k: float(v) for k, v in m["groups"][g].items()}
                           for g in range(3)}
            }
if rf_combined_metrics is not None:
    save_metrics["RF Combined"] = {}
    save_metrics["RF Combined"]["combined"] = {}
    for lv in [1, 2, 3, 4]:
        m = rf_combined_metrics[lv]
        save_metrics["RF Combined"]["combined"][f"level_{lv}"] = {
            "overall": {k: float(v) for k, v in m["overall"].items()},
            "groups": {group_names[g]: {k: float(v) for k, v in m["groups"][g].items()}
                       for g in range(3)}
        }

with open(os.path.join(OUTPUT_DIR, "error_bar_metrics.json"), "w") as f:
    json.dump(save_metrics, f, indent=2)

print(f"\nAll plots and metrics saved to: {OUTPUT_DIR}")
print("DONE!")
