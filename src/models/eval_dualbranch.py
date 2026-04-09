"""Compare DualBranchTransformer vs TemporalTransformer.

Both use FGI regime adaptive, 7.5/3 TP/SL.
Tracks per-regime stats for strategy analysis.

Usage:
    python3 -m src.models.eval_dualbranch
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(line_buffering=True)

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, EXPERIMENTS_DIR, LABEL_FGI_PATH
from src.features.pipeline import build_feature_matrix
from src.features.dataset import TimeSeriesDataset, make_loader
from src.models.architecture import DualBranchTransformerClassifier, TemporalTransformerClassifier
from src.models.trainer import train_model

FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
]

TP = 0.075
SL = 0.03

FGI_DF = pd.read_csv(LABEL_FGI_PATH, parse_dates=["date"])
FGI_LOOKUP = dict(zip(FGI_DF["date"].dt.date, FGI_DF["fgi_value"].astype(float)))

CONFIGS = [
    {"name": "DualBranch (VP + Candle)", "model_cls": DualBranchTransformerClassifier},
    {"name": "Temporal (VP only)", "model_cls": TemporalTransformerClassifier},
]


def get_regime_for_dataset(ds, df):
    regime = np.full(len(ds), np.nan)
    for si, vi in enumerate(ds.valid_indices):
        label_idx = vi + ds.lookback - 1
        dt = pd.Timestamp(df["date"].iloc[label_idx]).date()
        fgi = FGI_LOOKUP.get(dt, np.nan)
        if not np.isnan(fgi):
            regime[si] = 1.0 if fgi >= 50 else 0.0
    return regime


def run_walkforward(df, name, model_cls):
    import src.config as cfg
    import src.features.dataset as ds_mod

    cfg.LABEL_TP_PCT = TP
    cfg.LABEL_SL_PCT = SL
    cfg.LABEL_REGIME_ADAPTIVE = True
    cfg.LABEL_REGIME_MODE = "fgi"
    ds_mod.LABEL_TP_PCT = TP
    ds_mod.LABEL_SL_PCT = SL
    ds_mod.LABEL_REGIME_ADAPTIVE = True
    ds_mod.LABEL_REGIME_MODE = "fgi"

    all_preds = []
    all_labels = []
    all_regimes = []
    fold_accs = []

    for i in range(len(FOLD_BOUNDARIES) - 2):
        train_end = FOLD_BOUNDARIES[i]
        val_end = FOLD_BOUNDARIES[i + 1]
        test_end = FOLD_BOUNDARIES[i + 2]

        train_df = df[df["date"] < train_end].reset_index(drop=True)
        val_df = df[(df["date"] >= train_end) & (df["date"] < val_end)].reset_index(drop=True)
        test_df = df[(df["date"] >= val_end) & (df["date"] < test_end)].reset_index(drop=True)

        if len(train_df) < 1000 or len(val_df) < 100 or len(test_df) < 100:
            continue

        train_ds = TimeSeriesDataset(train_df, lookback=LOOKBACK_BARS_MODEL)
        val_ds = TimeSeriesDataset(val_df, lookback=LOOKBACK_BARS_MODEL)
        test_ds = TimeSeriesDataset(test_df, lookback=LOOKBACK_BARS_MODEL)

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        regime = get_regime_for_dataset(test_ds, test_df)

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        model = model_cls()
        train_model(model, train_loader, val_loader)

        model.eval()
        fold_p, fold_l = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                preds = (logits > 0).float()
                fold_p.extend(preds.tolist())
                fold_l.extend(y.tolist())

        fold_acc = np.mean(np.array(fold_p) == np.array(fold_l))
        fold_accs.append(fold_acc)
        all_preds.extend(fold_p)
        all_labels.extend(fold_l)
        all_regimes.extend(regime[:len(fold_p)].tolist())
        print(f"  Fold {i+1}: acc={fold_acc:.4f} ({len(fold_p)} samples)", flush=True)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    regimes = np.array(all_regimes)

    bull = regimes == 1.0
    bear = regimes == 0.0

    def confusion(p, l):
        tp = ((p == 1) & (l == 1)).sum()
        fp = ((p == 1) & (l == 0)).sum()
        tn = ((p == 0) & (l == 0)).sum()
        fn = ((p == 0) & (l == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn) if len(p) > 0 else 0
        return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "precision": float(prec), "npv": float(npv), "accuracy": float(acc),
                "n": int(len(p))}

    bull_stats = confusion(preds[bull], labels[bull])
    bear_stats = confusion(preds[bear], labels[bear])

    # EVs (with regime flip)
    ev_bull_long = bull_stats["precision"] * 7.5 - (1 - bull_stats["precision"]) * 3.0
    ev_bull_short = bull_stats["npv"] * 3.0 - (1 - bull_stats["npv"]) * 7.5
    ev_bear_long = bear_stats["precision"] * 3.0 - (1 - bear_stats["precision"]) * 7.5
    ev_bear_short = bear_stats["npv"] * 7.5 - (1 - bear_stats["npv"]) * 3.0

    overall_acc = (preds == labels).mean()
    folds_below_50 = sum(1 for a in fold_accs if a < 0.5)

    return {
        "name": name,
        "accuracy": float(overall_acc),
        "fold_accs": [float(a) for a in fold_accs],
        "folds_below_50": folds_below_50,
        "bull_stats": bull_stats,
        "bear_stats": bear_stats,
        "ev_bull_long": float(ev_bull_long),
        "ev_bull_short": float(ev_bull_short),
        "ev_bear_long": float(ev_bear_long),
        "ev_bear_short": float(ev_bear_short),
    }


def main():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []

    for config in CONFIGS:
        name = config["name"]
        model_cls = config["model_cls"]
        params = sum(p.numel() for p in model_cls().parameters())

        print(f"\n{'=' * 70}", flush=True)
        print(f"  {name}  ({params:,} params)", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        result = run_walkforward(df, name, model_cls)
        result["params"] = params
        results.append(result)

        print(f"\n  RESULT: {name}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f}, Folds <50%: {result['folds_below_50']}/10", flush=True)
        print(f"  Bull: prec={result['bull_stats']['precision']:.3f}, npv={result['bull_stats']['npv']:.3f}", flush=True)
        print(f"  Bear: prec={result['bear_stats']['precision']:.3f}, npv={result['bear_stats']['npv']:.3f}", flush=True)
        print(f"  EVs: bull_long={result['ev_bull_long']:+.3f}%, bear_long={result['ev_bear_long']:+.3f}%, "
              f"bear_short={result['ev_bear_short']:+.3f}%", flush=True)

    # Comparison
    print(f"\n\n{'=' * 80}", flush=True)
    print(f"  COMPARISON: DualBranch vs Temporal", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"{'Metric':<25} {'DualBranch':>15} {'Temporal':>15} {'Diff':>12}", flush=True)
    print("-" * 70, flush=True)

    r_d, r_t = results[0], results[1]
    for key, label in [
        ("params", "Params"),
        ("accuracy", "Accuracy"),
        ("folds_below_50", "Folds <50%"),
    ]:
        vd, vt = r_d[key], r_t[key]
        if isinstance(vd, float):
            print(f"{label:<25} {vd:>15.4f} {vt:>15.4f} {vd-vt:>+12.4f}", flush=True)
        else:
            print(f"{label:<25} {vd:>15,} {vt:>15,} {vd-vt:>+12,}", flush=True)

    for key, label in [
        ("ev_bull_long", "EV bull long"),
        ("ev_bull_short", "EV bull short"),
        ("ev_bear_long", "EV bear long"),
        ("ev_bear_short", "EV bear short"),
    ]:
        vd, vt = r_d[key], r_t[key]
        print(f"{label:<25} {vd:>+14.3f}% {vt:>+14.3f}% {vd-vt:>+11.3f}%", flush=True)

    # Strategy 4: long bull + both sides bear
    bull_p1_d = (r_d["bull_stats"]["tp"] + r_d["bull_stats"]["fp"]) / r_d["bull_stats"]["n"]
    bull_p1_t = (r_t["bull_stats"]["tp"] + r_t["bull_stats"]["fp"]) / r_t["bull_stats"]["n"]
    bear_p1_d = (r_d["bear_stats"]["tp"] + r_d["bear_stats"]["fp"]) / r_d["bear_stats"]["n"]
    bear_p1_t = (r_t["bear_stats"]["tp"] + r_t["bear_stats"]["fp"]) / r_t["bear_stats"]["n"]

    bull_frac_d = r_d["bull_stats"]["n"] / (r_d["bull_stats"]["n"] + r_d["bear_stats"]["n"])
    bull_frac_t = r_t["bull_stats"]["n"] / (r_t["bull_stats"]["n"] + r_t["bear_stats"]["n"])
    bear_frac_d = 1 - bull_frac_d
    bear_frac_t = 1 - bull_frac_t

    strat4_d = bull_frac_d * bull_p1_d * r_d["ev_bull_long"] + \
               bear_frac_d * bear_p1_d * r_d["ev_bear_long"] + \
               bear_frac_d * (1 - bear_p1_d) * r_d["ev_bear_short"]
    strat4_t = bull_frac_t * bull_p1_t * r_t["ev_bull_long"] + \
               bear_frac_t * bear_p1_t * r_t["ev_bear_long"] + \
               bear_frac_t * (1 - bear_p1_t) * r_t["ev_bear_short"]

    print(f"\n{'Strategy 4 daily EV':<25} {strat4_d:>+14.3f}% {strat4_t:>+14.3f}% {strat4_d-strat4_t:>+11.3f}%", flush=True)

    print(f"\n{'Fold':<8} {'DualBranch':>12} {'Temporal':>12} {'Diff':>12}", flush=True)
    print("-" * 46, flush=True)
    for i in range(min(len(r_d["fold_accs"]), len(r_t["fold_accs"]))):
        d, t = r_d["fold_accs"][i], r_t["fold_accs"][i]
        print(f"Fold {i+1:<3} {d:>11.1%} {t:>11.1%} {d-t:>+11.1%}", flush=True)

    # Save
    out = EXPERIMENTS_DIR / "dualbranch_compare_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)


if __name__ == "__main__":
    main()
