"""Compare adaptive vs fixed 7.5/3 TP/SL labels.

Model A: LABEL_REGIME_ADAPTIVE=True  — bull 7.5/3, bear 3/7.5 (flipped)
Model B: LABEL_REGIME_ADAPTIVE=False — fixed 7.5/3 everywhere

Same Transformer architecture, same walk-forward, same folds.
Also reports corrected EV (precision-based) for both.

Usage:
    python3 -m src.models.eval_regime_compare
"""

import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(line_buffering=True)

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, EXPERIMENTS_DIR
from src.features.pipeline import build_feature_matrix
from src.features.dataset import TimeSeriesDataset
from src.models.architecture import TransformerClassifier
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

CONFIGS = [
    {"name": "Adaptive (bull 7.5/3, bear 3/7.5)", "adaptive": True},
    {"name": "Fixed 7.5/3 (no flip)", "adaptive": False},
]


def run_walkforward(df, name, adaptive):
    import src.config as cfg
    import src.features.dataset as ds_mod

    cfg.LABEL_TP_PCT = TP
    cfg.LABEL_SL_PCT = SL
    cfg.LABEL_REGIME_ADAPTIVE = adaptive
    ds_mod.LABEL_TP_PCT = TP
    ds_mod.LABEL_SL_PCT = SL
    ds_mod.LABEL_REGIME_ADAPTIVE = adaptive

    all_preds = []
    all_labels = []
    all_logits = []
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

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TransformerClassifier()
        train_model(model, train_loader, val_loader)

        model.eval()
        fold_p, fold_l, fold_logits = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                preds = (logits > 0).float()
                fold_p.extend(preds.tolist())
                fold_l.extend(y.tolist())
                fold_logits.extend(logits.tolist())

        fold_acc = np.mean(np.array(fold_p) == np.array(fold_l))
        fold_accs.append(fold_acc)
        all_preds.extend(fold_p)
        all_labels.extend(fold_l)
        all_logits.extend(fold_logits)
        print(f"  Fold {i+1}: acc={fold_acc:.4f} ({len(fold_p)} samples)", flush=True)

    # Overall stats
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    logits = np.array(all_logits)

    acc = (preds == labels).mean()
    tp_count = ((preds == 1) & (labels == 1)).sum()
    fp_count = ((preds == 1) & (labels == 0)).sum()
    tn_count = ((preds == 0) & (labels == 0)).sum()
    fn_count = ((preds == 0) & (labels == 1)).sum()
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    npv = tn_count / (tn_count + fn_count) if (tn_count + fn_count) > 0 else 0

    # Prediction distribution
    n_pred_long = (preds == 1).sum()
    n_pred_short = (preds == 0).sum()
    pred_long_rate = n_pred_long / len(preds)

    # Base rate
    base_rate = labels.mean()

    # Corrected EV (precision-based, long-only)
    ev_long = float(precision) * TP * 100 - (1 - float(precision)) * SL * 100

    # For adaptive: bear side flips to TP=3%, SL=7.5%
    # Short EV uses NPV as win rate, with flipped ratios
    if adaptive:
        ev_short = float(npv) * SL * 100 - (1 - float(npv)) * TP * 100
    else:
        ev_short = float(npv) * SL * 100 - (1 - float(npv)) * TP * 100

    # Combined EV
    ev_combined = pred_long_rate * ev_long + (1 - pred_long_rate) * ev_short

    result = {
        "name": name,
        "adaptive": adaptive,
        "accuracy": float(acc),
        "precision": float(precision),
        "npv": float(npv),
        "recall": float(recall),
        "f1": float(f1),
        "base_rate": float(base_rate),
        "pred_long_rate": float(pred_long_rate),
        "n_pred_long": int(n_pred_long),
        "n_pred_short": int(n_pred_short),
        "ev_long_per_trade": float(ev_long),
        "ev_short_per_trade": float(ev_short),
        "ev_combined_per_trade": float(ev_combined),
        "total_samples": len(preds),
        "confusion": {
            "tp": int(tp_count), "fp": int(fp_count),
            "tn": int(tn_count), "fn": int(fn_count),
        },
        "fold_accs": [float(a) for a in fold_accs],
        "folds_below_50": sum(1 for a in fold_accs if a < 0.5),
    }

    return result


def main():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []

    for config in CONFIGS:
        name = config["name"]
        adaptive = config["adaptive"]

        print(f"\n{'=' * 70}", flush=True)
        print(f"  {name}", flush=True)
        print(f"  TP={TP*100:.1f}% SL={SL*100:.1f}% | Regime adaptive: {adaptive}", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        result = run_walkforward(df, name, adaptive)
        results.append(result)

        print(f"\n  RESULT: {name}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, "
              f"NPV: {result['npv']:.4f}, F1: {result['f1']:.4f}", flush=True)
        print(f"  Base rate (label=1): {result['base_rate']:.3f}, "
              f"Pred long rate: {result['pred_long_rate']:.3f}", flush=True)
        print(f"  Long EV/trade: {result['ev_long_per_trade']:+.3f}%", flush=True)
        print(f"  Pred long: {result['n_pred_long']}, Pred short: {result['n_pred_short']}", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)

    # Comparison
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  COMPARISON", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Metric':<25} {'Adaptive':>15} {'Fixed':>15}", flush=True)
    print("-" * 58, flush=True)

    r_a, r_f = results[0], results[1]
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision (long WR)"),
        ("npv", "NPV (short WR)"),
        ("f1", "F1"),
        ("base_rate", "Base rate (label=1)"),
        ("pred_long_rate", "Pred long rate"),
        ("total_samples", "Total samples"),
        ("ev_long_per_trade", "Long EV/trade"),
        ("folds_below_50", "Folds <50%"),
    ]:
        va, vf = r_a[key], r_f[key]
        if isinstance(va, float):
            if "ev" in key.lower():
                print(f"{label:<25} {va:>+14.3f}% {vf:>+14.3f}%", flush=True)
            else:
                print(f"{label:<25} {va:>15.4f} {vf:>15.4f}", flush=True)
        else:
            print(f"{label:<25} {va:>15} {vf:>15}", flush=True)

    # Fold-by-fold
    print(f"\n{'Fold':<8} {'Adaptive':>10} {'Fixed':>10} {'Diff':>10}", flush=True)
    print("-" * 40, flush=True)
    for i in range(min(len(r_a["fold_accs"]), len(r_f["fold_accs"]))):
        a, f = r_a["fold_accs"][i], r_f["fold_accs"][i]
        diff = f - a
        print(f"Fold {i+1:<3} {a:>9.1%} {f:>9.1%} {diff:>+9.1%}", flush=True)

    # Save
    out = EXPERIMENTS_DIR / "regime_compare_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)


if __name__ == "__main__":
    main()
