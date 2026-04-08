"""TP/SL ratio sweep — run overnight, results saved to file.

Usage:
    python3 -m src.models.sweep_tpsl
"""

import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

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

# Sweep: bull TP fixed at 3%, vary SL
SWEEP_CONFIGS = [
    {"name": "3/3 (1:1)",    "tp": 0.03, "sl": 0.03},
    {"name": "3/4 (1:1.33)", "tp": 0.03, "sl": 0.04},
    {"name": "3/5 (1:1.67)", "tp": 0.03, "sl": 0.05},
    {"name": "3/6 (1:2)",    "tp": 0.03, "sl": 0.06},
    {"name": "3/7.5 (1:2.5)","tp": 0.03, "sl": 0.075},
    {"name": "3/9 (1:3)",    "tp": 0.03, "sl": 0.09},
]


def run_sweep():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []

    for config in SWEEP_CONFIGS:
        tp, sl = config["tp"], config["sl"]
        name = config["name"]
        print(f"\n{'=' * 70}", flush=True)
        print(f"  SWEEP: {name} (bull TP={tp*100:.1f}% SL={sl*100:.1f}%, bear flipped)", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        # Override label config for this sweep
        import src.config as cfg
        cfg.LABEL_TP_PCT = tp
        cfg.LABEL_SL_PCT = sl

        # Also update the dataset module's imported values
        import src.features.dataset as ds_mod
        ds_mod.LABEL_TP_PCT = tp
        ds_mod.LABEL_SL_PCT = sl

        all_preds = []
        all_labels = []
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
            print(f"  Fold {i+1}: acc={fold_acc:.4f} ({len(fold_p)} samples)", flush=True)

        # Overall stats
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        acc = (preds == labels).mean()
        tp_count = ((preds == 1) & (labels == 1)).sum()
        fp_count = ((preds == 1) & (labels == 0)).sum()
        tn_count = ((preds == 0) & (labels == 0)).sum()
        fn_count = ((preds == 0) & (labels == 1)).sum()
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0

        # EV calculation
        win_rate = acc
        ev_per_trade = win_rate * tp * 100 - (1 - win_rate) * sl * 100

        result = {
            "name": name,
            "tp_pct": tp * 100,
            "sl_pct": sl * 100,
            "accuracy": float(acc),
            "precision": float(precision),
            "ev_per_trade": float(ev_per_trade),
            "total_samples": len(preds),
            "fold_accs": [float(a) for a in fold_accs],
            "folds_below_50": sum(1 for a in fold_accs if a < 0.5),
        }
        results.append(result)

        print(f"\n  RESULT: {name}", flush=True)
        print(f"  Accuracy: {acc:.4f}, Precision: {precision:.4f}", flush=True)
        print(f"  EV per trade: {ev_per_trade:+.4f}%", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)

    # Save results
    output_path = EXPERIMENTS_DIR / "sweep_tpsl_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  TP/SL SWEEP SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Config':<18} {'Acc':<8} {'Prec':<8} {'EV/trade':<10} {'Samples':<8} {'<50%':<6}", flush=True)
    print("-" * 58, flush=True)
    for r in results:
        print(f"{r['name']:<18} {r['accuracy']:<8.4f} {r['precision']:<8.4f} {r['ev_per_trade']:<+10.4f} {r['total_samples']:<8} {r['folds_below_50']:<6}", flush=True)

    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == "__main__":
    run_sweep()
