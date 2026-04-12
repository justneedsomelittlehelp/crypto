"""TP/SL sweep — WIDE TP side (TP > SL). Run after sweep_tpsl.py.

Tests whether lower win rate + bigger wins = positive EV.
Breakeven win rates: 2:1 needs 33%, 3:1 needs 25%.

Usage:
    python3 -m src.models.sweep_tpsl_wide
"""

import sys
import json
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

# Sweep: bull SL fixed at 3%, vary TP wider
SWEEP_CONFIGS = [
    {"name": "4/3 (1.33:1)", "tp": 0.04, "sl": 0.03},
    {"name": "5/3 (1.67:1)", "tp": 0.05, "sl": 0.03},
    {"name": "6/3 (2:1)",    "tp": 0.06, "sl": 0.03},
    {"name": "7.5/3 (2.5:1)","tp": 0.075,"sl": 0.03},
    {"name": "9/3 (3:1)",    "tp": 0.09, "sl": 0.03},
]


def run_sweep():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []

    for config in SWEEP_CONFIGS:
        tp, sl = config["tp"], config["sl"]
        name = config["name"]
        breakeven = sl / (tp + sl) * 100
        print(f"\n{'=' * 70}", flush=True)
        print(f"  SWEEP: {name} (bull TP={tp*100:.1f}% SL={sl*100:.1f}%, breakeven={breakeven:.1f}%)", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        import src.config as cfg
        cfg.LABEL_TP_PCT = tp
        cfg.LABEL_SL_PCT = sl

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

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        acc = (preds == labels).mean()
        tp_count = ((preds == 1) & (labels == 1)).sum()
        fp_count = ((preds == 1) & (labels == 0)).sum()
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0

        ev_per_trade = acc * tp * 100 - (1 - acc) * sl * 100

        result = {
            "name": name,
            "tp_pct": tp * 100,
            "sl_pct": sl * 100,
            "breakeven_winrate": float(breakeven),
            "accuracy": float(acc),
            "precision": float(precision),
            "ev_per_trade": float(ev_per_trade),
            "margin_above_breakeven": float(acc * 100 - breakeven),
            "total_samples": len(preds),
            "fold_accs": [float(a) for a in fold_accs],
            "folds_below_50": sum(1 for a in fold_accs if a < 0.5),
            "folds_below_breakeven": sum(1 for a in fold_accs if a < breakeven / 100),
        }
        results.append(result)

        print(f"\n  RESULT: {name}", flush=True)
        print(f"  Accuracy: {acc:.4f}, Breakeven: {breakeven:.1f}%, Margin: {result['margin_above_breakeven']:+.1f}%", flush=True)
        print(f"  EV per trade: {ev_per_trade:+.4f}%", flush=True)
        print(f"  Folds below breakeven: {result['folds_below_breakeven']}/10", flush=True)

    output_path = EXPERIMENTS_DIR / "sweep_tpsl_wide_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  WIDE TP SWEEP SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Config':<18} {'Acc':<8} {'BrkEvn':<8} {'Margin':<8} {'EV/trade':<10} {'<BrkEvn':<8}", flush=True)
    print("-" * 62, flush=True)
    for r in results:
        print(f"{r['name']:<18} {r['accuracy']:<8.4f} {r['breakeven_winrate']:<8.1f} {r['margin_above_breakeven']:<+8.1f} {r['ev_per_trade']:<+10.4f} {r['folds_below_breakeven']:<8}", flush=True)

    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == "__main__":
    run_sweep()
