"""Verify trading strategy EVs with actual model predictions.

Trains single FGI adaptive model (7.5/3), saves per-sample predictions
with regime labels, then computes exact EV for all strategy variants.

Usage:
    python3 -m src.models.eval_strategy
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
from src.features.dataset import TimeSeriesDataset
from src.models.architecture import TemporalTransformerClassifier
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


def get_regime_for_dataset(ds, df):
    regime = np.full(len(ds), np.nan)
    for si, vi in enumerate(ds.valid_indices):
        label_idx = vi + ds.lookback - 1
        dt = pd.Timestamp(df["date"].iloc[label_idx]).date()
        fgi = FGI_LOOKUP.get(dt, np.nan)
        if not np.isnan(fgi):
            regime[si] = 1.0 if fgi >= 50 else 0.0
    return regime


def main():
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

    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

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

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TemporalTransformerClassifier()
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

    # ── Per-regime confusion matrices ──
    def confusion(p, l):
        tp = ((p == 1) & (l == 1)).sum()
        fp = ((p == 1) & (l == 0)).sum()
        tn = ((p == 0) & (l == 0)).sum()
        fn = ((p == 0) & (l == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        acc = (tp + tn) / (tp + fp + tn + fn) if len(p) > 0 else 0
        pred1_rate = (p == 1).mean()
        return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "precision": float(prec), "npv": float(npv), "accuracy": float(acc),
                "pred1_rate": float(pred1_rate), "n": len(p)}

    bull_stats = confusion(preds[bull], labels[bull])
    bear_stats = confusion(preds[bear], labels[bear])

    print(f"\n{'=' * 70}")
    print(f"  PER-REGIME STATS (ACTUAL)")
    print(f"{'=' * 70}")
    print(f"\n  BULL ({bull_stats['n']} bars):")
    print(f"    Acc={bull_stats['accuracy']:.4f} Prec={bull_stats['precision']:.4f} NPV={bull_stats['npv']:.4f}")
    print(f"    Pred=1 rate: {bull_stats['pred1_rate']:.4f}")
    print(f"    TP={bull_stats['tp']} FP={bull_stats['fp']} TN={bull_stats['tn']} FN={bull_stats['fn']}")

    print(f"\n  BEAR ({bear_stats['n']} bars):")
    print(f"    Acc={bear_stats['accuracy']:.4f} Prec={bear_stats['precision']:.4f} NPV={bear_stats['npv']:.4f}")
    print(f"    Pred=1 rate: {bear_stats['pred1_rate']:.4f}")
    print(f"    TP={bear_stats['tp']} FP={bear_stats['fp']} TN={bear_stats['tn']} FN={bear_stats['fn']}")

    # ── EV per direction ──
    # Bull: TP=7.5% SL=3% (no flip). Long wins 7.5%, Short wins 3%
    # Bear: TP=3% SL=7.5% (flipped). Long wins 3%, Short wins 7.5%
    ev_bull_long = bull_stats["precision"] * 7.5 - (1 - bull_stats["precision"]) * 3.0
    ev_bull_short = bull_stats["npv"] * 3.0 - (1 - bull_stats["npv"]) * 7.5
    ev_bear_long = bear_stats["precision"] * 3.0 - (1 - bear_stats["precision"]) * 7.5
    ev_bear_short = bear_stats["npv"] * 7.5 - (1 - bear_stats["npv"]) * 3.0

    print(f"\n{'=' * 70}")
    print(f"  EV PER TRADE DIRECTION (ACTUAL)")
    print(f"{'=' * 70}")
    print(f"\n  {'Direction':<20} {'Bull regime':>15} {'Bear regime':>15}")
    print(f"  {'-' * 50}")
    print(f"  {'Long (pred=1)':<20} {ev_bull_long:>+14.3f}% {ev_bear_long:>+14.3f}%")
    print(f"  {'Short (pred=0)':<20} {ev_bull_short:>+14.3f}% {ev_bear_short:>+14.3f}%")

    # ── Strategy comparison ──
    bull_frac = bull.sum() / (bull.sum() + bear.sum())
    bear_frac = 1 - bull_frac
    bp1 = bull_stats["pred1_rate"]
    bp0 = 1 - bp1
    ep1 = bear_stats["pred1_rate"]
    ep0 = 1 - ep1

    strategies = {
        "1. Long bull only": bull_frac * bp1 * ev_bull_long,
        "2. Long bull, short bear": bull_frac * bp1 * ev_bull_long + bear_frac * ep0 * ev_bear_short,
        "3. Both sides, both regimes": (bull_frac * bp1 * ev_bull_long + bull_frac * bp0 * ev_bull_short
                                        + bear_frac * ep1 * ev_bear_long + bear_frac * ep0 * ev_bear_short),
        "4. Long bull, both sides bear": (bull_frac * bp1 * ev_bull_long
                                          + bear_frac * ep1 * ev_bear_long + bear_frac * ep0 * ev_bear_short),
        "5. Current (long both)": bull_frac * bp1 * ev_bull_long + bear_frac * ep1 * ev_bear_long,
    }

    print(f"\n{'=' * 70}")
    print(f"  STRATEGY RANKING (ACTUAL)")
    print(f"{'=' * 70}")
    for name, ev in sorted(strategies.items(), key=lambda x: -x[1]):
        annual = ev * 365
        print(f"  {name:<35} {ev:>+.3f}%/day   (~{annual:>+.0f}%/yr)", flush=True)

    # Save
    results = {
        "bull_stats": bull_stats,
        "bear_stats": bear_stats,
        "ev": {
            "bull_long": ev_bull_long, "bull_short": ev_bull_short,
            "bear_long": ev_bear_long, "bear_short": ev_bear_short,
        },
        "strategies": strategies,
        "fold_accs": [float(a) for a in fold_accs],
    }
    out = EXPERIMENTS_DIR / "strategy_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
