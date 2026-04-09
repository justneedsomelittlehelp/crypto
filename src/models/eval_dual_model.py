"""Dual-model evaluation: separate bull (long) and bear (short) models.

Bull model: TP=7.5% SL=3%, goes long when pred=1. Active when FGI >= 50.
Bear model: TP=3% SL=9%, goes short when pred=0. Active when FGI < 50.

Both models train on ALL data (no regime flip — fixed labels).
FGI decides which model is active at inference time.

Also compares against single adaptive model as baseline.

Usage:
    python3 -m src.models.eval_dual_model
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

# FGI lookup (daily)
FGI_DF = pd.read_csv(LABEL_FGI_PATH, parse_dates=["date"])
FGI_LOOKUP = dict(zip(FGI_DF["date"].dt.date, FGI_DF["fgi_value"].astype(float)))
FGI_THRESHOLD = 50


def get_fgi_regime_for_df(df):
    """Return boolean array: True=bull (FGI >= 50), False=bear."""
    dates = df["date"].values
    regime = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        dt = pd.Timestamp(d).date()
        fgi = FGI_LOOKUP.get(dt, np.nan)
        if not np.isnan(fgi):
            regime[i] = float(fgi >= FGI_THRESHOLD)
    return regime


def get_regime_for_dataset(ds, df):
    """Map FGI regime to each valid sample in the dataset."""
    regime_full = get_fgi_regime_for_df(df)
    regime_per_sample = np.full(len(ds), np.nan)
    for si, vi in enumerate(ds.valid_indices):
        label_idx = vi + ds.lookback - 1
        if label_idx < len(regime_full):
            regime_per_sample[si] = regime_full[label_idx]
    return regime_per_sample


def train_and_predict(df, tp, sl, fold_boundaries):
    """Train walk-forward and return per-sample predictions, labels, regimes."""
    import src.config as cfg
    import src.features.dataset as ds_mod

    cfg.LABEL_TP_PCT = tp
    cfg.LABEL_SL_PCT = sl
    cfg.LABEL_REGIME_ADAPTIVE = False  # Fixed labels, no flip
    cfg.LABEL_REGIME_MODE = "sma"
    ds_mod.LABEL_TP_PCT = tp
    ds_mod.LABEL_SL_PCT = sl
    ds_mod.LABEL_REGIME_ADAPTIVE = False
    ds_mod.LABEL_REGIME_MODE = "sma"

    all_preds = []
    all_labels = []
    all_regimes = []
    fold_accs = []

    for i in range(len(fold_boundaries) - 2):
        train_end = fold_boundaries[i]
        val_end = fold_boundaries[i + 1]
        test_end = fold_boundaries[i + 2]

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
        all_regimes.extend(regime[:len(fold_p)].tolist())
        print(f"    Fold {i+1}: acc={fold_acc:.4f} ({len(fold_p)} samples)", flush=True)

    return (np.array(all_preds), np.array(all_labels),
            np.array(all_regimes), fold_accs)


def main():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    # ── BULL MODEL ──
    print(f"{'=' * 70}", flush=True)
    print(f"  BULL MODEL: TP=7.5% SL=3% (long when pred=1)", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    bull_preds, bull_labels, bull_regimes, bull_fold_accs = \
        train_and_predict(df, tp=0.075, sl=0.03, fold_boundaries=FOLD_BOUNDARIES)

    # ── BEAR MODEL ──
    print(f"\n{'=' * 70}", flush=True)
    print(f"  BEAR MODEL: TP=3% SL=9% (short when pred=0)", flush=True)
    print(f"{'=' * 70}\n", flush=True)
    bear_preds, bear_labels, bear_regimes, bear_fold_accs = \
        train_and_predict(df, tp=0.03, sl=0.09, fold_boundaries=FOLD_BOUNDARIES)

    # ── COMBINE: use bull model in bull regime, bear model in bear regime ──
    print(f"\n{'=' * 70}", flush=True)
    print(f"  DUAL MODEL COMBINED RESULTS", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    # Bull model stats (only on bull-regime bars)
    bull_mask = bull_regimes == 1.0
    bear_mask_bull = bull_regimes == 0.0  # bear bars from bull model's perspective

    # Bear model stats (only on bear-regime bars)
    bear_mask = bear_regimes == 0.0
    bull_mask_bear = bear_regimes == 1.0  # bull bars from bear model's perspective

    # Bull model in bull regime: long when pred=1
    b_bull = bull_mask
    b_preds = bull_preds[b_bull]
    b_labels = bull_labels[b_bull]
    b_n = len(b_preds)
    b_tp = ((b_preds == 1) & (b_labels == 1)).sum()
    b_fp = ((b_preds == 1) & (b_labels == 0)).sum()
    b_long_trades = (b_preds == 1).sum()
    b_precision = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0
    b_acc = (b_preds == b_labels).mean() if b_n > 0 else 0
    # EV: long wins 7.5%, loses 3%
    ev_bull_long = float(b_precision) * 7.5 - (1 - float(b_precision)) * 3.0

    # Bear model in bear regime: short when pred=0
    s_bear = bear_mask
    s_preds = bear_preds[s_bear]
    s_labels = bear_labels[s_bear]
    s_n = len(s_preds)
    s_tn = ((s_preds == 0) & (s_labels == 0)).sum()
    s_fn = ((s_preds == 0) & (s_labels == 1)).sum()
    s_short_trades = (s_preds == 0).sum()
    s_npv = s_tn / (s_tn + s_fn) if (s_tn + s_fn) > 0 else 0
    s_acc = (s_preds == s_labels).mean() if s_n > 0 else 0
    # EV: short wins 9% (label=0, price dropped), loses 3% (label=1, price rose)
    ev_bear_short = float(s_npv) * 9.0 - (1 - float(s_npv)) * 3.0

    # Combined stats
    total_bull_bars = b_n
    total_bear_bars = s_n
    bull_frac = total_bull_bars / (total_bull_bars + total_bear_bars) if (total_bull_bars + total_bear_bars) > 0 else 0.5
    bear_frac = 1 - bull_frac
    ev_combined = bull_frac * ev_bull_long + bear_frac * ev_bear_short

    print(f"  BULL SIDE (FGI >= 50, long when pred=1):", flush=True)
    print(f"    Bars: {total_bull_bars}, Long trades: {int(b_long_trades)}", flush=True)
    print(f"    Accuracy: {b_acc:.4f}, Precision: {float(b_precision):.4f}", flush=True)
    print(f"    EV per long trade: {ev_bull_long:+.3f}%", flush=True)
    print(f"    Breakeven: 28.6%, Margin: {float(b_precision)*100 - 28.6:+.1f}%", flush=True)

    print(f"\n  BEAR SIDE (FGI < 50, short when pred=0):", flush=True)
    print(f"    Bars: {total_bear_bars}, Short trades: {int(s_short_trades)}", flush=True)
    print(f"    Accuracy: {s_acc:.4f}, NPV: {float(s_npv):.4f}", flush=True)
    print(f"    EV per short trade: {ev_bear_short:+.3f}%", flush=True)
    print(f"    Breakeven: 25.0%, Margin: {float(s_npv)*100 - 25.0:+.1f}%", flush=True)

    print(f"\n  COMBINED:", flush=True)
    print(f"    Bull fraction: {bull_frac:.1%}, Bear fraction: {bear_frac:.1%}", flush=True)
    print(f"    Weighted EV: {ev_combined:+.3f}%", flush=True)

    # ── FOLD-BY-FOLD ──
    print(f"\n  FOLD-BY-FOLD:", flush=True)
    print(f"  {'Fold':<8} {'Bull acc':>10} {'Bear acc':>10}", flush=True)
    print(f"  {'-' * 30}", flush=True)
    for i in range(min(len(bull_fold_accs), len(bear_fold_accs))):
        print(f"  Fold {i+1:<3} {bull_fold_accs[i]:>9.1%} {bear_fold_accs[i]:>9.1%}", flush=True)

    # ── COMPARISON WITH SINGLE ADAPTIVE MODEL ──
    # (reference: FGI single model from regime_fgi_compare_results.json)
    ref_path = EXPERIMENTS_DIR / "regime_fgi_compare_results.json"
    if ref_path.exists():
        with open(ref_path) as f:
            ref_data = json.load(f)
        fgi_ref = next((r for r in ref_data if "FGI" in r["name"]), None)
        if fgi_ref:
            print(f"\n  {'=' * 60}", flush=True)
            print(f"  DUAL MODEL vs SINGLE ADAPTIVE MODEL (FGI)", flush=True)
            print(f"  {'=' * 60}", flush=True)
            print(f"  {'Metric':<30} {'Dual':>12} {'Single':>12}", flush=True)
            print(f"  {'-' * 56}", flush=True)
            print(f"  {'EV bull':.<30} {ev_bull_long:>+11.3f}% {fgi_ref['ev_bull']:>+11.3f}%", flush=True)
            print(f"  {'EV bear':.<30} {ev_bear_short:>+11.3f}% {fgi_ref['ev_bear']:>+11.3f}%", flush=True)
            print(f"  {'EV combined':.<30} {ev_combined:>+11.3f}% {fgi_ref['ev_combined']:>+11.3f}%", flush=True)
            print(f"  {'Bull precision':.<30} {float(b_precision):>12.4f} {fgi_ref['prec_bull']:>12.4f}", flush=True)
            print(f"  {'Bear NPV/precision':.<30} {float(s_npv):>12.4f} {fgi_ref['prec_bear']:>12.4f}", flush=True)

    # Save results
    results = {
        "bull_model": {
            "tp_pct": 7.5, "sl_pct": 3.0,
            "n_bull_bars": int(total_bull_bars),
            "n_long_trades": int(b_long_trades),
            "accuracy": float(b_acc),
            "precision": float(b_precision),
            "ev_per_trade": float(ev_bull_long),
            "fold_accs": [float(a) for a in bull_fold_accs],
        },
        "bear_model": {
            "tp_pct": 3.0, "sl_pct": 9.0,
            "n_bear_bars": int(total_bear_bars),
            "n_short_trades": int(s_short_trades),
            "accuracy": float(s_acc),
            "npv": float(s_npv),
            "ev_per_trade": float(ev_bear_short),
            "fold_accs": [float(a) for a in bear_fold_accs],
        },
        "combined": {
            "bull_frac": float(bull_frac),
            "bear_frac": float(bear_frac),
            "ev_combined": float(ev_combined),
        },
    }
    out = EXPERIMENTS_DIR / "dual_model_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
