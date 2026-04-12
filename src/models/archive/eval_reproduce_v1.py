"""Reproduce the original Temporal Transformer baseline (61.9% target).

Uses frozen v2_temporal architecture + frozen v1_raw pipeline.
This is the "last known good" configuration from the pre-overhaul era.

Usage:
    python3 -m src.models.eval_reproduce_v1
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, EXPERIMENTS_DIR, LABEL_FGI_PATH
from src.features.pipelines.v1_raw import build_feature_matrix_v1, FEATURE_COLS_V1
from src.features.dataset import TimeSeriesDataset, make_loader
from src.models.architectures.v2_temporal import TemporalTransformerV2
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

    print("Loading data (FROZEN v1_raw pipeline)...", flush=True)
    df = build_feature_matrix_v1()
    print(f"Data: {len(df)} rows, {len(FEATURE_COLS_V1)} features\n", flush=True)

    all_preds = []
    all_labels = []
    all_regimes = []
    fold_accs = []
    start_time = time.time()

    for i in range(len(FOLD_BOUNDARIES) - 2):
        train_end = FOLD_BOUNDARIES[i]
        val_end = FOLD_BOUNDARIES[i + 1]
        test_end = FOLD_BOUNDARIES[i + 2]

        train_df = df[df["date"] < train_end].reset_index(drop=True)
        val_df = df[(df["date"] >= train_end) & (df["date"] < val_end)].reset_index(drop=True)
        test_df = df[(df["date"] >= val_end) & (df["date"] < test_end)].reset_index(drop=True)

        if len(train_df) < 1000 or len(val_df) < 100 or len(test_df) < 100:
            continue

        # Pass FEATURE_COLS_V1 explicitly — don't rely on the global
        train_ds = TimeSeriesDataset(train_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)
        val_ds = TimeSeriesDataset(val_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)
        test_ds = TimeSeriesDataset(test_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        regime = get_regime_for_dataset(test_ds, test_df)

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        # Frozen v2_temporal: 18 other features (10 derived + 8 VP structure)
        model = TemporalTransformerV2(n_other_features=18)

        train_model(
            model, train_loader, val_loader,
            lr=5e-4,
            lr_schedule="constant",
            grad_clip=0.0,
            seed=42,
        )

        model.eval()
        device = next(model.parameters()).device
        fold_p, fold_l = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = (logits > 0).float().cpu()
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
        return float(prec), float(npv)

    bull_prec, bull_npv = confusion(preds[bull], labels[bull])
    bear_prec, bear_npv = confusion(preds[bear], labels[bear])

    ev_bull_long = bull_prec * 7.5 - (1 - bull_prec) * 3.0
    ev_bear_long = bear_prec * 3.0 - (1 - bear_prec) * 7.5
    ev_bear_short = bear_npv * 7.5 - (1 - bear_npv) * 3.0

    result = {
        "name": "Reproduce v1 (TemporalV2 + v1_raw pipeline)",
        "accuracy": float((preds == labels).mean()),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_acc_std": float(np.std(fold_accs)),
        "folds_below_50": int(sum(1 for a in fold_accs if a < 0.5)),
        "bull_precision": bull_prec,
        "bear_precision": bear_prec,
        "bear_npv": bear_npv,
        "ev_bull_long": float(ev_bull_long),
        "ev_bear_long": float(ev_bear_long),
        "ev_bear_short": float(ev_bear_short),
        "runtime_sec": time.time() - start_time,
    }

    print(f"\n{'=' * 80}")
    print(f"  REPRODUCTION RESULT")
    print(f"{'=' * 80}")
    print(f"  Accuracy: {result['accuracy']:.4f}  (target: ~61.9%)")
    print(f"  Folds <50%: {result['folds_below_50']}/10")
    print(f"  Bull precision: {result['bull_precision']:.4f}")
    print(f"  Bear precision: {result['bear_precision']:.4f}")
    print(f"  EVs: bull_long={result['ev_bull_long']:+.3f}%, "
          f"bear_long={result['ev_bear_long']:+.3f}%, "
          f"bear_short={result['ev_bear_short']:+.3f}%")
    print(f"  Runtime: {result['runtime_sec']/60:.1f} min")

    out = EXPERIMENTS_DIR / "reproduce_v1_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {out}")


if __name__ == "__main__":
    main()
