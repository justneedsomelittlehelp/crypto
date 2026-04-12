"""Diagnostic sweep to find why LR sweep tanked.

Tests 5 configurations to isolate the regression:

1. v3 baseline (current): DualBranch + new pipeline + grad_clip=1.0
   → expected: matches LR sweep results (~52% acc)

2. Loosened grad clipping (max_norm=5.0)
   → tests if grad clipping was choking the model

3. No grad clipping
   → tests if grad clipping was the entire cause

4. Dual-branch + OLD pipeline (no z-score, no tanh, OHLC raw)
   → tests if pipeline overhaul is at fault

5. Temporal-only (no candle branch) + new pipeline
   → tests if dual-branch design is hurting with working candles

All use seed=42 + cosine_warmup_5e-4 for fair comparison.

Usage:
    python3 -m src.models.sweep_diagnose
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch

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

# 5 diagnostic configs
DIAGNOSTIC_CONFIGS = [
    {
        "name": "1. baseline (DualBranch + new pipeline + clip=1.0)",
        "model_cls": DualBranchTransformerClassifier,
        "grad_clip": 1.0,
        "old_pipeline": False,
    },
    {
        "name": "2. loosened clip (clip=5.0)",
        "model_cls": DualBranchTransformerClassifier,
        "grad_clip": 5.0,
        "old_pipeline": False,
    },
    {
        "name": "3. no clip (clip=0)",
        "model_cls": DualBranchTransformerClassifier,
        "grad_clip": 0.0,
        "old_pipeline": False,
    },
    {
        "name": "4. DualBranch + OLD pipeline (no scaling)",
        "model_cls": DualBranchTransformerClassifier,
        "grad_clip": 5.0,
        "old_pipeline": True,
    },
    {
        "name": "5. Temporal-only + new pipeline (no candle branch)",
        "model_cls": TemporalTransformerClassifier,
        "grad_clip": 5.0,
        "old_pipeline": False,
    },
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


def build_features_old_pipeline():
    """Rebuild feature matrix with the OLD pipeline behavior (no scaling)."""
    import src.features.pipeline as pipe_mod

    # Monkey-patch compute_derived_features to skip the scaling block
    original_func = pipe_mod.compute_derived_features

    def old_derived(df):
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["bar_range"] = (df["high"] - df["low"]) / df["close"]
        df["bar_body"] = (df["close"] - df["open"]) / df["open"]
        from src.config import VOLUME_COL, VOLUME_ROLL_WINDOW_BARS
        rolling_mean = df[VOLUME_COL].rolling(
            window=VOLUME_ROLL_WINDOW_BARS, min_periods=VOLUME_ROLL_WINDOW_BARS
        ).mean()
        df["volume_ratio"] = df[VOLUME_COL] / rolling_mean
        bar_height = (df["high"] - df["low"]).clip(lower=1e-10)
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / bar_height
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_height
        df["body_dir"] = np.sign(df["close"] - df["open"])
        df["ohlc_open_ratio"] = df["open"] / df["close"]
        df["ohlc_high_ratio"] = df["high"] / df["close"]
        df["ohlc_low_ratio"] = df["low"] / df["close"]
        return df

    pipe_mod.compute_derived_features = old_derived
    df = build_feature_matrix()
    pipe_mod.compute_derived_features = original_func  # restore
    return df


def run_walkforward(df, config):
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

        model = config["model_cls"]()
        train_model(
            model, train_loader, val_loader,
            lr=5e-4,
            lr_schedule="warmup_cosine",
            warmup_epochs=5,
            grad_clip=config["grad_clip"],
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

    return {
        "name": config["name"],
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
    }


def main():
    print("Loading NEW pipeline data...", flush=True)
    df_new = build_feature_matrix()
    print(f"NEW pipeline: {len(df_new)} rows\n", flush=True)

    print("Loading OLD pipeline data...", flush=True)
    df_old = build_features_old_pipeline()
    print(f"OLD pipeline: {len(df_old)} rows\n", flush=True)

    results = []
    start_time = time.time()

    for i, config in enumerate(DIAGNOSTIC_CONFIGS):
        df = df_old if config["old_pipeline"] else df_new
        print(f"\n{'=' * 80}", flush=True)
        print(f"  CONFIG {i+1}/{len(DIAGNOSTIC_CONFIGS)}: {config['name']}", flush=True)
        print(f"  model={config['model_cls'].__name__}, grad_clip={config['grad_clip']}, old_pipeline={config['old_pipeline']}", flush=True)
        print(f"{'=' * 80}\n", flush=True)

        config_start = time.time()
        result = run_walkforward(df, config)
        config_time = time.time() - config_start
        result["runtime_sec"] = config_time
        results.append(result)

        print(f"\n  RESULT: {config['name']}", flush=True)
        print(f"  Acc: {result['accuracy']:.4f} (std: {result['fold_acc_std']:.4f}), "
              f"Folds <50%: {result['folds_below_50']}/10", flush=True)
        print(f"  EVs: bull_long={result['ev_bull_long']:+.3f}%, "
              f"bear_long={result['ev_bear_long']:+.3f}%, "
              f"bear_short={result['ev_bear_short']:+.3f}%", flush=True)
        print(f"  Runtime: {config_time/60:.1f} min", flush=True)

        # Save intermediate results
        out = EXPERIMENTS_DIR / "diagnose_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n\n{'=' * 100}", flush=True)
    print(f"  DIAGNOSTIC SWEEP SUMMARY ({total_time/60:.1f} min total)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"{'Config':<55} {'Acc':>8} {'StdDev':>8} {'<50%':>6} {'EVbullL':>9} {'EVbearL':>9} {'EVbearS':>9}", flush=True)
    print("-" * 110, flush=True)
    for r in results:
        print(f"{r['name']:<55} {r['accuracy']:>7.4f} {r['fold_acc_std']:>7.4f} "
              f"{r['folds_below_50']:>5d} {r['ev_bull_long']:>+8.3f}% "
              f"{r['ev_bear_long']:>+8.3f}% {r['ev_bear_short']:>+8.3f}%", flush=True)


if __name__ == "__main__":
    main()
