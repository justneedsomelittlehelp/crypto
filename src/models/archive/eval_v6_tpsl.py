"""Evaluate v6_temporal_enriched with different TP/SL ratios.

Config 1: 2.5/1 asymmetric (FGI regime adaptive, same ratio as 7.5/3 but tighter)
Config 2: 1/1 symmetric (no regime, no FGI)
Config 3: 2.5/1 on v2_temporal baseline (to measure enrichment effect)

All use v1_raw pipeline, batch from config, seed=42.

Usage:
    python3 -m src.models.eval_v6_tpsl
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, EXPERIMENTS_DIR, LABEL_FGI_PATH
from src.features.pipelines.v1_raw import build_feature_matrix_v1, FEATURE_COLS_V1, feature_index_v1
from src.features.dataset import TimeSeriesDataset, make_loader
from src.models.architectures.v6_temporal_enriched import TemporalEnrichedV6
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

FGI_DF = pd.read_csv(LABEL_FGI_PATH, parse_dates=["date"])
FGI_LOOKUP = dict(zip(FGI_DF["date"].dt.date, FGI_DF["fgi_value"].astype(float)))


def make_v6_model():
    return TemporalEnrichedV6(
        ohlc_open_idx=feature_index_v1("ohlc_open_ratio"),
        ohlc_high_idx=feature_index_v1("ohlc_high_ratio"),
        ohlc_low_idx=feature_index_v1("ohlc_low_ratio"),
        log_return_idx=feature_index_v1("log_return"),
        volume_ratio_idx=feature_index_v1("volume_ratio"),
        vp_structure_start_idx=feature_index_v1("vp_ceiling_dist"),
        n_vp_structure=8,
        n_other_features=18,
    )


def make_v2_model():
    return TemporalTransformerV2(n_other_features=18)


CONFIGS = [
    {
        "name": "v6 enriched + 2.5/1 (FGI adaptive)",
        "model_fn": make_v6_model,
        "tp": 0.025, "sl": 0.01,
        "regime_adaptive": True,
    },
    {
        "name": "v6 enriched + 1/1 (no regime)",
        "model_fn": make_v6_model,
        "tp": 0.01, "sl": 0.01,
        "regime_adaptive": False,
    },
    {
        "name": "v2 temporal baseline + 2.5/1 (FGI adaptive)",
        "model_fn": make_v2_model,
        "tp": 0.025, "sl": 0.01,
        "regime_adaptive": True,
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


def run_walkforward(df, config):
    import src.config as cfg
    import src.features.dataset as ds_mod

    cfg.LABEL_TP_PCT = config["tp"]
    cfg.LABEL_SL_PCT = config["sl"]
    cfg.LABEL_REGIME_ADAPTIVE = config["regime_adaptive"]
    cfg.LABEL_REGIME_MODE = "fgi" if config["regime_adaptive"] else "sma"
    ds_mod.LABEL_TP_PCT = config["tp"]
    ds_mod.LABEL_SL_PCT = config["sl"]
    ds_mod.LABEL_REGIME_ADAPTIVE = config["regime_adaptive"]
    ds_mod.LABEL_REGIME_MODE = cfg.LABEL_REGIME_MODE

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

        train_ds = TimeSeriesDataset(train_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)
        val_ds = TimeSeriesDataset(val_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)
        test_ds = TimeSeriesDataset(test_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        regime = get_regime_for_dataset(test_ds, test_df)

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        model = config["model_fn"]()
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

    # Overall stats
    acc = (preds == labels).mean()
    tp_count = ((preds == 1) & (labels == 1)).sum()
    fp_count = ((preds == 1) & (labels == 0)).sum()
    tn_count = ((preds == 0) & (labels == 0)).sum()
    fn_count = ((preds == 0) & (labels == 1)).sum()
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    npv = tn_count / (tn_count + fn_count) if (tn_count + fn_count) > 0 else 0
    base_rate = labels.mean()
    pred1_rate = (preds == 1).mean()

    # EV per trade direction
    tp_pct = config["tp"] * 100
    sl_pct = config["sl"] * 100
    if config["regime_adaptive"]:
        # Bull: long wins TP%, loses SL%. Bear: flipped.
        bull = regimes == 1.0
        bear = regimes == 0.0
        bull_prec = ((preds[bull] == 1) & (labels[bull] == 1)).sum() / max(1, ((preds[bull] == 1)).sum())
        bear_prec = ((preds[bear] == 1) & (labels[bear] == 1)).sum() / max(1, ((preds[bear] == 1)).sum())
        bull_npv_val = ((preds[bull] == 0) & (labels[bull] == 0)).sum() / max(1, ((preds[bull] == 0)).sum())
        bear_npv_val = ((preds[bear] == 0) & (labels[bear] == 0)).sum() / max(1, ((preds[bear] == 0)).sum())

        ev_bull_long = float(bull_prec) * tp_pct - (1 - float(bull_prec)) * sl_pct
        ev_bear_long = float(bear_prec) * sl_pct - (1 - float(bear_prec)) * tp_pct
        ev_bear_short = float(bear_npv_val) * tp_pct - (1 - float(bear_npv_val)) * sl_pct
    else:
        # Symmetric: no regime split
        ev_bull_long = float(precision) * tp_pct - (1 - float(precision)) * sl_pct
        ev_bear_long = 0.0
        ev_bear_short = float(npv) * sl_pct - (1 - float(npv)) * tp_pct
        bull_prec = float(precision)
        bear_prec = 0.0
        bear_npv_val = float(npv)

    return {
        "name": config["name"],
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "regime_adaptive": config["regime_adaptive"],
        "accuracy": float(acc),
        "precision": float(precision),
        "npv": float(npv),
        "base_rate": float(base_rate),
        "pred1_rate": float(pred1_rate),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_acc_std": float(np.std(fold_accs)),
        "folds_below_50": int(sum(1 for a in fold_accs if a < 0.5)),
        "total_samples": len(preds),
        "bull_precision": float(bull_prec),
        "bear_precision": float(bear_prec),
        "bear_npv": float(bear_npv_val),
        "ev_bull_long": float(ev_bull_long),
        "ev_bear_long": float(ev_bear_long),
        "ev_bear_short": float(ev_bear_short),
    }


def main():
    print("Loading data (v1_raw pipeline)...", flush=True)
    df = build_feature_matrix_v1()
    print(f"Data: {len(df)} rows, {len(FEATURE_COLS_V1)} features\n", flush=True)

    results = []
    start_time = time.time()

    for i, config in enumerate(CONFIGS):
        print(f"\n{'=' * 80}", flush=True)
        params = sum(p.numel() for p in config["model_fn"]().parameters())
        print(f"  CONFIG {i+1}/{len(CONFIGS)}: {config['name']}", flush=True)
        print(f"  TP={config['tp']*100:.1f}% SL={config['sl']*100:.1f}% "
              f"regime={'FGI adaptive' if config['regime_adaptive'] else 'none'} "
              f"params={params:,}", flush=True)
        print(f"{'=' * 80}\n", flush=True)

        config_start = time.time()
        result = run_walkforward(df, config)
        config_time = time.time() - config_start
        result["runtime_sec"] = config_time
        result["params"] = params
        results.append(result)

        print(f"\n  RESULT: {config['name']}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f}, Precision (UP): {result['precision']:.4f}, "
              f"NPV (DOWN): {result['npv']:.4f}", flush=True)
        print(f"  Base rate: {result['base_rate']:.3f}, Pred UP rate: {result['pred1_rate']:.3f}", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)
        print(f"  EVs: bull_long={result['ev_bull_long']:+.3f}%, "
              f"bear_long={result['ev_bear_long']:+.3f}%, "
              f"bear_short={result['ev_bear_short']:+.3f}%", flush=True)
        print(f"  Runtime: {config_time/60:.1f} min", flush=True)

        # Save intermediate
        out = EXPERIMENTS_DIR / "v6_tpsl_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    total_time = time.time() - start_time
    print(f"\n\n{'=' * 100}", flush=True)
    print(f"  SUMMARY ({total_time/60:.1f} min total)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"{'Config':<45} {'Acc':>7} {'PrecUP':>8} {'NPVdn':>8} {'<50%':>5} "
          f"{'EVbullL':>9} {'EVbearL':>9} {'EVbearS':>9}", flush=True)
    print("-" * 105, flush=True)
    for r in results:
        print(f"{r['name']:<45} {r['accuracy']:>6.4f} {r['precision']:>7.4f} "
              f"{r['npv']:>7.4f} {r['folds_below_50']:>4d} "
              f"{r['ev_bull_long']:>+8.3f}% {r['ev_bear_long']:>+8.3f}% "
              f"{r['ev_bear_short']:>+8.3f}%", flush=True)

    print(f"\n  Fold-by-fold:")
    print(f"  {'Fold':<6}", end="", flush=True)
    for r in results:
        print(f" {r['name'][:20]:>20}", end="", flush=True)
    print()
    for fold_i in range(10):
        print(f"  {fold_i+1:<6}", end="", flush=True)
        for r in results:
            if fold_i < len(r["fold_accs"]):
                print(f" {r['fold_accs'][fold_i]:>19.1%}", end="", flush=True)
            else:
                print(f" {'N/A':>20}", end="", flush=True)
        print()

    print(f"\n  Saved to: {EXPERIMENTS_DIR / 'v6_tpsl_results.json'}", flush=True)


if __name__ == "__main__":
    main()
