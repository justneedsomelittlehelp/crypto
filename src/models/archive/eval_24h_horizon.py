"""Test 24h fixed-horizon labels: predict if price is higher or lower in 24 hours.

Simpler label = more training data + fixed trade duration + 1 trade per day.
Risk management (SL) is external, not baked into labels.

Tests v6 enriched and v2 temporal on 24h horizon.
No regime detection needed (symmetric labels, no TP/SL flip).

Usage:
    python3 -m src.models.eval_24h_horizon
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, EXPERIMENTS_DIR
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


def make_v6():
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


def make_v2():
    return TemporalTransformerV2(n_other_features=18)


CONFIGS = [
    {"name": "v6 enriched + 24h horizon", "model_fn": make_v6},
    {"name": "v2 temporal + 24h horizon", "model_fn": make_v2},
]


def run_walkforward(df, config):
    import src.config as cfg
    import src.features.dataset as ds_mod

    # Override to fixed-horizon mode
    cfg.LABEL_MODE = "fixed_horizon"
    cfg.LABEL_HORIZON_BARS = 24  # 24 hours
    cfg.LABEL_REGIME_ADAPTIVE = False
    ds_mod.LABEL_MODE = "fixed_horizon"
    ds_mod.LABEL_HORIZON_BARS = 24
    ds_mod.LABEL_REGIME_ADAPTIVE = False

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

        train_ds = TimeSeriesDataset(train_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)
        val_ds = TimeSeriesDataset(val_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)
        test_ds = TimeSeriesDataset(test_df, lookback=LOOKBACK_BARS_MODEL, feature_cols=FEATURE_COLS_V1)

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

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
        fold_p, fold_l, fold_logits = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                preds = (logits > 0).float().cpu()
                fold_p.extend(preds.tolist())
                fold_l.extend(y.tolist())
                fold_logits.extend(logits.cpu().tolist())

        fold_acc = np.mean(np.array(fold_p) == np.array(fold_l))
        fold_accs.append(fold_acc)
        all_preds.extend(fold_p)
        all_labels.extend(fold_l)
        all_logits.extend(fold_logits)
        print(f"  Fold {i+1}: acc={fold_acc:.4f} ({len(fold_p)} samples)", flush=True)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    logits = np.array(all_logits)

    # Overall stats
    acc = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    base_rate = labels.mean()
    pred1_rate = (preds == 1).mean()

    # Confidence analysis: accuracy at different thresholds
    confidence_thresholds = {}
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        logit_thresh = np.log(thresh / (1 - thresh))  # convert prob to logit
        high_conf_up = logits > logit_thresh
        high_conf_down = logits < -logit_thresh
        if high_conf_up.sum() > 0:
            up_acc = (labels[high_conf_up] == 1).mean()
        else:
            up_acc = 0
        if high_conf_down.sum() > 0:
            down_acc = (labels[high_conf_down] == 0).mean()
        else:
            down_acc = 0
        confidence_thresholds[f"{thresh:.2f}"] = {
            "n_up": int(high_conf_up.sum()),
            "up_acc": float(up_acc),
            "n_down": int(high_conf_down.sum()),
            "down_acc": float(down_acc),
        }

    return {
        "name": config["name"],
        "accuracy": float(acc),
        "precision": float(precision),
        "npv": float(npv),
        "recall": float(recall),
        "base_rate": float(base_rate),
        "pred1_rate": float(pred1_rate),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_acc_std": float(np.std(fold_accs)),
        "folds_below_50": int(sum(1 for a in fold_accs if a < 0.5)),
        "total_samples": len(preds),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "confidence_thresholds": confidence_thresholds,
    }


def main():
    print("Loading data (v1_raw pipeline)...", flush=True)
    df = build_feature_matrix_v1()
    print(f"Data: {len(df)} rows, {len(FEATURE_COLS_V1)} features\n", flush=True)

    results = []
    start_time = time.time()

    for i, config in enumerate(CONFIGS):
        params = sum(p.numel() for p in config["model_fn"]().parameters())
        print(f"\n{'=' * 80}", flush=True)
        print(f"  CONFIG {i+1}/{len(CONFIGS)}: {config['name']} ({params:,} params)", flush=True)
        print(f"  Labels: 24h fixed horizon (close[t+24] > close[t])", flush=True)
        print(f"  No regime detection, no TP/SL", flush=True)
        print(f"{'=' * 80}\n", flush=True)

        config_start = time.time()
        result = run_walkforward(df, config)
        config_time = time.time() - config_start
        result["runtime_sec"] = config_time
        result["params"] = params
        results.append(result)

        print(f"\n  RESULT: {config['name']}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f}", flush=True)
        print(f"  Precision (UP): {result['precision']:.4f}, NPV (DOWN): {result['npv']:.4f}", flush=True)
        print(f"  Base rate (% UP): {result['base_rate']:.3f}", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)
        print(f"  Runtime: {config_time/60:.1f} min", flush=True)

        print(f"\n  Confidence threshold analysis:", flush=True)
        print(f"  {'Threshold':<12} {'N_up':>8} {'UP_acc':>8} {'N_down':>8} {'DOWN_acc':>8}", flush=True)
        for thresh, stats in result["confidence_thresholds"].items():
            print(f"  {thresh:<12} {stats['n_up']:>8} {stats['up_acc']:>7.1%} "
                  f"{stats['n_down']:>8} {stats['down_acc']:>7.1%}", flush=True)

        # Save intermediate
        out = EXPERIMENTS_DIR / "horizon_24h_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    total_time = time.time() - start_time
    print(f"\n\n{'=' * 80}", flush=True)
    print(f"  SUMMARY ({total_time/60:.1f} min total)", flush=True)
    print(f"{'=' * 80}", flush=True)

    for r in results:
        print(f"\n  {r['name']}:", flush=True)
        print(f"    Acc={r['accuracy']:.4f}, Prec(UP)={r['precision']:.4f}, "
              f"NPV(DOWN)={r['npv']:.4f}, Folds<50%={r['folds_below_50']}", flush=True)
        print(f"    Folds: {' '.join(f'{a:.1%}' for a in r['fold_accs'])}", flush=True)

    if len(results) == 2:
        r6, r2 = results[0], results[1]
        print(f"\n  v6 vs v2 delta: {r6['accuracy'] - r2['accuracy']:+.4f} accuracy", flush=True)

    print(f"\n  Saved to: {EXPERIMENTS_DIR / 'horizon_24h_results.json'}", flush=True)


if __name__ == "__main__":
    main()
