"""Head-to-head: v6 enriched vs v2 temporal, same TP/SL as reproduction.

Both use:
- v1_raw pipeline (68 features, VP structure present)
- TP=7.5%, SL=3%, FGI adaptive (same as v1 reproduction)
- Batch from config, seed=42, constant LR 5e-4, no grad clipping

This isolates: does enriching day tokens with candle + VP structure help?

Usage:
    python3 -m src.models.eval_v6_vs_v2
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

TP = 0.075
SL = 0.03

FGI_DF = pd.read_csv(LABEL_FGI_PATH, parse_dates=["date"])
FGI_LOOKUP = dict(zip(FGI_DF["date"].dt.date, FGI_DF["fgi_value"].astype(float)))


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
    {"name": "v6 enriched temporal", "model_fn": make_v6},
    {"name": "v2 temporal (baseline)", "model_fn": make_v2},
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

    bull = regimes == 1.0
    bear = regimes == 0.0

    def confusion(p, l):
        tp = ((p == 1) & (l == 1)).sum()
        fp = ((p == 1) & (l == 0)).sum()
        tn = ((p == 0) & (l == 0)).sum()
        fn = ((p == 0) & (l == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        acc = (tp + tn) / len(p) if len(p) > 0 else 0
        return float(prec), float(npv), float(acc)

    bull_prec, bull_npv, bull_acc = confusion(preds[bull], labels[bull])
    bear_prec, bear_npv, bear_acc = confusion(preds[bear], labels[bear])
    overall_prec, overall_npv, _ = confusion(preds, labels)

    ev_bull_long = bull_prec * 7.5 - (1 - bull_prec) * 3.0
    ev_bear_long = bear_prec * 3.0 - (1 - bear_prec) * 7.5
    ev_bear_short = bear_npv * 7.5 - (1 - bear_npv) * 3.0

    return {
        "name": config["name"],
        "accuracy": float((preds == labels).mean()),
        "precision": float(overall_prec),
        "npv": float(overall_npv),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_acc_std": float(np.std(fold_accs)),
        "folds_below_50": int(sum(1 for a in fold_accs if a < 0.5)),
        "bull_acc": float(bull_acc),
        "bear_acc": float(bear_acc),
        "bull_precision": bull_prec,
        "bear_precision": bear_prec,
        "bear_npv": bear_npv,
        "ev_bull_long": float(ev_bull_long),
        "ev_bear_long": float(ev_bear_long),
        "ev_bear_short": float(ev_bear_short),
        "n_bull": int(bull.sum()),
        "n_bear": int(bear.sum()),
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
        print(f"  TP=7.5% SL=3% FGI adaptive, v1_raw pipeline", flush=True)
        print(f"{'=' * 80}\n", flush=True)

        config_start = time.time()
        result = run_walkforward(df, config)
        config_time = time.time() - config_start
        result["runtime_sec"] = config_time
        result["params"] = params
        results.append(result)

        print(f"\n  RESULT: {config['name']}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f} (std: {result['fold_acc_std']:.4f})", flush=True)
        print(f"  Precision (UP): {result['precision']:.4f}, NPV (DOWN): {result['npv']:.4f}", flush=True)
        print(f"  Bull acc: {result['bull_acc']:.4f}, Bear acc: {result['bear_acc']:.4f}", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)
        print(f"  EVs: bull_long={result['ev_bull_long']:+.3f}%, "
              f"bear_long={result['ev_bear_long']:+.3f}%, "
              f"bear_short={result['ev_bear_short']:+.3f}%", flush=True)
        print(f"  Runtime: {config_time/60:.1f} min", flush=True)

        # Save intermediate
        out = EXPERIMENTS_DIR / "v6_vs_v2_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    # Comparison
    total_time = time.time() - start_time
    r_v6, r_v2 = results[0], results[1]
    print(f"\n\n{'=' * 80}", flush=True)
    print(f"  COMPARISON ({total_time/60:.1f} min total)", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"{'Metric':<25} {'v6 enriched':>15} {'v2 baseline':>15} {'Diff':>12}", flush=True)
    print("-" * 70, flush=True)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision (UP)"),
        ("npv", "NPV (DOWN)"),
        ("folds_below_50", "Folds <50%"),
        ("ev_bull_long", "EV bull long"),
        ("ev_bear_long", "EV bear long"),
        ("ev_bear_short", "EV bear short"),
    ]:
        v6, v2 = r_v6[key], r_v2[key]
        if "ev" in key:
            print(f"{label:<25} {v6:>+14.3f}% {v2:>+14.3f}% {v6-v2:>+11.3f}%", flush=True)
        elif isinstance(v6, float):
            print(f"{label:<25} {v6:>15.4f} {v2:>15.4f} {v6-v2:>+12.4f}", flush=True)
        else:
            print(f"{label:<25} {v6:>15} {v2:>15} {v6-v2:>+12}", flush=True)

    print(f"\n{'Fold':<8} {'v6':>12} {'v2':>12} {'Diff':>12}", flush=True)
    print("-" * 46, flush=True)
    for fold_i in range(10):
        v6_a = r_v6["fold_accs"][fold_i]
        v2_a = r_v2["fold_accs"][fold_i]
        print(f"Fold {fold_i+1:<3} {v6_a:>11.1%} {v2_a:>11.1%} {v6_a-v2_a:>+11.1%}", flush=True)

    print(f"\n  Saved to: {EXPERIMENTS_DIR / 'v6_vs_v2_results.json'}", flush=True)


if __name__ == "__main__":
    main()
