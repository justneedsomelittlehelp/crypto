"""LR schedule sweep for DualBranch v3.

Tests multiple learning rate configurations to find what reduces
per-fold variance and stabilizes training.

Configs tested:
  1. Constant LR 5e-4 (baseline, current default)
  2. Constant LR 1e-4 (lower)
  3. Constant LR 1e-3 (higher)
  4. Cosine + warmup, base 5e-4 (5 epoch warmup)
  5. Cosine + warmup, base 1e-3 (5 epoch warmup)
  6. Cosine + warmup, base 5e-4, weight_decay 1e-4 (regularized)

Each config runs full 10-fold walk-forward on DualBranch v3 + FGI 7.5/3.

Usage:
    python3 -m src.models.sweep_lr
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
from src.models.architecture import DualBranchTransformerClassifier
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

# Sweep configs
SWEEP_CONFIGS = [
    {"name": "constant_5e-4 (baseline)",      "lr": 5e-4, "schedule": "constant",      "warmup": 0,  "weight_decay": 0.0},
    {"name": "constant_1e-4 (lower)",         "lr": 1e-4, "schedule": "constant",      "warmup": 0,  "weight_decay": 0.0},
    {"name": "constant_1e-3 (higher)",        "lr": 1e-3, "schedule": "constant",      "warmup": 0,  "weight_decay": 0.0},
    {"name": "cosine_warmup_5e-4",            "lr": 5e-4, "schedule": "warmup_cosine", "warmup": 5,  "weight_decay": 0.0},
    {"name": "cosine_warmup_1e-3",            "lr": 1e-3, "schedule": "warmup_cosine", "warmup": 5,  "weight_decay": 0.0},
    {"name": "cosine_warmup_5e-4 + wd1e-4",   "lr": 5e-4, "schedule": "warmup_cosine", "warmup": 5,  "weight_decay": 1e-4},
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

        train_ds = TimeSeriesDataset(train_df, lookback=LOOKBACK_BARS_MODEL)
        val_ds = TimeSeriesDataset(val_df, lookback=LOOKBACK_BARS_MODEL)
        test_ds = TimeSeriesDataset(test_df, lookback=LOOKBACK_BARS_MODEL)

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        regime = get_regime_for_dataset(test_ds, test_df)

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        model = DualBranchTransformerClassifier()
        train_model(
            model, train_loader, val_loader,
            lr=config["lr"],
            lr_schedule=config["schedule"],
            warmup_epochs=config["warmup"],
            weight_decay=config["weight_decay"],
            seed=42,  # fixed seed for fair comparison across configs
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
        "lr": config["lr"],
        "schedule": config["schedule"],
        "warmup": config["warmup"],
        "weight_decay": config["weight_decay"],
        "accuracy": float((preds == labels).mean()),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_acc_std": float(np.std(fold_accs)),
        "fold_acc_min": float(np.min(fold_accs)),
        "fold_acc_max": float(np.max(fold_accs)),
        "folds_below_50": int(sum(1 for a in fold_accs if a < 0.5)),
        "bull_precision": bull_prec,
        "bear_precision": bear_prec,
        "bear_npv": bear_npv,
        "ev_bull_long": float(ev_bull_long),
        "ev_bear_long": float(ev_bear_long),
        "ev_bear_short": float(ev_bear_short),
    }


def main():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []
    start_time = time.time()

    for i, config in enumerate(SWEEP_CONFIGS):
        print(f"\n{'=' * 80}", flush=True)
        print(f"  CONFIG {i+1}/{len(SWEEP_CONFIGS)}: {config['name']}", flush=True)
        print(f"  LR={config['lr']}, schedule={config['schedule']}, warmup={config['warmup']}, wd={config['weight_decay']}", flush=True)
        print(f"{'=' * 80}\n", flush=True)

        config_start = time.time()
        result = run_walkforward(df, config)
        config_time = time.time() - config_start
        result["runtime_sec"] = config_time
        results.append(result)

        print(f"\n  RESULT: {config['name']}", flush=True)
        print(f"  Acc: {result['accuracy']:.4f} (std: {result['fold_acc_std']:.4f}, "
              f"range: {result['fold_acc_min']:.3f}-{result['fold_acc_max']:.3f})", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)
        print(f"  EVs: bull_long={result['ev_bull_long']:+.3f}%, "
              f"bear_long={result['ev_bear_long']:+.3f}%, "
              f"bear_short={result['ev_bear_short']:+.3f}%", flush=True)
        print(f"  Runtime: {config_time/60:.1f} min", flush=True)

        # Save intermediate results after each config
        out = EXPERIMENTS_DIR / "lr_sweep_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {out}", flush=True)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n\n{'=' * 100}", flush=True)
    print(f"  LR SWEEP SUMMARY ({total_time/60:.1f} min total)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"{'Config':<35} {'Acc':>8} {'StdDev':>8} {'<50%':>6} {'EVbullL':>9} {'EVbearL':>9} {'EVbearS':>9}", flush=True)
    print("-" * 100, flush=True)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"{r['name']:<35} {r['accuracy']:>7.4f} {r['fold_acc_std']:>7.4f} "
              f"{r['folds_below_50']:>5d} {r['ev_bull_long']:>+8.3f}% "
              f"{r['ev_bear_long']:>+8.3f}% {r['ev_bear_short']:>+8.3f}%", flush=True)

    print(f"\nResults: {EXPERIMENTS_DIR / 'lr_sweep_results.json'}", flush=True)


if __name__ == "__main__":
    main()
