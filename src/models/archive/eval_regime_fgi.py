"""Compare regime detection methods: SMA vs FGI vs off (fixed).

All use 7.5/3 TP/SL. Same Transformer, same walk-forward.

Model A: SMA(90d) regime detection (current default)
Model B: Fear & Greed Index regime detection (FGI >= 50 = bull)
Model C: No regime flip (fixed 7.5/3 everywhere)

Usage:
    python3 -m src.models.eval_regime_fgi
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

TP = 0.075
SL = 0.03

CONFIGS = [
    {"name": "FGI(50)", "adaptive": True, "regime_mode": "fgi"},
    {"name": "SMA(90d)", "adaptive": True, "regime_mode": "sma"},
]


def get_regime_per_sample(test_ds, test_df, adaptive, regime_mode):
    """Get is_bull flag for each valid sample in the test dataset.

    Returns array aligned with test_ds iteration order: True=bull, False=bear, NaN=unknown.
    """
    n = len(test_df)
    if not adaptive:
        # Fixed mode: treat everything as bull (no flip)
        return np.ones(len(test_ds), dtype=np.float32)

    close = test_df["close"].values
    dates = test_df["date"].values if "date" in test_df.columns else None

    # Rebuild the same regime signal used during labeling
    is_bull_full = test_ds._build_regime_signal(close, dates)
    if is_bull_full is None:
        return np.ones(len(test_ds), dtype=np.float32)

    # Map valid_indices to regime: each sample's label_idx = valid_idx + lookback - 1
    regime_per_sample = np.full(len(test_ds), np.nan, dtype=np.float32)
    for si, vi in enumerate(test_ds.valid_indices):
        label_idx = vi + test_ds.lookback - 1
        if label_idx < len(is_bull_full):
            regime_per_sample[si] = is_bull_full[label_idx]
    return regime_per_sample


def run_walkforward(df, name, adaptive, regime_mode):
    import src.config as cfg
    import src.features.dataset as ds_mod

    cfg.LABEL_TP_PCT = TP
    cfg.LABEL_SL_PCT = SL
    cfg.LABEL_REGIME_ADAPTIVE = adaptive
    cfg.LABEL_REGIME_MODE = regime_mode
    ds_mod.LABEL_TP_PCT = TP
    ds_mod.LABEL_SL_PCT = SL
    ds_mod.LABEL_REGIME_ADAPTIVE = adaptive
    ds_mod.LABEL_REGIME_MODE = regime_mode

    all_preds = []
    all_labels = []
    all_regimes = []  # 1.0=bull, 0.0=bear per sample
    fold_accs = []
    fold_samples = []

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

        # Get regime (bull/bear) for each test sample
        regime = get_regime_per_sample(test_ds, test_df, adaptive, regime_mode)

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
        fold_samples.append(len(fold_p))
        all_preds.extend(fold_p)
        all_labels.extend(fold_l)
        all_regimes.extend(regime[:len(fold_p)].tolist())
        print(f"  Fold {i+1}: acc={fold_acc:.4f} ({len(fold_p)} samples)", flush=True)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    regimes = np.array(all_regimes)

    # Overall metrics
    acc = (preds == labels).mean()
    tp_count = ((preds == 1) & (labels == 1)).sum()
    fp_count = ((preds == 1) & (labels == 0)).sum()
    tn_count = ((preds == 0) & (labels == 0)).sum()
    fn_count = ((preds == 0) & (labels == 1)).sum()
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    npv = tn_count / (tn_count + fn_count) if (tn_count + fn_count) > 0 else 0
    base_rate = labels.mean()
    pred_long_rate = (preds == 1).mean()

    # Regime-split metrics for correct EV
    bull_mask = regimes == 1.0
    bear_mask = regimes == 0.0
    n_bull = bull_mask.sum()
    n_bear = bear_mask.sum()

    def safe_prec(p, l):
        tp = ((p == 1) & (l == 1)).sum()
        fp = ((p == 1) & (l == 0)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    prec_bull = safe_prec(preds[bull_mask], labels[bull_mask]) if n_bull > 0 else 0
    prec_bear = safe_prec(preds[bear_mask], labels[bear_mask]) if n_bear > 0 else 0
    acc_bull = (preds[bull_mask] == labels[bull_mask]).mean() if n_bull > 0 else 0
    acc_bear = (preds[bear_mask] == labels[bear_mask]).mean() if n_bear > 0 else 0

    # Corrected EV per long trade, split by regime
    # Bull: long wins TP%, loses SL%
    # Bear (adaptive): long wins SL% (flipped TP), loses TP% (flipped SL)
    # Bear (fixed): long wins TP%, loses SL% (no flip)
    ev_bull = float(prec_bull) * TP * 100 - (1 - float(prec_bull)) * SL * 100
    if adaptive:
        ev_bear = float(prec_bear) * SL * 100 - (1 - float(prec_bear)) * TP * 100
    else:
        ev_bear = float(prec_bear) * TP * 100 - (1 - float(prec_bear)) * SL * 100

    # Weighted combined EV
    bull_frac = n_bull / (n_bull + n_bear) if (n_bull + n_bear) > 0 else 1.0
    bear_frac = 1 - bull_frac
    ev_combined = bull_frac * ev_bull + bear_frac * ev_bear

    return {
        "name": name,
        "regime_mode": regime_mode,
        "adaptive": adaptive,
        "accuracy": float(acc),
        "precision": float(precision),
        "npv": float(npv),
        "recall": float(recall),
        "f1": float(f1),
        "base_rate": float(base_rate),
        "pred_long_rate": float(pred_long_rate),
        "total_samples": len(preds),
        "confusion": {
            "tp": int(tp_count), "fp": int(fp_count),
            "tn": int(tn_count), "fn": int(fn_count),
        },
        "fold_accs": [float(a) for a in fold_accs],
        "fold_samples": fold_samples,
        "folds_below_50": sum(1 for a in fold_accs if a < 0.5),
        # Regime-split stats
        "n_bull_samples": int(n_bull),
        "n_bear_samples": int(n_bear),
        "bull_frac": float(bull_frac),
        "acc_bull": float(acc_bull),
        "acc_bear": float(acc_bear),
        "prec_bull": float(prec_bull),
        "prec_bear": float(prec_bear),
        "ev_bull": float(ev_bull),
        "ev_bear": float(ev_bear),
        "ev_combined": float(ev_combined),
    }


def main():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []

    for config in CONFIGS:
        name = config["name"]
        print(f"\n{'=' * 70}", flush=True)
        print(f"  {name}  (TP={TP*100:.1f}% SL={SL*100:.1f}%)", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        result = run_walkforward(df, name, config["adaptive"], config["regime_mode"])
        results.append(result)

        print(f"\n  RESULT: {name}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, "
              f"NPV: {result['npv']:.4f}, F1: {result['f1']:.4f}", flush=True)
        print(f"  Base rate: {result['base_rate']:.3f}, Samples: {result['total_samples']}", flush=True)
        print(f"  Bull: acc={result['acc_bull']:.3f} prec={result['prec_bull']:.3f} "
              f"EV={result['ev_bull']:+.3f}% ({result['n_bull_samples']} samples)", flush=True)
        print(f"  Bear: acc={result['acc_bear']:.3f} prec={result['prec_bear']:.3f} "
              f"EV={result['ev_bear']:+.3f}% ({result['n_bear_samples']} samples)", flush=True)
        print(f"  Combined EV: {result['ev_combined']:+.3f}%", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)

    # Comparison table
    print(f"\n\n{'=' * 80}", flush=True)
    print(f"  COMPARISON: FGI vs SMA", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"{'Metric':<25} {'FGI(50)':>12} {'SMA(90d)':>12} {'Diff':>12}", flush=True)
    print("-" * 65, flush=True)

    r_f, r_s = results[0], results[1]
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("f1", "F1"),
        ("base_rate", "Base rate"),
        ("total_samples", "Samples"),
        ("folds_below_50", "Folds <50%"),
        ("acc_bull", "Acc (bull)"),
        ("acc_bear", "Acc (bear)"),
        ("prec_bull", "Prec (bull)"),
        ("prec_bear", "Prec (bear)"),
        ("ev_bull", "EV bull"),
        ("ev_bear", "EV bear"),
        ("ev_combined", "EV combined"),
    ]:
        vf, vs = r_f[key], r_s[key]
        if "ev" in key.lower():
            print(f"{label:<25} {vf:>+11.3f}% {vs:>+11.3f}% {vf-vs:>+11.3f}%", flush=True)
        elif isinstance(vf, float):
            print(f"{label:<25} {vf:>12.4f} {vs:>12.4f} {vf-vs:>+12.4f}", flush=True)
        else:
            print(f"{label:<25} {vf:>12} {vs:>12} {vf-vs:>+12}", flush=True)

    # Fold-by-fold
    print(f"\n{'Fold':<8} {'FGI':>10} {'SMA':>10} {'Diff':>10}", flush=True)
    print("-" * 42, flush=True)
    n_folds = min(len(r_f["fold_accs"]), len(r_s["fold_accs"]))
    for i in range(n_folds):
        f, s = r_f["fold_accs"][i], r_s["fold_accs"][i]
        diff = f - s
        print(f"Fold {i+1:<3} {f:>9.1%} {s:>9.1%} {diff:>+9.1%}", flush=True)

    # Save
    out = EXPERIMENTS_DIR / "regime_fgi_compare_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)


if __name__ == "__main__":
    main()
