"""Compare Temporal Transformer (Option B) vs current Transformer.

Both use FGI regime adaptive, 7.5/3 TP/SL.

Usage:
    python3 -m src.models.eval_temporal
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
from src.models.architecture import TransformerClassifier, TemporalTransformerClassifier
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
    {"name": "Temporal (Option B)", "model_cls": TemporalTransformerClassifier},
]


def run_walkforward(df, name, model_cls):
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

        model = model_cls()
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
    tn_count = ((preds == 0) & (labels == 0)).sum()
    fn_count = ((preds == 0) & (labels == 1)).sum()
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "name": name,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_samples": len(preds),
        "confusion": {
            "tp": int(tp_count), "fp": int(fp_count),
            "tn": int(tn_count), "fn": int(fn_count),
        },
        "fold_accs": [float(a) for a in fold_accs],
        "folds_below_50": sum(1 for a in fold_accs if a < 0.5),
    }


def main():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    print(f"Data: {len(df)} rows\n", flush=True)

    results = []

    for config in CONFIGS:
        name = config["name"]
        model_cls = config["model_cls"]
        params = sum(p.numel() for p in model_cls().parameters())

        print(f"\n{'=' * 70}", flush=True)
        print(f"  {name}  ({params:,} params)", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        result = run_walkforward(df, name, model_cls)
        result["params"] = params
        results.append(result)

        print(f"\n  RESULT: {name}", flush=True)
        print(f"  Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, "
              f"F1: {result['f1']:.4f}", flush=True)
        print(f"  Folds <50%: {result['folds_below_50']}/10", flush=True)

    # Comparison
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  COMPARISON: Temporal vs Current", flush=True)
    print(f"{'=' * 70}", flush=True)

    r_t, r_c = results[0], results[1]
    print(f"{'Metric':<25} {'Temporal':>12} {'Current':>12} {'Diff':>12}", flush=True)
    print("-" * 65, flush=True)
    for key, label in [
        ("params", "Params"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("f1", "F1"),
        ("total_samples", "Samples"),
        ("folds_below_50", "Folds <50%"),
    ]:
        vt, vc = r_t[key], r_c[key]
        if isinstance(vt, float):
            print(f"{label:<25} {vt:>12.4f} {vc:>12.4f} {vt-vc:>+12.4f}", flush=True)
        else:
            print(f"{label:<25} {vt:>12,} {vc:>12,} {vt-vc:>+12,}", flush=True)

    print(f"\n{'Fold':<8} {'Temporal':>10} {'Current':>10} {'Diff':>10}", flush=True)
    print("-" * 42, flush=True)
    for i in range(min(len(r_t["fold_accs"]), len(r_c["fold_accs"]))):
        t, c = r_t["fold_accs"][i], r_c["fold_accs"][i]
        print(f"Fold {i+1:<3} {t:>9.1%} {c:>9.1%} {t-c:>+9.1%}", flush=True)

    out = EXPERIMENTS_DIR / "temporal_compare_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)


if __name__ == "__main__":
    main()
