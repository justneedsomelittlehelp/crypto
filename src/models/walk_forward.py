"""Walk-forward retraining: retrain on expanding window, predict on next 3 months.

Usage:
    python -c "from src.models.walk_forward import run; run('cnn')"
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, LOOKBACK_BARS_MODEL, FEATURE_COLS
from src.features.pipeline import build_feature_matrix
from src.features.dataset import TimeSeriesDataset
from src.models.architecture import CNNClassifier, LSTMClassifier, RNNClassifier, TransformerClassifier
from src.models.trainer import train_model

MODELS = {
    "rnn": RNNClassifier,
    "lstm": LSTMClassifier,
    "cnn": CNNClassifier,
    "transformer": TransformerClassifier,
}

# Walk-forward windows: train up to date, val on next 3 months, test on 3 months after
# Start testing from 2020 onward (enough training data)
FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
]


def run(model_name: str = "cnn"):
    print("Loading data...")
    df = build_feature_matrix()

    all_preds = []
    all_labels = []
    fold_results = []

    for i in range(len(FOLD_BOUNDARIES) - 2):
        train_end = FOLD_BOUNDARIES[i]
        val_end = FOLD_BOUNDARIES[i + 1]
        test_end = FOLD_BOUNDARIES[i + 2]

        train_df = df[df["date"] < train_end].reset_index(drop=True)
        val_df = df[(df["date"] >= train_end) & (df["date"] < val_end)].reset_index(drop=True)
        test_df = df[(df["date"] >= val_end) & (df["date"] < test_end)].reset_index(drop=True)

        if len(train_df) < 1000 or len(val_df) < 100 or len(test_df) < 100:
            print(f"\nFold {i+1}: skipping (not enough data)")
            continue

        train_ds = TimeSeriesDataset(train_df, lookback=LOOKBACK_BARS_MODEL)
        val_ds = TimeSeriesDataset(val_df, lookback=LOOKBACK_BARS_MODEL)
        test_ds = TimeSeriesDataset(test_df, lookback=LOOKBACK_BARS_MODEL)

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            print(f"\nFold {i+1}: skipping (too few valid samples)")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\n{'=' * 60}")
        print(f"  Fold {i+1}: train <{train_end}, val {train_end}-{val_end}, test {val_end}-{test_end}")
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        print(f"{'=' * 60}")

        model = MODELS[model_name]()
        result = train_model(model, train_loader, val_loader)

        # Evaluate on test fold
        model.eval()
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                preds = (logits > 0).float()
                fold_preds.extend(preds.tolist())
                fold_labels.extend(y.tolist())

        fold_preds = np.array(fold_preds)
        fold_labels = np.array(fold_labels)
        acc = (fold_preds == fold_labels).mean()

        tp = int(((fold_preds == 1) & (fold_labels == 1)).sum())
        fp = int(((fold_preds == 1) & (fold_labels == 0)).sum())
        tn = int(((fold_preds == 0) & (fold_labels == 0)).sum())
        fn = int(((fold_preds == 0) & (fold_labels == 1)).sum())

        print(f"\n  Test accuracy: {acc:.4f}")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

        fold_results.append({
            "fold": i + 1,
            "train_end": train_end,
            "test_period": f"{val_end} - {test_end}",
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "accuracy": acc,
            "best_val_loss": result["best_val_loss"],
        })

        all_preds.extend(fold_preds.tolist())
        all_labels.extend(fold_labels.tolist())

    # Overall results
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = (all_preds == all_labels).mean()

    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())

    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD OVERALL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total test samples: {len(all_preds)}")
    print(f"  Overall accuracy:   {overall_acc:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':<6} {'Test Period':<28} {'Train':<8} {'Test':<8} {'Acc':<8} {'Val Loss':<10}")
    for r in fold_results:
        print(f"  {r['fold']:<6} {r['test_period']:<28} {r['train_samples']:<8} {r['test_samples']:<8} {r['accuracy']:<8.4f} {r['best_val_loss']:<10.4f}")
    print(f"{'=' * 60}")
