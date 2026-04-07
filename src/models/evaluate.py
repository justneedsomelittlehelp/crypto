import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, dataloader: DataLoader) -> dict:
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = (logits > 0).float()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    total = len(all_labels)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_samples": total,
    }


def print_results(metrics: dict, split_name: str = "Test"):
    cm = metrics["confusion_matrix"]
    print(f"\n{'=' * 40}")
    print(f"  {split_name} Set Results")
    print(f"{'=' * 40}")
    print(f"  Samples:   {metrics['total_samples']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred UP  Pred DOWN")
    print(f"  Actual UP    {cm['tp']:5d}    {cm['fn']:5d}")
    print(f"  Actual DOWN  {cm['fp']:5d}    {cm['tn']:5d}")
    print(f"{'=' * 40}")
