"""Rule-based strategy that encodes the user's decision flow directly.

No training — just applies the VP + candle + volume logic:
1. Near ceiling → predict DOWN
2. Near floor → predict UP
3. Mid-range → read candle pattern, weighted by volume
"""

import numpy as np
import pandas as pd

from src.config import (
    LABEL_HORIZON_BARS,
    LABEL_TP_PCT,
    LABEL_SL_PCT,
    LABEL_MAX_BARS,
    FEATURE_COLS,
    TRAIN_END,
    VAL_END,
)
from src.features.pipeline import build_feature_matrix
from src.features.dataset import TimeSeriesDataset


def predict(row: pd.Series, near_threshold: float = 0.3, volume_threshold: float = 1.0) -> int:
    """Apply rule-based strategy to a single row.

    Args:
        near_threshold: ceiling/floor dist below this = "near" the level
        volume_threshold: volume_ratio above this = trustworthy candle signal

    Returns:
        1 = predict UP, 0 = predict DOWN
    """
    ceil_dist = row["vp_ceiling_dist"]
    floor_dist = row["vp_floor_dist"]
    mid_range = row["vp_mid_range"]
    upper_wick = row["upper_wick"]
    lower_wick = row["lower_wick"]
    body_dir = row["body_dir"]
    vol_ratio = row["volume_ratio"]

    # Rule 1: Near ceiling → DOWN
    if ceil_dist < near_threshold and ceil_dist < floor_dist:
        return 0

    # Rule 2: Near floor → UP
    if floor_dist < near_threshold and floor_dist < ceil_dist:
        return 1

    # Rule 3: Mid-range → read candle pattern
    # Green hammer (bullish): close > open, long lower wick
    is_hammer = body_dir > 0 and lower_wick > 0.4
    # Red inverted hammer (bearish): open > close, long upper wick
    is_inv_hammer = body_dir < 0 and upper_wick > 0.4

    if is_hammer and vol_ratio > volume_threshold:
        return 1
    if is_inv_hammer and vol_ratio > volume_threshold:
        return 0

    # Weaker candle signals (lower volume)
    if is_hammer:
        return 1
    if is_inv_hammer:
        return 0

    # No clear signal — default to no trade (predict UP as neutral)
    # In live trading this would be "do nothing"
    return 1


def backtest(near_threshold: float = 0.3, volume_threshold: float = 1.0):
    """Run the rule-based strategy on train/val/test splits."""
    df = build_feature_matrix()

    # Compute first-hit labels
    close = df["close"].values
    num_peaks = df["vp_num_peaks"].values if "vp_num_peaks" in df.columns else None
    labels = TimeSeriesDataset._first_hit_labels(close, num_peaks)
    df["label"] = labels

    # Drop rows with NaN labels (neither TP nor SL hit)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Split
    train = df[df["date"] < TRAIN_END]
    val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)]
    test = df[df["date"] >= VAL_END]

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        preds = split.apply(lambda r: predict(r, near_threshold, volume_threshold), axis=1).values
        labels_split = split["label"].values

        tp = int(((preds == 1) & (labels_split == 1)).sum())
        fp = int(((preds == 1) & (labels_split == 0)).sum())
        tn = int(((preds == 0) & (labels_split == 0)).sum())
        fn = int(((preds == 0) & (labels_split == 1)).sum())
        total = len(labels_split)
        acc = (tp + tn) / total if total else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        # Count signal types
        n_near_ceil = int((split["vp_ceiling_dist"] < near_threshold).sum())
        n_near_floor = int((split["vp_floor_dist"] < near_threshold).sum())
        n_candle = total - n_near_ceil - n_near_floor

        print(f"\n{'=' * 50}")
        print(f"  {name} Set — Rule-Based Strategy")
        print(f"{'=' * 50}")
        print(f"  Samples:     {total}")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  F1:          {f1:.4f}")
        print(f"  Near ceiling: {n_near_ceil} ({n_near_ceil/total:.1%})")
        print(f"  Near floor:   {n_near_floor} ({n_near_floor/total:.1%})")
        print(f"  Mid-range:    {n_candle} ({n_candle/total:.1%})")
        print(f"\n  Confusion Matrix:")
        print(f"              Pred UP  Pred DOWN")
        print(f"  Actual UP    {tp:5d}    {fn:5d}")
        print(f"  Actual DOWN  {fp:5d}    {tn:5d}")
        print(f"{'=' * 50}")
