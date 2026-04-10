"""FROZEN: v2_scaled pipeline — z-score + tanh squashing, VP structure removed.

DO NOT MODIFY THIS FILE. Frozen snapshot of the pipeline overhaul.

── Provenance ──────────────────────────────────────────────────────────────
Created during DualBranch v3 iteration. Changes from v1_raw:
    1. Removed 8 VP structure features (spatial attention learns shape directly)
    2. Added walk-forward-safe expanding z-score for volume_ratio, bar_range, bar_body
    3. Added soft tanh squashing to bound outliers without losing magnitude
    4. log_return and OHLC ratios remain RAW (candle branch depends on them
       being actual multiplicative ratios for daily reconstruction)

── Output contract ─────────────────────────────────────────────────────────
Returns DataFrame with columns:
    ["date", "close"] + FEATURE_COLS
where FEATURE_COLS = 50 VP bins + 10 derived = 60 total. No VP structure.

Feature list (in order):
    0-49:   vp_rel_00 .. vp_rel_49         (VP bins, sum=1 normalized)
    50:     log_return                      (RAW — not scaled)
    51:     bar_range                       (z-score + tanh squash)
    52:     bar_body                        (z-score + tanh squash)
    53:     volume_ratio                    (log + z-score + tanh squash)
    54:     upper_wick                      (raw, 0-1)
    55:     lower_wick                      (raw, 0-1)
    56:     body_dir                        (raw, -1/0/+1)
    57:     ohlc_open_ratio                 (RAW — open/close, near 1.0)
    58:     ohlc_high_ratio                 (RAW — high/close, near 1.0)
    59:     ohlc_low_ratio                  (RAW — low/close, near 1.0)
"""

import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    VOLUME_COL,
    VOLUME_ROLL_WINDOW_BARS,
    VP_COL_NAMES,
    output_csv_name,
)

# Frozen feature column specification
DERIVED_FEATURE_COLS_V2 = [
    "log_return", "bar_range", "bar_body", "volume_ratio",
    "upper_wick", "lower_wick", "body_dir",
    "ohlc_open_ratio", "ohlc_high_ratio", "ohlc_low_ratio",
]

FEATURE_COLS_V2 = VP_COL_NAMES + DERIVED_FEATURE_COLS_V2


def _load_raw(csv_path=None):
    if csv_path is None:
        csv_path = PROJECT_ROOT / output_csv_name()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if "Ans" in df.columns:
        df = df.drop(columns=["Ans"])
    return df


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["bar_range"] = (df["high"] - df["low"]) / df["close"]
    df["bar_body"] = (df["close"] - df["open"]) / df["open"]

    rolling_mean = df[VOLUME_COL].rolling(
        window=VOLUME_ROLL_WINDOW_BARS, min_periods=VOLUME_ROLL_WINDOW_BARS
    ).mean()
    df["volume_ratio"] = df[VOLUME_COL] / rolling_mean

    bar_height = (df["high"] - df["low"]).clip(lower=1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / bar_height
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_height
    df["body_dir"] = np.sign(df["close"] - df["open"])

    # OHLC ratios kept RAW (candle branch depends on them being near 1.0)
    df["ohlc_open_ratio"] = df["open"] / df["close"]
    df["ohlc_high_ratio"] = df["high"] / df["close"]
    df["ohlc_low_ratio"] = df["low"] / df["close"]

    # Walk-forward-safe z-score with tanh squashing
    # NOTE: log_return is NOT scaled (candle branch uses cumsum + exp for reconstruction)
    min_periods = VOLUME_ROLL_WINDOW_BARS

    for col in ["volume_ratio", "bar_range", "bar_body"]:
        if col == "volume_ratio":
            base = np.log(df[col] + 1.0)
        else:
            base = df[col]
        exp_mean = base.expanding(min_periods=min_periods).mean()
        exp_std = base.expanding(min_periods=min_periods).std()
        z = (base - exp_mean) / exp_std.clip(lower=1e-8)
        df[col] = 5.0 * np.tanh(z / 5.0)

    return df


def build_feature_matrix_v2(csv_path=None) -> pd.DataFrame:
    """Build feature matrix using frozen v2_scaled pipeline."""
    df = _load_raw(csv_path)
    df = _compute_derived(df)
    df = df.dropna(subset=FEATURE_COLS_V2).reset_index(drop=True)
    return df[["date", "close"] + FEATURE_COLS_V2]


def feature_index_v2(col_name: str) -> int:
    """Get index of a column within the feature vector (for use with frozen architectures)."""
    return FEATURE_COLS_V2.index(col_name)
