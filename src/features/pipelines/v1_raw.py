"""FROZEN: v1_raw pipeline — original features, no scaling, VP structure present.

DO NOT MODIFY THIS FILE. Frozen snapshot of the pipeline that produced our
best validated accuracy result (Temporal Transformer, 61.9% on 1h data).

── Provenance ──────────────────────────────────────────────────────────────
This is the pre-overhaul pipeline. Returns raw ratios and counts without
any z-scoring or tanh squashing. Includes the 8 VP structure features that
were removed in the v2_scaled overhaul.

── Output contract ─────────────────────────────────────────────────────────
Returns DataFrame with columns:
    ["date", "close"] + FEATURE_COLS
where FEATURE_COLS = 50 VP bins + 10 derived + 8 VP structure = 68 total.

Feature list (in order):
    0-49:   vp_rel_00 .. vp_rel_49         (VP bins, sum=1 normalized per row)
    50:     log_return                      (raw log return)
    51:     bar_range                       (raw range / close)
    52:     bar_body                        (raw body / open)
    53:     volume_ratio                    (raw volume / 30d rolling mean)
    54:     upper_wick                      (proportion, 0-1)
    55:     lower_wick                      (proportion, 0-1)
    56:     body_dir                        (sign of close - open)
    57:     ohlc_open_ratio                 (open / close)
    58:     ohlc_high_ratio                 (high / close)
    59:     ohlc_low_ratio                  (low / close)
    60:     vp_ceiling_dist                 (distance to nearest peak above)
    61:     vp_floor_dist                   (distance to nearest peak below)
    62:     vp_num_peaks                    (count 0-5)
    63:     vp_ceiling_strength             (normalized peak value)
    64:     vp_floor_strength               (normalized peak value)
    65:     vp_ceiling_consistency          (fraction in shifted windows)
    66:     vp_floor_consistency            (fraction in shifted windows)
    67:     vp_mid_range                    (ratio of nearest peak distances)
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.config import (
    BARS_PER_DAY,
    HORIZON_24H_BARS,
    LOOKBACK_BARS_MODEL,
    PROJECT_ROOT,
    REL_BIN_COUNT,
    VOLUME_COL,
    VOLUME_ROLL_WINDOW_BARS,
    VP_COL_NAMES,
    output_csv_name,
)

# Frozen feature column specification
DERIVED_FEATURE_COLS_V1 = [
    "log_return", "bar_range", "bar_body", "volume_ratio",
    "upper_wick", "lower_wick", "body_dir",
    "ohlc_open_ratio", "ohlc_high_ratio", "ohlc_low_ratio",
]

VP_STRUCTURE_COLS_V1 = [
    "vp_ceiling_dist",
    "vp_floor_dist",
    "vp_num_peaks",
    "vp_ceiling_strength",
    "vp_floor_strength",
    "vp_ceiling_consistency",
    "vp_floor_consistency",
    "vp_mid_range",
]

FEATURE_COLS_V1 = VP_COL_NAMES + DERIVED_FEATURE_COLS_V1 + VP_STRUCTURE_COLS_V1


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

    df["ohlc_open_ratio"] = df["open"] / df["close"]
    df["ohlc_high_ratio"] = df["high"] / df["close"]
    df["ohlc_low_ratio"] = df["low"] / df["close"]

    return df


def _smooth_and_find_peaks(agg_vp, sigma=0.8, prominence=0.05):
    smooth = gaussian_filter1d(agg_vp.astype(np.float64), sigma=sigma)
    vp_max = smooth.max()
    if vp_max > 0:
        smooth /= vp_max
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=3)
    return peaks, smooth


def _compute_vp_structure(df: pd.DataFrame, window=LOOKBACK_BARS_MODEL) -> pd.DataFrame:
    df = df.copy()
    vp_matrix = df[VP_COL_NAMES].values
    mid_bin = REL_BIN_COUNT // 2
    n = len(df)

    cumsum = np.zeros((n + 1, REL_BIN_COUNT), dtype=np.float64)
    np.cumsum(vp_matrix, axis=0, out=cumsum[1:])

    def get_rolling_vp(idx, win):
        start = idx - win
        if start < 0:
            return None
        return cumsum[idx] - cumsum[start]

    shift_days = [3, 6, 9]
    shift_bars = [BARS_PER_DAY * d for d in shift_days]
    max_shift = max(shift_bars)
    peak_cache = {}

    def get_peaks_for_window_end(end_idx):
        if end_idx in peak_cache:
            return peak_cache[end_idx]
        agg_vp = get_rolling_vp(end_idx, window)
        if agg_vp is None:
            peak_cache[end_idx] = (np.array([]), None, None, None)
            return peak_cache[end_idx]
        peaks, norm = _smooth_and_find_peaks(agg_vp)
        above = peaks[peaks > mid_bin]
        below = peaks[peaks < mid_bin]
        ceil_bin = above[0] if len(above) > 0 else None
        floor_bin = below[-1] if len(below) > 0 else None
        peak_cache[end_idx] = (peaks, norm, ceil_bin, floor_bin)
        return peak_cache[end_idx]

    ceiling_dist = np.full(n, np.nan, dtype=np.float32)
    floor_dist = np.full(n, np.nan, dtype=np.float32)
    num_peaks_arr = np.full(n, np.nan, dtype=np.float32)
    ceiling_strength = np.full(n, np.nan, dtype=np.float32)
    floor_strength = np.full(n, np.nan, dtype=np.float32)
    ceiling_consistency = np.full(n, np.nan, dtype=np.float32)
    floor_consistency = np.full(n, np.nan, dtype=np.float32)

    start_idx = window + max_shift
    for i in range(window, n):
        peaks, norm, ceil_bin, floor_bin = get_peaks_for_window_end(i)
        num_peaks_arr[i] = len(peaks)

        if len(peaks) == 0:
            ceiling_dist[i] = 1.0
            floor_dist[i] = 1.0
            ceiling_strength[i] = 0.0
            floor_strength[i] = 0.0
        else:
            if ceil_bin is not None:
                ceiling_dist[i] = (ceil_bin - mid_bin) / mid_bin
                ceiling_strength[i] = norm[ceil_bin]
            else:
                ceiling_dist[i] = 1.0
                ceiling_strength[i] = 0.0
            if floor_bin is not None:
                floor_dist[i] = (mid_bin - floor_bin) / mid_bin
                floor_strength[i] = norm[floor_bin]
            else:
                floor_dist[i] = 1.0
                floor_strength[i] = 0.0

        if i >= start_idx:
            tolerance = 2
            ceil_matches = 0
            floor_matches = 0
            for shift in shift_bars:
                shifted_end = i - shift
                s_peaks, _, _, _ = get_peaks_for_window_end(shifted_end)
                if ceil_bin is not None and len(s_peaks) > 0:
                    if np.any(np.abs(s_peaks - ceil_bin) <= tolerance):
                        ceil_matches += 1
                if floor_bin is not None and len(s_peaks) > 0:
                    if np.any(np.abs(s_peaks - floor_bin) <= tolerance):
                        floor_matches += 1
            ceiling_consistency[i] = ceil_matches / len(shift_bars)
            floor_consistency[i] = floor_matches / len(shift_bars)

    c = ceiling_dist
    f = floor_dist
    valid = ~(np.isnan(c) | np.isnan(f))
    mid_range = np.full(n, np.nan, dtype=np.float32)
    mx = np.maximum(c[valid], f[valid])
    mn = np.minimum(c[valid], f[valid])
    mx = np.where(mx < 1e-8, 1.0, mx)
    mid_range[valid] = mn / mx

    df["vp_ceiling_dist"] = ceiling_dist
    df["vp_floor_dist"] = floor_dist
    df["vp_num_peaks"] = num_peaks_arr
    df["vp_ceiling_strength"] = ceiling_strength
    df["vp_floor_strength"] = floor_strength
    df["vp_ceiling_consistency"] = ceiling_consistency
    df["vp_floor_consistency"] = floor_consistency
    df["vp_mid_range"] = mid_range

    return df


def build_feature_matrix_v1(csv_path=None) -> pd.DataFrame:
    """Build feature matrix using frozen v1_raw pipeline."""
    df = _load_raw(csv_path)
    df = _compute_derived(df)
    df = _compute_vp_structure(df)
    df = df.dropna(subset=FEATURE_COLS_V1).reset_index(drop=True)
    return df[["date", "close"] + FEATURE_COLS_V1]


def feature_index_v1(col_name: str) -> int:
    """Get index of a column within the feature vector (for use with frozen architectures)."""
    return FEATURE_COLS_V1.index(col_name)
