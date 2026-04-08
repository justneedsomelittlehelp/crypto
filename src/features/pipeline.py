import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from src.config import (
    PROJECT_ROOT,
    VP_COL_NAMES,
    VOLUME_COL,
    VOLUME_ROLL_WINDOW_BARS,
    LOOKBACK_BARS_MODEL,
    BARS_PER_DAY,
    REL_BIN_COUNT,
    DERIVED_FEATURE_COLS,
    FEATURE_COLS,
    output_csv_name,
)


def load_raw_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = PROJECT_ROOT / output_csv_name()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if "Ans" in df.columns:
        df = df.drop(columns=["Ans"])
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-return: log(close / prev_close)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Bar range: (high - low) / close
    df["bar_range"] = (df["high"] - df["low"]) / df["close"]

    # Bar body: (close - open) / open
    df["bar_body"] = (df["close"] - df["open"]) / df["open"]

    # Volume ratio: volume / backward-looking rolling mean
    rolling_mean = df[VOLUME_COL].rolling(
        window=VOLUME_ROLL_WINDOW_BARS, min_periods=VOLUME_ROLL_WINDOW_BARS
    ).mean()
    df["volume_ratio"] = df[VOLUME_COL] / rolling_mean

    # Candle shape features
    bar_height = (df["high"] - df["low"]).clip(lower=1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / bar_height
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_height
    df["body_dir"] = np.sign(df["close"] - df["open"])

    return df


def compute_vp_structure_features(
    df: pd.DataFrame,
    window: int = LOOKBACK_BARS_MODEL,
) -> pd.DataFrame:
    """Aggregate VP bins over a rolling window and extract structural features.

    For each bar, sums the 50 VP bins over the preceding `window` bars to create
    one aggregated volume profile, then finds local maxima (support/resistance nodes).

    Features produced:
    - vp_ceiling_dist: distance (in bins) from current price (bin 25) to nearest peak above
    - vp_floor_dist: distance (in bins) from current price (bin 25) to nearest peak below
    - vp_num_peaks: number of prominent peaks in the aggregated VP
    - vp_ceiling_strength: volume at the ceiling peak (normalized)
    - vp_floor_strength: volume at the floor peak (normalized)
    """
    df = df.copy()
    vp_matrix = df[VP_COL_NAMES].values  # (n_rows, 50)
    mid_bin = REL_BIN_COUNT // 2  # bin 25 = current price

    n = len(df)
    ceiling_dist = np.full(n, np.nan, dtype=np.float32)
    floor_dist = np.full(n, np.nan, dtype=np.float32)
    num_peaks = np.full(n, np.nan, dtype=np.float32)
    ceiling_strength = np.full(n, np.nan, dtype=np.float32)
    floor_strength = np.full(n, np.nan, dtype=np.float32)

    for i in range(window, n):
        # Sum VP bins across the lookback window
        agg_vp = vp_matrix[i - window : i].sum(axis=0)

        # Smooth with Gaussian filter to remove bin-level noise
        agg_vp_smooth = gaussian_filter1d(agg_vp.astype(np.float64), sigma=0.8)

        # Normalize to [0, 1]
        vp_max = agg_vp_smooth.max()
        if vp_max > 0:
            agg_vp_norm = agg_vp_smooth / vp_max
        else:
            agg_vp_norm = agg_vp_smooth

        # Find prominent peaks (local maxima)
        peaks, properties = find_peaks(agg_vp_norm, prominence=0.05, distance=3)

        num_peaks[i] = len(peaks)

        if len(peaks) == 0:
            # No clear structure — set distances to max (normalized)
            ceiling_dist[i] = 1.0
            floor_dist[i] = 1.0
            ceiling_strength[i] = 0.0
            floor_strength[i] = 0.0
            continue

        # Peaks above current price (ceiling candidates)
        above = peaks[peaks > mid_bin]
        # Peaks below current price (floor candidates)
        below = peaks[peaks < mid_bin]

        if len(above) > 0:
            nearest_above = above[0]  # closest peak above
            ceiling_dist[i] = (nearest_above - mid_bin) / mid_bin  # normalized 0-1
            ceiling_strength[i] = agg_vp_norm[nearest_above]
        else:
            ceiling_dist[i] = 1.0  # no ceiling found, max distance
            ceiling_strength[i] = 0.0

        if len(below) > 0:
            nearest_below = below[-1]  # closest peak below
            floor_dist[i] = (mid_bin - nearest_below) / mid_bin  # normalized 0-1
            floor_strength[i] = agg_vp_norm[nearest_below]
        else:
            floor_dist[i] = 1.0  # no floor found, max distance
            floor_strength[i] = 0.0

    # --- Peak consistency: check if ceiling/floor persist across shifted windows ---
    # Shift offsets in bars (3d, 6d, 9d back)
    shift_bars = [BARS_PER_DAY * d for d in [3, 6, 9]]
    ceiling_consistency = np.full(n, np.nan, dtype=np.float32)
    floor_consistency = np.full(n, np.nan, dtype=np.float32)

    max_shift = max(shift_bars)
    for i in range(window + max_shift, n):
        # Current window's ceiling/floor bins
        curr_ceiling_bin = None
        curr_floor_bin = None

        agg_curr = vp_matrix[i - window : i].sum(axis=0)
        agg_curr_smooth = gaussian_filter1d(agg_curr.astype(np.float64), sigma=0.8)
        vp_max_curr = agg_curr_smooth.max()
        if vp_max_curr > 0:
            agg_curr_norm = agg_curr_smooth / vp_max_curr
        else:
            agg_curr_norm = agg_curr_smooth
        curr_peaks, _ = find_peaks(agg_curr_norm, prominence=0.05, distance=3)

        above_curr = curr_peaks[curr_peaks > mid_bin]
        below_curr = curr_peaks[curr_peaks < mid_bin]
        if len(above_curr) > 0:
            curr_ceiling_bin = above_curr[0]
        if len(below_curr) > 0:
            curr_floor_bin = below_curr[-1]

        ceiling_matches = 0
        floor_matches = 0
        tolerance = 2  # bins

        for shift in shift_bars:
            start = i - window - shift
            end = i - shift
            if start < 0:
                continue
            agg_shifted = vp_matrix[start:end].sum(axis=0)
            agg_shifted_smooth = gaussian_filter1d(agg_shifted.astype(np.float64), sigma=0.8)
            vp_max_s = agg_shifted_smooth.max()
            if vp_max_s > 0:
                agg_shifted_norm = agg_shifted_smooth / vp_max_s
            else:
                agg_shifted_norm = agg_shifted_smooth
            shifted_peaks, _ = find_peaks(agg_shifted_norm, prominence=0.05, distance=3)

            if curr_ceiling_bin is not None and len(shifted_peaks) > 0:
                if any(abs(p - curr_ceiling_bin) <= tolerance for p in shifted_peaks):
                    ceiling_matches += 1

            if curr_floor_bin is not None and len(shifted_peaks) > 0:
                if any(abs(p - curr_floor_bin) <= tolerance for p in shifted_peaks):
                    floor_matches += 1

        ceiling_consistency[i] = ceiling_matches / len(shift_bars)
        floor_consistency[i] = floor_matches / len(shift_bars)

    # Mid-range score: 1 when price is midway between ceiling and floor, 0 at extremes
    # Uses ceiling_dist and floor_dist: if both are similar, price is mid-range
    mid_range = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        c = ceiling_dist[i]
        f = floor_dist[i]
        if np.isnan(c) or np.isnan(f):
            continue
        total = c + f
        if total < 1e-8:
            mid_range[i] = 1.0
        else:
            # min/max ratio: 1 when equal (mid-range), 0 when one is much larger
            mid_range[i] = min(c, f) / max(c, f)

    df["vp_ceiling_dist"] = ceiling_dist
    df["vp_floor_dist"] = floor_dist
    df["vp_num_peaks"] = num_peaks
    df["vp_ceiling_strength"] = ceiling_strength
    df["vp_floor_strength"] = floor_strength
    df["vp_ceiling_consistency"] = ceiling_consistency
    df["vp_floor_consistency"] = floor_consistency
    df["vp_mid_range"] = mid_range

    return df


def build_feature_matrix(csv_path: str | None = None) -> pd.DataFrame:
    df = load_raw_data(csv_path)
    df = compute_derived_features(df)
    df = compute_vp_structure_features(df)

    # Drop warmup rows (NaNs from derived features + VP structure window)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Keep date + close (for label computation) + feature columns
    return df[["date", "close"] + FEATURE_COLS]
