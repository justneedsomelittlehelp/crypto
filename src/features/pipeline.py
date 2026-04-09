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

    # Candle shape features (1h bar — kept for backwards compat, mostly noise)
    bar_height = (df["high"] - df["low"]).clip(lower=1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / bar_height
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_height
    df["body_dir"] = np.sign(df["close"] - df["open"])

    # Normalized OHLC for in-model daily candle aggregation.
    # Centered at 0 (subtract 1) so they're symmetric around the close price.
    df["ohlc_open_ratio"] = df["open"] / df["close"] - 1.0
    df["ohlc_high_ratio"] = df["high"] / df["close"] - 1.0
    df["ohlc_low_ratio"] = df["low"] / df["close"] - 1.0

    # ── Walk-forward-safe z-score normalization ──
    # Use expanding window stats so each row only sees its own past data.
    # Prevents data leakage in walk-forward training.
    min_periods = VOLUME_ROLL_WINDOW_BARS  # 30 days warmup before any z-scoring

    for col in ["volume_ratio", "log_return", "bar_range", "bar_body"]:
        # Log-transform volume_ratio first (long-tail compression)
        if col == "volume_ratio":
            base = np.log(df[col] + 1.0)
        else:
            base = df[col]
        exp_mean = base.expanding(min_periods=min_periods).mean()
        exp_std = base.expanding(min_periods=min_periods).std()
        z = (base - exp_mean) / exp_std.clip(lower=1e-8)
        # Soft squash with tanh: smoothly bounds outliers while preserving
        # ordering. Black-swan magnitude is compressed but still distinguishable
        # from normal extreme moves. Killswitch handles real risk in production.
        df[col] = 5.0 * np.tanh(z / 5.0)

    return df


def _smooth_and_find_peaks(agg_vp: np.ndarray, sigma: float = 0.8, prominence: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a VP profile and find peaks. Returns (peaks_array, normalized_profile)."""
    smooth = gaussian_filter1d(agg_vp.astype(np.float64), sigma=sigma)
    vp_max = smooth.max()
    if vp_max > 0:
        smooth /= vp_max
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=3)
    return peaks, smooth


def compute_vp_structure_features(
    df: pd.DataFrame,
    window: int = LOOKBACK_BARS_MODEL,
) -> pd.DataFrame:
    """Aggregate VP bins over a rolling window and extract structural features.

    Optimized: uses rolling sum (add new row, subtract old) instead of
    recomputing the full sum each step. Pre-computes all rolling sums for
    shifted windows in one pass.
    """
    df = df.copy()
    vp_matrix = df[VP_COL_NAMES].values  # (n_rows, 50)
    mid_bin = REL_BIN_COUNT // 2  # bin 25 = current price
    n = len(df)

    # --- Pre-compute rolling VP sums using cumulative sum ---
    # cumsum[i] = sum of vp_matrix[0:i], so window sum = cumsum[i] - cumsum[i-window]
    cumsum = np.zeros((n + 1, REL_BIN_COUNT), dtype=np.float64)
    np.cumsum(vp_matrix, axis=0, out=cumsum[1:])

    def get_rolling_vp(idx, win):
        """Get aggregated VP for window ending at idx (exclusive) with length win."""
        start = idx - win
        if start < 0:
            return None
        return cumsum[idx] - cumsum[start]

    # --- Pass 1: compute peaks for all needed positions ---
    # We need peaks at: current window (i-window:i) and shifted windows
    shift_days = [3, 6, 9]
    shift_bars = [BARS_PER_DAY * d for d in shift_days]
    max_shift = max(shift_bars)

    # Cache: store (peaks, norm_profile, ceiling_bin, floor_bin) per position
    # Key = end index of the window
    peak_cache = {}

    def get_peaks_for_window_end(end_idx):
        """Get peak info for window ending at end_idx."""
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

    # --- Output arrays ---
    ceiling_dist = np.full(n, np.nan, dtype=np.float32)
    floor_dist = np.full(n, np.nan, dtype=np.float32)
    num_peaks_arr = np.full(n, np.nan, dtype=np.float32)
    ceiling_strength = np.full(n, np.nan, dtype=np.float32)
    floor_strength = np.full(n, np.nan, dtype=np.float32)
    ceiling_consistency = np.full(n, np.nan, dtype=np.float32)
    floor_consistency = np.full(n, np.nan, dtype=np.float32)

    # --- Main loop ---
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

        # --- Consistency check (only if enough history for shifts) ---
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

    # --- Mid-range score (vectorized) ---
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


def build_feature_matrix(csv_path: str | None = None) -> pd.DataFrame:
    df = load_raw_data(csv_path)
    df = compute_derived_features(df)
    df = compute_vp_structure_features(df)

    # Drop warmup rows (NaNs from derived features + VP structure window)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Keep date + close (for label computation) + feature columns
    return df[["date", "close"] + FEATURE_COLS]
