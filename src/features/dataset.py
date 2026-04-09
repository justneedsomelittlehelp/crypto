import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    FEATURE_COLS,
    LOOKBACK_BARS_MODEL,
    LABEL_HORIZON_BARS,
    LABEL_MODE,
    LABEL_TP_PCT,
    LABEL_SL_PCT,
    LABEL_MAX_BARS,
    LABEL_REGIME_ADAPTIVE,
    LABEL_REGIME_MODE,
    LABEL_REGIME_SMA_BARS,
    LABEL_FGI_THRESHOLD,
    LABEL_FGI_PATH,
    LABEL_NEUTRAL_MODE,
    LABEL_NEUTRAL_PEAKS_THRESHOLD,
    DATALOADER_WORKERS,
    TRAIN_END,
    VAL_END,
)


def create_splits(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < train_end].reset_index(drop=True)
    val = df[(df["date"] >= train_end) & (df["date"] < val_end)].reset_index(drop=True)
    test = df[df["date"] >= val_end].reset_index(drop=True)
    return train, val, test


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = LOOKBACK_BARS_MODEL,
        horizon: int = LABEL_HORIZON_BARS,
        feature_cols: list[str] = FEATURE_COLS,
        labeled: bool = True,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.labeled = labeled
        self.data = torch.from_numpy(df[feature_cols].values.astype(np.float32))

        if labeled:
            close = df["close"].values
            num_peaks = df["vp_num_peaks"].values if "vp_num_peaks" in df.columns else None
            dates = df["date"].values if "date" in df.columns else None
            is_bull = self._build_regime_signal(close, dates)
            if LABEL_MODE == "first_hit":
                self.labels = self._first_hit_labels(close, num_peaks, is_bull)
            else:
                # Fixed horizon: 1 if close[t+horizon] > close[t]
                self.labels = np.zeros(len(close), dtype=np.float32)
                for i in range(len(close) - horizon):
                    self.labels[i] = float(close[i + horizon] > close[i])

            # Convert labels to tensor
            self.labels = torch.from_numpy(self.labels)

            # Build valid index list: positions with a full lookback window and a valid label
            self.valid_indices = []
            max_start = len(self.data) - self.lookback
            for idx in range(max_start):
                label_idx = idx + self.lookback - 1
                if label_idx < len(self.labels) and not torch.isnan(self.labels[label_idx]):
                    self.valid_indices.append(idx)

    @staticmethod
    def _build_regime_signal(close: np.ndarray, dates: np.ndarray = None) -> np.ndarray | None:
        """Build a boolean is_bull array based on configured regime detection mode.

        Returns None if regime adaptation is disabled.
        """
        if not LABEL_REGIME_ADAPTIVE:
            return None

        n = len(close)

        if LABEL_REGIME_MODE == "fgi" and dates is not None:
            # Load FGI data and map to bars by date
            fgi_df = pd.read_csv(LABEL_FGI_PATH, parse_dates=["date"])
            fgi_lookup = dict(zip(fgi_df["date"].dt.date, fgi_df["fgi_value"].astype(float)))

            is_bull = np.full(n, np.nan)
            for i in range(n):
                dt = pd.Timestamp(dates[i]).date()
                fgi = fgi_lookup.get(dt, np.nan)
                if not np.isnan(fgi):
                    is_bull[i] = float(fgi >= LABEL_FGI_THRESHOLD)
            return is_bull

        # Default: SMA-based regime detection
        is_bull = np.full(n, np.nan)
        for i in range(LABEL_REGIME_SMA_BARS, n):
            sma = np.mean(close[i - LABEL_REGIME_SMA_BARS : i])
            is_bull[i] = float(close[i] >= sma)
        return is_bull

    @staticmethod
    def _first_hit_labels(close: np.ndarray, num_peaks: np.ndarray = None,
                          is_bull: np.ndarray = None) -> np.ndarray:
        """Label each bar: 1 if price hits +TP before -SL, 0 if SL hit first, NaN if neither.

        TP/SL are scaled by recent volatility (std of log-returns over past 30 bars).
        If is_bull is provided, TP/SL ratios flip in bear markets.
        If num_peaks is provided, bars with no VP structure use symmetric TP/SL or are skipped.
        """
        n = len(close)
        labels = np.full(n, np.nan, dtype=np.float32)

        # Compute rolling volatility (std of log-returns, 30-bar window)
        vol_window = 30
        log_returns = np.diff(np.log(close))
        rolling_vol = np.full(n, np.nan)
        for i in range(vol_window, len(log_returns)):
            rolling_vol[i + 1] = np.std(log_returns[i - vol_window + 1 : i + 1])

        # Median volatility for normalization
        valid_vol = rolling_vol[~np.isnan(rolling_vol)]
        if len(valid_vol) == 0:
            return labels
        median_vol = np.median(valid_vol)
        if median_vol < 1e-10:
            median_vol = 1e-10

        warmup = vol_window + 1

        for i in range(warmup, n - 1):
            if np.isnan(rolling_vol[i]):
                continue

            vol_scale = rolling_vol[i] / median_vol
            vol_scale = max(0.5, min(vol_scale, 3.0))

            # Check for neutral zone (no VP structure)
            is_neutral = (LABEL_NEUTRAL_MODE != "off"
                          and num_peaks is not None
                          and not np.isnan(num_peaks[i])
                          and num_peaks[i] <= LABEL_NEUTRAL_PEAKS_THRESHOLD)

            if is_neutral:
                if LABEL_NEUTRAL_MODE == "skip":
                    continue  # label stays NaN — sample will be excluded
                else:
                    # Symmetric TP/SL when no VP structure
                    sym_pct = (LABEL_TP_PCT + LABEL_SL_PCT) / 2
                    tp_pct = sym_pct * vol_scale
                    sl_pct = sym_pct * vol_scale
            elif is_bull is not None and not np.isnan(is_bull[i]):
                if is_bull[i]:
                    # Bull: normal TP/SL
                    tp_pct = LABEL_TP_PCT * vol_scale
                    sl_pct = LABEL_SL_PCT * vol_scale
                else:
                    # Bear: flip TP/SL
                    tp_pct = LABEL_SL_PCT * vol_scale
                    sl_pct = LABEL_TP_PCT * vol_scale
            else:
                tp_pct = LABEL_TP_PCT * vol_scale
                sl_pct = LABEL_SL_PCT * vol_scale

            entry = close[i]
            tp_level = entry * (1 + tp_pct)
            sl_level = entry * (1 - sl_pct)
            max_j = min(i + LABEL_MAX_BARS, n)
            for j in range(i + 1, max_j):
                if close[j] >= tp_level:
                    labels[i] = 1.0
                    break
                if close[j] <= sl_level:
                    labels[i] = 0.0
                    break
        return labels

    def __len__(self) -> int:
        if self.labeled:
            return len(self.valid_indices)
        return len(self.data) - self.lookback + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.labeled:
            real_idx = self.valid_indices[idx]
        else:
            real_idx = idx

        x = self.data[real_idx : real_idx + self.lookback]

        if self.labeled:
            label_idx = real_idx + self.lookback - 1
            y = self.labels[label_idx]
            return x, y

        return (x,)


def make_loader(ds, batch_size, shuffle=False):
    """Create DataLoader with optimal settings for GPU training."""
    import torch
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DATALOADER_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and DATALOADER_WORKERS > 0,
    )


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lookback: int = LOOKBACK_BARS_MODEL,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TimeSeriesDataset(train_df, lookback=lookback)
    val_ds = TimeSeriesDataset(val_df, lookback=lookback)
    test_ds = TimeSeriesDataset(test_df, lookback=lookback)

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size)
    test_loader = make_loader(test_ds, batch_size)

    return train_loader, val_loader, test_loader
