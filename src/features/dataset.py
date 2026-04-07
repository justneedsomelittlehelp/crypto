import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    FEATURE_COLS,
    LOOKBACK_BARS_MODEL,
    LABEL_HORIZON_BARS,
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
        self.data = df[feature_cols].values.astype(np.float32)

        if labeled:
            close = df["close"].values
            # Label: 1 if close[t+horizon] > close[t], else 0
            # t is the last bar in the lookback window
            self.labels = np.zeros(len(close), dtype=np.float32)
            for i in range(len(close) - horizon):
                self.labels[i] = float(close[i + horizon] > close[i])

    def __len__(self) -> int:
        if self.labeled:
            # Last position where we have both a full window AND a label
            return len(self.data) - self.lookback - self.horizon + 1
        return len(self.data) - self.lookback + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        window = self.data[idx : idx + self.lookback]
        x = torch.from_numpy(window)

        if self.labeled:
            # Label corresponds to the last bar in the window
            label_idx = idx + self.lookback - 1
            y = torch.tensor(self.labels[label_idx], dtype=torch.float32)
            return x, y

        return (x,)


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
