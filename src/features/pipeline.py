import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    VP_COL_NAMES,
    VOLUME_COL,
    VOLUME_ROLL_WINDOW_BARS,
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

    return df


def build_feature_matrix(csv_path: str | None = None) -> pd.DataFrame:
    df = load_raw_data(csv_path)
    df = compute_derived_features(df)

    # Drop warmup rows (NaNs from log_return + volume rolling window)
    df = df.dropna(subset=DERIVED_FEATURE_COLS).reset_index(drop=True)

    # Keep date + close (for label computation) + feature columns
    return df[["date", "close"] + FEATURE_COLS]
