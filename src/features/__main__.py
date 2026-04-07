"""
CLI entry point: python -m src.features

Usage:
    python -m src.features prepare   # build feature matrix, print split info
    python -m src.features check     # validate features (no NaNs, no leakage, shapes)
"""

import sys

from src.config import FEATURE_COLS, LOOKBACK_BARS_MODEL, TRAIN_END, VAL_END
from src.features.pipeline import build_feature_matrix
from src.features.dataset import create_splits, TimeSeriesDataset


def cmd_prepare():
    print("Building feature matrix...")
    df = build_feature_matrix()
    print(f"  Total rows: {len(df)}")
    print(f"  Features: {len(FEATURE_COLS)} ({len(FEATURE_COLS) - 4} VP + 4 derived)")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    train, val, test = create_splits(df)
    print(f"\nTime-based splits:")
    print(f"  Train: {len(train):,} rows  (... to {TRAIN_END})")
    print(f"  Val:   {len(val):,} rows  ({TRAIN_END} to {VAL_END})")
    print(f"  Test:  {len(test):,} rows  ({VAL_END} to ...)")

    ds = TimeSeriesDataset(train, lookback=LOOKBACK_BARS_MODEL)
    sample = ds[0]
    print(f"\nDataset sample shape: {list(sample.shape)}")
    print(f"  [lookback={LOOKBACK_BARS_MODEL}, features={len(FEATURE_COLS)}]")
    print("\nDone.")


def cmd_check():
    print("Validating features...")
    ok = True

    df = build_feature_matrix()

    # Check no NaNs
    nan_count = df[FEATURE_COLS].isna().sum().sum()
    if nan_count > 0:
        print(f"  FAIL: {nan_count} NaN values in feature matrix")
        ok = False
    else:
        print(f"  OK: No NaN values")

    # Check splits are chronological (no leakage)
    train, val, test = create_splits(df)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print(f"  FAIL: Empty split (train={len(train)}, val={len(val)}, test={len(test)})")
        ok = False
    else:
        train_max = train["date"].max()
        val_min = val["date"].min()
        val_max = val["date"].max()
        test_min = test["date"].min()
        if train_max >= val_min:
            print(f"  FAIL: Train/val overlap ({train_max} >= {val_min})")
            ok = False
        elif val_max >= test_min:
            print(f"  FAIL: Val/test overlap ({val_max} >= {test_min})")
            ok = False
        else:
            print(f"  OK: Splits are strictly chronological")

    # Check dataset shapes
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        ds = TimeSeriesDataset(split_df, lookback=LOOKBACK_BARS_MODEL)
        sample = ds[0]
        expected = (LOOKBACK_BARS_MODEL, len(FEATURE_COLS))
        if tuple(sample.shape) != expected:
            print(f"  FAIL: {name} shape {tuple(sample.shape)} != expected {expected}")
            ok = False
        else:
            print(f"  OK: {name} shape {tuple(sample.shape)}")

    if ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED.")
    return ok


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.features <command>")
        print("Commands: prepare, check")
        sys.exit(1)

    command = sys.argv[1]

    if command == "prepare":
        cmd_prepare()
    elif command == "check":
        ok = cmd_check()
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown command: {command}")
        print("Commands: prepare, check")
        sys.exit(1)


if __name__ == "__main__":
    main()
