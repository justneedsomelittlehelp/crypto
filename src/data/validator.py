"""
Data quality validation for the feature CSV.

Checks for timestamp gaps, zero/negative prices, VP normalization,
and other data integrity issues.
"""

import sys
import pandas as pd
import numpy as np

from src.config import (
    STEP_MS, VP_COL_NAMES, VOLUME_COL,
    output_csv_name, PROJECT_ROOT,
)


def validate(csv_path: str | None = None) -> bool:
    """
    Run all validation checks on the feature CSV.

    Args:
        csv_path: path to CSV file. Defaults to project root CSV.

    Returns:
        True if all checks pass, False otherwise.
    """
    if csv_path is None:
        csv_path = str(PROJECT_ROOT / output_csv_name())

    print(f"Validating: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  FAIL: file not found: {csv_path}")
        return False

    passed = True
    total_rows = len(df)
    print(f"  Rows: {total_rows}")

    # --- Check 1: Required columns exist ---
    required = ["ts", "date", "open", "high", "low", "close", VOLUME_COL] + VP_COL_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  FAIL: missing columns: {missing}")
        passed = False
    else:
        print(f"  OK: all {len(required)} required columns present")

    # --- Check 2: Timestamp gaps ---
    ts_diff = df["ts"].diff().dropna()
    expected_step = STEP_MS
    gaps = ts_diff[ts_diff != expected_step]
    if len(gaps) > 0:
        gap_count = len(gaps)
        gap_pct = gap_count / total_rows * 100
        print(f"  WARN: {gap_count} timestamp gaps ({gap_pct:.2f}% of rows)")
        # Show first few gaps
        for idx in gaps.head(5).index:
            gap_ms = int(ts_diff.loc[idx])
            gap_bars = gap_ms / expected_step
            print(f"    row {idx}: gap of {gap_bars:.1f} bars ({gap_ms}ms)")
    else:
        print(f"  OK: no timestamp gaps")

    # --- Check 3: Zero or negative prices ---
    for col in ["open", "high", "low", "close"]:
        bad = (df[col] <= 0).sum()
        if bad > 0:
            print(f"  FAIL: {bad} rows with {col} <= 0")
            passed = False
        else:
            print(f"  OK: {col} all positive")

    # --- Check 4: Zero volume ---
    zero_vol = (df[VOLUME_COL] <= 0).sum()
    if zero_vol > 0:
        print(f"  WARN: {zero_vol} rows with zero volume ({zero_vol / total_rows * 100:.2f}%)")
    else:
        print(f"  OK: all rows have positive volume")

    # --- Check 5: VP normalization ---
    vp_sums = df[VP_COL_NAMES].sum(axis=1)
    bad_norm = ((vp_sums - 1.0).abs() > 0.01).sum()
    zero_vp = (vp_sums == 0).sum()
    if zero_vp > 0:
        print(f"  WARN: {zero_vp} rows with all-zero VP")
    if bad_norm > 0:
        print(f"  FAIL: {bad_norm} rows where VP doesn't sum to ~1.0")
        passed = False
    else:
        print(f"  OK: VP normalization correct (all rows sum to ~1.0)")

    # --- Check 6: VP negative values ---
    neg_vp = (df[VP_COL_NAMES] < 0).any(axis=1).sum()
    if neg_vp > 0:
        print(f"  FAIL: {neg_vp} rows with negative VP values")
        passed = False
    else:
        print(f"  OK: no negative VP values")

    # --- Check 7: Date range ---
    date_min = df["date"].min()
    date_max = df["date"].max()
    print(f"  Date range: {date_min} → {date_max}")

    # --- Summary ---
    print()
    if passed:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED")

    return passed


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    ok = validate(csv_path)
    sys.exit(0 if ok else 1)
