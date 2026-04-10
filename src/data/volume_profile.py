"""
Relative Volume Profile computation.

Computes a 50-bin histogram of where volume traded relative to the
current price, over a configurable lookback window.
"""

import json
import numpy as np
import pandas as pd

from src.config import (
    SYMBOL, TIMEFRAME, START_DATE, EXCHANGES,
    LOOKBACK_DAYS, LOOKBACK_BARS, STEP_SECONDS, BARS_PER_DAY,
    HORIZON_24H_BARS, REL_SPAN_PCT, REL_BIN_COUNT, REL_NORMALIZE,
    REL_EDGES_LOG, REL_EDGES_PCT, REL_CENTERS_PCT,
    VOLUME_COL, VP_COL_NAMES,
    output_csv_name, output_meta_name,
)


def compute_relative_vp(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative volume profile for each row in the merged OHLCV data.

    For each candle at index i (where i >= LOOKBACK_BARS):
    - Takes past LOOKBACK_BARS candles as history window
    - Computes log(close_j / close_i) for each historical candle j
    - Bins these into REL_BIN_COUNT bins, weighted by volume
    - Optionally normalizes to sum=1

    Args:
        combined: Merged OHLCV DataFrame from scraper.merge_exchanges()

    Returns:
        DataFrame with OHLCV + VP columns for all valid rows.
    """
    final_rows = []

    print(
        f"Generating Relative VP: span=±{int(REL_SPAN_PCT * 100)}%, "
        f"bins={REL_BIN_COUNT}, lookback={LOOKBACK_DAYS} days ({LOOKBACK_BARS} bars) ..."
    )

    closes = combined["close"].values
    vols = combined["vol"].values

    for i in range(LOOKBACK_BARS, len(combined)):
        close_t = closes[i]

        # Log distance of each historical close vs current
        x = np.log(closes[i - LOOKBACK_BARS : i + 1] / close_t)
        w = vols[i - LOOKBACK_BARS : i + 1]

        # Weighted histogram
        vp, _ = np.histogram(x, bins=REL_EDGES_LOG, weights=w)

        if REL_NORMALIZE:
            vp_sum = vp.sum()
            if vp_sum > 0:
                vp = vp / vp_sum
            else:
                vp = np.zeros_like(vp, dtype=float)

        curr = combined.iloc[i]
        row = {
            "ts": int(curr["ts"]),
            "date": curr["date"].isoformat(),
            "open": float(curr["open"]),
            "high": float(curr["high"]),
            "low": float(curr["low"]),
            "close": float(curr["close"]),
            VOLUME_COL: float(curr["vol"]),
        }

        for k, name in enumerate(VP_COL_NAMES):
            row[name] = float(vp[k])

        final_rows.append(row)

        if i % 2000 == 0:
            print(f"  -> Processing: {i}/{len(combined)}", end="\r")

    print()
    return pd.DataFrame(final_rows)


def save_results(df: pd.DataFrame, output_dir: str = ".") -> tuple[str, str]:
    """
    Save feature DataFrame and metadata JSON.

    Args:
        df: DataFrame from compute_relative_vp()
        output_dir: directory to write files into

    Returns:
        Tuple of (csv_path, meta_path)
    """
    csv_name = output_csv_name()
    meta_name = output_meta_name()
    csv_path = f"{output_dir}/{csv_name}"
    meta_path = f"{output_dir}/{meta_name}"

    df.to_csv(csv_path, index=False)

    metadata = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "start_date": START_DATE,
        "exchanges": EXCHANGES,
        "lookback_days_for_vp": LOOKBACK_DAYS,
        "lookback_bars_for_vp": LOOKBACK_BARS,
        "step_seconds": STEP_SECONDS,
        "bars_per_day": BARS_PER_DAY,
        "horizon_24h_bars": HORIZON_24H_BARS,
        "relative_vp": {
            "span_pct": REL_SPAN_PCT,
            "bin_count": REL_BIN_COUNT,
            "normalize": REL_NORMALIZE,
            "axis": "log(price/close_t)",
            "edges_log": REL_EDGES_LOG.tolist(),
            "edges_pct": REL_EDGES_PCT.tolist(),
            "centers_pct": REL_CENTERS_PCT.tolist(),
            "notes": (
                "vp_rel_* columns represent weighted histogram of "
                "log(close_j/close_t) over lookback window, weighted by volume. "
                "If normalize=True, each row sums to 1 (distribution)."
            ),
        },
        "output": {
            "csv": csv_name,
            "metadata_json": meta_name,
            "columns": df.columns.tolist(),
            "vp_columns": VP_COL_NAMES,
            "volume_column": VOLUME_COL,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Final shape: {df.shape}")

    return csv_path, meta_path
