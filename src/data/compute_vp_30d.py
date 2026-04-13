"""Regenerate VP feature CSV with a 30-day lookback window.

Writes `BTC_1h_RELVP_30d.csv` (+ metadata) alongside the existing
`BTC_1h_RELVP.csv`. The existing 180-day file is left untouched.

Usage:
    python3 -m src.data.compute_vp_30d

Why a standalone script: config.LOOKBACK_DAYS is used throughout the
scraper/VP/pipeline code and feeds the default output filename. To avoid
clobbering the 180d CSV (still used by eval_v6_prime and legacy evals),
we patch cfg in-process and write to a parallel filename.

Shrinking lookback from 180d→30d means 3,600 fewer warmup rows are
discarded, so the output CSV gains ~3,600 rows at the head of the
series (going back to early 2016 instead of mid-2016).
"""

import json
import sys

import src.config as cfg

NEW_LOOKBACK_DAYS = 30

# Patch cfg BEFORE any downstream modules capture the values.
cfg.LOOKBACK_DAYS = NEW_LOOKBACK_DAYS
cfg.LOOKBACK_BARS = cfg.LOOKBACK_DAYS * cfg.BARS_PER_DAY

# Deferred imports (must come after the patch)
from src.data.scraper import fetch_and_merge  # noqa: E402
from src.data.volume_profile import compute_relative_vp  # noqa: E402


OUTPUT_CSV = "BTC_1h_RELVP_30d.csv"
OUTPUT_META = "BTC_1h_RELVP_30d_metadata.json"


def main():
    print(f"Computing VP with LOOKBACK_DAYS={cfg.LOOKBACK_DAYS} "
          f"({cfg.LOOKBACK_BARS} bars at {cfg.TIMEFRAME})")

    combined = fetch_and_merge()
    df = compute_relative_vp(combined)

    csv_path = cfg.PROJECT_ROOT / OUTPUT_CSV
    meta_path = cfg.PROJECT_ROOT / OUTPUT_META
    df.to_csv(csv_path, index=False)

    meta = {
        "symbol": cfg.SYMBOL,
        "timeframe": cfg.TIMEFRAME,
        "start_date": cfg.START_DATE,
        "exchanges": cfg.EXCHANGES,
        "lookback_days_for_vp": cfg.LOOKBACK_DAYS,
        "lookback_bars_for_vp": cfg.LOOKBACK_BARS,
        "bars_per_day": cfg.BARS_PER_DAY,
        "relative_vp": {
            "span_pct": cfg.REL_SPAN_PCT,
            "bin_count": cfg.REL_BIN_COUNT,
            "normalize": cfg.REL_NORMALIZE,
            "axis": "log(price/close_t)",
        },
        "note": (
            "Parallel VP file with 30-day lookback. Used by eval_v10 "
            "(90-day temporal transformer). Does not replace the 180d "
            "BTC_1h_RELVP.csv — both files coexist."
        ),
        "output": {
            "csv": OUTPUT_CSV,
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {csv_path} ({len(df):,} rows)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    sys.exit(main())
