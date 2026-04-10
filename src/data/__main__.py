"""
CLI entry point: python -m src.data

Usage:
    python -m src.data scrape            # fetch 1h data and compute VP
    python -m src.data scrape 15m        # fetch 15m data and compute VP
    python -m src.data validate          # validate existing CSV
    python -m src.data validate <path>   # validate specific CSV
"""

import sys


def _override_timeframe(timeframe: str):
    """Patch config module with new timeframe before other modules import from it."""
    import ccxt
    import src.config as cfg

    cfg.TIMEFRAME = timeframe
    cfg.STEP_SECONDS = ccxt.Exchange.parse_timeframe(timeframe)
    cfg.STEP_MS = cfg.STEP_SECONDS * 1000
    cfg.BARS_PER_DAY = int(round((24 * 3600) / cfg.STEP_SECONDS))

    # Sanity check
    if abs((cfg.BARS_PER_DAY * cfg.STEP_SECONDS) - (24 * 3600)) > 1:
        print(f"ERROR: TIMEFRAME='{timeframe}' does not evenly divide 24h.")
        sys.exit(1)

    # Rescale all bar-based params
    cfg.LOOKBACK_BARS = cfg.LOOKBACK_DAYS * cfg.BARS_PER_DAY
    cfg.HORIZON_24H_BARS = cfg.BARS_PER_DAY
    cfg.VOLUME_ROLL_WINDOW_BARS = cfg.VOLUME_ROLL_WINDOW_DAYS * cfg.BARS_PER_DAY
    cfg.LOOKBACK_BARS_MODEL = 30 * cfg.BARS_PER_DAY
    cfg.LABEL_HORIZON_BARS = cfg.BARS_PER_DAY
    cfg.LABEL_REGIME_SMA_BARS = 90 * cfg.BARS_PER_DAY
    cfg.LABEL_MAX_BARS = cfg.BARS_PER_DAY * 14
    cfg.VOLUME_COL = f"volume_{timeframe}".replace("/", "_")

    print(f"Timeframe override: {timeframe}")
    print(f"  BARS_PER_DAY={cfg.BARS_PER_DAY}, LOOKBACK_BARS={cfg.LOOKBACK_BARS}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.data <command> [timeframe]")
        print("Commands: scrape, validate")
        print("Example: python -m src.data scrape 15m")
        sys.exit(1)

    command = sys.argv[1]

    if command == "scrape":
        # Optional timeframe override (e.g., "15m", "5m")
        if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
            _override_timeframe(sys.argv[2])

        # Deferred imports — after config is patched
        from src.data.scraper import fetch_and_merge
        from src.data.volume_profile import compute_relative_vp, save_results

        combined = fetch_and_merge()
        df = compute_relative_vp(combined)
        save_results(df)
        print("\nDone.")

    elif command == "validate":
        from src.data.validator import validate
        csv_path = sys.argv[2] if len(sys.argv) > 2 else None
        ok = validate(csv_path)
        sys.exit(0 if ok else 1)

    else:
        print(f"Unknown command: {command}")
        print("Commands: scrape, validate")
        sys.exit(1)


if __name__ == "__main__":
    main()
