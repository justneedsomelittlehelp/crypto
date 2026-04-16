"""Fetch macro regime features for v12 regime-aware model.

Fetches:
  - VIX, DXY, GLD, USO at 1h resolution from yfinance (for regime encoder)
  - FFR (Fed funds rate), 10Y and 2Y treasury yields from FRED (daily, for day enrichment)

Outputs two CSVs:
  data/regime_hourly.csv  — VIX/DXY/GLD/USO at 1h, UTC-aligned
  data/regime_daily.csv   — all 6 daily regime features, UTC date-keyed

Usage:
  python3 -m src.data.fetch_regime [--fred-key YOUR_KEY]

Notes:
  - yfinance 1h data is limited to ~730 days of history. For dates before
    that, daily close is upsampled to hourly (flat within each day).
    This is acceptable because the regime encoder's job is detecting
    recent spikes, and historical data only feeds the temporal transformer
    via daily resolution anyway.
  - VIX/GLD/USO trade US market hours only. Off-hours are forward-filled.
  - DXY (DX-Y.NYB) can be spotty on yfinance; fallback to UUP ETF if needed.
  - FRED requires a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    If no key is provided, FFR and yield curve are filled with NaN and the
    model can still train (day enrichment will see zeros after normalization).
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def fetch_yfinance_hourly(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch 1h OHLCV from yfinance. Returns DataFrame with UTC datetime index."""
    import yfinance as yf

    tk = yf.Ticker(ticker)

    # yfinance 1h data limited to ~730 days; fetch in chunks if needed
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")

    # For the recent window (last ~730 days), get true 1h bars
    cutoff = end_dt - timedelta(days=725)
    hourly_start = max(start_dt, cutoff)

    dfs = []

    # Historical daily (before hourly cutoff) → upsample to 1h
    if start_dt < hourly_start:
        daily = tk.history(start=start_dt.strftime("%Y-%m-%d"),
                           end=hourly_start.strftime("%Y-%m-%d"),
                           interval="1d")
        if len(daily) > 0:
            daily.index = daily.index.tz_convert("UTC") if daily.index.tz else daily.index.tz_localize("UTC")
            daily_close = daily[["Close"]].rename(columns={"Close": "close"})
            # Upsample daily → hourly (forward-fill)
            hourly_idx = pd.date_range(start=daily_close.index[0],
                                       end=hourly_start,
                                       freq="h", tz="UTC")
            daily_up = daily_close.reindex(hourly_idx, method="ffill")
            dfs.append(daily_up)

    # Recent 1h bars
    hourly = tk.history(start=hourly_start.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        interval="1h")
    if len(hourly) > 0:
        hourly.index = hourly.index.tz_convert("UTC") if hourly.index.tz else hourly.index.tz_localize("UTC")
        hourly_close = hourly[["Close"]].rename(columns={"Close": "close"})
        dfs.append(hourly_close)

    if not dfs:
        print(f"  WARNING: no data for {ticker}")
        return pd.DataFrame(columns=["close"])

    result = pd.concat(dfs)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    return result


def fetch_yfinance_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close from yfinance."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    daily = tk.history(start=start, end=end, interval="1d")
    if len(daily) == 0:
        print(f"  WARNING: no daily data for {ticker}")
        return pd.DataFrame(columns=["close"])
    daily.index = daily.index.tz_convert("UTC") if daily.index.tz else daily.index.tz_localize("UTC")
    return daily[["Close"]].rename(columns={"Close": "close"})


def fetch_fred(series_id: str, start: str, end: str, api_key: str = None) -> pd.Series:
    """Fetch a FRED time series. Returns Series with UTC date index."""
    if not api_key:
        print(f"  SKIP {series_id}: no FRED API key")
        return pd.Series(dtype=float, name=series_id)

    from fredapi import Fred
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index).tz_localize("UTC")
    s.name = series_id
    return s


def build_regime_hourly(start: str, end: str) -> pd.DataFrame:
    """Build hourly regime features CSV."""
    tickers = {
        "vix":  "^VIX",
        "dxy":  "DX-Y.NYB",
        "gld":  "GLD",
        "uso":  "USO",
    }

    frames = {}
    for name, ticker in tickers.items():
        print(f"  Fetching {name} ({ticker}) hourly...")
        df = fetch_yfinance_hourly(ticker, start, end)
        frames[name] = df["close"] if "close" in df.columns else df.iloc[:, 0]

    combined = pd.DataFrame(frames)
    combined = combined.sort_index()
    combined = combined.ffill()

    # Resample to exact hourly grid, forward-fill gaps (weekends, holidays)
    hourly_idx = pd.date_range(
        start=combined.index[0].floor("h"),
        end=combined.index[-1].ceil("h"),
        freq="h", tz="UTC",
    )
    combined = combined.reindex(hourly_idx, method="ffill")
    combined.index.name = "datetime_utc"

    return combined


def build_regime_daily(start: str, end: str, fred_key: str = None) -> pd.DataFrame:
    """Build daily regime features CSV."""
    # yfinance daily
    tickers = {
        "vix":  "^VIX",
        "dxy":  "DX-Y.NYB",
        "gld":  "GLD",
        "uso":  "USO",
    }

    frames = {}
    for name, ticker in tickers.items():
        print(f"  Fetching {name} ({ticker}) daily...")
        df = fetch_yfinance_daily(ticker, start, end)
        frames[name] = df["close"] if "close" in df.columns else df.iloc[:, 0]

    # FRED
    fred_series = {
        "ffr":  "FEDFUNDS",
        "dgs10": "DGS10",
        "dgs2":  "DGS2",
    }
    for name, series_id in fred_series.items():
        print(f"  Fetching {name} ({series_id}) from FRED...")
        frames[name] = fetch_fred(series_id, start, end, fred_key)

    combined = pd.DataFrame(frames)
    combined = combined.sort_index()
    combined = combined.ffill()

    # Compute yield curve slope
    if "dgs10" in combined.columns and "dgs2" in combined.columns:
        combined["yield_curve"] = combined["dgs10"] - combined["dgs2"]
    else:
        combined["yield_curve"] = np.nan

    # Drop raw yields (keep slope)
    combined = combined.drop(columns=["dgs10", "dgs2"], errors="ignore")

    # Fill daily grid
    daily_idx = pd.date_range(
        start=combined.index[0].normalize(),
        end=combined.index[-1].normalize(),
        freq="D", tz="UTC",
    )
    combined = combined.reindex(daily_idx, method="ffill")
    combined.index.name = "date_utc"

    # Final columns: vix, dxy, gld, uso, ffr, yield_curve
    return combined


def main():
    p = argparse.ArgumentParser(description="Fetch regime features for v12")
    p.add_argument("--fred-key", type=str, default=None,
                   help="FRED API key. If omitted, FFR and yield curve will be NaN.")
    p.add_argument("--start", type=str, default="2016-01-01",
                   help="Start date (default: 2016-01-01)")
    p.add_argument("--end", type=str, default=None,
                   help="End date (default: today)")
    args = p.parse_args()

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Fetching hourly regime features ===")
    hourly = build_regime_hourly(args.start, end)
    hourly_path = DATA_DIR / "regime_hourly.csv"
    hourly.to_csv(hourly_path)
    print(f"  Wrote {hourly_path} ({len(hourly):,} rows)")
    print(f"  Range: {hourly.index[0]} → {hourly.index[-1]}")
    print(f"  Columns: {list(hourly.columns)}")
    print(f"  Nulls:\n{hourly.isnull().sum()}")

    print("\n=== Fetching daily regime features ===")
    daily = build_regime_daily(args.start, end, args.fred_key)
    daily_path = DATA_DIR / "regime_daily.csv"
    daily.to_csv(daily_path)
    print(f"  Wrote {daily_path} ({len(daily):,} rows)")
    print(f"  Range: {daily.index[0]} → {daily.index[-1]}")
    print(f"  Columns: {list(daily.columns)}")
    print(f"  Nulls:\n{daily.isnull().sum()}")


if __name__ == "__main__":
    main()
