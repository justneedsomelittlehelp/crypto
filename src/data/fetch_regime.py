"""Fetch macro regime features for v12 regime-aware model.

Fetches:
  - VIX, DXY, GLD, USO at 1h resolution from yfinance (for regime encoder)
  - FFR (Fed funds rate), 10Y and 2Y treasury yields from FRED (daily, for day enrichment)
  - Global outer-layer macro features (for hierarchical HMM global-outer variant):
      Credit spread (BAA10Y: Moody's Baa corp yield − 10Y Treasury) as
      risk-appetite proxy (substitute for HY OAS, which FRED truncated to
      ~3y rolling window in 2023 due to ICE/BofA licensing),
      Fed balance sheet net liquidity (WALCL - RRPONTSYD - WTREGEN),
      and G4 M2 YoY (US/EU/JP/CN equal-weighted in local currency).

Outputs three CSVs:
  data/regime_hourly.csv        — VIX/DXY/GLD/USO at 1h, UTC-aligned
  data/regime_daily.csv         — 6 daily regime features, UTC date-keyed
  data/regime_global_outer.csv  — dxy, hy_oas, net_fed_liq_yoy, global_m2_yoy,
                                  daily grid, with publication-lag safety applied

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
        "vix":   "^VIX",
        "dxy":   "DX-Y.NYB",
        "gld":   "GLD",
        "uso":   "USO",
        "cper":  "CPER",   # copper ETF (US Copper Index Fund)
        "corn":  "CORN",   # corn ETF (Teucrium Corn Fund)
        "soyb":  "SOYB",   # soybean ETF (Teucrium Soybean Fund)
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

    # Compute yield curve slope; keep raw 2Y as an explicit rate feature
    if "dgs10" in combined.columns and "dgs2" in combined.columns:
        combined["yield_curve"] = combined["dgs10"] - combined["dgs2"]
        combined = combined.rename(columns={"dgs2": "dgs2"})  # noop, be explicit
    else:
        combined["yield_curve"] = np.nan

    # Drop 10Y level (captured in slope + 2Y)
    combined = combined.drop(columns=["dgs10"], errors="ignore")

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


# ---------------------------------------------------------------------------
# Global outer-layer (for hierarchical HMM global-outer variant)
# ---------------------------------------------------------------------------

# Publication-lag days applied before forward-filling onto the daily grid.
# Conservative (never earlier than real release). See module docstring.
PUB_LAG = {
    "credit_spread": 1,  # BAA10Y, daily, 1d safety margin
    "dxy":      0,    # market data, same-day
    "walcl":    7,    # weekly (Thu release for prior Wed)
    "rrp":      1,    # daily
    "tga":      1,    # daily
    "m2_us":    30,   # monthly, ~2-3 week lag + safety
    "m2_eu":    30,
    "m2_jp":    30,
    "m2_cn":    30,
}


def fetch_fred_with_lag(series_id: str, start: str, end: str, api_key: str,
                        lag_days: int) -> pd.Series:
    """Fetch a FRED series and shift the index forward by `lag_days` so that
    the value on date T is one that was actually published by date T."""
    s = fetch_fred(series_id, start, end, api_key)
    if len(s) == 0:
        return s
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s


def build_regime_global_outer(start: str, end: str, fred_key: str) -> pd.DataFrame:
    """Build the 4-feature global outer panel on a daily grid.

    Features:
      dxy              — DX-Y.NYB close (yfinance)
      credit_spread    — Moody's Baa corp yield − 10Y Treasury (FRED BAA10Y);
                         proxy for HY OAS (which FRED truncated in 2023)
      net_fed_liq_yoy  — YoY %Δ of (WALCL − RRPONTSYD − WTREGEN)
      global_m2_yoy    — equal-weight avg of YoY %Δ of US/EU/JP/CN M2
                         (local currency, no FX conversion)

    All series are placed on a daily UTC index with publication-lag safety
    applied, then forward-filled. YoY transforms are computed in the native
    frequency of the underlying series before placing on the daily grid.
    """
    if not fred_key:
        raise ValueError("FRED API key required for global outer panel.")

    # --- DXY (yfinance daily) ---
    print("  Fetching DXY (DX-Y.NYB) daily...")
    dxy_df = fetch_yfinance_daily("DX-Y.NYB", start, end)
    dxy = dxy_df["close"].rename("dxy") if "close" in dxy_df.columns else pd.Series(dtype=float, name="dxy")

    # --- Credit spread (FRED BAA10Y, daily) ---
    # Substitute for HY OAS: FRED truncated BAMLH0A0HYM2 to ~3y rolling window
    # in 2023 due to ICE/BofA licensing. BAA10Y (Moody's Baa − 10Y Treasury)
    # has daily coverage back to 1986 and captures the same credit-risk-premium
    # signal for regime detection (correlation ~0.9 with HY OAS during stress).
    print("  Fetching credit spread (BAA10Y) from FRED...")
    credit_spread = fetch_fred_with_lag("BAA10Y", start, end, fred_key,
                                        PUB_LAG["credit_spread"]).rename("credit_spread")

    # --- Net Fed liquidity = WALCL − RRPONTSYD − WTREGEN ---
    # Native frequency differs; align, ffill to daily, then YoY.
    print("  Fetching WALCL / RRPONTSYD / WTREGEN from FRED...")
    walcl = fetch_fred_with_lag("WALCL",      start, end, fred_key, PUB_LAG["walcl"])
    rrp   = fetch_fred_with_lag("RRPONTSYD",  start, end, fred_key, PUB_LAG["rrp"])
    tga   = fetch_fred_with_lag("WTREGEN",    start, end, fred_key, PUB_LAG["tga"])
    liq_parts = pd.concat([walcl.rename("walcl"),
                           rrp.rename("rrp"),
                           tga.rename("tga")], axis=1).sort_index().ffill()
    net_liq = (liq_parts["walcl"] - liq_parts["rrp"] - liq_parts["tga"]).rename("net_fed_liq")
    # YoY % change (252 trading days ≈ 365 calendar days; using 365d shift on
    # daily ffill'd series gives a true year-over-year).
    net_liq_daily = net_liq.asfreq("D").ffill()
    net_fed_liq_yoy = (net_liq_daily / net_liq_daily.shift(365) - 1.0).rename("net_fed_liq_yoy")

    # --- Global M2 (equal-weight YoY, local currency) ---
    print("  Fetching G4 M2 (US/EU/JP/CN) from FRED...")
    m2_codes = {
        "m2_us": "M2SL",
        "m2_eu": "MYAGM2EZM196N",
        "m2_jp": "MYAGM2JPM189S",
        "m2_cn": "MYAGM2CNM189N",
    }
    m2_series = {}
    for name, code in m2_codes.items():
        s = fetch_fred_with_lag(code, start, end, fred_key, PUB_LAG[name])
        # YoY on monthly frequency = shift(12)
        yoy = (s / s.shift(12) - 1.0).rename(f"{name}_yoy")
        m2_series[name] = yoy

    m2_df = pd.concat(m2_series.values(), axis=1).sort_index()
    # Equal-weight across the four countries. Skip-na so a single late-posting
    # series doesn't blank out the composite.
    global_m2_yoy = m2_df.mean(axis=1, skipna=True).rename("global_m2_yoy")

    # --- Assemble on a daily grid ---
    daily_idx = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    frames = []
    for s in (dxy, credit_spread, net_fed_liq_yoy, global_m2_yoy):
        sx = s.copy()
        if sx.index.tz is None:
            sx.index = pd.to_datetime(sx.index).tz_localize("UTC")
        sx = sx.reindex(daily_idx, method="ffill")
        frames.append(sx)
    out = pd.concat(frames, axis=1)
    out.index.name = "date_utc"
    return out


def main():
    p = argparse.ArgumentParser(description="Fetch regime features for v12")
    p.add_argument("--fred-key", type=str, default=None,
                   help="FRED API key. If omitted, FFR and yield curve will be NaN.")
    p.add_argument("--start", type=str, default="2016-01-01",
                   help="Start date (default: 2016-01-01)")
    p.add_argument("--end", type=str, default=None,
                   help="End date (default: today)")
    p.add_argument("--global-outer", action="store_true",
                   help="Also build regime_global_outer.csv (dxy/hy_oas/net_fed_liq_yoy/global_m2_yoy).")
    p.add_argument("--global-outer-start", type=str, default="2003-01-01",
                   help="Start date for global outer panel (needs longer history "
                        "for 3y rolling ranks). Default: 2003-01-01.")
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

    if args.global_outer:
        print("\n=== Fetching global outer panel ===")
        go = build_regime_global_outer(args.global_outer_start, end, args.fred_key)
        go_path = DATA_DIR / "regime_global_outer.csv"
        go.to_csv(go_path)
        print(f"  Wrote {go_path} ({len(go):,} rows)")
        print(f"  Range: {go.index[0]} → {go.index[-1]}")
        print(f"  Columns: {list(go.columns)}")
        print(f"  Nulls (head):\n{go.isnull().sum()}")


if __name__ == "__main__":
    main()
