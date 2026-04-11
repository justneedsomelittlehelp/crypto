"""Funding rate data: merge Binance historical CSV + OKX API for gap fill.

Coverage:
  - Binance CSV (GitHub): 2020-01-01 → 2023-12-31 (pre-downloaded)
  - OKX API: 2024-01-01 → present (fetched on demand)

Output: Single CSV with columns [date, funding_rate] at 8h resolution,
forward-filled to match OHLCV bar frequency (1h or 15m).

Usage:
    from src.data.funding_rate import load_funding_rate
    fr = load_funding_rate()  # Returns DataFrame with [date, funding_rate]
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.request import urlopen, Request

from src.config import PROJECT_ROOT, DATA_DIR


BINANCE_CSV = DATA_DIR / "funding_rate_binance_2020_2024.csv"
GATE_CSV = DATA_DIR / "funding_rate_gate_2024_present.csv"
MERGED_CSV = DATA_DIR / "funding_rate_merged.csv"


def _load_binance_csv() -> pd.DataFrame:
    """Load pre-downloaded Binance funding rate CSV."""
    if not BINANCE_CSV.exists():
        raise FileNotFoundError(
            f"{BINANCE_CSV} not found. Download from:\n"
            "https://github.com/supervik/historical-funding-rates-fetcher/tree/main/data/BTC-USDT"
        )
    df = pd.read_csv(BINANCE_CSV)
    df = df.rename(columns={"Date": "date", "Funding Rate": "funding_rate"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "funding_rate"]].sort_values("date").reset_index(drop=True)
    return df


def _fetch_okx_funding(start_ts_ms: int = None) -> pd.DataFrame:
    """Fetch funding rate history from OKX public API.

    Paginates backward from most recent. Stops when reaching start_ts_ms
    or when API returns no more data.
    """
    base_url = "https://www.okx.com/api/v5/public/funding-rate-history"
    all_rows = []
    after = ""  # pagination cursor

    print("Fetching OKX funding rate history...")

    while True:
        url = f"{base_url}?instId=BTC-USDT-SWAP&limit=100"
        if after:
            url += f"&after={after}"

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  OKX API error: {e}")
            break

        if data["code"] != "0" or not data["data"]:
            break

        for item in data["data"]:
            ts = int(item["fundingTime"])
            rate = float(item["realizedRate"])
            all_rows.append({"date": pd.Timestamp(ts, unit="ms", tz="UTC"), "funding_rate": rate})

        # Pagination: use oldest timestamp as cursor
        oldest_ts = min(int(item["fundingTime"]) for item in data["data"])
        after = str(oldest_ts)

        if start_ts_ms and oldest_ts <= start_ts_ms:
            break

        last_date = pd.Timestamp(oldest_ts, unit="ms", tz="UTC")
        print(f"  → {last_date.strftime('%Y-%m-%d')}", end="\r")

        time.sleep(0.2)  # Rate limit

    print(f"\n  Fetched {len(all_rows)} funding rate records from OKX")

    if not all_rows:
        return pd.DataFrame(columns=["date", "funding_rate"])

    df = pd.DataFrame(all_rows)
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def _load_gate_csv() -> pd.DataFrame:
    """Load Gate.io funding rate CSV (2024-present)."""
    if not GATE_CSV.exists():
        raise FileNotFoundError(
            f"{GATE_CSV} not found. Fetch via Gate.io API or re-run scraper."
        )
    df = pd.read_csv(GATE_CSV)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "funding_rate"]].sort_values("date").reset_index(drop=True)
    return df


def build_merged_funding_rate(force_refresh: bool = False) -> pd.DataFrame:
    """Build merged funding rate from Binance CSV (2020-2024) + Gate.io CSV (2024-present).

    Caches result to MERGED_CSV. Use force_refresh=True to rebuild.
    """
    if MERGED_CSV.exists() and not force_refresh:
        print(f"Loading cached funding rate from {MERGED_CSV}")
        df = pd.read_csv(MERGED_CSV)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)
        return df

    # Load Binance historical (2020 → 2023-12-31)
    print("Loading Binance funding rate CSV...")
    binance = _load_binance_csv()
    print(f"  Binance: {len(binance)} records, {binance['date'].min()} → {binance['date'].max()}")

    # Load Gate.io (2024-01-01 → present)
    print("Loading Gate.io funding rate CSV...")
    gate = _load_gate_csv()
    print(f"  Gate.io: {len(gate)} records, {gate['date'].min()} → {gate['date'].max()}")

    # Only keep Gate.io data after Binance ends (avoid overlap)
    gate = gate[gate["date"] > binance["date"].max()]
    merged = pd.concat([binance, gate], ignore_index=True)

    merged = merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    print(f"  Merged: {len(merged)} records, {merged['date'].min()} → {merged['date'].max()}")

    # Cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)
    print(f"  Saved to {MERGED_CSV}")

    return merged


def load_funding_rate(ohlcv_dates: pd.Series = None) -> pd.DataFrame:
    """Load funding rate and optionally align to OHLCV bar dates.

    If ohlcv_dates is provided, forward-fills 8h funding rate to match
    bar frequency and computes derived features.

    Returns DataFrame with columns:
        - date
        - funding_rate (raw 8h rate, forward-filled)
        - funding_zscore (30-day rolling z-score)
        - funding_trend (24h change in funding rate)
    """
    fr = build_merged_funding_rate()

    if ohlcv_dates is None:
        return fr

    # Align to OHLCV dates via forward-fill
    ohlcv_df = pd.DataFrame({"date": pd.to_datetime(ohlcv_dates, utc=True)})
    fr_indexed = fr.set_index("date").sort_index()

    # Reindex to OHLCV dates, forward-fill (8h rate applies until next update)
    combined_idx = ohlcv_df["date"].sort_values().drop_duplicates()
    aligned = fr_indexed.reindex(combined_idx, method="ffill")

    # Merge back
    result = ohlcv_df.merge(aligned, left_on="date", right_index=True, how="left")

    # Derived features
    # Z-score: how extreme is current funding vs 30-day history
    # At 8h resolution, 30 days = 90 funding rate observations
    # But we're at bar resolution now, so use expanding window in bar count
    bars_30d = 90 * 3  # approximate: 90 funding periods × 3 bars per period at 1h (8h/1h ≈ 8, but ffilled)
    roll_mean = result["funding_rate"].expanding(min_periods=30).mean()
    roll_std = result["funding_rate"].expanding(min_periods=30).std()
    result["funding_zscore"] = ((result["funding_rate"] - roll_mean) / roll_std.clip(lower=1e-10))
    result["funding_zscore"] = result["funding_zscore"].clip(-5, 5)  # Cap extremes

    # Trend: change over 24h (in bar units — caller should adjust if not 1h)
    result["funding_trend"] = result["funding_rate"] - result["funding_rate"].shift(24)

    # Fill NaN for pre-funding era with 0 (neutral)
    for col in ["funding_rate", "funding_zscore", "funding_trend"]:
        result[col] = result[col].fillna(0.0)

    return result[["date", "funding_rate", "funding_zscore", "funding_trend"]]
