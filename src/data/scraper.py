"""
Multi-exchange OHLCV data fetcher and merger.

Fetches historical BTC/USD candles from multiple exchanges via ccxt,
then merges them into a single unified price series.
"""

import time
import ccxt
import pandas as pd
import numpy as np

from src.config import (
    SYMBOL, TIMEFRAME, START_DATE, EXCHANGES,
    STEP_MS, STEP_SECONDS,
)


def fetch_all_ohlcv(exchange_id: str, since_ms: int) -> pd.DataFrame:
    """
    Fetch all historical OHLCV data from a single exchange.

    Args:
        exchange_id: ccxt exchange id (e.g., "bitstamp", "coinbase")
        since_ms: start timestamp in milliseconds

    Returns:
        DataFrame with columns [ts, open, high, low, close, vol]
    """
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    all_batches = []
    current_since = since_ms

    print(f"[{exchange_id.upper()}] Collecting {SYMBOL} {TIMEFRAME} OHLCV ...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                SYMBOL, timeframe=TIMEFRAME, since=current_since, limit=1000
            )
            if not ohlcv:
                break

            all_batches.append(ohlcv)
            current_since = ohlcv[-1][0] + STEP_MS

            last_dt = pd.to_datetime(ohlcv[-1][0], unit="ms", utc=True)
            print(f"  -> Progress: {last_dt}", end="\r")

            if last_dt >= pd.Timestamp.now(tz="UTC"):
                break

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"\n[{exchange_id.upper()}] Error: {e}")
            break

    flat = [row for batch in all_batches for row in batch]
    df = pd.DataFrame(flat, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def merge_exchanges(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge OHLCV DataFrames from multiple exchanges.

    Prices: averaged across exchanges (non-zero only).
    Volume: summed across exchanges.

    Args:
        dfs: dict mapping exchange_id -> OHLCV DataFrame

    Returns:
        Merged DataFrame with columns [ts, date, open, high, low, close, vol]
    """
    exchange_ids = list(dfs.keys())
    ex0 = exchange_ids[0]

    m = dfs[ex0].copy()
    m = m.rename(columns={c: f"{c}_{ex0}" for c in ["open", "high", "low", "close", "vol"]})

    for ex_id in exchange_ids[1:]:
        d = dfs[ex_id].copy()
        d = d.rename(columns={c: f"{c}_{ex_id}" for c in ["open", "high", "low", "close", "vol"]})
        merge_cols = ["ts"] + [f"{c}_{ex_id}" for c in ["open", "high", "low", "close", "vol"]]
        m = pd.merge(m, d[merge_cols], on="ts", how="outer")

    m = m.sort_values("ts").fillna(0)

    # Average prices across non-zero contributors
    for col in ["open", "high", "low", "close"]:
        cols = [f"{col}_{ex}" for ex in exchange_ids]
        vals = m[cols].values
        nonzero = (vals != 0).astype(float)
        denom = nonzero.sum(axis=1)
        avg = np.where(denom > 0, (vals * nonzero).sum(axis=1) / denom, 0)
        m[col] = avg

    # Sum volumes
    vol_cols = [f"vol_{ex}" for ex in exchange_ids]
    m["vol"] = m[vol_cols].sum(axis=1)

    m["date"] = pd.to_datetime(m["ts"], unit="ms", utc=True)

    combined = m[["ts", "date", "open", "high", "low", "close", "vol"]].copy()
    combined = combined[combined["close"] != 0].reset_index(drop=True)

    return combined


def fetch_and_merge() -> pd.DataFrame:
    """
    Full pipeline: fetch from all configured exchanges and merge.

    Returns:
        Merged OHLCV DataFrame.
    """
    start_ts = ccxt.Exchange().parse8601(START_DATE)

    dfs = {}
    for ex_id in EXCHANGES:
        dfs[ex_id] = fetch_all_ohlcv(ex_id, start_ts)

    print("\nMerging exchanges ...")
    combined = merge_exchanges(dfs)
    print(f"Combined rows: {len(combined)}")

    return combined
