"""Absolute (visible-range) VP pipeline at 15m resolution, 30-day window.

Writes `BTC_15m_ABSVP_30d.csv` — the feature source for the v11 model.

── What changes vs the existing relative-VP pipeline ─────────────────────
1. TIMEFRAME patched to "15m" (vs default "1h").
2. VP axis is ABSOLUTE PRICE, not log-relative-to-close. For each bar t,
   bins are placed linearly between the 30-day trailing low/high (wicks,
   not closes). Same 50 bins, still volume-weighted, still normalized.
3. New 50-bin "self" channel (hard one-hot) marks the bin that the
   current close falls into. This gives the spatial transformer the
   positional anchor that was implicit (always bin 25) under the old
   relative VP.
4. Two new per-row scalars:
     - `price_pos`  = (close - lo) / (hi - lo)   ∈ [0, 1]  (precise)
     - `range_pct`  = (hi - lo) / close                    (regime width)

── Why ──────────────────────────────────────────────────────────────────
Matches how the user reads Kraken's visible-range VP: the 50 bins span
the actual 30-day high→low range, not a fixed ±25% window centred on
current price. Current price position is read at full precision via the
scalar, while the hard one-hot channel gives the spatial attention a
geometric anchor to reason about "N bins away from me". See the v11
design discussion (2026-04-13) and `arch-data-pipeline.md` for context.

Usage:
    python3 -m src.data.compute_absvp_15m_30d
"""

import json
import sys
import time

import numpy as np
import pandas as pd

import src.config as cfg

# ── Patch cfg BEFORE any downstream module captures values ──
cfg.TIMEFRAME = "15m"
import ccxt  # noqa: E402
cfg.STEP_SECONDS = ccxt.Exchange.parse_timeframe(cfg.TIMEFRAME)
cfg.STEP_MS = cfg.STEP_SECONDS * 1000
cfg.BARS_PER_DAY = int(round((24 * 3600) / cfg.STEP_SECONDS))  # 96
cfg.LOOKBACK_DAYS = 30
cfg.LOOKBACK_BARS = cfg.LOOKBACK_DAYS * cfg.BARS_PER_DAY       # 2880

# Deferred — these modules read cfg at import time.
from src.data.scraper import fetch_and_merge  # noqa: E402


N_BINS = 50
LOOKBACK_BARS = cfg.LOOKBACK_BARS  # 2880 bars = 30 days @ 15m

OUTPUT_CSV = "BTC_15m_ABSVP_30d.csv"
OUTPUT_META = "BTC_15m_ABSVP_30d_metadata.json"


def compute_absolute_vp(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute-range VP + self-channel + position scalars.

    For each row i where i >= LOOKBACK_BARS:
      - window = rows [i - LOOKBACK_BARS, i] inclusive (2,881 bars)
      - lo = min(low)  over window
      - hi = max(high) over window
      - 50 linear bins between lo and hi
      - vp_abs_k = normalized volume traded at close falling in bin k
      - self_k   = 1 if closes[i] is in bin k, else 0
      - price_pos = (closes[i] - lo) / (hi - lo)
      - range_pct = (hi - lo) / closes[i]

    Degenerate windows (hi == lo) are dropped.
    """
    n = len(combined)
    print(
        f"Computing absolute VP: {N_BINS} bins, {cfg.LOOKBACK_DAYS}-day window "
        f"({LOOKBACK_BARS} bars @ 15m), {n:,} total rows"
    )

    highs = combined["high"].values.astype(np.float64)
    lows = combined["low"].values.astype(np.float64)
    closes = combined["close"].values.astype(np.float64)
    vols = combined["vol"].values.astype(np.float64)

    # Rolling max-of-high and min-of-low over window of size LOOKBACK_BARS + 1.
    # Row i sees bars [i - LOOKBACK_BARS, i] inclusive → window size = LB + 1.
    win = LOOKBACK_BARS + 1
    hi_series = pd.Series(highs).rolling(win, min_periods=win).max().values
    lo_series = pd.Series(lows).rolling(win, min_periods=win).min().values

    out_rows = []
    t0 = time.time()
    start = LOOKBACK_BARS
    for i in range(start, n):
        hi = hi_series[i]
        lo = lo_series[i]
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            continue

        # Slice the window: histogram of closes weighted by volume.
        w_close = closes[i - LOOKBACK_BARS : i + 1]
        w_vol = vols[i - LOOKBACK_BARS : i + 1]

        edges = np.linspace(lo, hi, N_BINS + 1)
        vp, _ = np.histogram(w_close, bins=edges, weights=w_vol)
        s = vp.sum()
        if s > 0:
            vp = vp / s
        else:
            vp = np.zeros(N_BINS, dtype=np.float64)

        # Self channel: hard one-hot on bin containing closes[i].
        close_t = closes[i]
        pos_norm = (close_t - lo) / (hi - lo)
        bin_idx = int(np.clip(np.floor(pos_norm * N_BINS), 0, N_BINS - 1))
        self_ch = np.zeros(N_BINS, dtype=np.float64)
        self_ch[bin_idx] = 1.0

        price_pos = float(pos_norm)
        range_pct = float((hi - lo) / close_t)

        curr = combined.iloc[i]
        row = {
            "ts": int(curr["ts"]),
            "date": curr["date"].isoformat(),
            "open": float(curr["open"]),
            "high": float(curr["high"]),
            "low": float(curr["low"]),
            "close": float(curr["close"]),
            "volume_15m": float(curr["vol"]),
            "window_lo": float(lo),
            "window_hi": float(hi),
            "price_pos": price_pos,
            "range_pct": range_pct,
        }
        for k in range(N_BINS):
            row[f"vp_abs_{k:02d}"] = float(vp[k])
        for k in range(N_BINS):
            row[f"self_{k:02d}"] = float(self_ch[k])
        out_rows.append(row)

        if (i - start) % 5000 == 0 and i > start:
            elapsed = time.time() - t0
            done = i - start
            total = n - start
            eta = elapsed * (total - done) / done
            print(
                f"  -> {done:,}/{total:,} rows  "
                f"({100*done/total:.1f}%)  eta {eta/60:.1f} min",
                end="\r",
            )

    print()
    return pd.DataFrame(out_rows)


def main():
    print(
        f"Patched cfg: TIMEFRAME={cfg.TIMEFRAME}, BARS_PER_DAY={cfg.BARS_PER_DAY}, "
        f"LOOKBACK_BARS={cfg.LOOKBACK_BARS}"
    )

    combined = fetch_and_merge()
    print(f"Merged OHLCV rows: {len(combined):,}")
    print(f"  range: {combined['date'].iloc[0]} → {combined['date'].iloc[-1]}")

    df = compute_absolute_vp(combined)
    print(f"Output rows after dropping warmup/degenerate: {len(df):,}")

    csv_path = cfg.PROJECT_ROOT / OUTPUT_CSV
    meta_path = cfg.PROJECT_ROOT / OUTPUT_META
    df.to_csv(csv_path, index=False)

    vp_cols = [f"vp_abs_{k:02d}" for k in range(N_BINS)]
    self_cols = [f"self_{k:02d}" for k in range(N_BINS)]
    meta = {
        "symbol": cfg.SYMBOL,
        "timeframe": cfg.TIMEFRAME,
        "start_date": cfg.START_DATE,
        "exchanges": cfg.EXCHANGES,
        "lookback_days": cfg.LOOKBACK_DAYS,
        "lookback_bars": LOOKBACK_BARS,
        "bars_per_day": cfg.BARS_PER_DAY,
        "absolute_vp": {
            "bin_count": N_BINS,
            "range_source": "high/low wicks over rolling window",
            "axis": "absolute price, linear",
            "normalize": True,
            "self_channel": "hard one-hot on bin containing closes[i]",
            "scalars": ["price_pos (close − lo)/(hi − lo)", "range_pct (hi − lo)/close"],
        },
        "notes": (
            "Visible-range VP pipeline for v11. Bins span the actual "
            "30-day high/low range, not a fixed ±25% window. The self "
            "channel + price_pos scalar together replace the implicit "
            "'current price = bin 25' anchor used by the relative VP."
        ),
        "output": {
            "csv": OUTPUT_CSV,
            "rows": int(len(df)),
            "vp_columns": vp_cols,
            "self_columns": self_cols,
            "scalar_columns": ["price_pos", "range_pct"],
            "range_columns": ["window_lo", "window_hi"],
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {csv_path} ({len(df):,} rows)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    sys.exit(main())
