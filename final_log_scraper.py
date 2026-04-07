import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timezone

# =================================================================
# 1. CONFIGURATION & PARAMETERS
# 설정 및 파라미터 정의
# =================================================================
SYMBOL = "BTC/USD"
TIMEFRAME = "4h"                      # e.g., '1h', '4h', '15m'
START_DATE = "2016-01-01T00:00:00Z"

# Lookback used to compute VP at each row (매물대 계산에 포함할 과거)
LOOKBACK_DAYS = 180                   # e.g., 180 days

# Relative VP settings (현재가격 중심 상대 VP 설정)
REL_SPAN_PCT = 0.25                   # +/- 25% around current close (상하 25%)
REL_BIN_COUNT = 50                    # number of bins (총 bin 개수)
REL_NORMALIZE = True                  # True: convert to distribution (sum=1)

# Exchanges to combine (통합할 거래소)
EXCHANGES = ["bitstamp", "coinbase"]  # can add more if needed

# Output files
SAVE_CSV = f"BTC_{TIMEFRAME}_RELVP.csv".replace("/", "_")
SAVE_META = f"BTC_{TIMEFRAME}_RELVP_metadata.json".replace("/", "_")

# =================================================================
# 2. DERIVED PARAMETERS (do not hand-edit)
# TIMEFRAME에서 자동 계산되는 파라미터 (수동 수정 금지)
# =================================================================
STEP_SECONDS = ccxt.Exchange.parse_timeframe(TIMEFRAME)     # e.g., '4h' -> 14400
STEP_MS = STEP_SECONDS * 1000
BARS_PER_DAY = int(round((24 * 3600) / STEP_SECONDS))       # e.g., 4h -> 6 bars/day
LOOKBACK_BARS = LOOKBACK_DAYS * BARS_PER_DAY                # e.g., 180 days -> 1080 bars (for 4h)

# 24h horizon in bars (라벨 생성 시 참고용: 24시간 = 몇 개 bar?)
HORIZON_24H_BARS = int(round((24 * 3600) / STEP_SECONDS))

# Column name for the current candle volume (현재 봉 거래량 컬럼명)
VOLUME_COL = f"volume_{TIMEFRAME}".replace("/", "_")

# Sanity check: timeframe should divide 24h evenly
if abs((BARS_PER_DAY * STEP_SECONDS) - (24 * 3600)) > 1:
    raise ValueError(
        f"TIMEFRAME='{TIMEFRAME}' does not evenly divide 24h. "
        f"BARS_PER_DAY={BARS_PER_DAY}, STEP_SECONDS={STEP_SECONDS}"
    )

# =================================================================
# 3. RELATIVE VP BIN EDGES (fixed semantics across rows)
# 상대 VP bin 경계(모든 행에서 의미 고정)
# -----------------------------------------------------------------
# We build bins on log-return axis: x = log(price / close_t)
# and x-range corresponds to +/- REL_SPAN_PCT in price space.
#
# lo = -log(1+span), hi = +log(1+span)
# so that exp(lo) - 1 = -span/(1+span) (approximately -span for small span)
# and exp(hi) - 1 = +span
# =================================================================
EPS = 1e-12
LO_LOG = -np.log(1.0 + REL_SPAN_PCT)
HI_LOG =  np.log(1.0 + REL_SPAN_PCT)
REL_EDGES_LOG = np.linspace(LO_LOG, HI_LOG, REL_BIN_COUNT + 1)          # length = bins+1
REL_EDGES_PCT = np.expm1(REL_EDGES_LOG)                                 # convert log edges to % edges

# Bin center pct (for interpretation/debug/plot; not used by model necessarily)
REL_CENTERS_LOG = 0.5 * (REL_EDGES_LOG[:-1] + REL_EDGES_LOG[1:])
REL_CENTERS_PCT = np.expm1(REL_CENTERS_LOG)

# =================================================================
# 4. DATA ACQUISITION
# 데이터 수집 함수
# =================================================================
def fetch_all_ohlcv(exchange_id: str, since_ms: int) -> pd.DataFrame:
    """
    Fetch historical OHLCV for a given exchange and return DataFrame.
    거래소별 과거 OHLCV 데이터를 수집하여 DataFrame으로 반환.
    """
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    all_batches = []
    current_since = since_ms

    print(f"[{exchange_id.upper()}] Collecting {SYMBOL} {TIMEFRAME} OHLCV ...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=current_since, limit=1000)
            if not ohlcv:
                break

            all_batches.append(ohlcv)
            current_since = ohlcv[-1][0] + STEP_MS  # advance by exactly 1 bar (TIMEFRAME 한 개)

            last_dt = pd.to_datetime(ohlcv[-1][0], unit="ms", utc=True)
            print(f" -> Progress: {last_dt}", end="\r")

            # stop condition: reached (approx) now
            if last_dt >= pd.Timestamp.now(tz="UTC"):
                break

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"\n[{exchange_id.upper()}] Error: {e}")
            break

    flat = [row for batch in all_batches for row in batch]
    df = pd.DataFrame(flat, columns=["ts", "open", "high", "low", "close", "vol"])

    # De-dup by timestamp just in case (가끔 중복이 섞일 수 있어 제거)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


# Parse start date to ms
start_ts = ccxt.Exchange().parse8601(START_DATE)

# Fetch from each exchange
dfs = {}
for ex_id in EXCHANGES:
    dfs[ex_id] = fetch_all_ohlcv(ex_id, start_ts)

print("\nMerging exchanges ...")

# =================================================================
# 5. MULTI-EXCHANGE MERGE (OHLCV)
# 다중 거래소 데이터 통합
# -----------------------------------------------------------------
# - Outer merge on timestamp
# - Price columns: average if both present, otherwise take available
# - Volume: sum
# =================================================================
# Start merge with the first exchange
ex0 = EXCHANGES[0]
m = dfs[ex0].copy()
m = m.rename(columns={c: f"{c}_{ex0}" for c in ["open", "high", "low", "close", "vol"]})

for ex_id in EXCHANGES[1:]:
    d = dfs[ex_id].copy()
    d = d.rename(columns={c: f"{c}_{ex_id}" for c in ["open", "high", "low", "close", "vol"]})
    m = pd.merge(m, d[["ts"] + [f"{c}_{ex_id}" for c in ["open", "high", "low", "close", "vol"]]],
                 on="ts", how="outer")

m = m.sort_values("ts").fillna(0)

# Integrate OHLC
for col in ["open", "high", "low", "close"]:
    cols = [f"{col}_{ex}" for ex in EXCHANGES]
    # average across non-zero contributors
    vals = m[cols].values
    nonzero = (vals != 0).astype(float)
    denom = nonzero.sum(axis=1)
    # avoid divide by zero
    avg = np.where(denom > 0, (vals * nonzero).sum(axis=1) / denom, 0)
    m[col] = avg

# Integrate volume (sum across exchanges)
vol_cols = [f"vol_{ex}" for ex in EXCHANGES]
m["vol"] = m[vol_cols].sum(axis=1)

# Date column
m["date"] = pd.to_datetime(m["ts"], unit="ms", utc=True)

combined = m[["ts", "date", "open", "high", "low", "close", "vol"]].copy()

# Remove any rows with missing prices (close==0 implies no usable data)
combined = combined[combined["close"] != 0].reset_index(drop=True)

print(f"Combined rows: {len(combined)}")

# =================================================================
# 6. RELATIVE VOLUME PROFILE GENERATION
# 현재가격 중심 상대 VP 생성
# -----------------------------------------------------------------
# For each time index i:
# - hist_window = past LOOKBACK_BARS bars (including current bar)
# - define x_j = log(close_j / close_i)
# - histogram x_j into fixed log edges (REL_EDGES_LOG)
# - weight by volume_j
# - optional normalization to distribution (sum=1)
# Output: vp_rel_0 ... vp_rel_{REL_BIN_COUNT-1}
# =================================================================
vp_col_names = [f"vp_rel_{k:02d}" for k in range(REL_BIN_COUNT)]

final_rows = []

print(
    f"Generating Relative VP: span=±{int(REL_SPAN_PCT*100)}%, bins={REL_BIN_COUNT}, "
    f"lookback={LOOKBACK_DAYS} days ({LOOKBACK_BARS} bars) ..."
)

for i in range(len(combined)):
    if i < LOOKBACK_BARS:
        continue

    # Slice lookback window (과거 LOOKBACK_BARS 구간)
    hist = combined.iloc[i - LOOKBACK_BARS : i + 1]
    close_t = combined.at[i, "close"]

    # Relative log distance of each historical close vs current close
    # x = log(price / close_t)
    x = np.log(hist["close"].values / close_t)

    # Weighted histogram using fixed edges (고정된 상대축 bin)
    vp, _ = np.histogram(x, bins=REL_EDGES_LOG, weights=hist["vol"].values)

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

    # Add VP vector as flat numeric columns (PyTorch friendly)
    for k, name in enumerate(vp_col_names):
        row[name] = float(vp[k])

    final_rows.append(row)

    if i % 2000 == 0:
        print(f" -> Processing: {i}/{len(combined)}", end="\r")

final_df = pd.DataFrame(final_rows)

# =================================================================
# 7. EXPORT CSV + METADATA
# CSV 및 메타데이터(JSON) 저장
# =================================================================
final_df.to_csv(SAVE_CSV, index=False)

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
            "vp_rel_* columns represent weighted histogram of log(close_j/close_t) over lookback window, "
            "weighted by volume. If normalize=True, each row sums to 1 (distribution)."
        ),
    },
    "output": {
        "csv": SAVE_CSV,
        "metadata_json": SAVE_META,
        "columns": final_df.columns.tolist(),
        "vp_columns": vp_col_names,
        "volume_column": VOLUME_COL,
    }
}

with open(SAVE_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nDone.")
print(f"Saved CSV: {SAVE_CSV}")
print(f"Saved metadata: {SAVE_META}")
print(f"Final shape: {final_df.shape}")
