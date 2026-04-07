import numpy as np
import ccxt
from pathlib import Path

# =================================================================
# Project paths
# =================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# =================================================================
# Data pipeline configuration
# =================================================================
SYMBOL = "BTC/USD"
TIMEFRAME = "4h"
START_DATE = "2016-01-01T00:00:00Z"

# Exchanges to merge
EXCHANGES = ["bitstamp", "coinbase"]

# =================================================================
# Derived time parameters (from TIMEFRAME)
# =================================================================
STEP_SECONDS = ccxt.Exchange.parse_timeframe(TIMEFRAME)
STEP_MS = STEP_SECONDS * 1000
BARS_PER_DAY = int(round((24 * 3600) / STEP_SECONDS))

# Sanity check
if abs((BARS_PER_DAY * STEP_SECONDS) - (24 * 3600)) > 1:
    raise ValueError(
        f"TIMEFRAME='{TIMEFRAME}' does not evenly divide 24h. "
        f"BARS_PER_DAY={BARS_PER_DAY}, STEP_SECONDS={STEP_SECONDS}"
    )

# 24h horizon in bars
HORIZON_24H_BARS = BARS_PER_DAY

# =================================================================
# Volume Profile parameters
# =================================================================
LOOKBACK_DAYS = 180
LOOKBACK_BARS = LOOKBACK_DAYS * BARS_PER_DAY

REL_SPAN_PCT = 0.25       # +/- 25% around current close
REL_BIN_COUNT = 50
REL_NORMALIZE = True

# VP bin edges on log-return axis
EPS = 1e-12
LO_LOG = -np.log(1.0 + REL_SPAN_PCT)
HI_LOG = np.log(1.0 + REL_SPAN_PCT)
REL_EDGES_LOG = np.linspace(LO_LOG, HI_LOG, REL_BIN_COUNT + 1)
REL_EDGES_PCT = np.expm1(REL_EDGES_LOG)
REL_CENTERS_LOG = 0.5 * (REL_EDGES_LOG[:-1] + REL_EDGES_LOG[1:])
REL_CENTERS_PCT = np.expm1(REL_CENTERS_LOG)

# =================================================================
# Output file naming
# =================================================================
VOLUME_COL = f"volume_{TIMEFRAME}".replace("/", "_")
VP_COL_NAMES = [f"vp_rel_{k:02d}" for k in range(REL_BIN_COUNT)]

def output_csv_name() -> str:
    return f"BTC_{TIMEFRAME}_RELVP.csv".replace("/", "_")

def output_meta_name() -> str:
    return f"BTC_{TIMEFRAME}_RELVP_metadata.json".replace("/", "_")

# =================================================================
# Feature engineering configuration
# =================================================================
LOOKBACK_BARS_MODEL = 42                    # Model input window (42 bars = 1 week at 4h timeframe)
VOLUME_ROLL_WINDOW_DAYS = 30                # Rolling window for volume normalization
VOLUME_ROLL_WINDOW_BARS = VOLUME_ROLL_WINDOW_DAYS * BARS_PER_DAY

# Derived feature column names
DERIVED_FEATURE_COLS = ["log_return", "bar_range", "bar_body", "volume_ratio"]
FEATURE_COLS = VP_COL_NAMES + DERIVED_FEATURE_COLS

# =================================================================
# Train / Validation / Test split dates
# =================================================================
TRAIN_END = "2023-01-01T00:00:00+00:00"
VAL_END = "2024-01-01T00:00:00+00:00"

# =================================================================
# Label configuration
# =================================================================
LABEL_HORIZON_BARS = BARS_PER_DAY           # 6 bars = 24h lookahead for label

# =================================================================
# RNN model configuration
# =================================================================
RNN_HIDDEN_SIZES = [64, 32, 16]             # Hidden nodes per layer (decreasing)
RNN_DROPOUT = 0.2                           # Dropout between RNN layers
RNN_ACTIVATION = "tanh"                     # Activation function: "tanh" or "relu"

# =================================================================
# Training configuration
# =================================================================
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10                    # Stop if val loss doesn't improve for N epochs
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
