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
TIMEFRAME = "1h"
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
LOOKBACK_BARS_MODEL = 720                   # Model input window (720 bars = 30 days at 1h timeframe)
VOLUME_ROLL_WINDOW_DAYS = 30                # Rolling window for volume normalization
VOLUME_ROLL_WINDOW_BARS = VOLUME_ROLL_WINDOW_DAYS * BARS_PER_DAY

# Derived feature column names
DERIVED_FEATURE_COLS = [
    "log_return", "bar_range", "bar_body", "volume_ratio",
    "upper_wick", "lower_wick", "body_dir",
]
VP_STRUCTURE_COLS = [
    "vp_ceiling_dist",
    "vp_floor_dist",
    "vp_num_peaks",
    "vp_ceiling_strength",
    "vp_floor_strength",
    "vp_ceiling_consistency",
    "vp_floor_consistency",
    "vp_mid_range",
]
FEATURE_COLS = VP_COL_NAMES + DERIVED_FEATURE_COLS + VP_STRUCTURE_COLS

# =================================================================
# Train / Validation / Test split dates
# =================================================================
TRAIN_END = "2023-01-01T00:00:00+00:00"
VAL_END = "2024-01-01T00:00:00+00:00"

# =================================================================
# Label configuration
# =================================================================
LABEL_HORIZON_BARS = BARS_PER_DAY           # 6 bars = 24h lookahead (legacy, used as fallback)
LABEL_MODE = "first_hit"                    # "fixed_horizon" or "first_hit"
LABEL_TP_PCT = 0.025                        # Take-profit base (bull): 2.5%
LABEL_SL_PCT = 0.05                         # Stop-loss base (bull): 5%
LABEL_REGIME_ADAPTIVE = True                # Flip TP/SL ratio in bear markets
LABEL_REGIME_MODE = "sma"                   # "sma" = SMA-based, "fgi" = Fear & Greed Index
LABEL_REGIME_SMA_BARS = 90 * BARS_PER_DAY  # 90-day SMA for bull/bear detection
LABEL_FGI_THRESHOLD = 50                    # FGI >= threshold = bull, < threshold = bear
LABEL_FGI_PATH = DATA_DIR / "fear_greed_index.csv"
LABEL_NEUTRAL_MODE = "off"                  # "off" = no neutral filter, "symmetric" or "skip"
LABEL_NEUTRAL_PEAKS_THRESHOLD = 0           # num_peaks <= this = neutral (no VP structure)
LABEL_MAX_BARS = BARS_PER_DAY * 14          # Max lookahead: 14 days (84 bars)

# =================================================================
# RNN model configuration
# =================================================================
RNN_HIDDEN_SIZES = [64, 32, 16]             # Hidden nodes per layer (decreasing)
RNN_DROPOUT = 0.2                           # Dropout between RNN layers
RNN_ACTIVATION = "tanh"                     # Activation function: "tanh" or "relu"

# =================================================================
# LSTM model configuration
# =================================================================
LSTM_HIDDEN_SIZES = [8]                     # Single layer, small param count
LSTM_DROPOUT = 0.0                          # No dropout — model is underfitting

# =================================================================
# CNN model configuration
# =================================================================
CNN_CHANNELS = [8, 8]                       # Conv layer output channels
CNN_KERNEL_SIZE = 7                         # Conv kernel width (spans 7 VP bins)
CNN_FC_SIZE = 32                            # FC layer after conv + derived features
CNN_DROPOUT = 0.2                           # Moderate regularization

# =================================================================
# Training configuration
# =================================================================
LEARNING_RATE = 5e-4
EPOCHS = 50
BATCH_SIZE = 64
DATALOADER_WORKERS = 2                      # Parallel data loading threads
EARLY_STOP_PATIENCE = 15                    # Stop if val loss doesn't improve for N epochs
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
