# Reference

> **Read this when:** looking up file locations, data schema, configuration parameters, or Kraken API details.

---

## File Index

### Existing files (Phase 1 complete)
| Path | Purpose |
|------|---------|
| `src/config.py` | Centralized configuration — all parameters in one place |
| `src/data/__main__.py` | CLI entry point: `python3 -m src.data scrape\|validate` |
| `src/data/scraper.py` | Multi-exchange OHLCV fetching + merge |
| `src/data/volume_profile.py` | Relative VP computation + CSV/JSON export |
| `src/data/validator.py` | Data quality checks (gaps, prices, VP normalization) |
| `requirements.txt` | Pinned dependencies (ccxt, pandas, numpy, torch, python-dotenv) |
| `.env.example` | API key template |
| `final_log_scraper.py` | Original monolithic script (kept for reference) |
| `BTC_4h_RELVP.csv` | Generated feature data — 21,066 rows, 2016-06 to 2026-02 |
| `BTC_4h_RELVP_metadata.json` | Metadata: parameters, column names, bin edges |

### Phase 2 files (complete)
| Path | Purpose |
|------|---------|
| `src/features/__main__.py` | CLI: `python3 -m src.features prepare\|check` |
| `src/features/pipeline.py` | Load CSV, compute derived features (log_return, bar_range, bar_body, volume_ratio) |
| `src/features/dataset.py` | TimeSeriesDataset (configurable lookback), time-based splits, label generation |

### Phase 3 files (in progress)
| Path | Purpose |
|------|---------|
| `src/models/__main__.py` | CLI: `python3 -m src.models train --model rnn\|lstm\|cnn` |
| `src/models/architecture.py` | RNNClassifier, LSTMClassifier, CNNClassifier |
| `src/models/trainer.py` | Training loop with early stopping, checkpointing, weighted loss |
| `src/models/evaluate.py` | Accuracy, precision, recall, F1, confusion matrix |
| `src/models/walk_forward.py` | Walk-forward retraining (10 folds, 2020-2025) |
| `src/models/rule_based.py` | Rule-based strategy backtest (no ML) |
| `experiments/STRATEGY.md` | User's trading strategy and model iteration reasoning |
| `experiments/EVAL_VANILLA_RNN.md` | Vanilla RNN eval log (4 evals, exhausted) |
| `experiments/EVAL_LSTM.md` | LSTM eval log (7 evals, exhausted) |
| `experiments/EVAL_CNN.md` | **1D CNN eval log (26 evals, best: Eval 23 at 61.5%)** |
| `experiments/EVAL_TRANSFORMER.md` | **Transformer eval log (4 evals, best: 63.3% on 1h data)** |
| `experiments/RUN_INDEX.md` | Maps every run_* folder to its eval number |

### Planned files (future phases)
| Path | Phase | Purpose |
|------|-------|---------|
| `src/models/backtest.py` | 3 | Backtesting framework |
| `src/trading/kraken_client.py` | 4 | Kraken API wrapper |
| `src/trading/data_feed.py` | 4 | Real-time data feed |
| `src/trading/order_manager.py` | 4 | Order lifecycle management |
| `src/trading/strategy.py` | 5 | Trading strategy (user-defined logic) |
| `src/trading/position_manager.py` | 5 | Position tracking |
| `src/trading/risk_manager.py` | 5 | Risk checks and circuit breakers |
| `src/trading/bot.py` | 5 | Main execution loop |

## Data Schema

### BTC_4h_RELVP.csv columns
| Column | Type | Description |
|--------|------|-------------|
| `ts` | int | Unix timestamp (ms) |
| `date` | string | ISO 8601 datetime (UTC) |
| `open` | float | Merged open price |
| `high` | float | Merged high price |
| `low` | float | Merged low price |
| `close` | float | Merged close price |
| `volume_4h` | float | Merged volume (sum across exchanges) |
| `vp_rel_00`..`vp_rel_49` | float | Relative VP histogram bins (sum ≈ 1.0) |

### VP bin mapping
- Bin 00 = −25% from current price (lowest)
- Bin 25 = current price (center)
- Bin 49 = +25% from current price (highest)
- Axis: `log(close_j / close_t)`, linearly spaced in log space

## Configuration Parameters

### Data pipeline
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USD` | Trading pair |
| `TIMEFRAME` | `4h` | Candle interval |
| `START_DATE` | `2016-01-01` | Data collection start |
| `EXCHANGES` | `[bitstamp, coinbase]` | Data source exchanges |
| `LOOKBACK_DAYS` | 180 | VP computation lookback |
| `REL_SPAN_PCT` | 0.25 | VP price range (±25%) |
| `REL_BIN_COUNT` | 50 | VP histogram bins |

### Feature engineering
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOOKBACK_BARS_MODEL` | 720 | Model input window (720 bars = 30 days at 1h) |
| `VOLUME_ROLL_WINDOW_DAYS` | 30 | Rolling volume normalization window |
| `TRAIN_END` | `2023-01-01` | Train/val split boundary |
| `VAL_END` | `2024-01-01` | Val/test split boundary |
| `LABEL_HORIZON_BARS` | 6 | Lookahead for label (24h) |
| `VP_STRUCTURE_COLS` | 7 cols | VP ceiling/floor dist, strength, consistency, num_peaks |

### RNN model
| Parameter | Default | Description |
|-----------|---------|-------------|
| `RNN_HIDDEN_SIZES` | `[64, 32, 16]` | Hidden nodes per layer (decreasing) |
| `RNN_DROPOUT` | 0.2 | Dropout between layers |
| `RNN_ACTIVATION` | `tanh` | Activation function |

### LSTM model
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LSTM_HIDDEN_SIZES` | `[8]` | Hidden nodes per layer |
| `LSTM_DROPOUT` | 0.0 | Dropout between layers |

### Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `EPOCHS` | 50 | Max training epochs |
| `BATCH_SIZE` | 64 | Training batch size |
| `EARLY_STOP_PATIENCE` | 10 | Epochs without val improvement before stopping |

### Kraken trading
| Parameter | Value | Notes |
|-----------|-------|-------|
| Maker fee | 0.16% | Limit orders |
| Taker fee | 0.26% | Market orders |
| Min order (BTC) | 0.0001 | Minimum BTC order size |
| 4h candle times | 00, 04, 08, 12, 16, 20 UTC | Candle close times |

## Dependencies

| Package | Purpose |
|---------|---------|
| ccxt | Exchange API abstraction |
| pandas | Data manipulation |
| numpy | Numerical computation |
| torch | ML model training and inference |
| python-dotenv | Environment variable management |
