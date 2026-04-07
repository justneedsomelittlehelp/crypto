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

### Planned files (future phases)
| Path | Phase | Purpose |
|------|-------|---------|
| `src/features/indicators.py` | 2 | Technical indicator functions |
| `src/features/pipeline.py` | 2 | Feature pipeline orchestrator |
| `src/features/dataset.py` | 2 | PyTorch Dataset class |
| `src/models/architecture.py` | 3 | Model definition |
| `src/models/trainer.py` | 3 | Training loop |
| `src/models/evaluate.py` | 3 | Evaluation metrics |
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
