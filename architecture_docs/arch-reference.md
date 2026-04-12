# Reference

> **Read this when:** looking up file locations, data schema, configuration parameters, or Kraken API details.

> **For navigating the file tree:** every directory has a README.md. Start with the project root README.md and navigate down. This doc complements those by listing config parameters and data schema.

---

## File Index

For per-file inventory, see each directory's `README.md`. This section lists only the most-referenced files.

### Active eval scripts
| Path | Purpose |
|------|---------|
| `src/models/eval_v6_prime.py` | ⭐⭐ Current best — v6-prime + 3-seed ensemble + SWA + filter analysis |
| `src/models/run_backtest.py` | ⭐⭐ Realistic backtest engine driver |
| `src/models/eval_2plus1.py` | Reference: v7/v8 (2+1 layers) on 1h |
| `src/models/eval_15min.py` | Reference: v6/v8 on 15min data |

### Frozen architectures (`src/models/architectures/`)
| File | Architecture |
|------|--------------|
| `v6_prime_vp_labels.py` | ⭐⭐ Current best (TemporalEnrichedV6Prime) |
| `v6_temporal_enriched.py` | v6 baseline (1+1 enriched) |
| `v8_enriched_2plus1.py` | v8 (2+1 enriched) |
| `v7_simple_2plus1.py` | v7 (2+1 simple) |
| `v5_dualbranch_cls.py` | v5 (DualBranch, abandoned) |
| `v2_temporal.py` | v2 (historical baseline) |

### Frozen feature pipelines (`src/features/pipelines/`)
| File | Features | Used by |
|------|----------|---------|
| `v1_raw.py` | 68 (50 VP + 10 derived + 8 VP structure) | v6, v6-prime, v8 |
| `v2_scaled.py` | 60 (no VP structure, z-scored) | v7 |

### Backtest module (`src/backtest/`)
| File | Purpose |
|------|---------|
| `engine.py` | Portfolio simulator: BacktestConfig, Position, run_backtest() |
| `metrics.py` | compute_metrics() — total return, drawdown, Sharpe, etc. |

### Data module (`src/data/`)
| File | Purpose |
|------|---------|
| `__main__.py` | CLI: `python -m src.data scrape [timeframe]`, `validate` |
| `scraper.py` | Multi-exchange OHLCV fetcher |
| `volume_profile.py` | 50-bin VP computation |
| `funding_rate.py` | Binance + Gate.io merged funding rate |
| `validator.py` | CSV sanity checks |

### Documentation
| File | Purpose |
|------|---------|
| `experiments/MODEL_HISTORY.md` | ⭐ Living doc: every architecture decision (RNN → current best) |
| `experiments/STRATEGY.md` | User's manual trading strategy |
| `experiments/EVAL_TRANSFORMER.md` | Per-eval results (Eval 1-12) |
| `experiments/RUN_INDEX.md` | Maps run_* directories to evals |

## Data schema

### `BTC_<timeframe>_RELVP.csv` (e.g., `BTC_1h_RELVP.csv`)
| Column | Type | Description |
|--------|------|-------------|
| `ts` | int | Unix timestamp (ms) |
| `date` | string | ISO 8601 datetime (UTC) |
| `open` | float | Merged open price |
| `high` | float | Merged high price |
| `low` | float | Merged low price |
| `close` | float | Merged close price |
| `volume_<timeframe>` | float | Merged volume (sum across exchanges) |
| `vp_rel_00`..`vp_rel_49` | float | Relative VP histogram bins (sum ≈ 1.0) |

After running through `pipelines/v1_raw.py`, the dataframe also has:
- 10 derived features (log_return, bar_range, bar_body, volume_ratio, upper_wick, lower_wick, body_dir, ohlc_open_ratio, ohlc_high_ratio, ohlc_low_ratio)
- 8 VP structure features (vp_ceiling_dist, vp_floor_dist, vp_num_peaks, vp_ceiling_strength, vp_floor_strength, vp_ceiling_consistency, vp_floor_consistency, vp_mid_range)

### VP bin mapping
- Bin 00 = −25% from current price (lowest)
- Bin 25 = current price (center)
- Bin 49 = +25% from current price (highest)
- Axis: `log(close_j / close_t)`, linearly spaced in log space

### Funding rate CSVs (`data/funding_rate_*.csv`)
| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Funding settlement timestamp (UTC) |
| `funding_rate` | float | 8h funding rate as decimal (e.g., 0.0001 = 0.01%) |

## Configuration parameters (`src/config.py`)

### Data pipeline
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USD` | Trading pair |
| `TIMEFRAME` | `1h` | Candle interval (`1h`, `15m`, etc.) |
| `START_DATE` | `2016-01-01` | Data collection start |
| `EXCHANGES` | `[bitstamp, coinbase]` | Data source exchanges |
| `LOOKBACK_DAYS` | 180 | VP computation lookback (days) |
| `LOOKBACK_BARS` | 4320 (auto) | VP lookback in bars (= LOOKBACK_DAYS × BARS_PER_DAY) |
| `REL_SPAN_PCT` | 0.25 | VP price range (±25%) |
| `REL_BIN_COUNT` | 50 | VP histogram bins |

### Feature engineering / model input
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOOKBACK_BARS_MODEL` | 720 | Model input window (720 bars = 30 days at 1h) |
| `BARS_PER_DAY` | 24 (auto) | Auto-derived from TIMEFRAME |
| `VOLUME_ROLL_WINDOW_DAYS` | 30 | Rolling volume normalization window |

### Label configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LABEL_MODE` | `first_hit` | "fixed_horizon" or "first_hit" |
| `LABEL_TP_PCT` | 0.025 | TP base (used by older v6 baseline) |
| `LABEL_SL_PCT` | 0.05 | SL base (used by older v6 baseline) |
| `LABEL_REGIME_ADAPTIVE` | True | Flip TP/SL ratio in bear markets (legacy) |
| `LABEL_REGIME_MODE` | `sma` | "sma" or "fgi" |
| `LABEL_FGI_THRESHOLD` | 50 | FGI cutoff for bull/bear |
| `LABEL_MAX_BARS` | 336 | Max lookahead (14 days at 1h) |
| `LABEL_FGI_PATH` | `data/fear_greed_index.csv` | FGI data location |

**Note:** `eval_v6_prime.py` overrides these — it computes labels per-sample from VP structure, not from the global TP/SL config.

### Training configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 5e-4 | Optimizer learning rate |
| `EPOCHS` | 50 | Max training epochs |
| `BATCH_SIZE` | 64 (eval scripts override to 256-512) | Training batch size |
| `EARLY_STOP_PATIENCE` | 15 | Epochs without val improvement before stopping |
| `DATALOADER_WORKERS` | 2 | DataLoader parallelism |

### v6-prime additions (in `eval_v6_prime.py`, not config.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DROPOUT` | 0.3 | Dropout probability (overrides architecture default) |
| `WEIGHT_DECAY` | 1e-3 | AdamW weight decay |
| `LABEL_SMOOTHING` | 0.1 | Soft label smoothing |
| `N_SEEDS` | 3 | Number of seeds per fold (42, 43, 44) |
| `SWA_START_EPOCH` | 15 | SWA averaging starts at this epoch |
| `TP_RATIO` | 0.8 | VP-derived TP fraction (80% of way to ceiling) |
| `SL_RATIO` | 0.6 | VP-derived SL fraction (60% of way to floor) |
| `TP_MIN`, `TP_MAX` | 0.01, 0.15 | TP clipping bounds |
| `SL_MIN`, `SL_MAX` | 0.01, 0.15 | SL clipping bounds |
| `MIN_PEAKS` | 1 | Skip bars with fewer VP peaks |

### Backtest configuration (in `src/backtest/engine.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `starting_capital` | 5000.0 | Initial portfolio $ |
| `reserve_pct` | 0.30 | % of equity untouchable as safety |
| `position_size_pct` | 0.20 | Per-trade size when sizing_mode=fixed_pct |
| `sizing_mode` | `fixed_pct` | "fixed_pct", "dynamic", or "fixed_100" |
| `fee_taker` | 0.0026 | Kraken taker fee (0.26%) |
| `fee_maker` | 0.0016 | Kraken maker fee (0.16%) |
| `slippage_per_side` | 0.0005 | 0.05% slippage per fill |
| `max_hold_bars` | 336 | 14 days × 24 hours |
| `allow_pyramiding` | False | Stack same-direction positions |

## Walk-forward fold boundaries

Used by all eval scripts. Each fold trains on all data before `train_end`, validates on the next 6 months, tests on the next 6 months.

```
2020-01-01, 2020-07-01, 2021-01-01, 2021-07-01,
2022-01-01, 2022-07-01, 2023-01-01, 2023-07-01,
2024-01-01, 2024-07-01, 2025-01-01, 2025-07-01
```

10 test folds total (folds 1-10), test periods 2020 H2 → 2025 H1.

## Kraken trading

| Parameter | Value | Notes |
|-----------|-------|-------|
| Maker fee | 0.16% | Limit orders (TP exits) |
| Taker fee | 0.26% | Market orders (entries, SL exits) |
| Round trip (TP) | 0.42% | taker entry + maker exit |
| Round trip (SL) | 0.52% | taker entry + taker exit |
| Min order (BTC) | 0.0001 | Minimum BTC order size |
| Withdrawal | DISABLED | Use trade-only API keys |

## Dependencies

| Package | Purpose |
|---------|---------|
| ccxt | Exchange API abstraction |
| pandas | Data manipulation |
| numpy | Numerical computation |
| torch | ML model training and inference |
| scipy | Signal processing (Gaussian smoothing, peak detection) |
| python-dotenv | Environment variable management |
