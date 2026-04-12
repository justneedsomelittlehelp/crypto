# Source Code (`src/`)

All Python source for the trading bot. Each subdirectory is a Python package with its own README.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `data/` | Multi-exchange OHLCV scraping, volume profile computation, funding rate ingestion |
| `features/` | Feature engineering pipeline (50 VP bins + derived candle/volume features). Frozen versions in `features/pipelines/`. |
| `models/` | ML model definitions, training scripts, eval scripts, walk-forward backtest. Frozen architectures in `models/architectures/`. |
| `backtest/` | Realistic portfolio simulator with fees, slippage, capital constraints. Used by `models/run_backtest.py`. |
| `trading/` | (Planned) Kraken API integration, order execution, position management. Empty until Phase 4. |
| `monitoring/` | (Planned) Live trading monitoring, alerts, drift detection. Empty until Phase 6. |

## Top-level files

| File | Purpose |
|------|---------|
| `config.py` | Mutable configuration (timeframes, lookback bars, label parameters). New experiments should override values here. |

## Module dependency graph

```
data/        (scraping, no dependencies on others)
  ↓
features/    (uses scraped CSVs from data/)
  ↓
models/      (trains on features, saves predictions)
  ↓
backtest/    (consumes model predictions)
```

## Conventions

- **Frozen vs mutable**: `features/pipelines/` and `models/architectures/` contain frozen snapshots that should never be modified. New experiments create new versioned files alongside.
- **Eval scripts** live in `models/` and write results to `experiments/*.json`.
- **Active vs archive**: each subdirectory may have an `archive/` folder for historical/superseded files. Don't use archived files for new work.

## Reading order for newcomers

1. `data/README.md` — how raw data flows in
2. `features/README.md` — how features are built
3. `models/README.md` — what model scripts exist
4. `backtest/README.md` — how predictions become realistic P&L
