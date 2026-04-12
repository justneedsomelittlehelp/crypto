# Tests (`tests/`)

**Status: minimal — only `__init__.py` exists.** Tests will be added as live trading infrastructure is built (Phase 4+).

## Planned test structure

| File | What it tests |
|------|---------------|
| `test_data_pipeline.py` | OHLCV merging, VP computation correctness, edge cases (empty data, single exchange) |
| `test_features.py` | Frozen pipeline outputs match snapshot fixtures (catch accidental changes) |
| `test_backtest_engine.py` | Backtest engine: position management, fee accounting, edge cases (zero capital, max concurrent reached, etc.) |
| `test_kraken_client.py` | (Phase 4) Mock Kraken API responses, test error handling and retries |
| `test_risk_manager.py` | (Phase 5) Position size limits, circuit breakers, daily loss caps |

## Why so few tests now

Phase 3 (ML model dev) is research-heavy: every experiment is one-off. Writing tests for one-off scripts is wasted effort. As we move to live trading code, tests become essential because:
- Live code runs unsupervised
- Bugs cost real money
- Refactoring is risky without regression coverage

The backtest engine (`src/backtest/engine.py`) is the first piece of "production" code that should get full test coverage before live trading starts.
