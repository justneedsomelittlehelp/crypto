# Trading Module (`src/trading/`) — Phase 4-5 placeholder

**Status: empty.** This directory will hold the live trading code starting in Phase 4 (Kraken API Integration) and Phase 5 (Trading Strategy & Execution Engine).

## Planned files (when Phase 4-5 starts)

| File | Purpose |
|------|---------|
| `kraken_client.py` | Wrapper around ccxt's Kraken API. Authentication, rate limit handling, error retries. |
| `order_executor.py` | Places market/limit orders with TP/SL bracket. Handles partial fills, cancellation, error recovery. |
| `position_manager.py` | Tracks open positions, monitors TP/SL, applies exit logic. Mirrors `Portfolio` class from `src/backtest/engine.py` but for live trading. |
| `signal_runner.py` | Periodically queries the model, applies filters, dispatches trades to executor when signals fire. |
| `risk_manager.py` | Enforces position size limits, max drawdown circuit breaker, daily loss limit. |

## Why empty now

We're still in Phase 3 (model development). No point building trading infrastructure until we have a model worth deploying. Current best (v6-prime + ensemble + combined filter) is a candidate but still being validated.

See `architecture_docs/arch-trading-engine.md` for the planned design.
