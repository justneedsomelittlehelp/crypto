# Monitoring Module (`src/monitoring/`) — Phase 6 placeholder

**Status: empty.** This directory will hold live trading monitoring code starting in Phase 6 (Live Trading & Monitoring).

## Planned files (when Phase 6 starts)

| File | Purpose |
|------|---------|
| `metrics_logger.py` | Logs every trade, equity snapshot, and signal to a database (SQLite or similar) |
| `alert_dispatcher.py` | Sends alerts (Telegram/email/Slack) on circuit breaker triggers, large drawdowns, model staleness |
| `drift_detector.py` | Monitors feature distribution vs training data; raises alert if live data drifts from what model was trained on |
| `dashboard.py` | (Optional) Live dashboard server showing equity curve, open positions, recent signals |

## Why empty now

Same reason as `trading/` — Phase 3 (model) needs to be done first. Live monitoring is meaningless without a live trading system.

See `architecture_docs/arch-trading-engine.md` and `architecture_docs/arch-risk-safety.md` for monitoring design.
