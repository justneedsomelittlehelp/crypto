# Phase 6: Live Trading & Monitoring

## Goal
Deploy the bot for 24/7 operation with proper monitoring, alerting, and a paper-trading burn-in period.

## What This Phase Builds On
Phase 5 — complete execution engine with paper trading mode working.

## Implementation Steps

1. **VPS deployment**
   - Provision a Linux VPS (DigitalOcean, Hetzner, or Linode — $5-10/mo is sufficient)
   - Set up: Python environment, clone repo, install dependencies
   - Run bot as a systemd service (auto-restarts on crash)
   - Alternatively: run in a `tmux` session for simpler setup

2. **Structured logging** (`src/monitoring/logger.py`)
   - JSON-formatted logs for machine parseability
   - Log levels: INFO (trades, signals), WARNING (risk limits approached), ERROR (API failures)
   - Rotate log files (don't fill the disk)
   - Key events to log:
     - Every candle processed
     - Every signal generated (with model confidence)
     - Every order placed/filled/cancelled
     - Every risk check (pass or block)
     - Daily portfolio summary

3. **Alerting** (`src/monitoring/alerts.py`)
   - Notify on critical events:
     - Trade executed (buy/sell with amount and price)
     - Error that stops the bot
     - Daily P&L summary
     - Risk limit triggered
   - Channels: email (simplest), Telegram bot (popular in crypto), or Discord webhook

4. **Performance dashboard**
   - Simple approach: daily log-to-CSV that tracks equity, positions, trades
   - Can visualize with a Jupyter notebook or simple matplotlib script
   - Track: cumulative return, drawdown, win rate, trade count

5. **Paper trading burn-in**
   - Run in paper mode on live data for at least 2-4 weeks before going live
   - Compare paper trades against what the backtest predicted
   - Look for: execution timing issues, data feed gaps, unexpected model behavior

6. **Go-live checklist**
   - Paper trading results match backtest within tolerance
   - All risk limits are configured and tested
   - Alerting works (test each channel)
   - API keys have correct permissions (trade, but not withdraw)
   - Emergency shutdown procedure documented and tested

## Test When Done
- [ ] Bot runs as a service on VPS, auto-restarts after simulated crash
- [ ] Logs are structured and rotated
- [ ] Alerts fire correctly for test events
- [ ] Paper trading has run for the burn-in period with no unexpected behavior
- [ ] Go-live checklist is complete
