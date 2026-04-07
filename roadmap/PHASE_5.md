# Phase 5: Trading Strategy & Execution Engine

## Goal
Connect the ML model's predictions to actual trade execution through a strategy engine that manages signals, position sizing, and risk rules.

## What This Phase Builds On
Phase 3 — trained ML model. Phase 4 — Kraken API client and order management.

## Implementation Steps

1. **Strategy engine** (`src/trading/strategy.py`)
   - This is where the user's trading logic lives
   - Interface: receives current features + model prediction → outputs trading action
   - Actions: BUY, SELL, HOLD (with amount)
   - The user defines the signal-to-action mapping based on their strategy

2. **Position manager** (`src/trading/position_manager.py`)
   - Track current position: flat, long (amount of BTC held)
   - No short selling (spot only)
   - Position sizing logic: what fraction of portfolio to allocate per trade
   - State persistence: save position to disk so restarts don't lose track

3. **Risk management** (`src/trading/risk_manager.py`)
   - Pre-trade checks before any order is placed:
     - Max position size (e.g., never hold more than X% of portfolio in BTC)
     - Max daily loss threshold — stop trading if exceeded
     - Minimum time between trades (prevent rapid-fire orders on noisy signals)
   - Post-trade checks:
     - Portfolio drawdown monitoring
     - Trailing stop-loss (optional, strategy-dependent)

4. **Main execution loop** (`src/trading/bot.py`)
   - The orchestrator that ties everything together:
     1. Wait for new 4h candle close
     2. Fetch latest data → compute features → run model inference
     3. Pass prediction to strategy engine → get trading action
     4. Risk manager validates the action
     5. If approved, execute via order manager
     6. Log everything
   - Must be resumable: if bot restarts, it picks up from current state

5. **Configuration**
   - All tunable parameters (position sizes, thresholds, etc.) in config, not hardcoded
   - Separate configs for paper trading vs live trading

## Test When Done
- [ ] Full pipeline works end-to-end in paper trading mode
- [ ] Risk manager correctly blocks trades that violate limits
- [ ] Bot handles a simulated 24h period: waits for candles, generates signals, places paper trades
- [ ] Position state persists across bot restarts
- [ ] All actions are logged with timestamps and reasoning
