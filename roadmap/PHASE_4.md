# Phase 4: Kraken API Integration

## Goal
Connect to Kraken's API for account management, real-time data, and order execution.

## What This Phase Builds On
Phase 1 — project structure, env var setup for API keys.

## Implementation Steps

1. **Kraken API client** (`src/trading/kraken_client.py`)
   - Use ccxt's Kraken implementation (already a dependency)
   - Wrap ccxt calls in a client class with:
     - Authentication (API key + secret from `.env`)
     - Rate limiting (respect Kraken's limits)
     - Error handling and retry logic for transient failures
   - Key methods:
     - `get_balance()` — current USD and BTC balances
     - `get_ticker()` — current BTC/USD price
     - `get_orderbook()` — for slippage estimation
     - `place_order(side, amount, order_type)` — market or limit
     - `get_open_orders()` / `cancel_order()`
     - `get_trade_history()` — for reconciliation

2. **Real-time data feed** (`src/trading/data_feed.py`)
   - Fetch latest 4h candle from Kraken on schedule
   - Merge with historical data to compute current VP and features
   - Must handle partial candles (current bar not yet closed)

3. **Order management** (`src/trading/order_manager.py`)
   - Track order state: pending → filled / cancelled / failed
   - Handle partial fills
   - Log every order action (placed, filled, cancelled) with timestamp

4. **Paper trading mode**
   - A simulated order executor that behaves like the real one but doesn't send orders
   - Uses real market data, fake fills at current price + estimated slippage
   - Essential for testing Phases 5-6 without risking real money

## Test When Done
- [ ] `get_balance()` returns real account balances (with read-only API key)
- [ ] `get_ticker()` returns current BTC/USD price
- [ ] Paper trading mode can simulate a buy → hold → sell cycle
- [ ] All API calls log properly and handle rate limits
- [ ] API keys are loaded from `.env`, never hardcoded
