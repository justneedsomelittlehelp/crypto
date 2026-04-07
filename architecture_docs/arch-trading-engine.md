# Trading Engine

> **Read this when working on:** Kraken API integration, order execution, position management, the main bot loop
> **Related docs:** [arch-ml-model.md](arch-ml-model.md) (prediction source), [arch-risk-safety.md](arch-risk-safety.md) (pre-trade checks)

---

## Overview

The trading engine connects ML predictions to real order execution on Kraken. It manages the lifecycle: data feed → inference → strategy decision → risk check → order → position tracking.

## Kraken API Integration

### Authentication
- API key and secret stored in `.env` (never in code or git)
- Keys should have **trade** permission but **not withdraw** — this limits damage if keys are compromised
- ccxt handles Kraken's authentication (nonce, signature)

### Rate Limits
- Kraken's API has rate limits (varies by endpoint)
- ccxt's `enableRateLimit: True` handles basic throttling
- For burst scenarios (e.g., checking order status repeatedly), add explicit backoff

### Key Endpoints Used
| Action | ccxt Method | Notes |
|--------|-------------|-------|
| Get balance | `fetch_balance()` | Returns all currency balances |
| Current price | `fetch_ticker('BTC/USD')` | Mid price, bid, ask, volume |
| Place order | `create_order()` | Market or limit |
| Check order | `fetch_order(id)` | Status: open, closed, canceled |
| Cancel order | `cancel_order(id)` | For unfilled limit orders |
| Trade history | `fetch_my_trades()` | For reconciliation |

## Main Bot Loop

```
every 4h (on candle close):
    1. fetch latest OHLCV from Kraken
    2. append to historical data
    3. compute VP + features (same pipeline as training)
    4. run model inference → prediction
    5. strategy.decide(prediction, current_position) → action
    6. risk_manager.check(action, portfolio_state) → approved/blocked
    7. if approved: order_manager.execute(action)
    8. log everything
```

### Timing
- 4h candles close at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
- Bot should wait ~30 seconds after candle close for data to settle
- Use a scheduler (e.g., `schedule` library or cron) — don't sleep-loop

## Position Tracking

- **Spot only**: position is either flat (100% USD) or long (some % in BTC)
- No short selling
- Position state persisted to disk (`state/position.json`) for crash recovery
- On startup: reconcile disk state with Kraken's reported balances

## Paper Trading Mode

A drop-in replacement for the real order executor:
- Uses real market data (prices, orderbook)
- Simulates fills at current price + estimated slippage
- Tracks a virtual portfolio
- Identical logging to live mode

Essential for testing without risking money. Must be run for 2-4 weeks before going live.

## Order Types

- **Market orders**: simpler, guaranteed fill, but subject to slippage
- **Limit orders**: better price, but may not fill
- Start with market orders for simplicity. Optimize to limit orders in Phase 7 if slippage is material.

## Gotchas

- **Kraken maintenance windows** — Kraken occasionally goes down for maintenance. The bot must handle API errors gracefully (retry, skip candle, alert)
- **Partial fills** — limit orders may partially fill. The order manager must track remaining quantity
- **Nonce errors** — Kraken uses nonces for request ordering. If the bot crashes mid-request, the nonce may need to be reset. ccxt handles this but be aware
- **Minimum order size** — Kraken has minimum order sizes for BTC/USD (currently 0.0001 BTC). Check before placing small orders
