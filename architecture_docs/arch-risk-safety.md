# Risk & Safety

> **Read this when working on:** risk management, position limits, circuit breakers, loss thresholds, API key security
> **Related docs:** [arch-trading-engine.md](arch-trading-engine.md) (where risk checks are called), [security_check/SECURITY.md](../security_check/SECURITY.md) (security audit trail)

---

## Overview

This bot trades real money. Every component must be designed with the assumption that things will go wrong — bad predictions, API failures, bugs, and market crashes. Risk management is not optional; it's the most important part of the system.

## Pre-Trade Risk Checks

Before every order, the risk manager validates:

| Check | Rule | Rationale |
|-------|------|-----------|
| Max position size | Never hold more than X% of portfolio in BTC | Prevents overexposure |
| Max daily loss | Stop trading if daily P&L drops below -Y% | Limits damage from bad model behavior |
| Min time between trades | At least N candles between trades | Prevents rapid-fire trading on noisy signals |
| Order size sanity | Order value must be within [min, max] bounds | Catches bugs that produce absurd order sizes |
| Balance check | Sufficient USD/BTC to execute the trade | Prevents failed orders |

All thresholds are configurable in the trading config, not hardcoded.

## Circuit Breakers

Automatic trading halt conditions:

1. **Daily loss limit** — if cumulative daily loss exceeds threshold, halt until next day (or manual reset)
2. **Consecutive losses** — if N consecutive trades lose money, halt and alert
3. **Model confidence** — if model prediction confidence drops below threshold, skip the trade
4. **Market volatility** — if ATR or price movement exceeds N standard deviations from historical norm, halt

Circuit breakers require **manual intervention** to resume. This is intentional — automated recovery from extreme situations is risky.

## Position Limits

- **No leverage** — spot only, never borrow
- **No short selling** — can only be flat or long
- **Maximum BTC allocation** — configurable (e.g., 50% of portfolio)
- **Minimum cash reserve** — always keep some USD for fees and future trades

## Financial Safety Rules

- **Start small** — initial live trading should use a fraction of intended capital
- **Paper trade first** — minimum 2-4 weeks of paper trading before live
- **Separate trading account** — don't keep life savings in the trading account
- **API key permissions** — trade-only, no withdraw permission
- **No withdrawal automation** — never automate moving money out of Kraken

## Error Handling Philosophy

| Scenario | Response |
|----------|----------|
| API timeout | Retry with exponential backoff (max 3 attempts) |
| API error (non-transient) | Log, alert, skip this candle |
| Data feed gap | Log the gap, do not trade on stale data |
| Model inference error | Do not trade, alert |
| Unexpected exception | Catch at top level, log full traceback, alert, halt |

**Default to not trading.** If anything is uncertain, the safest action is to do nothing.

## Monitoring & Alerts

Critical events that must trigger alerts:
- Trade executed (every single one)
- Circuit breaker tripped
- API error that prevented trading
- Bot restart / crash recovery
- Daily P&L summary

## Gotchas

- **Flash crashes** — BTC can drop 10%+ in minutes. Market orders during a flash crash will fill at terrible prices. Consider using limit orders with a price bound.
- **Fee erosion** — frequent trading with small edges gets eaten by fees. Kraken taker fee is 0.26%; a round-trip costs ~0.52%. The model's edge must exceed this.
- **Slippage** — market orders on thin orderbooks can slip significantly. Check orderbook depth before placing large market orders.
- **Tax implications** — every trade is a taxable event in most jurisdictions. The trade log should be detailed enough for tax reporting.
