# Backtest Module (`src/backtest/`)

Realistic portfolio simulator that converts model predictions into a $-denominated equity curve. Designed to answer the question: **"if we had actually traded these signals, starting with $X, what would have happened?"**

## Files

| File | Purpose |
|------|---------|
| `engine.py` | Core simulator: `BacktestConfig`, `Position`, `ClosedTrade`, `Portfolio` class, `run_backtest()` function. Walks through bars chronologically, manages cash and open positions, applies fees and slippage. |
| `metrics.py` | `compute_metrics()` — turns a finished `Portfolio` into standard performance metrics (total return, drawdown, Sharpe, win rate, etc.) |
| `__init__.py` | Module marker |

## What the engine simulates

1. **Capital constraints** — only opens new positions when capital is available
2. **Position sizing** — fixed % of equity, dynamic, or full allocation (configurable)
3. **Reserve buffer** — keeps a configurable % of equity untouchable as safety
4. **Fees** — taker on entry (market order), maker on TP exit (limit order), taker on SL exit (stop-loss = market)
5. **Slippage** — applied per side
6. **Pyramiding control** — stack same-direction positions or skip
7. **Max hold time** — timeout exit if neither TP nor SL hits
8. **Mark-to-market equity tracking** — equity updated every bar

## Configuration

All settings live in `BacktestConfig` dataclass. Default values match the user's chosen settings:
- `starting_capital = 5000.0`
- `reserve_pct = 0.30`
- `position_size_pct = 0.20`
- `sizing_mode = "fixed_pct"`  ("fixed_pct" / "dynamic" / "fixed_100")
- `fee_taker = 0.0026`
- `fee_maker = 0.0016`
- `slippage_per_side = 0.0005`
- `max_hold_bars = 14 * 24` (14 days at 1h)
- `allow_pyramiding = False`
- `direction = "long"`

## Usage

The driver script `src/models/run_backtest.py` is the normal entry point. To use programmatically:

```python
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import compute_metrics

config = BacktestConfig(
    starting_capital=5000,
    sizing_mode="fixed_pct",
    position_size_pct=0.50,
    min_confidence=0.65,
    min_asymmetry=1.5,
    allow_pyramiding=True,
)

portfolio, summary = run_backtest(
    dates=dates_array,
    close_prices=close_array,
    probs=probs_array,
    tp_pct=tp_pct_array,
    sl_pct=sl_pct_array,
    config=config,
)

metrics = compute_metrics(portfolio, config.starting_capital)
print(f"Final equity: ${metrics['final_equity']:,.0f}")
print(f"Sharpe: {metrics['sharpe_annualized']:.2f}")
```

## What the metrics tell you

| Metric | What it means |
|--------|---------------|
| `total_return_pct` | Final equity vs starting, as % |
| `annualized_return_pct` | CAGR (compound annual growth rate) |
| `max_drawdown_pct` | Worst peak-to-trough decline (% of peak equity) |
| `sharpe_annualized` | Risk-adjusted return, annualized. >1 = good, >2 = very good. |
| `win_rate` | Fraction of trades that closed in profit |
| `avg_hold_days` | Mean trade duration |
| `total_fees` | Cumulative dollars paid in fees |
| `max_consec_losses` | Longest losing streak (drawdown risk indicator) |
| `exit_reasons` | Histogram of TP / SL / timeout / end-of-test |
| `monthly_summary` | Recent 12 months of P&L breakdown |

## Why a separate engine vs the eval script?

The `eval_v6_prime.py` script computes per-trade EV but assumes infinite parallel capital. That's useful for understanding model edge but doesn't reflect real trading. The backtest engine adds the constraints that matter for live deployment:
- You can't open infinite positions
- Each trade locks dollars until close
- Fees compound
- Drawdowns matter for risk management

## Currently produced results

`run_backtest.py` runs 12 combinations (3 filters × 4 sizing strategies) and writes to `experiments/backtest_results.json`. The best combination becomes a candidate for live trading after Phase 4 (Kraken integration).
