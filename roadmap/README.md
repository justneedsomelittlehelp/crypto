# Roadmap — Crypto Trading Bot

> Automated BTC/USD spot trading bot using Kraken API with ML-driven signals (PyTorch).

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.12+ | Ecosystem for ML + exchange APIs |
| ML Framework | PyTorch | Industry standard at quant firms, flexible for custom architectures |
| Exchange API | Kraken (via ccxt) | User's target exchange; ccxt already used for data collection |
| Data | pandas, numpy | Already in use for feature engineering |
| Deployment | Local dev → Linux VPS for live | Bot must run 24/7; VPS is simplest path |

## Constraints

- **BTC/USD spot only** — single pair, no margin/futures
- **User owns the trading strategy** — implementation support, not strategy design
- **Data source**: multi-exchange OHLCV merged + relative volume profile (50-bin histogram)
- **4h timeframe** candles as base resolution

## Phase Overview

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Project Setup & Data Pipeline | **Complete** |
| 2 | Feature Engineering | **Complete** |
| 3 | ML Model Development | **Audit complete — shelved** (2026-04-12). See `experiments/EVAL_AUDIT.md`. |
| 4 | Kraken API Integration | Not started — not green-lit |
| 5 | Trading Strategy & Execution Engine | Not started |
| 6 | Live Trading & Monitoring | Not started |
| 7 | Optimization & Hardening | Not started |

## Project status (2026-04-12)

**Paused at end of Phase 3.** The first hypothesis — *VP-only features on 1h BTC, long-only, can produce an automated strategy that beats passive SPY* — was rejected after a pre-deployment audit uncovered walk-forward label leakage and test-set parameter tuning. Honest best strategy profile is +6.0% CAGR / −18.4% DD, which does not justify the operational overhead versus passive index exposure. Infrastructure (walk-forward with embargo, holdout protocol, backtest engine, VP features, transformer pipeline) is preserved for any future hypothesis.
