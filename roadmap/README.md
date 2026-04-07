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
| 3 | ML Model Development | **In progress** |
| 4 | Kraken API Integration | Not started |
| 5 | Trading Strategy & Execution Engine | Not started |
| 6 | Live Trading & Monitoring | Not started |
| 7 | Optimization & Hardening | Not started |
