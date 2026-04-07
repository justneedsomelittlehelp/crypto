# Crypto Trading Bot

> Automated BTC/USD spot trading bot — Kraken API + PyTorch ML model, driven by volume profile features.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.12+ |
| ML Framework | PyTorch |
| Exchange API | Kraken (via ccxt) |
| Data Processing | pandas, numpy |
| Config | python-dotenv for secrets |
| Deployment | Local dev → Linux VPS for 24/7 operation |

## CLI Commands

```bash
python3 -m src.data scrape       # fetch all exchange data + compute VP → CSV
python3 -m src.data validate     # validate existing CSV (gaps, prices, VP normalization)
```

## Architecture Docs (in `architecture_docs/`)

Read the relevant doc before modifying code in that domain:

| When working on | Read |
|----------------|------|
| Data collection, OHLCV scraping, volume profile computation | `arch-data-pipeline.md` |
| Model architecture, training, evaluation, backtesting | `arch-ml-model.md` |
| Kraken API, order execution, bot loop, position tracking | `arch-trading-engine.md` |
| Risk limits, circuit breakers, loss thresholds, safety rules | `arch-risk-safety.md` |
| Looking up a file path, data column, config parameter | `arch-reference.md` |

## Key Data

- **Primary feature**: 50-bin relative volume profile (VP) per 4h candle
- **VP bins**: `vp_rel_00` (−25% from price) through `vp_rel_49` (+25% from price), bin 25 = current price
- **Data source**: Bitstamp + Coinbase merged OHLCV, from 2016-01-01
- **Timeframe**: 4h candles (6 bars/day)

## Security Rules

- **Never commit `.env`** — API keys live there
- **Kraken API keys**: trade-only permission, no withdraw
- **Never log API keys** — sanitize error output
- See `security_check/SECURITY.md` for full threat model

## Coding Rules

- All times in UTC
- Financial data: use `float` (sufficient for BTC/USD prices at this scale)
- Feature computation in inference must exactly match training pipeline — any divergence silently degrades predictions
- Time-series splits only (never random shuffle) for train/val/test
- Default to not trading when uncertain — the safest action is to do nothing

## Build Status

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Project Setup & Data Pipeline | **Complete** |
| 2 | Feature Engineering | Not started |
| 3 | ML Model Development | Not started |
| 4 | Kraken API Integration | Not started |
| 5 | Trading Strategy & Execution | Not started |
| 6 | Live Trading & Monitoring | Not started |
| 7 | Optimization & Hardening | Not started |

## Harness Maintenance

This project uses a structured documentation harness. Follow these rules to keep it accurate:

### When to Update What

| Event | Update |
|-------|--------|
| Architecture decision made or changed | Update the relevant `architecture_docs/arch-{domain}.md` with the decision and reasoning |
| New doc or reference file added | Add a routing entry to the "When working on / Read" table above |
| Phase completed | Mark phase as done in `roadmap/README.md` and Build Status section above |
| New phase or scope change | Create/update `roadmap/PHASE_N.md`, update roadmap README |
| Non-obvious decision worth preserving across sessions | Add a brief pointer to MEMORY.md |
| New source file created | Update `arch-reference.md` file index |
| New config parameter added | Update `arch-reference.md` configuration table |
| Security vulnerability found or fixed | Update `security_check/SECURITY.md` with numbered entry |

