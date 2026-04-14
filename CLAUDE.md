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
| User's trading strategy, model iteration reasoning | `experiments/STRATEGY.md` |
| Vanilla RNN evals (4 evals, exhausted) | `experiments/EVAL_VANILLA_RNN.md` |
| LSTM evals (7 evals, exhausted) | `experiments/EVAL_LSTM.md` |
| 1D CNN evals (26 evals, best: 61.5% on 4h) | `experiments/EVAL_CNN.md` |
| **Transformer evals (6 evals, best: 63.3% on 1h, 2+1 tested)** | `experiments/EVAL_TRANSFORMER.md` |
| **TP/SL sweep (11 configs, long+short, FGI vs SMA regime)** | `experiments/EVAL_TPSL_SWEEP.md` |
| **Dual-branch Transformer (VP + Candle, separate attention)** | `experiments/EVAL_DUAL_BRANCH.md` |
| **Model evolution history (living doc)** | `experiments/MODEL_HISTORY.md` |
| **⚠ Audit post-mortem — retracts Eval 11/12/17/18** | `experiments/EVAL_AUDIT.md` |
| **v10: 90-day temporal × 30-day VP (post-audit experiment)** | `architecture_docs/arch-ml-model.md` §v10 |
| **Label redesign — triple-barrier options, v11 post-mortem** | `experiments/LABEL_REDESIGN.md` |
| Run folder → eval mapping | `experiments/RUN_INDEX.md` |

## Key Data

- **Primary feature**: 50-bin relative volume profile (VP) per candle
- **VP structure features**: ceiling/floor distance, peak strength, peak consistency (7 cols)
- **VP bins**: `vp_rel_00` (−25% from price) through `vp_rel_49` (+25% from price), bin 25 = current price
- **Data source**: Bitstamp + Coinbase merged OHLCV, from 2016-01-01
- **Timeframe**: 1h candles (24 bars/day) — switched from 4h for more training data
- **Lookback**: 720 bars (30 days at 1h)
- **Fear & Greed Index**: daily FGI data from 2018-02, used for regime detection (`data/fear_greed_index.csv`)
- **Trading strategy doc**: `experiments/STRATEGY.md` — user's VP support/resistance approach

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
| 2 | Feature Engineering | **Complete** |
| 3 | ML Model Development | **Phase 3 reopened 2026-04-14 after decisive v11 triple-barrier ablation returned first positive VP result in project history.** Prior: 5 post-audit experiments rejected, audit retracted Eval 11/12/17/18. v11 identified the binding constraint (all prior label formulas shared inputs with VP features, making the VP hypothesis unfalsifiable). 2026-04-14: triple-barrier labels + v11-full vs v11-nopv ablation → full beats nopv on every split, larger lift on holdout than in-sample, **v11-full at conf≥0.80 produces +11.6% holdout CAGR / 8.2% DD / 58.6% WR** (first positive holdout CAGR + first holdout DD below v10 honest's 18.4%). Signal is weak (+2–5 pp accuracy) but regime-robust. Next: multi-asset v11-tb on BTC + ETH + SOL to test universality vs BTC-specific microstructure. See `experiments/LABEL_REDESIGN.md` §Results, `EVAL_AUDIT.md` §9 Stage 6, `MODEL_HISTORY.md` §§30–31. |
| 4 | Kraken API Integration | Not started — not green-lit |
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

