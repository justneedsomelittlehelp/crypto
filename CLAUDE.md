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
| Vanilla RNN evals (4 evals, exhausted) — archived | `experiments/archive/historical_evals/EVAL_VANILLA_RNN.md` |
| LSTM evals (7 evals, exhausted) — archived | `experiments/archive/historical_evals/EVAL_LSTM.md` |
| 1D CNN evals (26 evals, best: 61.5% on 4h) — archived | `experiments/archive/historical_evals/EVAL_CNN.md` |
| **Transformer evals (active, through Eval 20 cadence ablation)** | `experiments/EVAL_TRANSFORMER.md` |
| TP/SL sweep (11 configs, long+short, FGI vs SMA regime) — archived | `experiments/archive/historical_evals/EVAL_TPSL_SWEEP.md` |
| Dual-branch Transformer (VP + Candle, separate attention) — archived | `experiments/archive/historical_evals/EVAL_DUAL_BRANCH.md` |
| **Model evolution history (living doc)** | `experiments/MODEL_HISTORY.md` |
| **⚠ Audit post-mortem — retracts Eval 11/12/17/18** | `experiments/EVAL_AUDIT.md` |
| **v10: 90-day temporal × 30-day VP (post-audit experiment)** | `architecture_docs/arch-ml-model.md` §v10 |
| **Label redesign — triple-barrier options, v11 post-mortem** | `experiments/LABEL_REDESIGN.md` |
| **⭐ Prediction cadence ablation — 1h vs 15m, match-epochs + matched-gradient-steps** | `experiments/EVAL_CADENCE.md` |
| **⭐ v12 regime-aware model (conv encoder + macro day enrichment)** | `src/models/architectures/v12_regime.py`, `src/models/eval_v12.py` |
| **Regime data pipeline (VIX/DXY/GLD/USO hourly + FRED daily)** | `src/data/fetch_regime.py` |
| **Multi-asset plan → reframed as regime features plan** | `experiments/MULTI_ASSET_PLAN.md` |
| **⭐ Feature statistical testing (Stage 1 IC screening, methodology + results)** | `stat_test/README.md`, `stat_test/how_to_stat_test.md` |
| **⭐ HMM macro-regime detector (frozen 2026-04-20) — iteration log, frozen config, key lessons** | `stat_test/HMM_LOG.md` |
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
| 3 | ML Model Development | **2026-04-16: Stage 8 matched-gradient-steps (200 epochs, 3 independent seeds) CONFIRMED undertraining hypothesis and produced FIRST all-seeds-positive holdout CAGR.** conf75_pyr holdout: +20.0/+6.2/+4.0% (mean +10.1%, 135/116/123 trades). conf80_pyr holdout (newly reachable): +9.5/+0.5/+3.7% (mean +4.6%, 114/94/96 trades). Non-pyramid conf75: +1.2/+0.2/+0.7% (mean +0.7%, 1pp spread, 60.7% WR, ~20 trades) — **tightest reproducible signal in project history**. Confidence ceiling rose 0.77→0.94, threshold trade counts stabilized, fold 4 mode collapse resolved. **Red flag: fold 12 (2026 Q1) raw accuracy crashed to ~30% on all seeds** — model appropriately more uncertain on this regime (lower p50) but high-confidence precision is seed-dependent (59%/0%/18% at conf80). This is the regime-awareness problem, not calibration. Walk-forward acc slightly degraded (0.5135 mean vs 0.5276 on match-epochs — mild overfit). **Prior (2026-04-15)**: Stage 7 single-seed "+6.1% holdout CAGR" retracted after 3-seed match-epochs replication showed seed noise dominated (51/0/6 trades at conf75). **Binding constraint has shifted** from calibration instability (solved) to regime awareness (unsolved). **Next**: regime conditioning features (VIX/DXY/yield-curve/FFR, per `MULTI_ASSET_PLAN.md §REFRAME`) — promoted from #4 to #1 priority. Directly attacks the fold 12 problem. Then rolling retraining, then envelope dynamics. See `experiments/EVAL_CADENCE.md §10` for matched-gradient-steps writeup, `MODEL_HISTORY.md §33`. |
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

