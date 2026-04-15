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
| **⭐ Prediction cadence ablation — 1h vs 15m (first positive holdout, single seed)** | `experiments/EVAL_CADENCE.md` |
| **Multi-asset plan — next experiment, BTC+ETH Stage 1 scope** | `experiments/MULTI_ASSET_PLAN.md` |
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
| 3 | ML Model Development | **⚠ 2026-04-15 evening: Stage 7 single-seed "+6.1% holdout CAGR" RETRACTED.** 3-seed independent replication (base seeds 42/43/44, commit `647cc24` adds `--base-seed` arg) did **not** reproduce the positive holdout result. At `conf75_pause24_pyr` holdout: seed42 = +6.1% (51 trades) → seed43 = 0.0% (**0 trades**) → seed44 = −3.0% (**6 trades**), mean +1.0%, spread 9pp. At `conf70_pause24_pyr` holdout (where trade counts are stable ~60 across seeds): seed42 = −10.6%, seed43 = +9.8%, seed44 = −6.3%, **mean −2.4%** (spread 20pp). Seed42 was simultaneously the best walk-forward acc *and* the best conf75_pyr holdout *and* the worst conf70_pyr holdout — classic signature of seed noise dominating signal. **What's stable across seeds**: walk-forward acc (0.5318/0.5224/0.5286, mean 0.5276), full-period CAGR at conf75_pyr (+6.6/+5.2/+6.5, mean +6.1%), full-period drawdown reduction vs 15m (−10 to −14% at 1h vs ~−40% at 15m pyr). **What's not stable**: holdout CAGR, confidence-threshold trade counts, per-fold n_long. **Fold 4 mode collapse is now a known model failure**, not seed noise — all three seeds produced n_long=0 on fold 4 (4061 samples each, 2022 H1 bear). **Retracted claims**: "first positive real-engine holdout CAGR", "beats v6-prime on holdout by ~11pp", any fixed-threshold comparison between 15m and 1h. **Kept**: walk-forward acc equivalence, methodological separation of data resolution from prediction cadence, 1h operational speedup (3–4×). **Prior (2026-04-14)**: v11-full vs v11-nopv triple-barrier ablation validated VP features (Δ +1.5 to +8.5 pp CAGR on every real-engine holdout filter under the single-seed framing). v6-prime honest (+6.0% full-period CAGR, −18.4% DD, holdout ~−5%) is back to nominal best. **Revised next**: matched-gradient-steps at 1h cadence with **3 independent seeds from the start** (200 epochs, `--base-seed {42,43,44}`). Hypothesis: under-convergence produces the calibration instability, so full training should (a) raise confidence ceiling above 0.80 and (b) stabilize conf75/80 holdout trade counts across seeds. If this fails, the conf-threshold framing is fundamentally fragile and we abandon it. **Envelope dynamics and regime features are deferred until we have a reproducible baseline.** See `experiments/EVAL_CADENCE.md §9` for the full 3-seed post-mortem, `MODEL_HISTORY.md §32` evening addendum, `LABEL_REDESIGN.md §2026-04-15`. |
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

