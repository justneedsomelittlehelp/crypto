# Crypto Trading Bot

> Automated BTC/USD spot trading bot using Kraken API + PyTorch ML model driven by volume profile features.

## Quick start

```bash
# Scrape data (run once, or whenever you want fresh data)
python -m src.data scrape         # 1h data (default)
python -m src.data scrape 15m     # 15min data

# Train current best model (~45 min on Colab A100)
python -m src.models.eval_v6_prime

# Run backtest on cached predictions (~5 sec)
python -m src.models.run_backtest
```

## Directory map

| Directory | Purpose |
|-----------|---------|
| `src/` | All Python source code (data scraping, features, models, backtest engine) |
| `experiments/` | Eval result JSONs, model history, eval logs (`EVAL_*.md`), per-fold checkpoints |
| `architecture_docs/` | Domain architecture docs — read these before modifying code in that domain |
| `roadmap/` | Phase-by-phase build plan (Phase 1-7) |
| `security_check/` | Threat model and security audit doc |
| `data/` | Raw scraped data (gitignored except merged CSVs) |
| `logs/` | Training run logs (gitignored) |
| `tests/` | Unit and integration tests |

## Key project docs

- **`CLAUDE.md`** — instructions for AI coding assistants. Routes to architecture docs and eval logs by topic.
- **`README.md`** (this file) — human entry point.
- **`experiments/MODEL_HISTORY.md`** — living document of every architecture decision from RNN → current. Read this for full context.
- **`experiments/STRATEGY.md`** — user's manual trading strategy (the model is built around this).
- **`roadmap/README.md`** — phase progression and tech stack rationale.

## Current state (2026-04-12)

- **Phase 3 (ML model)**: ⭐⭐ v6-prime + 3-seed ensemble + SWA + combined filter → +3.98% per trade, 78.4% precision, Sharpe 0.97
- Backtest engine in place — converts model predictions into realistic portfolio simulation with fees, slippage, and capital constraints
- Phases 4-7 (Kraken API, live trading, monitoring): not yet started

## How to navigate

If you're new to the project, read in this order:

1. `CLAUDE.md` — high-level orientation
2. `experiments/MODEL_HISTORY.md` — model evolution from RNN to current best
3. `experiments/STRATEGY.md` — what the user is actually trying to trade
4. `architecture_docs/arch-ml-model.md` — model architecture spec
5. `src/models/README.md` — what each script in src/models/ does
6. `src/backtest/README.md` — how the backtest engine works
