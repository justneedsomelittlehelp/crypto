# Models Module (`src/models/`)

ML model definitions, training scripts, eval scripts, walk-forward backtest infrastructure.

## ⭐ Active scripts (use these)

| File | Purpose |
|------|---------|
| **`eval_v6_prime.py`** | ⭐⭐ Current best — v6-prime + 3-seed ensemble + SWA + filter analysis. Saves predictions cache for backtest. |
| **`run_backtest.py`** | ⭐⭐ Realistic backtest engine driver. Loads predictions cache, runs 12 filter×sizing combinations. |
| `eval_2plus1.py` | Reference — v7/v8 (2 spatial + 1 temporal) on 1h data |
| `eval_15min.py` | Reference — v6/v8 on 15min data |

## Infrastructure (used by all scripts)

| File | Purpose |
|------|---------|
| `__init__.py` | Module marker |
| `__main__.py` | CLI entry point (not heavily used; most scripts run via `python -m`) |
| `architecture.py` | **Mutable** architectures: RNN, LSTM, CNN, original Transformer. Used by archived scripts. |
| `trainer.py` | Generic `train_model()` function with early stopping, checkpointing |
| `walk_forward.py` | Generic walk-forward fold utilities |
| `evaluate.py` | Standard metric helpers |
| `rule_based.py` | Baseline rule-based model for comparison |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `architectures/` | **FROZEN** model snapshots (v2, v5, v6, v6-prime, v7, v8). See `architectures/README.md`. |
| `archive/` | **Historical** eval scripts from earlier evals (1-9). Kept for reference, do not use for new work. See `archive/README.md`. |

## Workflow

```bash
# Train current best (~45 min on Colab A100)
python -m src.models.eval_v6_prime

# Run backtest on cached predictions (~5 sec)
python -m src.models.run_backtest

# Reference experiments (also run via `python -m`)
python -m src.models.eval_2plus1
python -m src.models.eval_15min
```

## Active vs archive split

- **Active scripts** at this level produce results referenced in `experiments/MODEL_HISTORY.md`.
- **Archived scripts** in `archive/` produced earlier results that have been superseded. They're kept so historical numbers can be reproduced if needed.

For full file inventory and decision history, see `experiments/MODEL_HISTORY.md`.
