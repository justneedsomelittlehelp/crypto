# Phase 3: ML Model Development

> **Status (2026-04-12): Audit complete — strategy shelved.** A pre-deployment audit uncovered a walk-forward label leak (no embargo between folds, first-hit labels looking 14 days ahead) and a test-set sweep of the post-SL pause parameter. The Eval 17 "deployable strategy" (+34% CAGR / −15% DD / 72% win rate) does not reproduce under clean labels. Honest best config: `conf_70 + tp/sl guard + 1x / 20% + 24h pause` → **+6.0% CAGR / −18.4% DD / 65% win rate / holdout ≈ −5%**, which does not beat passive SPY. Project paused at end of Phase 3 with infrastructure preserved. See `experiments/EVAL_AUDIT.md` for the full post-mortem.

## Goal
Build, train, evaluate, and iterate on a PyTorch model that generates trading signals from the feature set.

## What This Phase Builds On
Phase 2 — feature engineering pipeline, PyTorch Dataset, train/val/test splits.

## Implementation Steps

1. **Define model architecture** (`src/models/architecture.py`)
   - ~~Vanilla RNN~~ — exhausted, 4 evals, ~50% accuracy (random)
   - ~~LSTM~~ — exhausted, 7 evals, best 53.5% val acc but didn't generalize
   - **1D CNN** — next, treats VP as spatial signal matching user's strategy
   - Key insight: user reads VP as a histogram shape (peaks = support/resistance), not a time series. Sequential models (RNN/LSTM) are the wrong paradigm. See `experiments/STRATEGY.md`

2. **Training loop** (`src/models/trainer.py`)
   - Standard PyTorch training: DataLoader, optimizer (Adam), loss function, LR scheduler
   - Loss function depends on the task (MSE for regression, CrossEntropy for classification)
   - Logging: loss curves, validation metrics per epoch
   - Early stopping based on validation loss
   - Model checkpointing (save best model weights)

3. **Evaluation & metrics** (`src/models/evaluate.py`)
   - Financial-specific metrics beyond accuracy:
     - Sharpe ratio of predicted signals (simulated)
     - Max drawdown of hypothetical trades
     - Win rate, profit factor
     - Confusion matrix for directional predictions
   - Walk-forward validation: retrain periodically on expanding window

4. **Backtesting framework** (`src/models/backtest.py`)
   - Simulate trades based on model predictions on the test set
   - Account for:
     - Trading fees (Kraken maker/taker fees)
     - Slippage estimation
     - Position sizing
   - Output: equity curve, trade log, performance summary

5. **Experiment tracking**
   - Simple approach: save hyperparams + metrics to JSON per run
   - Directory: `experiments/{run_id}/` with model weights, config, metrics
   - Optional: integrate Weights & Biases or MLflow later

## Test When Done
- [ ] Model trains without errors on the full training set
- [ ] Validation loss decreases and stabilizes (model is learning, not just memorizing)
- [ ] Backtest produces a trade log and equity curve on test data
- [ ] Model checkpoint can be loaded and used for inference
- [ ] Results are reproducible (fixed random seeds)
