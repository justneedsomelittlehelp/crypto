# Phase 3: ML Model Development

## Goal
Build, train, evaluate, and iterate on a PyTorch model that generates trading signals from the feature set.

## What This Phase Builds On
Phase 2 — feature engineering pipeline, PyTorch Dataset, train/val/test splits.

## Implementation Steps

1. **Define model architecture** (`src/models/architecture.py`)
   - Architecture depends on the user's strategy and signal type
   - Common starting points for time-series financial data:
     - **LSTM/GRU**: good baseline for sequential patterns
     - **Transformer**: better at capturing long-range dependencies in VP data
     - **1D CNN + LSTM hybrid**: fast local pattern extraction + temporal modeling
   - The 50-bin VP vector per timestep is naturally suited to attention or conv layers
   - Start simple (LSTM), iterate toward complexity only if justified by validation metrics

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
