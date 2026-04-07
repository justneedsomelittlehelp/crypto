# ML Model

> **Read this when working on:** model architecture, training, evaluation, backtesting, inference pipeline
> **Related docs:** [arch-data-pipeline.md](arch-data-pipeline.md) (data source), [arch-trading-engine.md](arch-trading-engine.md) (consumes predictions), [arch-risk-safety.md](arch-risk-safety.md) (confidence thresholds)

---

## Overview

PyTorch model that takes historical features (VP + technical indicators) and outputs trading signals. The user owns the strategy and model design — this doc describes the infrastructure around it.

## Input Shape

Per timestep, the model receives:
- **50 VP bins** — the relative volume profile (primary feature)
- **N technical indicators** — added in Phase 2 (RSI, MACD, moving averages, etc.)
- **OHLCV values** — raw price/volume data

The model may see a **sequence of timesteps** (lookback window), making the input shape: `(batch, sequence_length, features)`.

## Training Principles

### Time-series split (critical)
- **Never shuffle** financial time-series data for train/test split
- Train: earliest data → cutoff date
- Validation: cutoff → second cutoff
- Test: second cutoff → end
- Walk-forward validation for robustness: retrain on expanding window

### Preventing overfitting
- Financial data is noisy — overfitting is the primary risk
- Regularization: dropout, weight decay, early stopping
- Keep model capacity proportional to signal-to-noise ratio (start small)
- Monitor: if training loss drops but validation loss doesn't, the model is memorizing

### Label design
- Labels encode the user's trading objective (direction, magnitude, threshold-based)
- Must be computed from **future** price data that is strictly excluded from features
- The label function is pluggable — defined by the user's strategy

## Inference Pipeline

For live trading:
1. New 4h candle closes
2. Fetch latest OHLCV data
3. Compute current VP + indicators using same pipeline as training
4. Feed into model → get prediction
5. Pass prediction to strategy engine

**Critical**: the feature computation in inference must exactly match training. Any difference (different normalization, missing features, different lookback) will degrade predictions silently.

## Experiment Management

Each training run saves to `experiments/{run_id}/`:
- `config.json` — hyperparameters, data splits, feature list
- `model.pt` — best model checkpoint (by validation metric)
- `metrics.json` — training/validation loss curves, final metrics
- `backtest.json` — simulated trading performance

## Backtesting

Simulated trading on the test set to evaluate model quality in financial terms:
- Apply the strategy engine to model predictions
- Account for Kraken fees (maker: 0.16%, taker: 0.26%)
- Estimate slippage
- Output: equity curve, trade log, Sharpe ratio, max drawdown, win rate

## Gotchas

- **Look-ahead bias** — the most common and most dangerous mistake. If any feature leaks future information, backtest results will be wildly optimistic and live performance will be terrible. Audit the feature pipeline carefully.
- **Survivorship bias** — BTC/USD is a survivor; the model hasn't seen coins that went to zero. Keep this in mind when interpreting results.
- **Regime change** — a model trained on 2016-2023 data saw a very different market than 2024+. Walk-forward retraining helps but doesn't eliminate this.
- **Reproducibility** — set random seeds for PyTorch, numpy, and Python's random module. Use deterministic CUDA operations if using GPU.
