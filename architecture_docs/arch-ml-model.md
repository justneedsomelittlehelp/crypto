# ML Model

> **Read this when working on:** model architecture, training, evaluation, backtesting, inference pipeline
> **Related docs:** [arch-data-pipeline.md](arch-data-pipeline.md) (data source), [arch-trading-engine.md](arch-trading-engine.md) (consumes predictions), [arch-risk-safety.md](arch-risk-safety.md) (confidence thresholds)

---

## Overview

PyTorch model that takes historical features (VP + technical indicators) and outputs trading signals. The user owns the strategy and model design — this doc describes the infrastructure around it.

## Input Shape

Per timestep, the model receives **54 features**:
- **50 VP bins** — the relative volume profile (primary feature, no scaling)
- **4 derived features** — `log_return`, `bar_range`, `bar_body`, `volume_ratio` (scale-invariant, no scaler needed)

No technical indicators — user's thesis is that VP alone carries the signal.

Input shape: `(batch, 42, 54)` — 42 bars (1 week) lookback, 54 features per bar. Lookback is configurable via `LOOKBACK_BARS_MODEL`.

## Current Model: RNN

Stacked RNN layers with decreasing hidden sizes: `54 → 64 → 32 → 16 → 1`
- 3 RNN layers (each a separate `nn.RNN` to allow different sizes)
- tanh activation, 0.2 dropout between layers
- Last timestep's hidden state → linear → logit
- BCEWithLogitsLoss for binary classification
- **Label**: 1 if price up 24h later, 0 otherwise

### Baseline Results (RNN)
- ~52% test accuracy (near random — expected for vanilla RNN on financial data)
- Model biases toward predicting UP
- Next: try LSTM, Transformer, CNN, or Markov chain

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
- Binary label: `1` if `close[t+6] > close[t]`, `0` otherwise (next-24h direction)
- Computed in `TimeSeriesDataset` from close prices — not stored in CSV
- Must use strictly future data excluded from features

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
