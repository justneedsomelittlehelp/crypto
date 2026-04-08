# ML Model

> **Read this when working on:** model architecture, training, evaluation, backtesting, inference pipeline
> **Related docs:** [arch-data-pipeline.md](arch-data-pipeline.md) (data source), [arch-trading-engine.md](arch-trading-engine.md) (consumes predictions), [arch-risk-safety.md](arch-risk-safety.md) (confidence thresholds)

---

## Overview

PyTorch model that takes historical features (VP + derived) and outputs trading signals. The user owns the strategy — see `experiments/STRATEGY.md` for the full strategy-to-model translation.

## Input Shape

Per timestep, the model receives **61 features**:
- **50 VP bins** — the relative volume profile (primary feature, no scaling)
- **4 derived features** — `log_return`, `bar_range`, `bar_body`, `volume_ratio` (scale-invariant)
- **7 VP structure features** — `vp_ceiling_dist`, `vp_floor_dist`, `vp_num_peaks`, `vp_ceiling_strength`, `vp_floor_strength`, `vp_ceiling_consistency`, `vp_floor_consistency`

No technical indicators — user's thesis is that VP alone carries the signal.

Input shape: `(batch, 180, 61)` — 180 bars (30 days) lookback, 61 features per bar. Lookback is configurable via `LOOKBACK_BARS_MODEL`.

## VP Structure Features

Derived from the user's actual trading strategy (see `experiments/STRATEGY.md`):
1. **Aggregated VP** — sum 50 VP bins across 30d window, smooth with Gaussian filter (sigma=2)
2. **Peak detection** — `scipy.signal.find_peaks` on smoothed profile (prominence=0.15, distance=3)
3. **Ceiling/floor** — nearest peak above/below mid-bin (bin 25 = current price), distance normalized 0-1
4. **Peak consistency** — check if same peaks appear in shifted 30d windows (3d, 6d, 9d back). Higher = more robust level

## Models

### RNN (baseline, exhausted)
- Stacked RNN layers: `54 → 64 → 32 → 16 → 1`, tanh, dropout 0.2
- **Result:** ~50% accuracy across all hyperparameter combinations. Majority-class collapse.
- **Why it failed:** Sequential processing of raw VP bins — no concept of VP as a spatial shape
- See `experiments/EVAL_VANILLA_RNN.md` (4 evals)

### LSTM (exhausted)
- Single/stacked LSTM layers, various hidden sizes [8] to [64,32,16]
- Weighted BCEWithLogitsLoss to fix majority-class collapse
- **Result:** Best val acc 53.5%, but didn't generalize to test set
- **Why it failed:** Even with VP structure features, LSTM processes temporally not spatially. VP ceiling/floor features get diluted across 180 timesteps
- See `experiments/EVAL_LSTM.md` (7 evals)

### 1D CNN (best walk-forward: 61.5%)
- Treats 50-bin VP profile as a spatial signal (like a 1D image)
- Conv filters learn smoothing + peak detection from data
- Best config: Conv1d(1→4, k=5) × 2, flatten, FC(215→16→1), 3,581 params
- Gaussian smoothing sigma=0.8, prominence=0.05 for VP peak detection
- See `experiments/EVAL_CNN.md` (26 evals across 8 phases)

### Transformer (63.3% walk-forward — BEST MODEL)
- Self-attention across 50 VP bins — each bin attends to all others
- Can learn peak-pair relationships (ceiling/floor) directly
- Best config: embed=32, 4 heads, 2 layers, FC(47→64→1), 21,889 params
- Trained on **1h data** (84k rows, 720-bar lookback = 30 days)
- 2 attention layers: first finds peaks, second finds peak-pair relationships
- See `experiments/EVAL_TRANSFORMER.md` (4 evals)

### Key findings across all models
- **Label design matters most:** First-hit labels with regime-adaptive TP/SL was the biggest improvement (+15% over fixed horizon)
- **Data resolution matters:** 1h data (6x more samples) broke past the 4h ceiling (61.5% → 63.3%)
- **Model capacity must match data:** Small Transformer on 1h was worse; larger Transformer on 1h was best
- **Next direction:** 15min data (4x more than 1h) + pipeline optimization

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
