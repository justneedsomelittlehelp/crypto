# Phase 2: Feature Engineering

## Goal
Build a feature engineering pipeline that computes technical indicators and prepares training-ready datasets for the ML model.

## What This Phase Builds On
Phase 1 — clean project structure, refactored data pipeline, validated OHLCV + VP data.

## Implementation Steps

1. **Design feature set**
   - The 50-bin relative VP is already the core feature
   - Add standard technical indicators as supplementary features:
     - Moving averages (SMA/EMA at multiple windows)
     - RSI, MACD, Bollinger Bands
     - ATR (Average True Range) for volatility
     - Volume-based indicators (OBV, VWAP approximation)
   - All features must be computed from data available at prediction time (no future leakage)

2. **Create `src/features/indicators.py`**
   - Functions for each indicator, operating on pandas DataFrames
   - Each function takes OHLCV columns and returns new column(s)

3. **Create `src/features/pipeline.py`**
   - Orchestrates: load raw data → compute indicators → merge with VP → output feature matrix
   - Handle NaN rows from indicator warmup periods (drop or forward-fill, document which)

4. **Create `src/features/dataset.py`**
   - PyTorch `Dataset` class for the feature matrix
   - Configurable lookback window (how many past bars the model sees)
   - Proper train/validation/test split — **time-based, not random** (critical for financial data)
   - Split strategy: e.g., train on 2016-2023, validate on 2023-2024, test on 2024+

5. **Label generation**
   - This depends on the user's trading strategy
   - Provide a pluggable label function interface: given future price data, return a label
   - Example labels: next-bar return direction, next-24h return magnitude, custom thresholds

6. **Feature normalization**
   - StandardScaler or MinMaxScaler fitted on training set only
   - Save scaler parameters for inference time
   - VP columns may not need scaling (already normalized to sum=1)

## Test When Done
- [ ] Feature pipeline produces a complete feature matrix with no NaNs in the usable range
- [ ] Train/val/test splits are strictly chronological — no future data in training
- [ ] PyTorch Dataset loads batches correctly; tensor shapes match expectations
- [ ] Features are reproducible: same input → same output
