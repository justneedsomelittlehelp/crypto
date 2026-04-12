# Features Module (`src/features/`)

Feature engineering: derives candle/volume/VP-structure features from raw OHLCV+VP data, then provides a `TimeSeriesDataset` for model training.

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | **Mutable** active pipeline. Use frozen versions in `pipelines/` for experiments. |
| `dataset.py` | `TimeSeriesDataset` class — slices feature DataFrame into (lookback, features) windows with first-hit TP/SL labels. Used by all model training scripts. |
| `vp_structure.py` | Helpers for VP peak detection (Gaussian smoothing, peak finding). Used by frozen v1_raw pipeline. |

## Subdirectory

| Directory | Purpose |
|-----------|---------|
| `pipelines/` | **FROZEN** pipeline snapshots. See `pipelines/README.md`. |

## Pipeline versioning

- **`pipelines/v1_raw.py`** — 68 features (50 VP + 10 derived + 8 VP structure). Used by v6, v6-prime, v8.
- **`pipelines/v2_scaled.py`** — 60 features (no VP structure, z-score + tanh). Used by v7.

## Usage

```python
# Load features
from src.features.pipelines.v1_raw import build_feature_matrix_v1, FEATURE_COLS_V1
df = build_feature_matrix_v1()

# Build dataset
from src.features.dataset import TimeSeriesDataset
ds = TimeSeriesDataset(df, lookback=720, feature_cols=FEATURE_COLS_V1)
```

## Why frozen pipelines?

Each model architecture is paired with a specific pipeline version. If we changed `pipeline.py` after running v6 evals, those results would no longer be reproducible. Frozen versions guarantee that any model can be retrained from scratch and produce the same features.

The mutable `pipeline.py` is for experimenting with new feature ideas. Once results validate, the new logic gets frozen as `pipelines/vN_*.py`.
