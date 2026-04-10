# Frozen Feature Pipelines

Each file here is a **frozen snapshot** of a feature pipeline. Once committed
with results, it should never be modified.

## Rules

1. **No edits to frozen files.** If you need to iterate, create a new `vN_*.py`
   file next to the existing ones.
2. **Import explicitly.** Eval scripts should import the specific version they
   want, along with that version's `FEATURE_COLS` list:
   ```python
   from src.features.pipelines.v1_raw import build_feature_matrix_v1, FEATURE_COLS_V1
   ```
3. **Pass feature_cols to TimeSeriesDataset.** Don't rely on the global
   `FEATURE_COLS` from `src.config` — pass the pipeline's own list explicitly:
   ```python
   ds = TimeSeriesDataset(df, feature_cols=FEATURE_COLS_V1)
   ```
4. **Document provenance.** Each file's docstring must list:
   - What features it produces (with indices)
   - What changed vs the previous version
   - Which architectures it's compatible with

## Current versions

| File | Features | Key differences | Best pairing |
|------|----------|-----------------|--------------|
| `v1_raw.py` | 68 (50 VP + 10 derived + 8 VP structure) | Original, no scaling | `v2_temporal` (61.9%) |
| `v2_scaled.py` | 60 (50 VP + 10 derived) | No VP structure, z-score + tanh, expanding window | `v5_dualbranch_cls` |

## Naming convention

- `vN_shortname.py` where N is the chronological version.
- Function name inside = `build_feature_matrix_v{N}`.
- Feature list constant = `FEATURE_COLS_V{N}`.
- Helper: `feature_index_v{N}(col_name)` to get index of a column.
