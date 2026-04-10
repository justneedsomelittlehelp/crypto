# Frozen Architectures

Each file here is a **frozen snapshot** of an architecture. Once a file is
committed with results, it should never be modified.

## Rules

1. **No edits to frozen files.** If you need to iterate, create a new `vN_*.py`
   file next to the existing ones.
2. **Import by exact class name.** Eval scripts should import the specific
   version they want: `from src.models.architectures.v2_temporal import TemporalTransformerV2`.
3. **Hardcode dimensions.** Frozen architectures should NOT read from
   `src.config` mutable globals. Dimensions should be constructor args or
   class constants.
4. **Document provenance.** Each file's docstring must list:
   - Its best validated result (accuracy, pipeline used, date)
   - What changed vs the previous version
   - The input contract (expected feature count, ordering)

## Current versions

| File | Architecture | Best result | Pipeline |
|------|--------------|-------------|----------|
| `v2_temporal.py` | Temporal Transformer (VP only) | 61.9% acc, 0/10 folds <50% | v1_raw |
| `v5_dualbranch_cls.py` | Dual-branch with CLS + volume | 60.5% acc, Strategy 4 +0.71%/day | v2_scaled |

## Naming convention

- `vN_shortname.py` where N is the chronological version and shortname
  describes the key feature (e.g., `v2_temporal`, `v5_dualbranch_cls`).
- Class name inside = `{CapitalizedShortname}V{N}` (e.g., `TemporalTransformerV2`).
