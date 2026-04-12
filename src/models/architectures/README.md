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
| `v2_temporal.py` | Temporal Transformer (VP only, 1 spatial + 1 temporal) | 61.9% acc, 0/10 folds <50% | v1_raw |
| `v5_dualbranch_cls.py` | Dual-branch with CLS + volume (abandoned: candle branch hurt -3%) | 60.5% acc | v2_scaled |
| `v6_temporal_enriched.py` | Single-path temporal + day enrichment (1 spatial + 1 temporal) | 58.4% acc, **bull long EV +1.57%** | v1_raw |
| `v7_simple_2plus1.py` | 2 spatial + 1 temporal, no enrichment, mean pool | 56.2% acc — extra spatial layer not worth it | v2_scaled |
| `v8_enriched_2plus1.py` | 2 spatial + 1 temporal + enrichment + CLS pool | 58.0% acc on 1h, 59.6% on 15min | v1_raw |
| `v6_prime_vp_labels.py` | ⭐⭐ v6 architecture for VP-derived labels (dropout 0.3 default) | **+3.98% EV/trade with combined filter** (Eval 12) | v1_raw |

## Active vs historical

**Currently used in active eval scripts:**
- `v6_prime_vp_labels.py` ← `eval_v6_prime.py` (current best)
- `v6_temporal_enriched.py` ← `eval_15min.py` (reference)
- `v8_enriched_2plus1.py` ← `eval_15min.py`, `eval_2plus1.py` (reference)
- `v7_simple_2plus1.py` ← `eval_2plus1.py` (reference)

**Historical only (referenced in archived eval scripts):**
- `v2_temporal.py`, `v5_dualbranch_cls.py`

## Naming convention

- `vN_shortname.py` where N is the chronological version and shortname
  describes the key feature (e.g., `v2_temporal`, `v5_dualbranch_cls`).
- Class name inside = `{CapitalizedShortname}V{N}` (e.g., `TemporalTransformerV2`).
