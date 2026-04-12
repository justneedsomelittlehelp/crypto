# Architecture Documentation (`architecture_docs/`)

Domain-specific design documents. Read the relevant doc before modifying code in that domain — these capture decisions and constraints that aren't obvious from the code alone.

## Files

| File | Domain |
|------|--------|
| `arch-data-pipeline.md` | Data scraping, OHLCV merging, volume profile computation, CSV format |
| `arch-ml-model.md` | Model architecture spec, training methodology, walk-forward eval design |
| `arch-trading-engine.md` | (Phase 4-5) Kraken API integration, order execution, position tracking, bot loop |
| `arch-risk-safety.md` | (Phase 4+) Risk limits, circuit breakers, loss thresholds, safety rules |
| `arch-reference.md` | Lookup index: file paths, data column names, config parameters |

## Convention

These docs are **append-only living documents**. When a major architectural decision is made, add a new section explaining:
- What we decided
- Why (the reasoning)
- What alternatives we considered
- What it means for future code

Avoid deleting old sections — mark them as superseded if needed.

## Relationship to MODEL_HISTORY.md

- `experiments/MODEL_HISTORY.md` — chronological narrative of model evolution (when, what, why)
- `architecture_docs/arch-ml-model.md` — current architecture spec (the "what is" — should always reflect the latest design)

The history doc tells the story; the architecture doc describes the present state.

## Maintenance rule

⚠️ **These docs go stale fast if neglected.** When making a major change (new architecture, new pipeline, new metric, new training procedure):
1. Update `MODEL_HISTORY.md` with the decision and reasoning
2. **Also update the relevant `arch-*.md`** to reflect the new current state
3. Update memory (`milestone_*.md` or `project_phase3_status.md`) with the savepoint

If you skip step 2, these docs become dangling. They were last fully refreshed 2026-04-12 to reflect Eval 12 (v6-prime + ensemble + filter).
