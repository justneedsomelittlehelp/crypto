# Models Archive (`src/models/archive/`)

Historical eval and sweep scripts from earlier experiments. These produced results that have been superseded — kept here for reproducibility, NOT for new work.

## Why archive instead of delete

- Old results in `experiments/archive/*.json` reference these scripts
- If you ever need to reproduce a historical eval, the script is still here
- Some patterns (e.g., regime detection logic) may be useful for new experiments

## What's here

### Eval scripts (Eval 1-9 era)
| File | What it tested | Result era |
|------|----------------|------------|
| `eval_strategy.py` | Pre-Eval 9: strategy verification with temporal model on 7.5/3 labels | TPSL sweep era |
| `eval_temporal.py` | v2 temporal vs CNN comparison | Eval 1-2 era |
| `eval_dualbranch.py` | DualBranch v1: separate VP + candle attention paths | Pre-Eval 5, abandoned |
| `eval_dual_model.py` | DualBranch v3: pipeline overhaul attempt | Pre-Eval 5, broken |
| `eval_v6_vs_v2.py` | v6 enriched vs v2 baseline head-to-head | Eval 6 era |
| `eval_v6_tpsl.py` | v6 with various TP/SL ratios | Pre-Eval 6 |
| `eval_24h_horizon.py` | 24h fixed-horizon labels (NOT first-hit) | Found: VP cannot predict 24h direction (50% accuracy) |
| `eval_regime_compare.py` | SMA vs FGI regime detection | Pre-Eval 5 |
| `eval_regime_fgi.py` | FGI-only regime test | Pre-Eval 5 |
| `eval_reproduce_v1.py` | Pipeline reproduction sanity check | Pre-Eval 6 |
| `eval_finetune_funding.py` | Eval 9: v6 + funding rate fine-tune. Mixed results, fold 6 catastrophic collapse. **Abandoned.** |

### Sweep scripts
| File | What it swept |
|------|---------------|
| `sweep_tpsl.py` | TP/SL grid search → produced `EVAL_TPSL_SWEEP.md` findings (7.5/3 best) |
| `sweep_tpsl_wide.py` | Wider TP/SL search range |
| `sweep_lr.py` | Learning rate sweep (found: LR has minimal impact, regularization matters more) |
| `sweep_diagnose.py` | Diagnostic for sweep failures |
| `analyze_sweep.py` | Sweep result analysis utility |

## How to find what produced a specific result

1. Check `experiments/MODEL_HISTORY.md` for the eval-by-eval narrative
2. Cross-reference with `experiments/RUN_INDEX.md` for run folder mapping
3. Look at the corresponding archived script for implementation details

## Restoration

If you ever need to actively use one of these:
1. Read its docstring to understand what config it expects
2. Move it back to `src/models/` (just `git mv`)
3. Update `src/models/README.md` to add it back to the active list
4. Update memory's `project_structure.md` to reflect the change
