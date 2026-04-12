# Experiments Archive (`experiments/archive/`)

Historical/superseded result JSONs from earlier experiments. Kept for reference, NOT for active analysis.

## Why archive instead of delete

- Some are referenced by `experiments/MODEL_HISTORY.md` for the historical narrative
- If you want to compare current results vs an old eval, the raw numbers are still here
- Old sweep results contain hyperparameter search data that may inform future tuning

## Files

| File | Originating script | Era |
|------|--------------------|-----|
| `diagnose_results.json` | `archive/sweep_diagnose.py` | Pre-Eval 5 (DualBranch troubleshooting) |
| `dual_model_results.json` | `archive/eval_dual_model.py` | Pre-Eval 5 (DualBranch v3) |
| `dualbranch_compare_results.json` | `archive/eval_dualbranch.py` | Pre-Eval 5 |
| `dualbranch_v3_results.json` | `archive/eval_dualbranch.py` | Pre-Eval 5 |
| `eval_finetune_funding_results.json` | `archive/eval_finetune_funding.py` | Eval 9 (funding rate, abandoned) |
| `horizon_24h_results.json` | `archive/eval_24h_horizon.py` | Pre-Eval 8 (24h labels = random) |
| `regime_compare_results.json` | `archive/eval_regime_compare.py` | Pre-Eval 5 (SMA vs FGI) |
| `regime_fgi_compare_results.json` | `archive/eval_regime_fgi.py` | Pre-Eval 5 |
| `reproduce_v1_results.json` | `archive/eval_reproduce_v1.py` | Pipeline reproduction sanity check |
| `strategy_results.json` | `archive/eval_strategy.py` | TP/SL sweep era — used by EVAL_TPSL_SWEEP.md |
| `sweep_analysis.json` | `archive/analyze_sweep.py` | TP/SL sweep results |
| `sweep_analysis_combined.json` | `archive/analyze_sweep.py` | Combined sweep analysis |
| `sweep_tpsl_results.json` | `archive/sweep_tpsl.py` | TP/SL grid search (found 7.5/3 best) |
| `sweep_tpsl_wide_results.json` | `archive/sweep_tpsl_wide.py` | Wider TP/SL search |
| `temporal_compare_results.json` | `archive/eval_temporal.py` | v2 temporal vs CNN comparison |

## Key findings preserved here

- **24h direction labels = random** (`horizon_24h_results.json`): VP cannot predict 24h direction. Validates first-hit approach.
- **DualBranch hurts** (`dualbranch_*.json`): separate candle attention adds noise. Validates v6 single-path enrichment.
- **TP/SL 7.5/3 optimal** (`sweep_tpsl_*.json`): wide TP + tight SL maximizes EV in bull markets, validated across 11 configs.
- **Funding rate brittle** (`eval_finetune_funding_results.json`): worked on most folds but fold 6 catastrophic. Need better gating.
