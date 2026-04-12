# Logs (`logs/`)

Training run logs from past Colab sessions. **All files are gitignored.**

## What's here

Plain text logs captured during long training runs (LR sweeps, walk-forward evals, etc.). Useful for diagnosing failures or tracing what hyperparameters produced what results.

Naming convention: `<eval_name>_<context>.log`

Examples:
- `dual_model_output.log` — DualBranch v3 run
- `dualbranch_colab.log` — DualBranch on Colab A100
- `lr_sweep_results.log` — LR sweep across 6 configs
- `regime_compare_output.log` — SMA vs FGI regime comparison
- `strategy_temporal_output.log` — Strategy verification with temporal model

## Why gitignored

- Large (some are 10-50 MB)
- Mostly redundant with the result JSONs
- Useful only for ad-hoc debugging, not reproducibility

## How they're produced

When running on Colab, output gets piped to a `.log` file via `&> logs/<name>.log` or similar. The most recent runs use `print(... flush=True)` directly to the notebook output, so no separate log file is created.

If you want to capture future runs:
```bash
python -m src.models.eval_v6_prime 2>&1 | tee logs/eval_v6_prime_$(date +%Y%m%d).log
```
