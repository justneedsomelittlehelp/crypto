# Training Runs (`experiments/runs/`)

Per-fold training checkpoints from walk-forward evaluations. Each `run_<unix_timestamp>/` directory is one model training run produced by `train_model()` in `src/models/trainer.py`.

## Structure of each run directory

```
run_1775617546/
├── model.pt        # PyTorch state_dict of best-val-loss checkpoint
├── metrics.json    # Per-epoch train/val loss and accuracy
└── config.json     # Hyperparameters used (lr, epochs, model class, etc.)
```

## Why so many directories

Walk-forward evaluation produces 10 folds × 1+ runs each. Most multi-evaluation experiments produced 10+ runs. With ~30+ experiments over the project's history, that's 300-500+ run directories.

## How to identify which eval produced a run

See `experiments/RUN_INDEX.md` — it maps each `run_<timestamp>` to the eval it belongs to.

## Cleanup considerations

- These directories are large (each ~5-10 MB depending on model size)
- They're mostly historical — current best (Eval 12) generated its own runs
- Can safely delete runs from archived eval scripts if disk space matters
- Keep recent runs in case you need to reload a checkpoint for analysis

## Gitignored

This entire directory is **gitignored** (via `experiments/runs/` in `.gitignore`). The numerous binary checkpoint files would bloat the repo. They're regenerated whenever you re-run an eval script.
