"""Eval v9: v6-prime binary classification pipeline + v9 wall-aware architecture.

Stage 2 of the post-audit experiment. Philosophy D (continuous regression
labels) was tested in Stage 1 and failed on heterogeneous folds — the model
couldn't learn a regression surface across 6 years of mixed regimes. We
reverted to v6-prime's binary first-hit classification loss, which is the
structurally correct match for the bimodal TP/SL label distribution, AND
tested it with the v9 architecture that gives the spatial attention layer
explicit access to VP structure features via a context token.

Key differences from eval_v6_prime:
  - Imports `TemporalEnrichedV9WallAware` instead of `TemporalEnrichedV6Prime`.
  - Output paths: eval_v9_results.json, v9_predictions.npz.
  - Everything else (labels, loss, training loop, walk-forward, embargo,
    holdout, SWA, multi-seed ensemble) is identical to v6-prime, imported
    directly. Same pipeline, one architectural knob turned.

Usage:
    python -m src.models.eval_v9
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

from src.config import (
    LOOKBACK_BARS_MODEL, BARS_PER_DAY, EXPERIMENTS_DIR, LABEL_FGI_PATH,
    LABEL_MAX_BARS, EPOCHS, EARLY_STOP_PATIENCE, REL_SPAN_PCT,
)
from src.features.pipelines.v1_raw import (
    build_feature_matrix_v1, FEATURE_COLS_V1, VP_STRUCTURE_COLS_V1,
    DERIVED_FEATURE_COLS_V1, feature_index_v1,
)
from src.models.architectures.v9_wall_aware import TemporalEnrichedV9WallAware

# Reuse everything that isn't model-specific from eval_v6_prime. This is a
# deliberate dependency: Stage 2 is a strict architectural A/B against the
# v6-prime baseline, so the rest of the pipeline must be bit-for-bit
# identical. If eval_v6_prime.py ever needs modification, it should be a
# new frozen copy (eval_v6_prime2.py etc.), not edited in place.
from src.models.eval_v6_prime import (
    # Config constants (shared with v6-prime baseline)
    BATCH_SIZE, LR, WEIGHT_DECAY, LABEL_SMOOTHING, DROPOUT,
    NUM_WORKERS, USE_COMPILE,
    TP_RATIO, SL_RATIO, TP_MIN, TP_MAX, SL_MIN, SL_MAX, MIN_PEAKS,
    N_SEEDS, SWA_START_EPOCH, SWA_UPDATE_FREQ,
    FOLD_BOUNDARIES, HOLDOUT_START, EMBARGO_BARS,
    # Utility functions
    precompute_vp_labels,
    FastDataset,
    build_regime_array,
    get_device,
    make_loader,
    train_fold,
    evaluate_fold,
    count_params,
)


# ═══════════════════════════════════════════════════════════════════
# Model builder — v9 wall-aware
# ═══════════════════════════════════════════════════════════════════
def build_v9():
    return TemporalEnrichedV9WallAware(
        ohlc_open_idx=feature_index_v1("ohlc_open_ratio"),
        ohlc_high_idx=feature_index_v1("ohlc_high_ratio"),
        ohlc_low_idx=feature_index_v1("ohlc_low_ratio"),
        log_return_idx=feature_index_v1("log_return"),
        volume_ratio_idx=feature_index_v1("volume_ratio"),
        vp_structure_start_idx=feature_index_v1("vp_ceiling_dist"),
        n_vp_structure=len(VP_STRUCTURE_COLS_V1),
        n_other_features=len(DERIVED_FEATURE_COLS_V1) + len(VP_STRUCTURE_COLS_V1),
        dropout=DROPOUT,
    )


# ═══════════════════════════════════════════════════════════════════
# Main walk-forward
# ═══════════════════════════════════════════════════════════════════
def main():
    device = get_device()
    use_amp = device.type == "cuda"

    print("=" * 70)
    print("  v9 WALL-AWARE (Stage 2: structure context token)")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

    # Sanity-print param count
    _sanity_model = build_v9()
    n_params = count_params(_sanity_model)
    del _sanity_model
    print(f"\nModel: v9 wall-aware ({n_params:,} params)")

    print(f"\nConfig:")
    print(f"  Labels: VP-derived first-hit binary (v6-prime, unchanged)")
    print(f"  Loss: BCEWithLogitsLoss + label_smoothing={LABEL_SMOOTHING}")
    print(f"  Regularization: dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}")
    print(f"  Optimizer: AdamW, LR={LR}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  Multi-seed: {N_SEEDS} seeds per fold (ensemble by logit averaging)")
    print(f"  SWA: start at epoch {SWA_START_EPOCH}, update every {SWA_UPDATE_FREQ}")
    print(f"  Embargo: {EMBARGO_BARS} bars ({EMBARGO_BARS//BARS_PER_DAY} days)")
    print(f"  Folds: {len(FOLD_BOUNDARIES) - 2} (holdout starts {HOLDOUT_START.date()})")

    # ─── Load data ───
    t0 = time.time()
    print("\nLoading v1 features...")
    df = build_feature_matrix_v1()
    print(f"  {len(df)} rows, {len(FEATURE_COLS_V1)} features")

    # ─── Compute VP-derived binary labels ───
    print("\nComputing VP-derived first-hit labels (binary)...")
    labels, tp_pct_arr, sl_pct_arr, label_stats = precompute_vp_labels(
        close=df["close"].values,
        ceiling_dist=df["vp_ceiling_dist"].values,
        floor_dist=df["vp_floor_dist"].values,
        num_peaks=df["vp_num_peaks"].values,
    )
    n_valid = label_stats["n_valid"]
    print(f"  Valid labels: {n_valid:,}")
    print(f"  Label 1 (TP first): {label_stats['n_label_1']:,} ({label_stats['n_label_1']/max(n_valid,1)*100:.1f}%)")
    print(f"  Label 0 (SL first): {label_stats['n_label_0']:,} ({label_stats['n_label_0']/max(n_valid,1)*100:.1f}%)")
    print(f"  Skipped (no peaks): {label_stats['n_skipped_nopeaks']:,}")
    print(f"  TP pct: mean={label_stats['tp_pct_mean']*100:.2f}% median={label_stats['tp_pct_median']*100:.2f}%")
    print(f"  SL pct: mean={label_stats['sl_pct_mean']*100:.2f}% median={label_stats['sl_pct_median']*100:.2f}%")

    regime = build_regime_array(df["date"].values)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    features = torch.from_numpy(df[FEATURE_COLS_V1].values.astype(np.float32))
    labels_t = torch.from_numpy(labels)
    dates = pd.to_datetime(df["date"]).values

    # ─── Walk-forward ───
    all_preds, all_labels, all_regimes = [], [], []
    all_tp_pct, all_sl_pct, all_logits = [], [], []
    all_dates, all_close = [], []
    fold_results = []

    close_prices = df["close"].values.astype(np.float64)

    print(f"\nStarting walk-forward...")
    total_start = time.time()

    embargo_td = pd.Timedelta(hours=EMBARGO_BARS)

    for i in range(len(FOLD_BOUNDARIES) - 2):
        fold_start = time.time()
        train_end = pd.Timestamp(FOLD_BOUNDARIES[i])
        val_end = pd.Timestamp(FOLD_BOUNDARIES[i + 1])
        test_end = pd.Timestamp(FOLD_BOUNDARIES[i + 2])

        train_mask = dates < (train_end - embargo_td)
        val_mask = (dates >= train_end) & (dates < (val_end - embargo_td))
        test_mask = (dates >= val_end) & (dates < test_end)

        if train_mask.sum() < 1000 or val_mask.sum() < 100 or test_mask.sum() < 100:
            continue

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        train_ds = FastDataset(
            features[train_idx[0]:train_idx[-1]+1],
            labels_t[train_idx[0]:train_idx[-1]+1],
        )
        val_ds = FastDataset(
            features[val_idx[0]:val_idx[-1]+1],
            labels_t[val_idx[0]:val_idx[-1]+1],
        )
        test_ds = FastDataset(
            features[test_idx[0]:test_idx[-1]+1],
            labels_t[test_idx[0]:test_idx[-1]+1],
            regime[test_idx[0]:test_idx[-1]+1],
            tp_pct=tp_pct_arr[test_idx[0]:test_idx[-1]+1],
            sl_pct=sl_pct_arr[test_idx[0]:test_idx[-1]+1],
        )

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        print(f"\n  Fold {i + 1} ({FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]})")
        print(f"    Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
        print(f"    Training {N_SEEDS} seeds with SWA...")

        seed_logits = []
        total_epochs_run = 0

        for seed_idx in range(N_SEEDS):
            seed = 42 + seed_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"    [Seed {seed}]")
            model = build_v9()
            if USE_COMPILE and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                except Exception:
                    pass

            model, epochs_run = train_fold(model, train_loader, val_loader, device, use_amp)
            _, _, seed_fold_logits = evaluate_fold(model, test_loader, device, use_amp)
            seed_logits.append(seed_fold_logits)
            total_epochs_run += epochs_run

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Ensemble: average logits across seeds (matches v6-prime protocol)
        fold_logits = np.mean(np.stack(seed_logits), axis=0)
        preds = (fold_logits > 0).astype(np.float32)

        fold_labels = np.array([
            test_ds.labels[test_ds.valid_indices[j] + test_ds.lookback - 1].item()
            for j in range(len(test_ds))
        ])

        fold_regimes = np.array([test_ds.get_regime(j) for j in range(len(test_ds))])
        fold_tp_pct = np.array([test_ds.get_tp_pct(j) for j in range(len(test_ds))])
        fold_sl_pct = np.array([test_ds.get_sl_pct(j) for j in range(len(test_ds))])

        test_slice_start = test_idx[0]
        fold_bar_idx = np.array([
            test_slice_start + test_ds.valid_indices[j] + test_ds.lookback - 1
            for j in range(len(test_ds))
        ])
        fold_dates = dates[fold_bar_idx]
        fold_close = close_prices[fold_bar_idx]

        fold_acc = (preds == fold_labels).mean()
        fold_time = time.time() - fold_start

        # Per-fold real EV: long when preds==1, PnL from VP-derived tp/sl
        long_mask = preds == 1
        long_wins = long_mask & (fold_labels == 1)
        long_losses = long_mask & (fold_labels == 0)
        n_long = long_mask.sum()
        if n_long > 0:
            fold_long_ev = (
                fold_tp_pct[long_wins].sum() - fold_sl_pct[long_losses].sum()
            ) / n_long * 100
        else:
            fold_long_ev = 0.0

        fold_results.append({
            "fold": i + 1,
            "period": f"{FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]}",
            "acc": float(fold_acc),
            "n_test": len(preds),
            "n_seeds": N_SEEDS,
            "total_epochs": total_epochs_run,
            "time_sec": round(fold_time, 1),
            "n_long_trades": int(n_long),
            "long_ev_real": round(float(fold_long_ev), 3),
            "avg_tp_pct": round(float(np.nanmean(fold_tp_pct)), 4),
            "avg_sl_pct": round(float(np.nanmean(fold_sl_pct)), 4),
        })
        all_preds.extend(preds.tolist())
        all_labels.extend(fold_labels.tolist())
        all_regimes.extend(fold_regimes.tolist())
        all_tp_pct.extend(fold_tp_pct.tolist())
        all_sl_pct.extend(fold_sl_pct.tolist())
        all_logits.extend(fold_logits.tolist())
        all_dates.extend(fold_dates.tolist())
        all_close.extend(fold_close.tolist())

        print(
            f"    [Ensemble] Acc: {fold_acc:.4f} "
            f"Long trades: {int(n_long)}, Real EV/trade: {fold_long_ev:+.3f}% "
            f"({N_SEEDS} seeds, {total_epochs_run} epochs, {fold_time:.0f}s)"
        )

    total_time = time.time() - total_start

    # ─── Overall stats ───
    preds = np.array(all_preds)
    labels_all = np.array(all_labels)
    regimes = np.array(all_regimes)
    tp_pct_all = np.array(all_tp_pct)
    sl_pct_all = np.array(all_sl_pct)
    logits_all = np.array(all_logits)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))

    def confusion(p, l):
        tp = int(((p == 1) & (l == 1)).sum())
        fp = int(((p == 1) & (l == 0)).sum())
        tn = int(((p == 0) & (l == 0)).sum())
        fn = int(((p == 0) & (l == 1)).sum())
        precision = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "npv": round(npv, 4),
            "accuracy": round(acc, 4),
            "n": int(tp + fp + tn + fn),
        }

    overall_conf = confusion(preds, labels_all)

    # Long-only real EV
    long_mask_all = preds == 1
    if long_mask_all.sum() > 0:
        long_wins_all = long_mask_all & (labels_all == 1)
        long_losses_all = long_mask_all & (labels_all == 0)
        long_only_ev = (
            tp_pct_all[long_wins_all].sum() - sl_pct_all[long_losses_all].sum()
        ) / long_mask_all.sum() * 100
    else:
        long_only_ev = 0.0

    print(f"\n{'=' * 70}")
    print(f"OVERALL (all folds, ensembled):")
    print(f"  n          = {overall_conf['n']:,}")
    print(f"  accuracy   = {overall_conf['accuracy']*100:.2f}%")
    print(f"  precision  = {overall_conf['precision']*100:.2f}%  (of pred=1, fraction with label=1)")
    print(f"  NPV        = {overall_conf['npv']*100:.2f}%  (of pred=0, fraction with label=0)")
    print(f"  long-only real EV = {long_only_ev:+.3f}% per trade")

    print(f"\nLogit distribution (diagnostic):")
    print(f"  mean={logits_all.mean():+.3f} std={logits_all.std():.3f}")
    print(f"  q10={np.percentile(logits_all, 10):+.3f} q25={np.percentile(logits_all, 25):+.3f}")
    print(f"  q75={np.percentile(logits_all, 75):+.3f} q90={np.percentile(logits_all, 90):+.3f}")
    print(f"\nTotal time: {total_time / 60:.1f} min")

    output = {
        "config": {
            "model": "v9 wall-aware (TemporalEnrichedV9WallAware)",
            "architecture_change": "+1 structure context token in spatial attention",
            "labels": "VP-derived first-hit binary (same as v6-prime)",
            "tp_ratio": TP_RATIO,
            "sl_ratio": SL_RATIO,
            "tp_range": [TP_MIN, TP_MAX],
            "sl_range": [SL_MIN, SL_MAX],
            "min_peaks": MIN_PEAKS,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "optimizer": "AdamW",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "n_seeds": N_SEEDS,
            "swa_start_epoch": SWA_START_EPOCH,
            "swa_update_freq": SWA_UPDATE_FREQ,
            "embargo_bars": EMBARGO_BARS,
            "n_params": n_params,
        },
        "label_stats": label_stats,
        "overall": overall_conf,
        "long_only_real_ev_pct": round(float(long_only_ev), 4),
        "logit_stats": {
            "mean": float(logits_all.mean()),
            "std": float(logits_all.std()),
            "median": float(np.median(logits_all)),
            "q10": float(np.percentile(logits_all, 10)),
            "q25": float(np.percentile(logits_all, 25)),
            "q75": float(np.percentile(logits_all, 75)),
            "q90": float(np.percentile(logits_all, 90)),
        },
        "folds": fold_results,
        "total_time_min": round(total_time / 60, 1),
    }
    out_path = EXPERIMENTS_DIR / "eval_v9_results.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save predictions cache for backtest engine
    predictions_path = EXPERIMENTS_DIR / "v9_predictions.npz"
    dates_iso = np.array([pd.Timestamp(d).isoformat() for d in all_dates], dtype=object)
    np.savez(
        predictions_path,
        dates=dates_iso,
        close=np.array(all_close, dtype=np.float64),
        logits=logits_all,
        probs=probs_all,
        preds=preds.astype(np.int8),
        labels=labels_all.astype(np.int8),
        tp_pct=tp_pct_all,
        sl_pct=sl_pct_all,
        regimes=regimes,
    )
    print(f"Predictions cache saved to {predictions_path}")


if __name__ == "__main__":
    main()
