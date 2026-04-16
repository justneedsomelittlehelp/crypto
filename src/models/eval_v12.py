"""Eval v12 — v11 + regime-aware encoder (macro context at hourly + daily).

Extends eval_v11 with regime features:
  - Day enrichment: 6 daily regime scalars (VIX, DXY, GLD, USO, FFR,
    yield_curve) appended to each day token. Temporal transformer sees
    the 90-day macro trajectory.
  - Regime encoder: 1D conv stack over last 72 hours of VIX/DXY/GLD/USO
    at 1h resolution. Captures intraday regime shifts (VIX spikes etc).
    Injected at the final FC.

Usage (Colab):
    # fetch regime data first (only need once):
    !python3 -m src.data.fetch_regime

    # train with regime features:
    !python3 -m src.models.eval_v12 --stride 4 --seeds 1 --base-seed 42 \\
        --epochs 200 --patience 60 --swa-start 60 --tag v12_regime_seed42

    # ablation (regime zeroed out — same model capacity, no regime signal):
    !python3 -m src.models.eval_v12 --regime none --stride 4 --seeds 1 \\
        --base-seed 42 --epochs 200 --patience 60 --swa-start 60 \\
        --tag v12_noregime_seed42
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

from src.models.architectures.v12_regime import AbsVPv12
from src.models.eval_v11 import (
    PROJECT_ROOT, CSV_PATH, EXPERIMENTS_DIR,
    N_DAYS, BARS_PER_DAY, LOOKBACK_BARS,
    LR, WEIGHT_DECAY, LABEL_SMOOTHING, DROPOUT,
    EPOCHS, EARLY_STOP_PATIENCE, SWA_START_EPOCH,
    BATCH_SIZE, STRIDE_BARS_DEFAULT,
    LABEL_MAX_BARS, EMBARGO,
    TB_VOL_WINDOW, TB_HORIZON_BARS, TB_BARRIER_K, TB_BARRIER_MIN, TB_BARRIER_MAX,
    FOLD_BOUNDARIES,
    get_device, build_features,
    iterate_batches, smooth_labels, count_params,
    N_SEEDS,
)

# ═══════════════════════════════════════════════════════════════════
# Regime data config
# ═══════════════════════════════════════════════════════════════════
REGIME_HOURLY_PATH = PROJECT_ROOT / "data" / "regime_hourly.csv"
REGIME_DAILY_PATH = PROJECT_ROOT / "data" / "regime_daily.csv"
REGIME_HOURLY_LOOKBACK = 72   # hours
REGIME_HOURLY_COLS = ["vix", "dxy", "gld", "uso"]
REGIME_DAILY_COLS = ["vix", "dxy", "gld", "uso", "ffr", "yield_curve"]


def _results_path(tag: str) -> Path:
    return EXPERIMENTS_DIR / f"eval_v12_{tag}_results.json"


def _predictions_path(tag: str) -> Path:
    return EXPERIMENTS_DIR / f"v12_{tag}_predictions.npz"


def load_regime_features(dates: np.ndarray, regime_mode: str = "full"):
    """Load and align regime features to the main CSV's bar timestamps.

    Returns:
        regime_daily:  (N, 6) float32 — daily regime scalars per bar
        regime_hourly: (N_hours, 4) float32 — hourly regime data (full timeline)
        hourly_dates:  (N_hours,) datetime64 — hourly timestamps
    """
    dates_utc = pd.DatetimeIndex(dates, tz="UTC")
    n = len(dates_utc)

    if regime_mode == "none":
        regime_daily = np.zeros((n, len(REGIME_DAILY_COLS)), dtype=np.float32)
        n_hours = n // 4 + 1
        regime_hourly_arr = np.zeros((n_hours, len(REGIME_HOURLY_COLS)), dtype=np.float32)
        hourly_dates = pd.date_range(
            start=dates_utc[0], periods=n_hours, freq="h", tz="UTC"
        ).values
        return regime_daily, regime_hourly_arr, hourly_dates

    # ─── Daily regime ───
    print(f"  Loading regime daily from {REGIME_DAILY_PATH.name}...")
    df_daily = pd.read_csv(REGIME_DAILY_PATH, parse_dates=["date_utc"], index_col="date_utc")
    if df_daily.index.tz is None:
        df_daily.index = df_daily.index.tz_localize("UTC")

    # Align to bar dates (each bar gets its calendar date's regime values)
    bar_dates = dates_utc.normalize()
    daily_aligned = df_daily.reindex(bar_dates, method="ffill")
    regime_daily = daily_aligned[REGIME_DAILY_COLS].values.astype(np.float32)
    regime_daily = np.nan_to_num(regime_daily, nan=0.0)

    # Normalize daily features: z-score with expanding window to avoid lookahead
    for col_idx in range(regime_daily.shape[1]):
        col = regime_daily[:, col_idx]
        s = pd.Series(col)
        rolling_mean = s.expanding(min_periods=BARS_PER_DAY * 30).mean().values
        rolling_std = s.expanding(min_periods=BARS_PER_DAY * 30).std().values
        rolling_std = np.where(rolling_std < 1e-8, 1.0, rolling_std)
        regime_daily[:, col_idx] = (col - rolling_mean) / rolling_std
    regime_daily = np.nan_to_num(regime_daily, nan=0.0).astype(np.float32)

    # ─── Hourly regime ───
    print(f"  Loading regime hourly from {REGIME_HOURLY_PATH.name}...")
    df_hourly = pd.read_csv(REGIME_HOURLY_PATH, parse_dates=["datetime_utc"], index_col="datetime_utc")
    if df_hourly.index.tz is None:
        df_hourly.index = df_hourly.index.tz_localize("UTC")

    hourly_data = df_hourly[REGIME_HOURLY_COLS]
    hourly_data = hourly_data.ffill().bfill()

    # Normalize hourly: compute log-returns (what matters is change, not level)
    hourly_logret = np.log(hourly_data / hourly_data.shift(1)).fillna(0.0)
    regime_hourly_arr = hourly_logret.values.astype(np.float32)
    regime_hourly_arr = np.nan_to_num(regime_hourly_arr, nan=0.0)
    hourly_dates = hourly_data.index.values

    print(f"  Regime daily:  {regime_daily.shape}, nulls={np.isnan(regime_daily).sum()}")
    print(f"  Regime hourly: {regime_hourly_arr.shape}, range={hourly_dates[0]} → {hourly_dates[-1]}")

    return regime_daily, regime_hourly_arr, hourly_dates


def build_hourly_index(bar_dates: np.ndarray, hourly_dates: np.ndarray) -> np.ndarray:
    """For each 15m bar, find the index of the most recent hourly regime row.

    Returns (N,) int64 array where bar_hourly_idx[i] is the index into
    regime_hourly that is the latest hourly timestamp <= bar_dates[i].
    """
    bar_ts = bar_dates.astype("datetime64[ns]").astype(np.int64)
    hourly_ts = hourly_dates.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(hourly_ts, bar_ts, side="right") - 1
    idx = np.clip(idx, 0, len(hourly_ts) - 1)
    return idx.astype(np.int64)


def gather_batch_v12(
    batch_idx: torch.Tensor,
    day_rows_gpu: torch.Tensor,
    last_bar_gpu: torch.Tensor,
    labels_gpu: torch.Tensor,
    day_offsets: torch.Tensor,
    regime_hourly_gpu: torch.Tensor,
    bar_hourly_idx_gpu: torch.Tensor,
    regime_lookback: int = REGIME_HOURLY_LOOKBACK,
):
    """Like gather_batch but also gathers the 72h regime window."""
    day_indices = batch_idx[:, None] + day_offsets[None, :]
    day_tokens = day_rows_gpu[day_indices]
    last = last_bar_gpu[batch_idx]
    y = labels_gpu[batch_idx]

    # Gather 72h regime window ending at each sample's timestamp
    hourly_end_idx = bar_hourly_idx_gpu[batch_idx]        # (B,)
    # Build (B, 72) index array: [end-71, end-70, ..., end]
    offsets = torch.arange(-regime_lookback + 1, 1, device=batch_idx.device)
    hourly_indices = hourly_end_idx[:, None] + offsets[None, :]
    hourly_indices = hourly_indices.clamp(min=0)
    regime_window = regime_hourly_gpu[hourly_indices]      # (B, 72, 4)

    return day_tokens, last, y, regime_window


def train_one_seed_v12(
    seed: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
    regime_hourly_gpu, bar_hourly_idx_gpu,
    device, use_amp: bool, pos_weight: torch.Tensor,
    n_spatial: int, n_temporal: int,
    max_epochs: int = EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    swa_start: int = SWA_START_EPOCH,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AbsVPv12(
        n_days=N_DAYS,
        n_spatial_layers=n_spatial,
        n_temporal_layers=n_temporal,
        dropout=DROPOUT,
    ).to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    swa_weights = None
    swa_count = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch_idx in iterate_batches(train_idx, BATCH_SIZE, shuffle=True):
            day_tokens, last, y, regime_w = gather_batch_v12(
                batch_idx, day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
                regime_hourly_gpu, bar_hourly_idx_gpu,
            )
            y_smooth = smooth_labels(y, LABEL_SMOOTHING)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(day_tokens, last, regime_w)
                loss = criterion(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * y.shape[0]
            train_n += y.shape[0]
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx in iterate_batches(val_idx, BATCH_SIZE, shuffle=False):
                day_tokens, last, y, regime_w = gather_batch_v12(
                    batch_idx, day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
                    regime_hourly_gpu, bar_hourly_idx_gpu,
                )
                y_smooth = smooth_labels(y, LABEL_SMOOTHING)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(day_tokens, last, regime_w)
                    loss = criterion(logits, y_smooth)
                val_loss_sum += loss.item() * y.shape[0]
                val_correct += ((logits > 0).float() == y).sum().item()
                val_n += y.shape[0]
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)

        if epoch == 1 or epoch % 5 == 0:
            print(f"    Epoch {epoch:3d} | TrL {train_loss:.4f} | VaL {val_loss:.4f} | VaAcc {val_acc:.4f}")

        if epoch >= swa_start:
            current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = current_state
                swa_count = 1
            else:
                for k in swa_weights:
                    swa_weights[k] = (swa_weights[k] * swa_count + current_state[k]) / (swa_count + 1)
                swa_count += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    if swa_weights is not None and swa_count >= 2:
        model.load_state_dict(swa_weights)
        print(f"    SWA: averaged {swa_count} checkpoints")
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_v12(
    model, test_idx: torch.Tensor,
    day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
    regime_hourly_gpu, bar_hourly_idx_gpu,
    device, use_amp: bool,
):
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch_idx in iterate_batches(test_idx, BATCH_SIZE, shuffle=False):
            day_tokens, last, _, regime_w = gather_batch_v12(
                batch_idx, day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
                regime_hourly_gpu, bar_hourly_idx_gpu,
            )
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(day_tokens, last, regime_w)
            all_logits.append(logits.float().cpu())
    return torch.cat(all_logits).numpy()


def parse_args():
    p = argparse.ArgumentParser(description="v12 walk-forward eval (regime-aware)")
    p.add_argument("--spatial", type=int, default=1)
    p.add_argument("--temporal", type=int, default=1)
    p.add_argument("--features", choices=["full", "nopv"], default="full")
    p.add_argument("--regime", choices=["full", "none"], default="full",
                   help="'full' = regime features active, 'none' = zeroed "
                        "(same model capacity, no regime signal — for ablation)")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--swa-start", type=int, default=SWA_START_EPOCH)
    p.add_argument("--stride", type=int, default=STRIDE_BARS_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    n_spatial = args.spatial
    n_temporal = args.temporal
    features_mode = args.features
    regime_mode = args.regime
    stride_bars = args.stride
    assert stride_bars >= 1
    cadence_minutes = stride_bars * 15
    cadence_suffix = "" if stride_bars == 1 else f"_c{cadence_minutes}m"

    if args.tag is not None:
        tag = args.tag
    else:
        regime_suffix = "" if regime_mode == "full" else "_noregime"
        tag = f"v12_{features_mode}{regime_suffix}{cadence_suffix}"

    n_seeds = args.seeds
    results_path = _results_path(tag)
    predictions_path = _predictions_path(tag)

    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    print("\nConfig:")
    print(f"  CSV:         {CSV_PATH.name}")
    print(f"  Model:       AbsVPv12 (regime-aware)")
    print(f"  Layers:      spatial={n_spatial}  temporal={n_temporal}  tag={tag}")
    print(f"  Labels:      triple_barrier")
    print(f"  Features:    {features_mode}")
    print(f"  Regime:      {regime_mode}")
    print(f"  Cadence:     stride={stride_bars} bars ({cadence_minutes}m)")
    print(f"  Epochs:      max={args.epochs}  patience={args.patience}  "
          f"swa_start={args.swa_start}")
    print(f"  Seeds/fold:  {n_seeds}  base_seed={args.base_seed}")
    print(f"  Results →    {results_path.name}")
    print(f"  Preds   →    {predictions_path.name}")

    # ─── Load features (reuse v11's build_features) ───
    feats = build_features(CSV_PATH, labels_mode="triple_barrier", features_mode=features_mode)
    day_rows_v11 = feats["day_rows"]       # (N, 110)
    last_bar = feats["last_bar"]           # (N, 4)
    labels_np = feats["labels"]
    dates = feats["dates"]
    tp_pct_np = feats["tp_pct"]
    sl_pct_np = feats["sl_pct"]
    close_np = feats["close"]
    n = len(day_rows_v11)

    # ─── Load regime features ───
    print("\nLoading regime features...")
    regime_daily, regime_hourly_arr, hourly_dates = load_regime_features(dates, regime_mode)

    # Widen day_rows: (N, 110) → (N, 116) by appending daily regime
    day_rows = np.concatenate([day_rows_v11, regime_daily], axis=1).astype(np.float32)
    assert day_rows.shape[1] == AbsVPv12.DAY_TOKEN_WIDTH, (
        f"day_rows width {day_rows.shape[1]} != expected {AbsVPv12.DAY_TOKEN_WIDTH}"
    )

    # Build hourly index mapping: for each 15m bar, which hourly row?
    bar_hourly_idx = build_hourly_index(dates, hourly_dates)

    # ─── Upload to GPU ───
    print("\nUploading feature tensors to device...")
    day_rows_gpu = torch.from_numpy(day_rows).to(device)
    last_bar_gpu = torch.from_numpy(last_bar).to(device)
    labels_gpu = torch.from_numpy(labels_np).to(device)
    day_offsets = (torch.arange(N_DAYS, device=device) - (N_DAYS - 1)) * BARS_PER_DAY
    day_offsets = day_offsets.to(torch.long)
    regime_hourly_gpu = torch.from_numpy(regime_hourly_arr).to(device)
    bar_hourly_idx_gpu = torch.from_numpy(bar_hourly_idx).to(device)

    print(f"  day_rows: {day_rows_gpu.shape}  ({day_rows_gpu.element_size() * day_rows_gpu.numel() / 1e6:.1f} MB)")
    print(f"  regime_hourly: {regime_hourly_gpu.shape}  ({regime_hourly_gpu.element_size() * regime_hourly_gpu.numel() / 1e6:.1f} MB)")

    # ─── Eligibility mask ───
    min_history = (N_DAYS - 1) * BARS_PER_DAY
    has_history = np.arange(n) >= min_history
    has_label = ~np.isnan(labels_np)
    first_minute = pd.Timestamp(dates[0]).minute
    assert first_minute % 15 == 0
    anchor_offset = ((-first_minute // 15) % stride_bars)
    cadence_mask = ((np.arange(n) - anchor_offset) % stride_bars) == 0
    eligible_mask = has_history & has_label & cadence_mask
    eligible_idx_np = np.where(eligible_mask)[0].astype(np.int64)
    print(f"\nEligible samples: {len(eligible_idx_np):,} / {n:,}")

    # ─── Model probe ───
    probe = AbsVPv12(n_days=N_DAYS, n_spatial_layers=n_spatial,
                     n_temporal_layers=n_temporal, dropout=DROPOUT)
    n_params = count_params(probe)
    print(f"\nModel params: {n_params:,}")
    del probe

    # ─── Walk-forward ───
    all_results = []
    all_preds, all_labels = [], []
    all_logits, all_dates, all_close = [], [], []
    all_tp, all_sl = [], []

    print("\nStarting walk-forward...")
    total_start = time.time()

    for i in range(len(FOLD_BOUNDARIES) - 2):
        fold_start = time.time()
        train_end = pd.Timestamp(FOLD_BOUNDARIES[i], tz="UTC")
        val_end = pd.Timestamp(FOLD_BOUNDARIES[i + 1], tz="UTC")
        test_end = pd.Timestamp(FOLD_BOUNDARIES[i + 2], tz="UTC")

        train_mask = dates < (train_end - EMBARGO)
        val_mask = (dates >= train_end) & (dates < (val_end - EMBARGO))
        test_mask = (dates >= val_end) & (dates < test_end)

        train_np = eligible_idx_np[train_mask[eligible_idx_np]]
        val_np = eligible_idx_np[val_mask[eligible_idx_np]]
        test_np = eligible_idx_np[test_mask[eligible_idx_np]]

        if len(train_np) < 1000 or len(val_np) < 100 or len(test_np) < 100:
            print(f"  Fold {i + 1}: skipping (insufficient data)")
            continue

        train_idx = torch.from_numpy(train_np).to(device)
        val_idx = torch.from_numpy(val_np).to(device)
        test_idx = torch.from_numpy(test_np).to(device)

        fold_labels_train = labels_np[train_np]
        n_pos = int((fold_labels_train == 1).sum())
        n_neg = int((fold_labels_train == 0).sum())
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], device=device, dtype=torch.float32
        )

        print(f"\n  Fold {i + 1} ({FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]})")
        print(f"    Train: {len(train_np):,}  Val: {len(val_np):,}  Test: {len(test_np):,}")
        print(f"    pos_weight: {pos_weight.item():.3f}")

        seed_logits = []
        for seed_offset in range(n_seeds):
            seed = args.base_seed + seed_offset
            print(f"    [Seed {seed}]")
            model = train_one_seed_v12(
                seed, train_idx, val_idx,
                day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
                regime_hourly_gpu, bar_hourly_idx_gpu,
                device, use_amp, pos_weight,
                n_spatial=n_spatial, n_temporal=n_temporal,
                max_epochs=args.epochs,
                patience=args.patience,
                swa_start=args.swa_start,
            )
            fold_logits = evaluate_v12(
                model, test_idx,
                day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
                regime_hourly_gpu, bar_hourly_idx_gpu,
                device, use_amp,
            )
            seed_logits.append(fold_logits)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ensemble_logits = np.mean(np.stack(seed_logits), axis=0)
        preds = (ensemble_logits > 0).astype(np.float32)
        fold_y = labels_np[test_np].astype(np.float32)
        fold_tp = tp_pct_np[test_np]
        fold_sl = sl_pct_np[test_np]
        fold_dates = feats["dates"][test_np]
        fold_close = close_np[test_np]

        acc = float((preds == fold_y).mean())
        n_long = int((preds == 1).sum())
        if n_long > 0:
            long_wins = (preds == 1) & (fold_y == 1)
            long_losses = (preds == 1) & (fold_y == 0)
            ev = float((fold_tp[long_wins].sum() - fold_sl[long_losses].sum()) / n_long * 100)
        else:
            ev = 0.0

        fold_time = time.time() - fold_start
        print(f"    → acc {acc:.4f}  n_long {n_long:,}  EV/trade {ev:.3f}%  ({fold_time / 60:.1f} min)")

        all_results.append({
            "fold": i + 1,
            "period": f"{FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]}",
            "n_test": int(len(preds)),
            "acc": acc,
            "n_long": n_long,
            "long_ev_real": ev,
            "time_sec": round(fold_time, 1),
        })
        all_preds.extend(preds.tolist())
        all_labels.extend(fold_y.tolist())
        all_logits.extend(ensemble_logits.tolist())
        all_dates.extend(fold_dates.tolist())
        all_close.extend(fold_close.tolist())
        all_tp.extend(fold_tp.tolist())
        all_sl.extend(fold_sl.tolist())

    total_time = time.time() - total_start
    print(f"\nTotal walk-forward time: {total_time / 60:.1f} min")

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    overall_acc = float((all_preds_np == all_labels_np).mean()) if len(all_preds_np) else 0.0

    print(f"\n{'=' * 60}")
    print(f"  V12 WALK-FORWARD SUMMARY (regime={regime_mode})")
    print(f"{'=' * 60}")
    print(f"  Total test samples: {len(all_preds_np):,}")
    print(f"  Overall accuracy:   {overall_acc:.4f}")
    print(f"{'=' * 60}")
    print(f"  Fold  Period                       Test       Acc     n_long   EV/trade")
    for r in all_results:
        print(
            f"  {r['fold']:<5} {r['period']:<28} "
            f"{r['n_test']:<10,} {r['acc']:.4f}  {r['n_long']:<8,} {r['long_ev_real']:>7.3f}%"
        )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval": f"v12 tag={tag} — regime-aware VP @ 15m × 90-day temporal",
        "csv": CSV_PATH.name,
        "model": "AbsVPv12",
        "n_spatial_layers": n_spatial,
        "n_temporal_layers": n_temporal,
        "labels_mode": "triple_barrier",
        "features_mode": features_mode,
        "regime_mode": regime_mode,
        "tag": tag,
        "n_seeds": n_seeds,
        "base_seed": args.base_seed,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "swa_start": args.swa_start,
        "n_days": N_DAYS,
        "bars_per_day": BARS_PER_DAY,
        "stride_bars": stride_bars,
        "cadence_minutes": cadence_minutes,
        "label_max_bars": LABEL_MAX_BARS,
        "embargo_days": 14,
        "triple_barrier_k": TB_BARRIER_K,
        "triple_barrier_vol_window": TB_VOL_WINDOW,
        "n_params": n_params,
        "overall_acc": overall_acc,
        "total_time_min": round(total_time / 60, 1),
        "folds": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        predictions_path,
        preds=np.array(all_preds),
        labels=np.array(all_labels),
        logits=np.array(all_logits),
        dates=np.array(all_dates, dtype="datetime64[ns]"),
        close=np.array(all_close),
        tp_pct=np.array(all_tp),
        sl_pct=np.array(all_sl),
        stride_bars=np.int64(stride_bars),
        cadence_minutes=np.int64(cadence_minutes),
    )
    print(f"\nWrote {results_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
