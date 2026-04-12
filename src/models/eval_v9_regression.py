"""Eval v9-regression: v6-prime architecture, regression labels (Philosophy D).

Stage 1 of the post-audit experiment. Goal: isolate the effect of switching
from binary first-hit labels to continuous realized-PnL labels, with
everything else (architecture, walk-forward, embargo, holdout) held constant.

Key changes vs eval_v6_prime.py:
  - Labels: continuous realized P&L under VP-derived TP/SL exit model.
    TP hit → +tp_pct. SL hit → -sl_pct. Timeout → close-to-close return
    over the window. Pre-fee, pre-slippage (engine applies them).
  - Loss: HuberLoss(delta=0.02) instead of BCEWithLogitsLoss.
  - No class balance / pos_weight (regression, not classification).
  - No label smoothing.
  - Output: model's raw scalar is the predicted return (no sigmoid).
  - Metrics: mean/std of predictions, Pearson corr with labels,
    "precision @ 1.5%" (of bars where pred > 1.5%, fraction with positive
    realized return) — the closest thing to a real deployment signal.
  - Predictions cache key: `predicted_return` (was `logits` / `probs`).

Same as v6-prime:
  - v6-prime architecture (TemporalEnrichedV6Prime), unchanged.
  - 14-day embargo at fold boundaries.
  - Walk-forward with holdout folds 11-12 (2025-07 → 2026-04-08).
  - 3-seed ensemble + SWA averaging.

Usage:
    python -m src.models.eval_v9_regression
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(line_buffering=True)

from src.config import (
    LOOKBACK_BARS_MODEL, BARS_PER_DAY, EXPERIMENTS_DIR, LABEL_FGI_PATH,
    LABEL_MAX_BARS, EPOCHS, EARLY_STOP_PATIENCE, REL_SPAN_PCT,
)
from src.features.pipelines.v1_raw import (
    build_feature_matrix_v1, FEATURE_COLS_V1, VP_STRUCTURE_COLS_V1,
    DERIVED_FEATURE_COLS_V1, feature_index_v1,
)
from src.models.architectures.v6_prime_vp_labels import TemporalEnrichedV6Prime

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
BATCH_SIZE = 512
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
NUM_WORKERS = 4
USE_COMPILE = True

# Regression-specific
# Huber delta: must be larger than typical label magnitude or the loss
# acts as L1 over the entire dataset, and L1's optimum is the conditional
# median (slightly negative here since SL hits outnumber TP hits 60/40),
# which collapses the model to a constant predictor.
# Median |label| is ~0.05-0.075, so delta=0.10 keeps L2 behavior over the
# typical range and only clips the rare ±0.15 cap hits.
HUBER_DELTA = 0.10
PRECISION_AT_THRESHOLD = 0.015  # 1.5% — for "precision @ threshold" metric

# VP-derived label parameters (UNCHANGED from v6-prime — Stage 1 isolates labels only)
TP_RATIO = 0.8
SL_RATIO = 0.6
TP_MIN = 0.01
TP_MAX = 0.15
SL_MIN = 0.01
SL_MAX = 0.15
MIN_PEAKS = 1

# Multi-seed + SWA
N_SEEDS = 3
SWA_START_EPOCH = 15
SWA_UPDATE_FREQ = 1

FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
    "2026-01-01", "2026-04-08",
]

HOLDOUT_START = pd.Timestamp("2025-07-01")
EMBARGO_BARS = LABEL_MAX_BARS


# ═══════════════════════════════════════════════════════════════════
# Regime array (analytics only, not used for labels)
# ═══════════════════════════════════════════════════════════════════
def build_regime_array(dates: np.ndarray) -> np.ndarray:
    n = len(dates)
    fgi_df = pd.read_csv(LABEL_FGI_PATH, parse_dates=["date"])
    fgi_lookup = dict(zip(fgi_df["date"].dt.date, fgi_df["fgi_value"].astype(float)))
    is_bull = np.full(n, np.nan)
    for i in range(n):
        dt = pd.Timestamp(dates[i]).date()
        fgi = fgi_lookup.get(dt, np.nan)
        if not np.isnan(fgi):
            is_bull[i] = float(fgi >= 50)
    return is_bull


# ═══════════════════════════════════════════════════════════════════
# Regression labels: realized P&L under fixed VP-derived TP/SL exit
# ═══════════════════════════════════════════════════════════════════
def precompute_vp_regression_labels(
    close: np.ndarray,
    ceiling_dist: np.ndarray,
    floor_dist: np.ndarray,
    num_peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-sample continuous label = realized P&L under fixed VP TP/SL exit.

    For each bar:
      - Compute tp_pct, sl_pct from VP geometry (same formula as v6-prime).
      - Walk forward up to LABEL_MAX_BARS bars (close prices only).
      - If tp_level hit first  → label = +tp_pct
      - If sl_level hit first  → label = -sl_pct
      - If neither hits        → label = (close[end] - entry) / entry
                                  (close-to-close return over window)
      - Bars with num_peaks < MIN_PEAKS or NaN structure → NaN label.

    Labels are pre-fee, pre-slippage. The backtest engine applies fees/slip
    on top during simulation. This way the learning target is the raw
    geometric P&L the price action would have delivered.

    Returns:
        labels:   (n,) float32, signed P&L in decimal (e.g. 0.075 = 7.5%)
        tp_pct:   (n,) float32, per-sample TP (NaN where skipped)
        sl_pct:   (n,) float32, per-sample SL (NaN where skipped)
        stats:    summary dict
    """
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)
    tp_pct_arr = np.full(n, np.nan, dtype=np.float32)
    sl_pct_arr = np.full(n, np.nan, dtype=np.float32)

    # Vol scaling (rolling 30-bar realized vol on log returns, normalized)
    vol_window = 30
    log_returns = np.diff(np.log(close))
    rolling_vol = np.full(n, np.nan)
    lr_series = pd.Series(log_returns)
    rolling_std = lr_series.rolling(window=vol_window, min_periods=vol_window).std().values
    rolling_vol[vol_window + 1:] = rolling_std[vol_window:]

    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) == 0:
        median_vol = 1.0
    else:
        median_vol = float(np.median(valid_vol))
    vol_scale = np.where(np.isnan(rolling_vol), 1.0, rolling_vol / median_vol)

    max_bars = LABEL_MAX_BARS

    skipped_nopeaks = 0
    skipped_nanstruct = 0
    skipped_novol = 0
    n_tp_hit = 0
    n_sl_hit = 0
    n_timeout = 0
    label_values = []

    for i in range(n - 1):
        if np.isnan(rolling_vol[i]):
            skipped_novol += 1
            continue
        if np.isnan(ceiling_dist[i]) or np.isnan(floor_dist[i]) or np.isnan(num_peaks[i]):
            skipped_nanstruct += 1
            continue
        if num_peaks[i] < MIN_PEAKS:
            skipped_nopeaks += 1
            continue

        ceiling_pct = ceiling_dist[i] * REL_SPAN_PCT
        floor_pct = floor_dist[i] * REL_SPAN_PCT

        tp_pct = ceiling_pct * TP_RATIO * vol_scale[i]
        sl_pct = floor_pct * SL_RATIO * vol_scale[i]

        tp_pct = np.clip(tp_pct, TP_MIN, TP_MAX)
        sl_pct = np.clip(sl_pct, SL_MIN, SL_MAX)

        tp_pct_arr[i] = tp_pct
        sl_pct_arr[i] = sl_pct

        entry = close[i]
        tp_level = entry * (1 + tp_pct)
        sl_level = entry * (1 - sl_pct)
        end = min(i + max_bars, n)
        future = close[i + 1:end]

        if len(future) == 0:
            continue

        tp_hits = future >= tp_level
        sl_hits = future <= sl_level
        tp_first = np.argmax(tp_hits) if tp_hits.any() else len(future)
        sl_first = np.argmax(sl_hits) if sl_hits.any() else len(future)

        if tp_first < sl_first:
            labels[i] = float(tp_pct)
            n_tp_hit += 1
        elif sl_first < tp_first:
            labels[i] = float(-sl_pct)
            n_sl_hit += 1
        else:
            # Neither hit within window — use close-to-close return
            labels[i] = float((future[-1] - entry) / entry)
            n_timeout += 1
        label_values.append(labels[i])

    label_arr = np.array(label_values, dtype=np.float64)
    stats = {
        "n_valid": int((~np.isnan(labels)).sum()),
        "n_tp_hit": n_tp_hit,
        "n_sl_hit": n_sl_hit,
        "n_timeout": n_timeout,
        "n_skipped_nopeaks": skipped_nopeaks,
        "n_skipped_nanstruct": skipped_nanstruct,
        "n_skipped_novol": skipped_novol,
        "label_mean": float(label_arr.mean()) if len(label_arr) else 0.0,
        "label_median": float(np.median(label_arr)) if len(label_arr) else 0.0,
        "label_std": float(label_arr.std()) if len(label_arr) else 0.0,
        "label_min": float(label_arr.min()) if len(label_arr) else 0.0,
        "label_max": float(label_arr.max()) if len(label_arr) else 0.0,
        "label_pos_frac": float((label_arr > 0).mean()) if len(label_arr) else 0.0,
    }
    return labels, tp_pct_arr, sl_pct_arr, stats


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class FastDataset(Dataset):
    def __init__(self, features, labels, regime=None, tp_pct=None, sl_pct=None,
                 lookback=LOOKBACK_BARS_MODEL):
        self.features = features
        self.labels = labels
        self.regime = regime
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.lookback = lookback
        max_start = len(features) - lookback
        if max_start <= 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            candidates = np.arange(max_start)
            label_indices = candidates + lookback - 1
            valid_mask = (label_indices < len(labels)) & ~torch.isnan(labels[label_indices]).numpy()
            self.valid_indices = candidates[valid_mask]

    def get_regime(self, idx):
        if self.regime is None:
            return np.nan
        real_idx = self.valid_indices[idx]
        li = real_idx + self.lookback - 1
        return self.regime[li] if li < len(self.regime) else np.nan

    def get_tp_pct(self, idx):
        if self.tp_pct is None:
            return np.nan
        real_idx = self.valid_indices[idx]
        li = real_idx + self.lookback - 1
        return self.tp_pct[li] if li < len(self.tp_pct) else np.nan

    def get_sl_pct(self, idx):
        if self.sl_pct is None:
            return np.nan
        real_idx = self.valid_indices[idx]
        li = real_idx + self.lookback - 1
        return self.sl_pct[li] if li < len(self.sl_pct) else np.nan

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.features[real_idx:real_idx + self.lookback]
        y = self.labels[real_idx + self.lookback - 1]
        return x, y


# ═══════════════════════════════════════════════════════════════════
# Training (regression flavor)
# ═══════════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(ds, batch_size, shuffle=False):
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and NUM_WORKERS > 0,
    )


def regression_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute regression diagnostics. preds and labels are 1D numpy arrays."""
    if len(preds) == 0:
        return {"mean_pred": 0.0, "std_pred": 0.0, "mean_label": 0.0,
                "corr": 0.0, "precision_at_threshold": 0.0,
                "n_above_threshold": 0}

    mean_pred = float(preds.mean())
    std_pred = float(preds.std())
    mean_label = float(labels.mean())

    if std_pred > 1e-9 and labels.std() > 1e-9:
        corr = float(np.corrcoef(preds, labels)[0, 1])
    else:
        corr = 0.0

    above = preds > PRECISION_AT_THRESHOLD
    n_above = int(above.sum())
    if n_above > 0:
        precision_at = float((labels[above] > 0).mean())
    else:
        precision_at = 0.0

    return {
        "mean_pred": mean_pred,
        "std_pred": std_pred,
        "mean_label": mean_label,
        "corr": corr,
        "precision_at_threshold": precision_at,
        "n_above_threshold": n_above,
    }


def train_fold_regression(model, train_loader, val_loader, device, use_amp=False):
    """Train v6-prime as a regressor with Huber loss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    swa_weights = None
    swa_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_preds_buf, train_labels_buf = [], []

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(y)
            train_count += len(y)
            train_preds_buf.append(pred.detach().float().cpu().numpy())
            train_labels_buf.append(y.detach().float().cpu().numpy())

        train_loss = train_loss_sum / train_count
        train_preds = np.concatenate(train_preds_buf)
        train_labels = np.concatenate(train_labels_buf)
        train_metrics = regression_metrics(train_preds, train_labels)

        # ─── Validation ───
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds_buf, val_labels_buf = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss_sum += loss.item() * len(y)
                val_count += len(y)
                val_preds_buf.append(pred.float().cpu().numpy())
                val_labels_buf.append(y.float().cpu().numpy())

        val_loss = val_loss_sum / max(val_count, 1)
        val_preds = np.concatenate(val_preds_buf) if val_preds_buf else np.array([])
        val_labels = np.concatenate(val_labels_buf) if val_labels_buf else np.array([])
        val_metrics = regression_metrics(val_preds, val_labels)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    Ep {epoch:3d} | "
                f"tr huber={train_loss:.5f} mp={train_metrics['mean_pred']:+.4f} "
                f"sp={train_metrics['std_pred']:.4f} corr={train_metrics['corr']:+.3f} "
                f"p@1.5={train_metrics['precision_at_threshold']:.3f} "
                f"(n={train_metrics['n_above_threshold']:>5d}) "
                f"| val huber={val_loss:.5f} corr={val_metrics['corr']:+.3f} "
                f"p@1.5={val_metrics['precision_at_threshold']:.3f} "
                f"(n={val_metrics['n_above_threshold']:>5d})"
            )

        # SWA
        if epoch >= SWA_START_EPOCH and epoch % SWA_UPDATE_FREQ == 0:
            current_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = current_state
                swa_count = 1
            else:
                for k in swa_weights:
                    swa_weights[k] = (swa_weights[k] * swa_count + current_state[k]) / (swa_count + 1)
                swa_count += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"    Early stop at epoch {epoch} (best val_loss at epoch {epoch - epochs_no_improve}, SWA count={swa_count})")
                break

    if swa_weights is not None and swa_count >= 2:
        model.load_state_dict(swa_weights)
        print(f"    SWA applied: averaged {swa_count} checkpoints")
    elif best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, epoch


def evaluate_fold_regression(model, test_loader, device, use_amp=False):
    """Return (predictions, labels) — both continuous floats."""
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(x)
            all_preds.extend(pred.float().cpu().tolist())
            all_labels.extend(y.tolist())
    return np.array(all_preds, dtype=np.float32), np.array(all_labels, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Model builder (same v6-prime architecture, unchanged)
# ═══════════════════════════════════════════════════════════════════
def build_v6_prime():
    return TemporalEnrichedV6Prime(
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    device = get_device()
    use_amp = device.type == "cuda"

    print("=" * 70)
    print("  v9-REGRESSION (Stage 1: Philosophy D)")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

    print(f"\nConfig:")
    print(f"  Labels: continuous realized P&L (TP_RATIO={TP_RATIO}, SL_RATIO={SL_RATIO})")
    print(f"  Loss: HuberLoss(delta={HUBER_DELTA})")
    print(f"  Embargo: {EMBARGO_BARS} bars ({EMBARGO_BARS//BARS_PER_DAY} days)")
    print(f"  Holdout starts: {HOLDOUT_START.date()}")
    print(f"  Multi-seed: {N_SEEDS} seeds per fold (ensemble by mean of predictions)")
    print(f"  SWA: from epoch {SWA_START_EPOCH}, every {SWA_UPDATE_FREQ}")
    print(f"  Precision-at metric threshold: {PRECISION_AT_THRESHOLD}")

    # ─── Load data ───
    t0 = time.time()
    print("\nLoading v1 features...")
    df = build_feature_matrix_v1()
    print(f"  {len(df)} rows, {len(FEATURE_COLS_V1)} features")

    # ─── Compute regression labels ───
    print("\nComputing VP-derived REGRESSION labels...")
    labels, tp_pct_arr, sl_pct_arr, label_stats = precompute_vp_regression_labels(
        close=df["close"].values,
        ceiling_dist=df["vp_ceiling_dist"].values,
        floor_dist=df["vp_floor_dist"].values,
        num_peaks=df["vp_num_peaks"].values,
    )
    print(f"  Valid labels: {label_stats['n_valid']:,}")
    print(f"  TP hit:       {label_stats['n_tp_hit']:,} ({label_stats['n_tp_hit']/max(label_stats['n_valid'],1)*100:.1f}%)")
    print(f"  SL hit:       {label_stats['n_sl_hit']:,} ({label_stats['n_sl_hit']/max(label_stats['n_valid'],1)*100:.1f}%)")
    print(f"  Timeout:      {label_stats['n_timeout']:,} ({label_stats['n_timeout']/max(label_stats['n_valid'],1)*100:.1f}%)")
    print(f"  Label mean:   {label_stats['label_mean']*100:+.3f}%")
    print(f"  Label median: {label_stats['label_median']*100:+.3f}%")
    print(f"  Label std:    {label_stats['label_std']*100:.3f}%")
    print(f"  Label range:  [{label_stats['label_min']*100:+.2f}%, {label_stats['label_max']*100:+.2f}%]")
    print(f"  Pos frac:     {label_stats['label_pos_frac']*100:.1f}%")

    regime = build_regime_array(df["date"].values)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    features = torch.from_numpy(df[FEATURE_COLS_V1].values.astype(np.float32))
    labels_t = torch.from_numpy(labels)
    dates = pd.to_datetime(df["date"]).values

    # ─── Walk-forward ───
    all_preds, all_labels, all_regimes = [], [], []
    all_tp_pct, all_sl_pct = [], []
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

        seed_preds = []
        total_epochs_run = 0

        for seed_idx in range(N_SEEDS):
            seed = 42 + seed_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"    [Seed {seed}]")
            model = build_v6_prime()
            if USE_COMPILE and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                except Exception:
                    pass

            model, epochs_run = train_fold_regression(model, train_loader, val_loader, device, use_amp)
            seed_test_preds, _ = evaluate_fold_regression(model, test_loader, device, use_amp)
            seed_preds.append(seed_test_preds)
            total_epochs_run += epochs_run

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Ensemble: average predictions across seeds
        fold_preds = np.mean(np.stack(seed_preds), axis=0)

        fold_labels = np.array([
            test_ds.labels[test_ds.valid_indices[j] + test_ds.lookback - 1].item()
            for j in range(len(test_ds))
        ], dtype=np.float32)

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

        fold_metrics = regression_metrics(fold_preds, fold_labels)
        fold_time = time.time() - fold_start

        # Real EV at threshold
        above = fold_preds > PRECISION_AT_THRESHOLD
        n_above = int(above.sum())
        if n_above > 0:
            real_ev_at_threshold = float(fold_labels[above].mean())
        else:
            real_ev_at_threshold = 0.0

        fold_results.append({
            "fold": i + 1,
            "period": f"{FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]}",
            "n_test": len(fold_preds),
            "n_seeds": N_SEEDS,
            "total_epochs": total_epochs_run,
            "time_sec": round(fold_time, 1),
            "mean_pred": round(fold_metrics["mean_pred"], 5),
            "std_pred": round(fold_metrics["std_pred"], 5),
            "corr": round(fold_metrics["corr"], 4),
            "precision_at_threshold": round(fold_metrics["precision_at_threshold"], 4),
            "n_above_threshold": int(fold_metrics["n_above_threshold"]),
            "real_ev_at_threshold_pct": round(real_ev_at_threshold * 100, 4),
            "avg_tp_pct": round(float(np.nanmean(fold_tp_pct)), 4),
            "avg_sl_pct": round(float(np.nanmean(fold_sl_pct)), 4),
        })
        all_preds.extend(fold_preds.tolist())
        all_labels.extend(fold_labels.tolist())
        all_regimes.extend(fold_regimes.tolist())
        all_tp_pct.extend(fold_tp_pct.tolist())
        all_sl_pct.extend(fold_sl_pct.tolist())
        all_dates.extend(fold_dates.tolist())
        all_close.extend(fold_close.tolist())

        print(
            f"    [Ensemble] corr={fold_metrics['corr']:+.3f} "
            f"p@1.5={fold_metrics['precision_at_threshold']:.3f} "
            f"(n={n_above}) real_EV@1.5={real_ev_at_threshold*100:+.3f}% "
            f"({total_epochs_run} epochs, {fold_time:.0f}s)"
        )

    total_time = time.time() - total_start

    # ─── Overall stats ───
    preds = np.array(all_preds, dtype=np.float32)
    labels_all = np.array(all_labels, dtype=np.float32)
    tp_pct_all = np.array(all_tp_pct, dtype=np.float32)
    sl_pct_all = np.array(all_sl_pct, dtype=np.float32)
    overall = regression_metrics(preds, labels_all)

    print(f"\n{'=' * 70}")
    print(f"OVERALL (all folds, ensembled):")
    print(f"  n               = {len(preds):,}")
    print(f"  mean_pred       = {overall['mean_pred']*100:+.3f}%")
    print(f"  std_pred        = {overall['std_pred']*100:.3f}%")
    print(f"  mean_label      = {overall['mean_label']*100:+.3f}%")
    print(f"  corr(pred,real) = {overall['corr']:+.4f}")
    print(f"  precision @ {PRECISION_AT_THRESHOLD*100:.1f}% = {overall['precision_at_threshold']:.4f} (n={overall['n_above_threshold']:,})")

    above = preds > PRECISION_AT_THRESHOLD
    if above.sum() > 0:
        ev_above = float(labels_all[above].mean())
        print(f"  real EV @ {PRECISION_AT_THRESHOLD*100:.1f}%   = {ev_above*100:+.4f}% per trade")

    output = {
        "config": {
            "model": "v6-prime architecture (regression head)",
            "labels": "VP-derived continuous P&L (Philosophy D)",
            "tp_ratio": TP_RATIO,
            "sl_ratio": SL_RATIO,
            "tp_range": [TP_MIN, TP_MAX],
            "sl_range": [SL_MIN, SL_MAX],
            "loss": f"HuberLoss(delta={HUBER_DELTA})",
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "n_seeds": N_SEEDS,
            "swa_start_epoch": SWA_START_EPOCH,
            "swa_update_freq": SWA_UPDATE_FREQ,
            "embargo_bars": EMBARGO_BARS,
            "precision_at_threshold": PRECISION_AT_THRESHOLD,
        },
        "label_stats": label_stats,
        "overall_metrics": overall,
        "folds": fold_results,
        "total_time_min": round(total_time / 60, 1),
    }
    out_path = EXPERIMENTS_DIR / "eval_v9_regression_results.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save predictions cache for backtest engine
    predictions_path = EXPERIMENTS_DIR / "v9_regression_predictions.npz"
    dates_iso = np.array([pd.Timestamp(d).isoformat() for d in all_dates], dtype=object)
    np.savez(
        predictions_path,
        dates=dates_iso,
        close=np.array(all_close, dtype=np.float64),
        predicted_return=preds,
        labels=labels_all,
        tp_pct=tp_pct_all,
        sl_pct=sl_pct_all,
        regimes=np.array(all_regimes),
    )
    print(f"Predictions cache saved to {predictions_path}")


if __name__ == "__main__":
    main()
