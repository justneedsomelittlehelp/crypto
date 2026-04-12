"""Eval v6-prime: v6 architecture with VP-derived TP/SL labels + regularization.

Key changes vs v6 baseline:
  - Labels: VP-derived per-sample TP/SL (aimed at VP peaks, not fixed %)
  - Regularization: dropout=0.3, weight_decay=1e-3, label_smoothing=0.1
  - Optimizer: AdamW (proper weight decay)

Hypothesis: Labels aligned with what the model predicts (VP barriers)
should give cleaner training signal than fixed-% labels. Regularization
overhaul should fix the "converges in 1-2 epochs" issue.

Usage:
    python -m src.models.eval_v6_prime
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
WEIGHT_DECAY = 1e-3         # NEW: L2 regularization via AdamW
LABEL_SMOOTHING = 0.1       # NEW: soft labels
DROPOUT = 0.3               # NEW: up from 0.15
NUM_WORKERS = 4
USE_COMPILE = True

# VP-derived label parameters
TP_RATIO = 0.8              # Aim 80% of the way to VP peak
SL_RATIO = 0.6              # Stop 60% of the way to opposite VP peak
TP_MIN = 0.01               # Minimum 1%
TP_MAX = 0.15               # Maximum 15%
SL_MIN = 0.01
SL_MAX = 0.15
MIN_PEAKS = 1               # Require at least 1 peak (ceiling OR floor)

# Reference fixed TP/SL for EV calculations (approximation — actual labels are per-sample)
TP_REF = 0.075
SL_REF = 0.03

# Multi-seed + SWA config
N_SEEDS = 3                 # Number of seeds per fold (3 = good balance of quality/time)
SWA_START_EPOCH = 15        # Start SWA averaging after this epoch
SWA_UPDATE_FREQ = 1         # Update SWA every N epochs (1 = every epoch)

FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
]


# ═══════════════════════════════════════════════════════════════════
# Regime array (still used for analytics, not labels)
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
# VP-derived TP/SL labels
# ═══════════════════════════════════════════════════════════════════
def precompute_vp_labels(
    close: np.ndarray,
    ceiling_dist: np.ndarray,
    floor_dist: np.ndarray,
    num_peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-sample VP-derived TP/SL first-hit labels.

    For each bar:
      TP = entry × (1 + ceiling_dist × REL_SPAN × TP_RATIO)
      SL = entry × (1 - floor_dist × REL_SPAN × SL_RATIO)

    TP/SL are clipped to [TP_MIN, TP_MAX] and [SL_MIN, SL_MAX].
    Bars with num_peaks < MIN_PEAKS get NaN labels (skipped).

    Returns:
        labels: (n,) float32 array with {0, 1, NaN}
        tp_pct: (n,) float32 array with per-sample TP % (NaN for skipped)
        sl_pct: (n,) float32 array with per-sample SL % (NaN for skipped)
        stats: dict with label distribution info
    """
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)
    tp_pct_arr = np.full(n, np.nan, dtype=np.float32)
    sl_pct_arr = np.full(n, np.nan, dtype=np.float32)

    # Vol scaling (still applied — scales TP/SL with current market vol)
    vol_window = 30
    log_returns = np.diff(np.log(close))
    rolling_vol = np.full(n, np.nan)
    lr_series = pd.Series(log_returns)
    rolling_std = lr_series.rolling(window=vol_window, min_periods=vol_window).std().values
    rolling_vol[vol_window + 1:] = rolling_std[vol_window:]

    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) == 0:
        return labels, tp_pct_arr, sl_pct_arr, {"error": "no valid vol"}
    median_vol = np.median(valid_vol)
    if median_vol < 1e-10:
        median_vol = 1e-10
    vol_scale = np.clip(rolling_vol / median_vol, 0.5, 3.0)

    warmup = vol_window + 1
    max_bars = LABEL_MAX_BARS

    # Stats
    tp_pct_list = []
    sl_pct_list = []
    skipped_nopeaks = 0
    skipped_nanstruct = 0
    skipped_novol = 0

    for i in range(warmup, n - 1):
        if np.isnan(vol_scale[i]):
            skipped_novol += 1
            continue
        if np.isnan(ceiling_dist[i]) or np.isnan(floor_dist[i]) or np.isnan(num_peaks[i]):
            skipped_nanstruct += 1
            continue
        if num_peaks[i] < MIN_PEAKS:
            skipped_nopeaks += 1
            continue

        # VP peak distances → percentages
        ceiling_pct = ceiling_dist[i] * REL_SPAN_PCT  # 0-0.25
        floor_pct = floor_dist[i] * REL_SPAN_PCT

        # Apply TP/SL ratios and vol scaling
        tp_pct = ceiling_pct * TP_RATIO * vol_scale[i]
        sl_pct = floor_pct * SL_RATIO * vol_scale[i]

        # Clip
        tp_pct = np.clip(tp_pct, TP_MIN, TP_MAX)
        sl_pct = np.clip(sl_pct, SL_MIN, SL_MAX)

        tp_pct_list.append(tp_pct)
        sl_pct_list.append(sl_pct)

        # Store per-sample TP/SL (always, even if label ends up NaN)
        tp_pct_arr[i] = tp_pct
        sl_pct_arr[i] = sl_pct

        # First-hit logic
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

        if tp_first < len(future) or sl_first < len(future):
            labels[i] = 1.0 if tp_first <= sl_first else 0.0

    stats = {
        "n_valid": int((~np.isnan(labels)).sum()),
        "n_label_1": int((labels == 1.0).sum()),
        "n_label_0": int((labels == 0.0).sum()),
        "n_skipped_nopeaks": skipped_nopeaks,
        "n_skipped_nanstruct": skipped_nanstruct,
        "n_skipped_novol": skipped_novol,
        "tp_pct_mean": float(np.mean(tp_pct_list)) if tp_pct_list else 0,
        "tp_pct_median": float(np.median(tp_pct_list)) if tp_pct_list else 0,
        "tp_pct_std": float(np.std(tp_pct_list)) if tp_pct_list else 0,
        "sl_pct_mean": float(np.mean(sl_pct_list)) if sl_pct_list else 0,
        "sl_pct_median": float(np.median(sl_pct_list)) if sl_pct_list else 0,
        "sl_pct_std": float(np.std(sl_pct_list)) if sl_pct_list else 0,
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
        self.tp_pct = tp_pct    # np array or None
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
# Training
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


def smooth_labels(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing: 0 → smoothing, 1 → 1 - smoothing."""
    return y * (1 - smoothing) + 0.5 * smoothing


def train_fold(model, train_loader, val_loader, device, use_amp=False):
    """Train v6-prime with regularization overhaul."""
    # Compute pos_weight for class balance
    all_labels = torch.cat([y for _, y in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    model = model.to(device)
    # AdamW with weight decay (proper L2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    # SWA state
    swa_weights = None
    swa_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Apply label smoothing
            y_smooth = smooth_labels(y, LABEL_SMOOTHING)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, y_smooth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(y)
            preds = (logits.detach() > 0).float()
            train_correct += (preds == y).sum().item()  # Accuracy on hard labels
            train_total += len(y)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                y_smooth = smooth_labels(y, LABEL_SMOOTHING)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y_smooth)
                val_loss_sum += loss.item() * len(y)
                preds = (logits > 0).float()
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Train {train_acc:.4f} | Val {val_acc:.4f} | Val Loss {val_loss:.4f}")

        # ─── SWA weight averaging (after warmup epoch) ───
        if epoch >= SWA_START_EPOCH and epoch % SWA_UPDATE_FREQ == 0:
            current_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = current_state
                swa_count = 1
            else:
                # Running average: swa = (swa * count + current) / (count + 1)
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

    # Use SWA weights if available (averaged over last epochs)
    # Falls back to best_state if SWA never activated (short training)
    if swa_weights is not None and swa_count >= 2:
        model.load_state_dict(swa_weights)
        print(f"    SWA applied: averaged {swa_count} checkpoints")
    elif best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, epoch


def evaluate_fold(model, test_loader, device, use_amp=False):
    """Return predictions, labels, and raw logits for confidence filtering."""
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x)
            logits_cpu = logits.float().cpu()
            preds = (logits_cpu > 0).float()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
            all_logits.extend(logits_cpu.tolist())
    return np.array(all_preds), np.array(all_labels), np.array(all_logits)


# ═══════════════════════════════════════════════════════════════════
# Model builder
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
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

    print(f"\nConfig:")
    print(f"  Labels: VP-derived TP/SL (TP_RATIO={TP_RATIO}, SL_RATIO={SL_RATIO})")
    print(f"  Clip: TP [{TP_MIN*100}%, {TP_MAX*100}%], SL [{SL_MIN*100}%, {SL_MAX*100}%]")
    print(f"  Regularization: dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}, label_smoothing={LABEL_SMOOTHING}")
    print(f"  Optimizer: AdamW, LR={LR}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  Multi-seed: {N_SEEDS} seeds per fold (ensemble by logit averaging)")
    print(f"  SWA: start at epoch {SWA_START_EPOCH}, update every {SWA_UPDATE_FREQ}")

    # ─── Load data ───
    t0 = time.time()
    print("\nLoading v1 features...")
    df = build_feature_matrix_v1()
    print(f"  {len(df)} rows, {len(FEATURE_COLS_V1)} features")

    # ─── Compute VP-derived labels ───
    print("\nComputing VP-derived labels...")
    labels, tp_pct_arr, sl_pct_arr, label_stats = precompute_vp_labels(
        close=df["close"].values,
        ceiling_dist=df["vp_ceiling_dist"].values,
        floor_dist=df["vp_floor_dist"].values,
        num_peaks=df["vp_num_peaks"].values,
    )
    print(f"  Valid labels: {label_stats['n_valid']:,}")
    print(f"  Label 1: {label_stats['n_label_1']:,} ({label_stats['n_label_1']/max(label_stats['n_valid'],1)*100:.1f}%)")
    print(f"  Label 0: {label_stats['n_label_0']:,} ({label_stats['n_label_0']/max(label_stats['n_valid'],1)*100:.1f}%)")
    print(f"  Skipped (no peaks): {label_stats['n_skipped_nopeaks']:,}")
    print(f"  Skipped (NaN structure): {label_stats['n_skipped_nanstruct']:,}")
    print(f"  TP pct: mean={label_stats['tp_pct_mean']*100:.2f}%, median={label_stats['tp_pct_median']*100:.2f}%, std={label_stats['tp_pct_std']*100:.2f}%")
    print(f"  SL pct: mean={label_stats['sl_pct_mean']*100:.2f}%, median={label_stats['sl_pct_median']*100:.2f}%, std={label_stats['sl_pct_std']*100:.2f}%")

    regime = build_regime_array(df["date"].values)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Convert
    features = torch.from_numpy(df[FEATURE_COLS_V1].values.astype(np.float32))
    labels_t = torch.from_numpy(labels)
    dates = pd.to_datetime(df["date"]).values

    # ─── Walk-forward ───
    all_preds, all_labels, all_regimes = [], [], []
    all_tp_pct, all_sl_pct, all_logits = [], [], []
    all_dates, all_close = [], []     # For backtest caching
    fold_results = []

    close_prices = df["close"].values.astype(np.float64)

    print(f"\nStarting walk-forward...")
    total_start = time.time()

    for i in range(len(FOLD_BOUNDARIES) - 2):
        fold_start = time.time()
        train_end = pd.Timestamp(FOLD_BOUNDARIES[i])
        val_end = pd.Timestamp(FOLD_BOUNDARIES[i + 1])
        test_end = pd.Timestamp(FOLD_BOUNDARIES[i + 2])

        train_mask = dates < train_end
        val_mask = (dates >= train_end) & (dates < val_end)
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

        # ─── Multi-seed training with SWA ───
        seed_logits = []   # Collect logits from each seed for ensembling
        total_epochs_run = 0

        for seed_idx in range(N_SEEDS):
            seed = 42 + seed_idx  # seeds 42, 43, 44, ...
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

            model, epochs_run = train_fold(model, train_loader, val_loader, device, use_amp)
            _, _, seed_fold_logits = evaluate_fold(model, test_loader, device, use_amp)
            seed_logits.append(seed_fold_logits)
            total_epochs_run += epochs_run

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Ensemble: average logits across seeds
        fold_logits = np.mean(np.stack(seed_logits), axis=0)
        preds = (fold_logits > 0).astype(np.float32)

        # Get labels from test_ds (same for all seeds)
        fold_labels = np.array([test_ds.labels[test_ds.valid_indices[j] + test_ds.lookback - 1].item()
                                for j in range(len(test_ds))])

        fold_regimes = np.array([test_ds.get_regime(j) for j in range(len(test_ds))])
        fold_tp_pct = np.array([test_ds.get_tp_pct(j) for j in range(len(test_ds))])
        fold_sl_pct = np.array([test_ds.get_sl_pct(j) for j in range(len(test_ds))])

        # Absolute bar indices in df (for date/close lookup)
        test_slice_start = test_idx[0]
        fold_bar_idx = np.array([
            test_slice_start + test_ds.valid_indices[j] + test_ds.lookback - 1
            for j in range(len(test_ds))
        ])
        fold_dates = dates[fold_bar_idx]
        fold_close = close_prices[fold_bar_idx]

        fold_acc = (preds == fold_labels).mean()
        fold_time = time.time() - fold_start

        # Per-fold real EV (long-only strategy: trade when pred=1)
        long_mask = preds == 1
        long_wins = long_mask & (fold_labels == 1)
        long_losses = long_mask & (fold_labels == 0)
        n_long = long_mask.sum()
        if n_long > 0:
            fold_long_ev = (
                fold_tp_pct[long_wins].sum() - fold_sl_pct[long_losses].sum()
            ) / n_long * 100  # Convert to percentage
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

        print(f"    [Ensemble] Acc: {fold_acc:.4f} ({N_SEEDS} seeds, {total_epochs_run} total epochs, {fold_time:.0f}s)")
        print(f"    Long trades: {int(n_long)}, Real EV/trade: {fold_long_ev:+.3f}%")

    total_time = time.time() - total_start

    # ─── Overall stats ───
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    regimes = np.array(all_regimes)
    tp_pct_all = np.array(all_tp_pct)
    sl_pct_all = np.array(all_sl_pct)
    logits_all = np.array(all_logits)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))   # sigmoid

    def confusion(p, l):
        tp = int(((p == 1) & (l == 1)).sum())
        fp = int(((p == 1) & (l == 0)).sum())
        tn = int(((p == 0) & (l == 0)).sum())
        fn = int(((p == 0) & (l == 1)).sum())
        prec = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        acc = (tp + tn) / max(len(p), 1)
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": round(prec, 4), "npv": round(npv, 4),
                "accuracy": round(acc, 4), "n": len(p)}

    overall = confusion(preds, labels)
    bull = regimes == 1.0
    bear = regimes == 0.0
    bull_stats = confusion(preds[bull], labels[bull]) if bull.sum() > 0 else None
    bear_stats = confusion(preds[bear], labels[bear]) if bear.sum() > 0 else None

    # ─── REAL per-sample EV ───
    # Strategies:
    #   Long only: trade when pred=1. Win = actual tp_pct, Loss = -actual sl_pct
    #   Short only: trade when pred=0. Win = actual sl_pct, Loss = -actual tp_pct
    #   Both sides: trade every prediction
    def compute_real_ev(mask, pred, lbl, tp, sl):
        """Compute real EV for a strategy mask (which samples to trade on).
        mask can be (pred == 1) for long-only, (pred == 0) for short-only, etc.
        """
        if mask.sum() == 0:
            return 0.0, 0
        selected_pred = pred[mask]
        selected_lbl = lbl[mask]
        selected_tp = tp[mask]
        selected_sl = sl[mask]

        profits = np.zeros(len(selected_pred))
        for j in range(len(selected_pred)):
            if selected_pred[j] == 1:  # long trade
                if selected_lbl[j] == 1:  # TP hit
                    profits[j] = selected_tp[j]
                else:  # SL hit
                    profits[j] = -selected_sl[j]
            else:  # short trade
                if selected_lbl[j] == 0:  # SL hit (price went down — short wins)
                    profits[j] = selected_sl[j]
                else:  # TP hit (price went up — short loses)
                    profits[j] = -selected_tp[j]

        return float(np.mean(profits) * 100), int(mask.sum())  # Return as percentage

    # Long-only: trade when pred=1
    long_mask = preds == 1
    long_ev, long_n = compute_real_ev(long_mask, preds, labels, tp_pct_all, sl_pct_all)

    # Short-only: trade when pred=0
    short_mask = preds == 0
    short_ev, short_n = compute_real_ev(short_mask, preds, labels, tp_pct_all, sl_pct_all)

    # Both sides
    both_ev = (long_ev * long_n + short_ev * short_n) / max(long_n + short_n, 1)

    # Per-regime real EV
    real_regime_ev = {}
    for regime_name, mask_base in [("bull", bull), ("bear", bear)]:
        # Long
        m = mask_base & (preds == 1)
        ev, n = compute_real_ev(m, preds, labels, tp_pct_all, sl_pct_all)
        real_regime_ev[f"{regime_name}_long"] = round(ev, 3)
        real_regime_ev[f"{regime_name}_long_n"] = n
        # Short
        m = mask_base & (preds == 0)
        ev, n = compute_real_ev(m, preds, labels, tp_pct_all, sl_pct_all)
        real_regime_ev[f"{regime_name}_short"] = round(ev, 3)
        real_regime_ev[f"{regime_name}_short_n"] = n

    print(f"\n{'=' * 70}")
    print(f"  v6-PRIME RESULTS (VP-derived labels + regularization)")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Overall: Acc={overall['accuracy']:.4f}  Prec={overall['precision']:.4f}  NPV={overall['npv']:.4f}")
    print(f"  TP={overall['tp']} FP={overall['fp']} TN={overall['tn']} FN={overall['fn']}")

    print(f"\n  REAL EV (per-sample TP/SL):")
    print(f"    Long-only  (pred=1, {long_n:,} trades):  {long_ev:+.3f}% per trade")
    print(f"    Short-only (pred=0, {short_n:,} trades):  {short_ev:+.3f}% per trade")
    print(f"    Both sides  ({long_n+short_n:,} trades):  {both_ev:+.3f}% per trade")

    print(f"\n  Per-fold:")
    print(f"  {'Fold':<6} {'Period':<28} {'Acc':>7} {'Epochs':>7} {'Long#':>7} {'Long EV':>9} {'TP avg':>8} {'SL avg':>8}")
    for fr in fold_results:
        print(f"  {fr['fold']:<6} {fr['period']:<28} {fr['acc']*100:6.1f}% {fr.get('total_epochs', 0):>7} {fr.get('n_long_trades', 0):>7,} {fr.get('long_ev_real', 0):+8.3f}% {fr.get('avg_tp_pct', 0)*100:7.2f}% {fr.get('avg_sl_pct', 0)*100:7.2f}%")

    if bull_stats:
        print(f"\n  BULL ({bull_stats['n']} bars): Prec={bull_stats['precision']:.4f}  NPV={bull_stats['npv']:.4f}")
        print(f"    Long-only:  {real_regime_ev['bull_long']:+.3f}% ({real_regime_ev['bull_long_n']:,} trades)")
        print(f"    Short-only: {real_regime_ev['bull_short']:+.3f}% ({real_regime_ev['bull_short_n']:,} trades)")
    if bear_stats:
        print(f"\n  BEAR ({bear_stats['n']} bars): Prec={bear_stats['precision']:.4f}  NPV={bear_stats['npv']:.4f}")
        print(f"    Long-only:  {real_regime_ev['bear_long']:+.3f}% ({real_regime_ev['bear_long_n']:,} trades)")
        print(f"    Short-only: {real_regime_ev['bear_short']:+.3f}% ({real_regime_ev['bear_short_n']:,} trades)")

    # Keep the legacy fake EV for comparison with older runs
    regime_ev = {}
    if bull_stats:
        regime_ev["bull_long_fake"] = round(bull_stats["precision"] * 7.5 - (1 - bull_stats["precision"]) * 3.0, 2)
        regime_ev["bull_short_fake"] = round(bull_stats["npv"] * 3.0 - (1 - bull_stats["npv"]) * 7.5, 2)
    if bear_stats:
        regime_ev["bear_long_fake"] = round(bear_stats["precision"] * 3.0 - (1 - bear_stats["precision"]) * 7.5, 2)
        regime_ev["bear_short_fake"] = round(bear_stats["npv"] * 7.5 - (1 - bear_stats["npv"]) * 3.0, 2)

    # ═══════════════════════════════════════════════════════════════════
    # Filter analysis — try different selection strategies
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  FILTER ANALYSIS — long-only, filtered")
    print(f"{'=' * 70}")

    def compute_filter_stats(filter_mask):
        """Compute full trading stats for samples matching filter.

        Returns dict with:
          n_trades, precision, avg_win, avg_loss,
          ev_arith (arithmetic mean = EV/trade),
          ev_geom (geometric mean return per trade, compound-adjusted),
          compound_total (what $1 becomes after all trades),
          max_consec_losses, sharpe
        """
        m = filter_mask & (preds == 1)
        n = int(m.sum())
        if n == 0:
            return {"n_trades": 0, "precision": 0.0, "avg_win": 0.0,
                    "avg_loss": 0.0, "ev_arith": 0.0, "ev_geom": 0.0,
                    "compound_total": 1.0, "max_consec_losses": 0, "sharpe": 0.0}

        wins_mask = m & (labels == 1)
        losses_mask = m & (labels == 0)
        n_wins = int(wins_mask.sum())
        n_losses = int(losses_mask.sum())

        # Per-trade returns (signed)
        returns = np.zeros(n)
        sel_preds = preds[m]
        sel_labels = labels[m]
        sel_tp = tp_pct_all[m]
        sel_sl = sl_pct_all[m]
        for j in range(n):
            if sel_labels[j] == 1:
                returns[j] = sel_tp[j]   # win
            else:
                returns[j] = -sel_sl[j]  # loss

        # Sort by chronological order for streak/compound calculation
        # (samples are already in fold order which is chronological)

        # Arithmetic mean (EV per trade)
        ev_arith = returns.mean() * 100

        # Geometric mean: (prod(1+r))^(1/n) - 1
        # Use log-sum-exp for numerical stability
        log_returns = np.log1p(returns)
        ev_geom = (np.exp(log_returns.mean()) - 1) * 100

        # Compound total: what $1 becomes
        compound_total = float(np.exp(log_returns.sum()))

        # Win rate = precision of long predictions
        precision = n_wins / n

        # Avg win / avg loss (magnitudes, in %)
        avg_win = float(sel_tp[sel_labels == 1].mean() * 100) if n_wins > 0 else 0.0
        avg_loss = float(sel_sl[sel_labels == 0].mean() * 100) if n_losses > 0 else 0.0

        # Max consecutive losses
        is_loss = (returns < 0).astype(int)
        max_streak = 0
        cur = 0
        for x in is_loss:
            if x == 1:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0

        # Sharpe-like: mean / std (per-trade, not annualized)
        sharpe = float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0

        return {
            "n_trades": n,
            "precision": round(precision, 4),
            "avg_win": round(avg_win, 3),
            "avg_loss": round(avg_loss, 3),
            "ev_arith": round(float(ev_arith), 3),
            "ev_geom": round(float(ev_geom), 3),
            "compound_total": round(compound_total, 3),
            "max_consec_losses": max_streak,
            "sharpe": round(sharpe, 3),
        }

    filter_results = {}

    def print_row(label, stats):
        """Print a filter's stats in a compact row."""
        print(f"  {label:<18} {stats['n_trades']:>7,} trades  "
              f"prec={stats['precision']*100:5.1f}%  "
              f"EV(arith)={stats['ev_arith']:+6.2f}%  "
              f"EV(geom)={stats['ev_geom']:+6.2f}%  "
              f"×{stats['compound_total']:.1f}")

    # 1. Raw confidence thresholds
    print(f"\n  (A) Raw confidence (sigmoid(logit) > threshold):")
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = probs_all > th
        stats = compute_filter_stats(mask)
        filter_results[f"conf_{th}"] = stats
        print_row(f"conf > {th}", stats)

    # 2. Expected EV filter
    print(f"\n  (B) Expected EV filter (P(tp)*tp - P(sl)*sl > threshold):")
    expected_ev = probs_all * tp_pct_all - (1 - probs_all) * sl_pct_all
    for th in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        mask = expected_ev > th
        stats = compute_filter_stats(mask)
        filter_results[f"expev_{th}"] = stats
        print_row(f"expEV > {th}", stats)

    # 3. Asymmetry filter
    print(f"\n  (C) Asymmetry filter (tp_pct / sl_pct > ratio):")
    safe_ratio = tp_pct_all / np.clip(sl_pct_all, 1e-8, None)
    for ratio in [1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = safe_ratio > ratio
        stats = compute_filter_stats(mask)
        filter_results[f"asym_{ratio}"] = stats
        print_row(f"tp/sl > {ratio}", stats)

    # 4. Combined filters
    print(f"\n  (D) Combined filters:")
    mask = (probs_all > 0.65) & (safe_ratio > 1.5)
    stats = compute_filter_stats(mask)
    filter_results["combined_65_15"] = stats
    print_row("conf>0.65 + asym>1.5", stats)

    mask = (probs_all > 0.65) & (expected_ev > 0.02)
    stats = compute_filter_stats(mask)
    filter_results["combined_65_exev02"] = stats
    print_row("conf>0.65 + expEV>0.02", stats)

    mask = (probs_all > 0.60) & (safe_ratio > 2.0)
    stats = compute_filter_stats(mask)
    filter_results["combined_60_asym20"] = stats
    print_row("conf>0.60 + asym>2.0", stats)

    # Logit distribution diagnostic
    print(f"\n  Logit distribution:")
    print(f"    Mean:   {logits_all.mean():+.3f}")
    print(f"    Std:    {logits_all.std():.3f}")
    print(f"    Median: {np.median(logits_all):+.3f}")
    print(f"    Q25/Q75: {np.percentile(logits_all, 25):+.3f} / {np.percentile(logits_all, 75):+.3f}")
    print(f"    Q10/Q90: {np.percentile(logits_all, 10):+.3f} / {np.percentile(logits_all, 90):+.3f}")
    print(f"  Softer distribution = better calibration. Label smoothing should keep |logit| small.")

    # Save
    output = {
        "config": {
            "model": "v6-prime (TemporalEnrichedV6Prime)",
            "labels": "VP-derived TP/SL",
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
        },
        "label_stats": label_stats,
        "overall": overall,
        "bull_stats": bull_stats,
        "bear_stats": bear_stats,
        "real_ev": {
            "long_only": round(long_ev, 3),
            "long_trades": long_n,
            "short_only": round(short_ev, 3),
            "short_trades": short_n,
            "both_sides": round(both_ev, 3),
            **real_regime_ev,
        },
        "legacy_regime_ev_fake": regime_ev,
        "filter_results": filter_results,
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
    out_path = EXPERIMENTS_DIR / "eval_v6_prime_results.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save predictions cache for backtest engine
    predictions_path = EXPERIMENTS_DIR / "v6_prime_predictions.npz"
    dates_iso = np.array([pd.Timestamp(d).isoformat() for d in all_dates], dtype=object)
    np.savez(
        predictions_path,
        dates=dates_iso,
        close=np.array(all_close, dtype=np.float64),
        logits=logits_all,
        probs=probs_all,
        preds=preds.astype(np.int8),
        labels=labels.astype(np.int8),
        tp_pct=tp_pct_all,
        sl_pct=sl_pct_all,
        regimes=regimes,
    )
    print(f"Predictions cache saved to {predictions_path}")


if __name__ == "__main__":
    main()
