"""Eval v11 — absolute-range VP @ 15m × 90-day temporal window.

The "final iteration" for Phase 3:
  - Feature source: BTC_15m_ABSVP_30d.csv (visible-range VP, self-channel,
    continuous price_pos + range_pct scalars).
  - Architecture:   AbsVPv11 (2-channel spatial attention, otherwise same
    2+1 transformer shape as v10).
  - Labels:         range-derived TP/SL, first-hit, long-only.
    TP = (window_hi − close) / close × 0.8, clipped to [1%, 15%]
    SL = (close − window_lo) / close × 0.6, clipped to [1%, 15%]
  - Walk-forward:   same calendar boundaries as v6-prime/v10.
  - Embargo:        14 days wall-clock (resolution-independent).
  - Throughput:     all features pre-loaded to GPU as flat tensors; per
    batch we index-gather 90 day tokens per sample on-device. No
    DataLoader, no num_workers, no per-batch CPU↔GPU copies.

Usage (Colab, H100/A100):
    !python3 -m src.models.eval_v11
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

from src.models.architectures.v11_abs_vp import AbsVPv11

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "BTC_15m_ABSVP_30d.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_PATH = EXPERIMENTS_DIR / "eval_v11_results.json"
PREDICTIONS_PATH = EXPERIMENTS_DIR / "v11_predictions.npz"

# Resolution-specific
BARS_PER_DAY = 96                      # 15m
N_DAYS = 90                            # temporal window
LOOKBACK_BARS = N_DAYS * BARS_PER_DAY  # 8640 (used only for sample eligibility)
LABEL_MAX_BARS = 14 * BARS_PER_DAY     # 1344 bars = 14 days
EMBARGO = pd.Timedelta(days=14)        # wall-clock, resolution-independent

# Training
BATCH_SIZE = 1024
LR = 5e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
EPOCHS = 50
EARLY_STOP_PATIENCE = 15
N_SEEDS = 1          # Single seed for the first go/no-go run. Bump to 3
                     # for a final reported number if v11 shows signal.
SWA_START_EPOCH = 15

# Label formula (mirrors v6-prime ratios; source features differ)
TP_RATIO = 0.8
SL_RATIO = 0.6
TP_MIN, TP_MAX = 0.01, 0.15
SL_MIN, SL_MAX = 0.01, 0.15

# Same calendar boundaries as v6-prime / v10.
FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
    "2026-01-01", "2026-04-08",
]
HOLDOUT_START = pd.Timestamp("2025-07-01", tz="UTC")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════
# Feature construction — run once on CPU, uploaded to GPU as flat tensors.
# ═══════════════════════════════════════════════════════════════════
def build_features(csv_path: Path):
    """Load CSV and return flat numpy arrays ready for GPU upload.

    Returns dict with:
      day_rows:   (N, 110) float32 — per-bar "day token" row (vp+self+candle+scalars)
      last_bar:   (N, 4)   float32 — per-bar last-bar features
      close:      (N,)     float64 — per-bar close (for label computation)
      dates:      (N,)     datetime64[ns, UTC]
      labels:     (N,)     float32 — first-hit label, NaN where invalid
      tp_pct:     (N,)     float32 — per-sample TP, NaN where invalid
      sl_pct:     (N,)     float32 — per-sample SL, NaN where invalid
    """
    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    n = len(df)
    print(f"  {n:,} rows, {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    # Raw columns
    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    vol = df["volume_15m"].values.astype(np.float64)
    win_hi = df["window_hi"].values.astype(np.float64)
    win_lo = df["window_lo"].values.astype(np.float64)
    price_pos = df["price_pos"].values.astype(np.float32)
    range_pct = df["range_pct"].values.astype(np.float32)

    vp_cols = [f"vp_abs_{k:02d}" for k in range(50)]
    self_cols = [f"self_{k:02d}" for k in range(50)]
    vp_abs = df[vp_cols].values.astype(np.float32)
    self_ch = df[self_cols].values.astype(np.float32)

    # ─── Derived per-bar features ───
    # log_return: log(close / close_prev)
    log_return = np.zeros(n, dtype=np.float32)
    log_return[1:] = np.log(close[1:] / close[:-1]).astype(np.float32)

    # volume_ratio: vol / rolling mean vol (30-day window = 30*96 = 2880 bars)
    vol_roll_bars = 30 * BARS_PER_DAY
    vol_mean = pd.Series(vol).rolling(vol_roll_bars, min_periods=vol_roll_bars).mean().values
    volume_ratio = np.where(vol_mean > 0, vol / vol_mean, 0.0).astype(np.float32)

    # OHLC-vs-close ratios (same semantics as v1_raw: open/close, high/close, low/close)
    ohlc_open_r = (open_ / close).astype(np.float32)
    ohlc_high_r = (high / close).astype(np.float32)
    ohlc_low_r = (low / close).astype(np.float32)

    # ─── Daily candle features (8 per bar, using last 96 bars) ───
    # Compute per-bar "day ending at bar i" candle features from the 96
    # intraday bars leading up to (and including) bar i. Vectorised via
    # rolling ops where possible; body/wicks via trailing max/min.
    t0 = time.time()
    print("  Computing per-bar daily candle aggregates...")

    day_open = np.full(n, np.nan, dtype=np.float32)
    day_high = np.full(n, np.nan, dtype=np.float32)
    day_low = np.full(n, np.nan, dtype=np.float32)

    # Rolling max-of-high / min-of-low over 96 bars
    high_roll_max = pd.Series(high).rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).max().values
    low_roll_min = pd.Series(low).rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).min().values
    # day_open = open of the bar 95 steps back from current (first bar of the 96-bar window)
    day_open[BARS_PER_DAY - 1:] = open_[: n - BARS_PER_DAY + 1].astype(np.float32)
    day_high[BARS_PER_DAY - 1:] = high_roll_max[BARS_PER_DAY - 1:].astype(np.float32)
    day_low[BARS_PER_DAY - 1:] = low_roll_min[BARS_PER_DAY - 1:].astype(np.float32)
    day_close = close.astype(np.float32)

    # Normalise OHLC to current close → ratios (matches v6-prime's internal
    # _aggregate_daily_candles convention where day_close is implicit 1.0).
    day_open_r = day_open / day_close
    day_high_r = day_high / day_close
    day_low_r = day_low / day_close
    day_close_r = np.ones_like(day_close)

    bar_height = np.maximum(day_high_r - day_low_r, 1e-8)
    body = day_close_r - day_open_r
    upper_wick = (day_high_r - np.maximum(day_open_r, day_close_r)) / bar_height
    lower_wick = (np.minimum(day_open_r, day_close_r) - day_low_r) / bar_height

    # Day volume ratio = mean(volume_ratio) over the last 96 bars
    day_vol_ratio = (
        pd.Series(volume_ratio)
        .rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY)
        .mean()
        .values
        .astype(np.float32)
    )

    candle = np.stack(
        [day_open_r, day_close_r, day_high_r, day_low_r, body, upper_wick, lower_wick, day_vol_ratio],
        axis=1,
    ).astype(np.float32)
    candle = np.nan_to_num(candle, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Daily candle aggregates done in {time.time() - t0:.1f}s")

    # ─── Pack day_rows: (N, 110) = vp(50) + self(50) + candle(8) + scalars(2) ───
    scalars = np.stack([price_pos, range_pct], axis=1)
    day_rows = np.concatenate([vp_abs, self_ch, candle, scalars], axis=1).astype(np.float32)

    # ─── Pack last_bar: (N, 4) = log_return, volume_ratio, price_pos, range_pct ───
    last_bar = np.stack([log_return, volume_ratio, price_pos, range_pct], axis=1).astype(np.float32)

    # ─── Labels ───
    print("  Computing range-derived TP/SL first-hit labels...")
    t0 = time.time()
    labels, tp_pct, sl_pct = compute_labels(close, win_hi, win_lo)
    n_valid = int(np.sum(~np.isnan(labels)))
    n_pos = int(np.nansum(labels == 1))
    n_neg = int(np.nansum(labels == 0))
    print(
        f"  Labels done in {time.time() - t0:.1f}s: "
        f"valid={n_valid:,} (pos={n_pos:,}, neg={n_neg:,}, "
        f"pos_rate={n_pos / max(n_valid, 1) * 100:.1f}%)"
    )

    return {
        "day_rows": day_rows,
        "last_bar": last_bar,
        "close": close,
        "dates": df["date"].values,
        "labels": labels.astype(np.float32),
        "tp_pct": tp_pct.astype(np.float32),
        "sl_pct": sl_pct.astype(np.float32),
    }


def compute_labels(close: np.ndarray, win_hi: np.ndarray, win_lo: np.ndarray):
    """Range-derived long-only TP/SL first-hit labels.

    For each bar i:
      TP_pct = clip((win_hi[i] − close[i]) / close[i] × TP_RATIO, TP_MIN, TP_MAX)
      SL_pct = clip((close[i] − win_lo[i]) / close[i] × SL_RATIO, SL_MIN, SL_MAX)
      Scan closes[i+1 : i+1+LABEL_MAX_BARS] for first touch of TP or SL.
      Label 1 if TP hit first, 0 if SL hit first, NaN if neither in window.
    """
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)
    tp_arr = np.full(n, np.nan, dtype=np.float32)
    sl_arr = np.full(n, np.nan, dtype=np.float32)

    for i in range(n - 1):
        hi = win_hi[i]
        lo = win_lo[i]
        c = close[i]
        # Must be a valid range that straddles current price
        if not (np.isfinite(hi) and np.isfinite(lo) and hi > c > lo):
            continue

        tp_pct = np.clip((hi - c) / c * TP_RATIO, TP_MIN, TP_MAX)
        sl_pct = np.clip((c - lo) / c * SL_RATIO, SL_MIN, SL_MAX)

        tp_level = c * (1 + tp_pct)
        sl_level = c * (1 - sl_pct)

        end = min(i + 1 + LABEL_MAX_BARS, n)
        future = close[i + 1:end]
        if len(future) == 0:
            continue

        tp_hits = future >= tp_level
        sl_hits = future <= sl_level
        tp_first = np.argmax(tp_hits) if tp_hits.any() else len(future) + 1
        sl_first = np.argmax(sl_hits) if sl_hits.any() else len(future) + 1

        tp_arr[i] = tp_pct
        sl_arr[i] = sl_pct

        if tp_first <= len(future) or sl_first <= len(future):
            labels[i] = 1.0 if tp_first < sl_first else 0.0

    return labels, tp_arr, sl_arr


# ═══════════════════════════════════════════════════════════════════
# On-GPU batching — no DataLoader, no workers
# ═══════════════════════════════════════════════════════════════════
def iterate_batches(idx_tensor: torch.Tensor, batch_size: int, shuffle: bool):
    """Yield batches of sample indices directly from a GPU tensor."""
    n = idx_tensor.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=idx_tensor.device)
        idx_tensor = idx_tensor[perm]
    for start in range(0, n, batch_size):
        yield idx_tensor[start:start + batch_size]


def gather_batch(
    batch_idx: torch.Tensor,
    day_rows_gpu: torch.Tensor,
    last_bar_gpu: torch.Tensor,
    labels_gpu: torch.Tensor,
    day_offsets: torch.Tensor,
):
    """Build (day_tokens, last_bar, label) tensors for a batch via index gather.

    batch_idx:     (B,) int64, absolute row indices (end-of-sample bars)
    day_offsets:   (N_DAYS,) int64, precomputed [−89×96, −88×96, …, 0]
    """
    day_indices = batch_idx[:, None] + day_offsets[None, :]   # (B, 90)
    day_tokens = day_rows_gpu[day_indices]                    # (B, 90, 110)
    last = last_bar_gpu[batch_idx]                            # (B, 4)
    y = labels_gpu[batch_idx]                                 # (B,)
    return day_tokens, last, y


# ═══════════════════════════════════════════════════════════════════
# Training / evaluation
# ═══════════════════════════════════════════════════════════════════
def smooth_labels(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    return y * (1 - smoothing) + 0.5 * smoothing


def train_one_seed(
    seed: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
    device, use_amp: bool, pos_weight: torch.Tensor,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AbsVPv11(n_days=N_DAYS, dropout=DROPOUT).to(device)
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

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch_idx in iterate_batches(train_idx, BATCH_SIZE, shuffle=True):
            day_tokens, last, y = gather_batch(
                batch_idx, day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets
            )
            y_smooth = smooth_labels(y, LABEL_SMOOTHING)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(day_tokens, last)
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
                day_tokens, last, y = gather_batch(
                    batch_idx, day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets
                )
                y_smooth = smooth_labels(y, LABEL_SMOOTHING)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(day_tokens, last)
                    loss = criterion(logits, y_smooth)
                val_loss_sum += loss.item() * y.shape[0]
                val_correct += ((logits > 0).float() == y).sum().item()
                val_n += y.shape[0]
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)

        if epoch == 1 or epoch % 5 == 0:
            print(f"    Epoch {epoch:3d} | TrL {train_loss:.4f} | VaL {val_loss:.4f} | VaAcc {val_acc:.4f}")

        # SWA
        if epoch >= SWA_START_EPOCH:
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
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    if swa_weights is not None and swa_count >= 2:
        model.load_state_dict(swa_weights)
        print(f"    SWA: averaged {swa_count} checkpoints")
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(
    model, test_idx: torch.Tensor,
    day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
    device, use_amp: bool,
):
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch_idx in iterate_batches(test_idx, BATCH_SIZE, shuffle=False):
            day_tokens, last, _ = gather_batch(
                batch_idx, day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets
            )
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(day_tokens, last)
            all_logits.append(logits.float().cpu())
    return torch.cat(all_logits).numpy()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 support: {torch.cuda.is_bf16_supported()}")
        # head_dim = embed_dim / n_heads = 32 / 4 = 8, which Flash Attention
        # does not support on most kernels (needs head_dim >= 16). Disable
        # flash SDP and let torch fall back to mem-efficient / math backends.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    print("\nConfig:")
    print(f"  CSV:         {CSV_PATH.name}")
    print(f"  N_DAYS:      {N_DAYS}  (lookback {LOOKBACK_BARS} bars @ 15m)")
    print(f"  LABEL_MAX:   {LABEL_MAX_BARS} bars (14 days)")
    print(f"  EMBARGO:     {EMBARGO}")
    print(f"  BATCH:       {BATCH_SIZE}")
    print(f"  LR:          {LR}  (AdamW, wd={WEIGHT_DECAY})")
    print(f"  Dropout:     {DROPOUT}, label_smoothing={LABEL_SMOOTHING}")
    print(f"  Seeds/fold:  {N_SEEDS}, SWA from epoch {SWA_START_EPOCH}")

    # ─── Load + build features ───
    feats = build_features(CSV_PATH)
    day_rows = feats["day_rows"]
    last_bar = feats["last_bar"]
    labels_np = feats["labels"]
    # Re-localize to UTC: df["date"].values strips tz info to datetime64[ns].
    dates = pd.DatetimeIndex(feats["dates"]).tz_localize("UTC")
    tp_pct_np = feats["tp_pct"]
    sl_pct_np = feats["sl_pct"]
    close_np = feats["close"]
    n = len(day_rows)

    # Sanity: day_rows columns
    assert day_rows.shape[1] == AbsVPv11.DAY_TOKEN_WIDTH, (
        f"day_rows width {day_rows.shape[1]} != expected {AbsVPv11.DAY_TOKEN_WIDTH}"
    )

    # ─── Upload to GPU ───
    print("\nUploading feature tensors to device...")
    day_rows_gpu = torch.from_numpy(day_rows).to(device)           # ~157 MB
    last_bar_gpu = torch.from_numpy(last_bar).to(device)           # ~6 MB
    labels_gpu = torch.from_numpy(labels_np).to(device)            # ~1.5 MB
    day_offsets = (torch.arange(N_DAYS, device=device) - (N_DAYS - 1)) * BARS_PER_DAY
    day_offsets = day_offsets.to(torch.long)
    print(f"  day_rows: {day_rows_gpu.shape}  {day_rows_gpu.dtype}  "
          f"({day_rows_gpu.element_size() * day_rows_gpu.numel() / 1e6:.1f} MB)")
    print(f"  last_bar: {last_bar_gpu.shape}  {last_bar_gpu.dtype}")
    print(f"  labels:   {labels_gpu.shape}  {labels_gpu.dtype}")

    # ─── Eligibility mask (has 90 days of history AND a non-NaN label) ───
    # Need i - (N_DAYS - 1) * BARS_PER_DAY >= 0 so the earliest day-offset
    # index lands on row 0. That's 89 × 96 = 8544 at 15m, not 8640.
    min_history = (N_DAYS - 1) * BARS_PER_DAY
    has_history = np.arange(n) >= min_history
    has_label = ~np.isnan(labels_np)
    eligible_mask = has_history & has_label
    eligible_idx_np = np.where(eligible_mask)[0].astype(np.int64)
    print(f"\nEligible samples: {len(eligible_idx_np):,} / {n:,}")

    # ─── Walk-forward ───
    all_results = []
    all_preds, all_labels = [], []
    all_logits, all_dates, all_close = [], [], []
    all_tp, all_sl = [], []

    # Build a param count + sample/param diagnostic before starting
    probe = AbsVPv11(n_days=N_DAYS, dropout=DROPOUT)
    n_params = count_params(probe)
    print(f"\nModel params: {n_params:,}")
    del probe

    print("\nStarting walk-forward...")
    total_start = time.time()

    for i in range(len(FOLD_BOUNDARIES) - 2):
        fold_start = time.time()
        train_end = pd.Timestamp(FOLD_BOUNDARIES[i], tz="UTC")
        val_end = pd.Timestamp(FOLD_BOUNDARIES[i + 1], tz="UTC")
        test_end = pd.Timestamp(FOLD_BOUNDARIES[i + 2], tz="UTC")

        # Embargo: shrink training/validation tails by 14 days wall-clock
        # so a sample's forward label window cannot overlap the next segment.
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

        # Class balance on this fold
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
        for seed_offset in range(N_SEEDS):
            seed = 42 + seed_offset
            print(f"    [Seed {seed}]")
            model = train_one_seed(
                seed, train_idx, val_idx,
                day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
                device, use_amp, pos_weight,
            )
            fold_logits = evaluate(
                model, test_idx,
                day_rows_gpu, last_bar_gpu, labels_gpu, day_offsets,
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

    # ─── Summary ───
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    overall_acc = float((all_preds_np == all_labels_np).mean()) if len(all_preds_np) else 0.0

    print(f"\n{'=' * 60}")
    print(f"  V11 WALK-FORWARD SUMMARY")
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

    # ─── Save results + predictions ───
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval": "v11 — absolute-range VP @ 15m × 90-day temporal",
        "csv": CSV_PATH.name,
        "n_days": N_DAYS,
        "bars_per_day": BARS_PER_DAY,
        "label_max_bars": LABEL_MAX_BARS,
        "embargo_days": 14,
        "n_params": n_params,
        "overall_acc": overall_acc,
        "total_time_min": round(total_time / 60, 1),
        "folds": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        PREDICTIONS_PATH,
        preds=np.array(all_preds),
        labels=np.array(all_labels),
        logits=np.array(all_logits),
        dates=np.array(all_dates, dtype="datetime64[ns]"),
        close=np.array(all_close),
        tp_pct=np.array(all_tp),
        sl_pct=np.array(all_sl),
    )
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
