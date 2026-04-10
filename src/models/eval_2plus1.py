"""Eval 2+1: Spatial-Temporal Transformer comparison.

Compares against Eval 4 (spatial-only, 63.3%):
  A) TemporalTransformerClassifier(n_spatial=2, n_temporal=1) — simple 2+1
  B) TemporalEnrichedV6(n_spatial=2, n_temporal=1) — enriched 2+1

Walk-forward, FGI adaptive 7.5/3 labels, 1h data.

Training optimizations (vs eval_strategy.py):
  - Labels precomputed once on full df (not per-fold)
  - Batch size 512
  - Mixed precision (bf16 on CUDA, fp32 otherwise)
  - torch.compile when available
  - pin_memory + num_workers=4
  - Vectorized label + index construction

Usage:
    python -m src.models.eval_2plus1
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
    LABEL_MAX_BARS, EPOCHS, EARLY_STOP_PATIENCE, REL_BIN_COUNT,
)
from src.features.pipelines.v1_raw import (
    build_feature_matrix_v1, FEATURE_COLS_V1, VP_STRUCTURE_COLS_V1,
    DERIVED_FEATURE_COLS_V1, feature_index_v1,
)
from src.features.pipelines.v2_scaled import (
    build_feature_matrix_v2, FEATURE_COLS_V2, DERIVED_FEATURE_COLS_V2,
)
from src.models.architectures.v7_simple_2plus1 import SimpleTemporalV7
from src.models.architectures.v8_enriched_2plus1 import EnrichedTemporalV8

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
TP = 0.075
SL = 0.03
BATCH_SIZE = 512
LR = 5e-4
NUM_WORKERS = 4
USE_COMPILE = True

FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
]


# ═══════════════════════════════════════════════════════════════════
# Precompute labels (once, on full df)
# ═══════════════════════════════════════════════════════════════════
def build_regime_array(dates: np.ndarray) -> np.ndarray:
    """Build FGI-based regime array: 1.0=bull, 0.0=bear, NaN=unknown."""
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


def precompute_labels(close: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Vectorized first-hit labels with FGI regime adaptation."""
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)

    is_bull = build_regime_array(dates)

    # Rolling volatility
    vol_window = 30
    log_returns = np.diff(np.log(close))
    rolling_vol = np.full(n, np.nan)
    # Vectorized rolling std
    lr_series = pd.Series(log_returns)
    rolling_std = lr_series.rolling(window=vol_window, min_periods=vol_window).std().values
    rolling_vol[vol_window + 1:] = rolling_std[vol_window:]

    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) == 0:
        return labels
    median_vol = np.median(valid_vol)
    if median_vol < 1e-10:
        median_vol = 1e-10

    # Compute vol_scale for all bars at once
    vol_scale = np.clip(rolling_vol / median_vol, 0.5, 3.0)

    # Compute TP/SL percentages per bar
    tp_pct = np.full(n, np.nan)
    sl_pct = np.full(n, np.nan)
    bull_mask = is_bull == 1.0
    bear_mask = is_bull == 0.0
    tp_pct[bull_mask] = TP * vol_scale[bull_mask]
    sl_pct[bull_mask] = SL * vol_scale[bull_mask]
    tp_pct[bear_mask] = SL * vol_scale[bear_mask]  # flipped in bear
    sl_pct[bear_mask] = TP * vol_scale[bear_mask]

    warmup = vol_window + 1
    max_bars = LABEL_MAX_BARS

    # Main loop — inner search is vectorized with numpy
    for i in range(warmup, n - 1):
        if np.isnan(tp_pct[i]):
            continue

        entry = close[i]
        tp_level = entry * (1 + tp_pct[i])
        sl_level = entry * (1 - sl_pct[i])
        end = min(i + max_bars, n)
        future = close[i + 1:end]

        if len(future) == 0:
            continue

        tp_hits = future >= tp_level
        sl_hits = future <= sl_level

        tp_first = np.argmax(tp_hits) if tp_hits.any() else len(future)
        sl_first = np.argmax(sl_hits) if sl_hits.any() else len(future)

        if tp_first < len(future) or sl_first < len(future):
            if tp_first <= sl_first:
                labels[i] = 1.0
            else:
                labels[i] = 0.0

    return labels


# ═══════════════════════════════════════════════════════════════════
# Fast dataset (precomputed labels, vectorized indices)
# ═══════════════════════════════════════════════════════════════════
class FastDataset(Dataset):
    """TimeSeriesDataset with precomputed labels — no per-init computation."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 regime: np.ndarray = None, lookback: int = LOOKBACK_BARS_MODEL):
        self.features = features
        self.labels = labels
        self.regime = regime
        self.lookback = lookback

        # Vectorized valid index construction
        max_start = len(features) - lookback
        if max_start <= 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            candidates = np.arange(max_start)
            label_indices = candidates + lookback - 1
            valid_mask = (label_indices < len(labels)) & ~torch.isnan(labels[label_indices]).numpy()
            self.valid_indices = candidates[valid_mask]

    def get_regime(self, idx):
        """Get regime for sample idx (used after prediction, not in training)."""
        if self.regime is None:
            return np.nan
        real_idx = self.valid_indices[idx]
        label_idx = real_idx + self.lookback - 1
        if label_idx < len(self.regime):
            return self.regime[label_idx]
        return np.nan

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.features[real_idx:real_idx + self.lookback]
        y = self.labels[real_idx + self.lookback - 1]
        return x, y


# ═══════════════════════════════════════════════════════════════════
# Optimized training loop
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
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and NUM_WORKERS > 0,
    )


def train_fold(model, train_loader, val_loader, device, use_amp=False):
    """Train one fold with optional mixed precision."""
    # Compute class weight
    all_labels = []
    for _, y in train_loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels)
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # AMP setup
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(y)
            preds = (logits.detach() > 0).float()
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss_sum += loss.item() * len(y)
                preds = (logits > 0).float()
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Train {train_acc:.4f} | Val {val_acc:.4f} | Val Loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model


def evaluate_fold(model, test_loader, device, use_amp=False):
    """Evaluate model on test set, return predictions and labels."""
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x)
            preds = (logits.cpu() > 0).float()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    return np.array(all_preds), np.array(all_labels)


# ═══════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════
def build_model_a():
    """Simple 2+1: SimpleTemporalV7 (frozen)."""
    return SimpleTemporalV7()


def build_model_b():
    """Enriched 2+1: EnrichedTemporalV8 (frozen)."""
    return EnrichedTemporalV8(
        ohlc_open_idx=feature_index_v1("ohlc_open_ratio"),
        ohlc_high_idx=feature_index_v1("ohlc_high_ratio"),
        ohlc_low_idx=feature_index_v1("ohlc_low_ratio"),
        log_return_idx=feature_index_v1("log_return"),
        volume_ratio_idx=feature_index_v1("volume_ratio"),
        vp_structure_start_idx=feature_index_v1("vp_ceiling_dist"),
        n_vp_structure=len(VP_STRUCTURE_COLS_V1),
        n_other_features=len(DERIVED_FEATURE_COLS_V1) + len(VP_STRUCTURE_COLS_V1),
        embed_dim=32,
        n_heads=4,
        n_spatial_layers=2,
        n_temporal_layers=1,
        fc_size=64,
        dropout=0.15,
        n_days=30,
        bars_per_day=BARS_PER_DAY,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# Walk-forward runner
# ═══════════════════════════════════════════════════════════════════
def run_walk_forward(model_name, model_builder, features_tensor, labels_tensor,
                     dates, regime_array, device, use_amp):
    """Run full walk-forward for one model config."""
    print(f"\n{'=' * 70}")
    print(f"  {model_name}")
    print(f"{'=' * 70}")

    sample_model = model_builder()
    n_params = count_params(sample_model)
    print(f"  Params: {n_params:,}")
    del sample_model

    all_preds, all_labels, all_regimes = [], [], []
    fold_results = []

    for i in range(len(FOLD_BOUNDARIES) - 2):
        fold_start = time.time()
        train_end = pd.Timestamp(FOLD_BOUNDARIES[i])
        val_end = pd.Timestamp(FOLD_BOUNDARIES[i + 1])
        test_end = pd.Timestamp(FOLD_BOUNDARIES[i + 2])

        # Date masks
        train_mask = dates < train_end
        val_mask = (dates >= train_end) & (dates < val_end)
        test_mask = (dates >= val_end) & (dates < test_end)

        if train_mask.sum() < 1000 or val_mask.sum() < 100 or test_mask.sum() < 100:
            continue

        # Build datasets from precomputed tensors
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Slice contiguous ranges
        train_ds = FastDataset(
            features_tensor[train_idx[0]:train_idx[-1] + 1],
            labels_tensor[train_idx[0]:train_idx[-1] + 1],
            regime_array[train_idx[0]:train_idx[-1] + 1],
        )
        val_ds = FastDataset(
            features_tensor[val_idx[0]:val_idx[-1] + 1],
            labels_tensor[val_idx[0]:val_idx[-1] + 1],
            regime_array[val_idx[0]:val_idx[-1] + 1],
        )
        test_ds = FastDataset(
            features_tensor[test_idx[0]:test_idx[-1] + 1],
            labels_tensor[test_idx[0]:test_idx[-1] + 1],
            regime_array[test_idx[0]:test_idx[-1] + 1],
        )

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        print(f"\n  Fold {i + 1} ({FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]})")
        print(f"    Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

        model = model_builder()

        # torch.compile (PyTorch 2.0+)
        if USE_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                if i == 0:
                    print("    [torch.compile enabled]")
            except Exception:
                pass

        model = train_fold(model, train_loader, val_loader, device, use_amp=use_amp)
        preds, labels = evaluate_fold(model, test_loader, device, use_amp=use_amp)

        # Collect regime for each test sample
        fold_regimes = np.array([test_ds.get_regime(j) for j in range(len(test_ds))])

        fold_acc = (preds == labels).mean()
        fold_time = time.time() - fold_start

        fold_results.append({
            "fold": i + 1,
            "period": f"{FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]}",
            "acc": float(fold_acc),
            "n_test": len(preds),
            "time_sec": round(fold_time, 1),
        })
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_regimes.extend(fold_regimes.tolist())

        print(f"    Acc: {fold_acc:.4f} ({len(preds)} samples, {fold_time:.0f}s)")

        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Overall stats
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    regimes = np.array(all_regimes)

    def confusion(p, l):
        tp = int(((p == 1) & (l == 1)).sum())
        fp = int(((p == 1) & (l == 0)).sum())
        tn = int(((p == 0) & (l == 0)).sum())
        fn = int(((p == 0) & (l == 1)).sum())
        prec = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        acc = (tp + tn) / max(len(p), 1)
        pred1_rate = float((p == 1).mean())
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": round(prec, 4), "npv": round(npv, 4),
                "accuracy": round(acc, 4), "pred1_rate": round(pred1_rate, 4),
                "n": len(p)}

    all_stats = confusion(preds, labels)
    acc = all_stats["accuracy"]
    prec = all_stats["precision"]
    recall = all_stats["tp"] / max(all_stats["tp"] + all_stats["fn"], 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-8)

    # Per-regime stats
    bull = regimes == 1.0
    bear = regimes == 0.0
    bull_stats = confusion(preds[bull], labels[bull]) if bull.sum() > 0 else None
    bear_stats = confusion(preds[bear], labels[bear]) if bear.sum() > 0 else None

    # EV calculation
    # Bull regime: TP=7.5% SL=3% (no flip). Long wins 7.5%, loses 3%. Short wins 3%, loses 7.5%.
    # Bear regime: TP=3% SL=7.5% (flipped). Long wins 3%, loses 7.5%. Short wins 7.5%, loses 3%.
    regime_ev = {}
    if bull_stats:
        ev_bull_long = bull_stats["precision"] * 7.5 - (1 - bull_stats["precision"]) * 3.0
        ev_bull_short = bull_stats["npv"] * 3.0 - (1 - bull_stats["npv"]) * 7.5
        regime_ev["bull_long"] = round(ev_bull_long, 2)
        regime_ev["bull_short"] = round(ev_bull_short, 2)
    if bear_stats:
        ev_bear_long = bear_stats["precision"] * 3.0 - (1 - bear_stats["precision"]) * 7.5
        ev_bear_short = bear_stats["npv"] * 7.5 - (1 - bear_stats["npv"]) * 3.0
        regime_ev["bear_long"] = round(ev_bear_long, 2)
        regime_ev["bear_short"] = round(ev_bear_short, 2)

    overall = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": all_stats["tp"], "fp": all_stats["fp"],
        "tn": all_stats["tn"], "fn": all_stats["fn"],
        "n_samples": len(preds),
        "n_params": count_params(model_builder()),
        "bull_stats": bull_stats,
        "bear_stats": bear_stats,
        "regime_ev": regime_ev,
    }

    print(f"\n  OVERALL: Acc={acc:.4f}  Prec={prec:.4f}  F1={f1:.4f}")
    print(f"  TP={all_stats['tp']} FP={all_stats['fp']} TN={all_stats['tn']} FN={all_stats['fn']}")

    if bull_stats:
        print(f"\n  BULL ({bull_stats['n']} bars):")
        print(f"    Acc={bull_stats['accuracy']:.4f}  Prec={bull_stats['precision']:.4f}  NPV={bull_stats['npv']:.4f}  Pred=1 rate={bull_stats['pred1_rate']:.4f}")
        print(f"    Long EV:  {regime_ev['bull_long']:+.2f}%   Short EV: {regime_ev['bull_short']:+.2f}%")

    if bear_stats:
        print(f"\n  BEAR ({bear_stats['n']} bars):")
        print(f"    Acc={bear_stats['accuracy']:.4f}  Prec={bear_stats['precision']:.4f}  NPV={bear_stats['npv']:.4f}  Pred=1 rate={bear_stats['pred1_rate']:.4f}")
        print(f"    Long EV:  {regime_ev['bear_long']:+.2f}%   Short EV: {regime_ev['bear_short']:+.2f}%")

    # Strategy EVs (weighted by regime fraction)
    if bull_stats and bear_stats:
        bull_frac = bull.sum() / max(bull.sum() + bear.sum(), 1)
        bear_frac = 1 - bull_frac
        strategies = {
            "S1: Long only (both regimes)": bull_frac * regime_ev["bull_long"] + bear_frac * regime_ev["bear_long"],
            "S2: Long bull, skip bear": bull_frac * regime_ev["bull_long"],
            "S3: Both sides both regimes": bull_frac * (regime_ev["bull_long"] + regime_ev["bull_short"]) / 2 + bear_frac * (regime_ev["bear_long"] + regime_ev["bear_short"]) / 2,
            "S4: Long bull + both bear": bull_frac * regime_ev["bull_long"] + bear_frac * (regime_ev["bear_long"] + regime_ev["bear_short"]) / 2,
        }
        print(f"\n  STRATEGY EVs (bull={bull_frac:.1%}, bear={bear_frac:.1%}):")
        for name, ev in strategies.items():
            print(f"    {name}: {ev:+.2f}%")
        overall["strategies"] = {k: round(v, 2) for k, v in strategies.items()}

    return {"model": model_name, "overall": overall, "folds": fold_results}


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
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mixed precision: {use_amp}")
    print(f"torch.compile: {USE_COMPILE and hasattr(torch, 'compile')}")

    # ─── Load data ───
    t0 = time.time()
    print("\nLoading v2 features (for Model A — simple 2+1)...")
    df_v2 = build_feature_matrix_v2()
    print(f"  {len(df_v2)} rows, {len(FEATURE_COLS_V2)} features")

    print("Loading v1 features (for Model B — enriched 2+1)...")
    df_v1 = build_feature_matrix_v1()
    print(f"  {len(df_v1)} rows, {len(FEATURE_COLS_V1)} features")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ─── Precompute labels (once, on v2 df — same close prices) ───
    t0 = time.time()
    print("\nPrecomputing labels (FGI adaptive, TP=7.5%, SL=3%)...")
    labels_v2 = precompute_labels(df_v2["close"].values, df_v2["date"].values)
    n_valid = (~np.isnan(labels_v2)).sum()
    n_pos = (labels_v2 == 1.0).sum()
    print(f"  {n_valid} valid labels ({n_pos} positive = {n_pos/max(n_valid,1)*100:.1f}%)")

    # Labels for v1 — same close prices, different row count due to VP structure warmup
    labels_v1 = precompute_labels(df_v1["close"].values, df_v1["date"].values)
    n_valid_v1 = (~np.isnan(labels_v1)).sum()
    print(f"  v1: {n_valid_v1} valid labels")
    print(f"  Labels computed in {time.time() - t0:.1f}s")

    # ─── Precompute regime arrays ───
    print("Building regime arrays...")
    regime_v2 = build_regime_array(df_v2["date"].values)
    regime_v1 = build_regime_array(df_v1["date"].values)

    # ─── Convert to tensors ───
    features_v2 = torch.from_numpy(df_v2[FEATURE_COLS_V2].values.astype(np.float32))
    labels_v2_t = torch.from_numpy(labels_v2)
    dates_v2 = pd.to_datetime(df_v2["date"]).values

    features_v1 = torch.from_numpy(df_v1[FEATURE_COLS_V1].values.astype(np.float32))
    labels_v1_t = torch.from_numpy(labels_v1)
    dates_v1 = pd.to_datetime(df_v1["date"]).values

    # ─── Run Model A: Simple 2+1 ───
    t0 = time.time()
    results_a = run_walk_forward(
        "Model A: Simple 2+1 (v7, n_spatial=2, n_temporal=1)",
        build_model_a, features_v2, labels_v2_t, dates_v2, regime_v2, device, use_amp,
    )
    results_a["total_time_min"] = round((time.time() - t0) / 60, 1)
    print(f"\n  Total time: {results_a['total_time_min']} min")

    # ─── Run Model B: Enriched 2+1 ───
    t0 = time.time()
    results_b = run_walk_forward(
        "Model B: Enriched 2+1 (v8, n_spatial=2, n_temporal=1)",
        build_model_b, features_v1, labels_v1_t, dates_v1, regime_v1, device, use_amp,
    )
    results_b["total_time_min"] = round((time.time() - t0) / 60, 1)
    print(f"\n  Total time: {results_b['total_time_min']} min")

    # ─── Comparison table ───
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON (vs Eval 4 baseline: 63.3% acc, 65.8% prec)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<50} {'Acc':>6} {'Prec':>6} {'F1':>6} {'Params':>8}")
    print(f"  {'-' * 50} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 8}")
    print(f"  Eval 4 (spatial-only, 2 layers)                   63.30% 65.80%  0.702   21,889")
    for r in [results_a, results_b]:
        o = r["overall"]
        name = r["model"][:50]
        print(f"  {name:<50} {o['accuracy']*100:5.2f}% {o['precision']*100:5.2f}% {o['f1']:5.3f} {o['n_params']:>8,}")

    # Per-fold comparison
    print(f"\n  Per-fold accuracy:")
    print(f"  {'Fold':<6} {'Period':<30} {'Eval 4':>8} {'Model A':>8} {'Model B':>8}")
    print(f"  {'-' * 6} {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")
    eval4_folds = [88.4, 57.5, 44.4, 69.0, 66.4, 58.4, 81.0, 54.8, 72.9, 42.0]
    for fi in range(min(len(results_a["folds"]), len(results_b["folds"]))):
        fa = results_a["folds"][fi]
        fb = results_b["folds"][fi]
        e4 = eval4_folds[fi] if fi < len(eval4_folds) else 0
        print(f"  {fa['fold']:<6} {fa['period']:<30} {e4:7.1f}% {fa['acc']*100:7.1f}% {fb['acc']*100:7.1f}%")

    # ─── Save results ───
    output = {
        "config": {
            "tp": TP, "sl": SL, "batch_size": BATCH_SIZE, "lr": LR,
            "epochs": EPOCHS, "patience": EARLY_STOP_PATIENCE,
            "lookback": LOOKBACK_BARS_MODEL, "bars_per_day": BARS_PER_DAY,
            "mixed_precision": use_amp,
            "device": str(device),
        },
        "model_a": results_a,
        "model_b": results_b,
    }
    out_path = EXPERIMENTS_DIR / "eval_2plus1_results.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
