"""Eval 15min: v6 vs v8 on 15-minute resolution data.

Tests whether 4x more training data (340k vs 84k rows) allows:
  A) v6 enriched (1+1, 24,737 params) to improve with better data/param ratio
  B) v8 enriched (2+1, 33,281 params) to finally leverage its extra capacity

Walk-forward, FGI adaptive 7.5/3 labels, 15min data (96 bars/day).

Usage:
    python -m src.models.eval_15min
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

# ═══════════════════════════════════════════════════════════════════
# Patch config for 15min BEFORE importing pipelines/architectures
# ═══════════════════════════════════════════════════════════════════
import ccxt
import src.config as cfg

cfg.TIMEFRAME = "15m"
cfg.STEP_SECONDS = ccxt.Exchange.parse_timeframe("15m")
cfg.STEP_MS = cfg.STEP_SECONDS * 1000
cfg.BARS_PER_DAY = int(round((24 * 3600) / cfg.STEP_SECONDS))  # 96
cfg.LOOKBACK_BARS = cfg.LOOKBACK_DAYS * cfg.BARS_PER_DAY        # 17280
cfg.LOOKBACK_BARS_MODEL = 30 * cfg.BARS_PER_DAY                 # 2880
cfg.HORIZON_24H_BARS = cfg.BARS_PER_DAY
cfg.VOLUME_ROLL_WINDOW_BARS = cfg.VOLUME_ROLL_WINDOW_DAYS * cfg.BARS_PER_DAY
cfg.LABEL_HORIZON_BARS = cfg.BARS_PER_DAY
cfg.LABEL_REGIME_SMA_BARS = 90 * cfg.BARS_PER_DAY
cfg.LABEL_MAX_BARS = cfg.BARS_PER_DAY * 14                      # 1344
cfg.VOLUME_COL = "volume_15m"

# Now safe to import pipeline and architecture modules
from src.features.pipelines.v1_raw import (
    build_feature_matrix_v1, FEATURE_COLS_V1, VP_STRUCTURE_COLS_V1,
    DERIVED_FEATURE_COLS_V1, feature_index_v1,
)
from src.models.architectures.v6_temporal_enriched import TemporalEnrichedV6
from src.models.architectures.v8_enriched_2plus1 import EnrichedTemporalV8

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
TP = 0.075
SL = 0.03
LR = 5e-4
NUM_WORKERS = 4
USE_COMPILE = True

# Auto-detect GPU memory and set batch size
def _auto_batch_size():
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_gb >= 40:    # A100 (80GB) or similar
            return 256
        elif mem_gb >= 14:  # T4 (16GB)
            return 64
        else:
            return 32
    return 32  # CPU/MPS fallback

BATCH_SIZE = _auto_batch_size()
BARS_PER_DAY = 96
LOOKBACK = 2880           # 30 days × 96 bars/day
N_DAYS = 30
EXPERIMENTS_DIR = cfg.EXPERIMENTS_DIR
LABEL_FGI_PATH = cfg.LABEL_FGI_PATH
LABEL_MAX_BARS = cfg.LABEL_MAX_BARS
EPOCHS = cfg.EPOCHS
EARLY_STOP_PATIENCE = cfg.EARLY_STOP_PATIENCE

FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    "2021-01-01", "2021-07-01",
    "2022-01-01", "2022-07-01",
    "2023-01-01", "2023-07-01",
    "2024-01-01", "2024-07-01",
    "2025-01-01", "2025-07-01",
]


# ═══════════════════════════════════════════════════════════════════
# Precompute labels (same as eval_2plus1.py)
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
    lr_series = pd.Series(log_returns)
    rolling_std = lr_series.rolling(window=vol_window, min_periods=vol_window).std().values
    rolling_vol[vol_window + 1:] = rolling_std[vol_window:]

    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) == 0:
        return labels
    median_vol = np.median(valid_vol)
    if median_vol < 1e-10:
        median_vol = 1e-10

    vol_scale = np.clip(rolling_vol / median_vol, 0.5, 3.0)

    tp_pct = np.full(n, np.nan)
    sl_pct = np.full(n, np.nan)
    bull_mask = is_bull == 1.0
    bear_mask = is_bull == 0.0
    tp_pct[bull_mask] = TP * vol_scale[bull_mask]
    sl_pct[bull_mask] = SL * vol_scale[bull_mask]
    tp_pct[bear_mask] = SL * vol_scale[bear_mask]
    sl_pct[bear_mask] = TP * vol_scale[bear_mask]

    warmup = vol_window + 1
    max_bars = LABEL_MAX_BARS

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
# Fast dataset
# ═══════════════════════════════════════════════════════════════════
class FastDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 regime: np.ndarray = None, lookback: int = LOOKBACK):
        self.features = features
        self.labels = labels
        self.regime = regime
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


def train_fold(model, train_loader, val_loader, device, use_amp=False):
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

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
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

        train_acc = train_correct / train_total

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
def build_v6():
    """v6 enriched (1+1) with 15min bars_per_day=96."""
    return TemporalEnrichedV6(
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
        n_spatial_layers=1,
        n_temporal_layers=1,
        fc_size=64,
        dropout=0.15,
        n_days=N_DAYS,
        bars_per_day=BARS_PER_DAY,
    )


def build_v8():
    """v8 enriched (2+1) with 15min bars_per_day=96."""
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
        n_days=N_DAYS,
        bars_per_day=BARS_PER_DAY,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# Walk-forward runner
# ═══════════════════════════════════════════════════════════════════
def run_walk_forward(model_name, model_builder, features_tensor, labels_tensor,
                     dates, regime_array, device, use_amp):
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

        train_mask = dates < train_end
        val_mask = (dates >= train_end) & (dates < val_end)
        test_mask = (dates >= val_end) & (dates < test_end)

        if train_mask.sum() < 1000 or val_mask.sum() < 100 or test_mask.sum() < 100:
            continue

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

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

        if USE_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                if i == 0:
                    print("    [torch.compile enabled]")
            except Exception:
                pass

        model = train_fold(model, train_loader, val_loader, device, use_amp=use_amp)
        preds, labels = evaluate_fold(model, test_loader, device, use_amp=use_amp)

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

    bull = regimes == 1.0
    bear = regimes == 0.0
    bull_stats = confusion(preds[bull], labels[bull]) if bull.sum() > 0 else None
    bear_stats = confusion(preds[bear], labels[bear]) if bear.sum() > 0 else None

    regime_ev = {}
    if bull_stats:
        regime_ev["bull_long"] = round(bull_stats["precision"] * 7.5 - (1 - bull_stats["precision"]) * 3.0, 2)
        regime_ev["bull_short"] = round(bull_stats["npv"] * 3.0 - (1 - bull_stats["npv"]) * 7.5, 2)
    if bear_stats:
        regime_ev["bear_long"] = round(bear_stats["precision"] * 3.0 - (1 - bear_stats["precision"]) * 7.5, 2)
        regime_ev["bear_short"] = round(bear_stats["npv"] * 7.5 - (1 - bear_stats["npv"]) * 3.0, 2)

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
    print(f"Resolution: 15min (bars_per_day={BARS_PER_DAY}, lookback={LOOKBACK})")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mixed precision: {use_amp}")

    # ─── Load data ───
    t0 = time.time()
    print("\nLoading v1 features (15min)...")
    csv_path = cfg.PROJECT_ROOT / "BTC_15m_RELVP.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 'python -m src.data scrape 15m' first.")
        sys.exit(1)
    df = build_feature_matrix_v1(csv_path=csv_path)
    print(f"  {len(df)} rows, {len(FEATURE_COLS_V1)} features")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ─── Precompute labels ───
    t0 = time.time()
    print("\nPrecomputing labels (FGI adaptive, TP=7.5%, SL=3%)...")
    labels = precompute_labels(df["close"].values, df["date"].values)
    n_valid = (~np.isnan(labels)).sum()
    n_pos = (labels == 1.0).sum()
    print(f"  {n_valid} valid labels ({n_pos} positive = {n_pos/max(n_valid,1)*100:.1f}%)")
    print(f"  Labels computed in {time.time() - t0:.1f}s")

    # ─── Precompute regime ───
    print("Building regime array...")
    regime = build_regime_array(df["date"].values)

    # ─── Convert to tensors ───
    features = torch.from_numpy(df[FEATURE_COLS_V1].values.astype(np.float32))
    labels_t = torch.from_numpy(labels)
    dates = pd.to_datetime(df["date"]).values

    # ─── Data/param ratio check ───
    fold1_train = (dates < pd.Timestamp("2020-01-01")).sum()
    approx_samples = max(fold1_train - LOOKBACK, 0)
    print(f"\n  Fold 1 train samples: ~{approx_samples:,}")
    print(f"  v6 samples/param: {approx_samples/24737:.1f}:1")
    print(f"  v8 samples/param: {approx_samples/33281:.1f}:1")

    # ─── Run v6 (1+1) ───
    t0 = time.time()
    results_v6 = run_walk_forward(
        "v6 enriched (1+1) on 15min",
        build_v6, features, labels_t, dates, regime, device, use_amp,
    )
    results_v6["total_time_min"] = round((time.time() - t0) / 60, 1)
    print(f"\n  Total time: {results_v6['total_time_min']} min")

    # ─── Run v8 (2+1) ───
    t0 = time.time()
    results_v8 = run_walk_forward(
        "v8 enriched (2+1) on 15min",
        build_v8, features, labels_t, dates, regime, device, use_amp,
    )
    results_v8["total_time_min"] = round((time.time() - t0) / 60, 1)
    print(f"\n  Total time: {results_v8['total_time_min']} min")

    # ─── Comparison ───
    print(f"\n{'=' * 70}")
    print(f"  15min vs 1h COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Model':<35} {'Acc':>6} {'Prec':>6} {'Bull long':>10} {'Bear long':>10} {'Bear short':>11}")
    print(f"  {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 11}")
    # 1h baselines
    print(f"  {'v6 1h (baseline)':<35} {'58.4%':>6} {'68.3%':>6} {'+1.57%':>10} {'+1.02%':>10} {'-0.04%':>11}")
    print(f"  {'v8 1h (baseline)':<35} {'58.0%':>6} {'60.4%':>6} {'+1.08%':>10} {'+1.07%':>10} {'+0.62%':>11}")
    for r in [results_v6, results_v8]:
        o = r["overall"]
        ev = o.get("regime_ev", {})
        name = r["model"][:35]
        bl = f"{ev.get('bull_long', 0):+.2f}%"
        brl = f"{ev.get('bear_long', 0):+.2f}%"
        brs = f"{ev.get('bear_short', 0):+.2f}%"
        print(f"  {name:<35} {o['accuracy']*100:5.1f}% {o['precision']*100:5.1f}% {bl:>10} {brl:>10} {brs:>11}")

    # Per-fold
    print(f"\n  Per-fold accuracy:")
    print(f"  {'Fold':<6} {'Period':<30} {'v6 15m':>8} {'v8 15m':>8}")
    print(f"  {'-' * 6} {'-' * 30} {'-' * 8} {'-' * 8}")
    for fi in range(min(len(results_v6["folds"]), len(results_v8["folds"]))):
        f6 = results_v6["folds"][fi]
        f8 = results_v8["folds"][fi]
        print(f"  {f6['fold']:<6} {f6['period']:<30} {f6['acc']*100:7.1f}% {f8['acc']*100:7.1f}%")

    # ─── Save ───
    output = {
        "config": {
            "timeframe": "15m", "tp": TP, "sl": SL, "batch_size": BATCH_SIZE,
            "lr": LR, "epochs": EPOCHS, "patience": EARLY_STOP_PATIENCE,
            "lookback": LOOKBACK, "bars_per_day": BARS_PER_DAY,
            "mixed_precision": use_amp, "device": str(device),
        },
        "v6": results_v6,
        "v8": results_v8,
    }
    out_path = EXPERIMENTS_DIR / "eval_15min_results.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
