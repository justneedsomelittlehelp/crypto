"""Fine-tune v6 backbone with funding rate features.

Approach:
  1. Train v6 backbone fresh per fold (VP-only, same as before)
  2. Freeze all backbone weights
  3. Replace FC head with wider one: (32 + 18 + 3) → 64 → 1
  4. Train only FC head with funding features added

The backbone learns VP barrier prediction from full history.
The FC head learns how funding rate modifies the VP signal.

Usage:
    python -m src.models.eval_finetune_funding
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
    LABEL_MAX_BARS, EPOCHS, EARLY_STOP_PATIENCE,
)
from src.features.pipelines.v1_raw import (
    build_feature_matrix_v1, FEATURE_COLS_V1, VP_STRUCTURE_COLS_V1,
    DERIVED_FEATURE_COLS_V1, feature_index_v1,
)
from src.models.architectures.v6_temporal_enriched import TemporalEnrichedV6
from src.data.funding_rate import load_funding_rate

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
TP = 0.075
SL = 0.03
BATCH_SIZE = 512
LR_BACKBONE = 5e-4        # For stage 1 (backbone training)
LR_FINETUNE = 1e-3        # For stage 2 (FC head only — higher LR, fewer params)
NUM_WORKERS = 4
USE_COMPILE = True
N_FUNDING_FEATURES = 3    # funding_rate, funding_zscore, funding_trend
FINETUNE_EPOCHS = 30      # Fewer epochs needed for tiny FC head
FINETUNE_PATIENCE = 10

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
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)
    is_bull = build_regime_array(dates)

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
    tp_pct[is_bull == 1.0] = TP * vol_scale[is_bull == 1.0]
    sl_pct[is_bull == 1.0] = SL * vol_scale[is_bull == 1.0]
    tp_pct[is_bull == 0.0] = SL * vol_scale[is_bull == 0.0]
    sl_pct[is_bull == 0.0] = TP * vol_scale[is_bull == 0.0]

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
            labels[i] = 1.0 if tp_first <= sl_first else 0.0

    return labels


# ═══════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════
class FastDataset(Dataset):
    """VP features only (for backbone training)."""
    def __init__(self, features, labels, regime=None, lookback=LOOKBACK_BARS_MODEL):
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
        return self.regime[real_idx + self.lookback - 1] if (real_idx + self.lookback - 1) < len(self.regime) else np.nan

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.features[real_idx:real_idx + self.lookback]
        y = self.labels[real_idx + self.lookback - 1]
        return x, y


class FundingDataset(Dataset):
    """VP features + funding features (for fine-tuning)."""
    def __init__(self, features, funding, labels, regime=None, lookback=LOOKBACK_BARS_MODEL):
        self.features = features
        self.funding = funding    # (n, 3) tensor
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
        return self.regime[real_idx + self.lookback - 1] if (real_idx + self.lookback - 1) < len(self.regime) else np.nan

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.features[real_idx:real_idx + self.lookback]
        f = self.funding[real_idx + self.lookback - 1]  # funding at prediction time
        y = self.labels[real_idx + self.lookback - 1]
        return x, f, y


# ═══════════════════════════════════════════════════════════════════
# Training helpers
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


def build_v6():
    return TemporalEnrichedV6(
        ohlc_open_idx=feature_index_v1("ohlc_open_ratio"),
        ohlc_high_idx=feature_index_v1("ohlc_high_ratio"),
        ohlc_low_idx=feature_index_v1("ohlc_low_ratio"),
        log_return_idx=feature_index_v1("log_return"),
        volume_ratio_idx=feature_index_v1("volume_ratio"),
        vp_structure_start_idx=feature_index_v1("vp_ceiling_dist"),
        n_vp_structure=len(VP_STRUCTURE_COLS_V1),
        n_other_features=len(DERIVED_FEATURE_COLS_V1) + len(VP_STRUCTURE_COLS_V1),
    )


def count_params(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Train backbone (VP-only, same as before)
# ═══════════════════════════════════════════════════════════════════
def train_backbone(model, train_loader, val_loader, device, use_amp=False):
    """Train full v6 model on VP features. Returns trained model."""
    all_labels = torch.cat([y for _, y in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BACKBONE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
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

        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss_sum += loss.item() * len(y)
                val_total += len(y)
        val_loss = val_loss_sum / max(val_total, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Fine-tune FC head with funding features
# ═══════════════════════════════════════════════════════════════════
class FinetunedV6(nn.Module):
    """v6 backbone (frozen) + funding features → new FC head."""

    def __init__(self, backbone: TemporalEnrichedV6, n_funding: int = N_FUNDING_FEATURES):
        super().__init__()
        self.backbone = backbone
        self.n_funding = n_funding

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # New FC head: embed_dim + n_other_features + n_funding → fc_size → 1
        embed_dim = backbone.embed_dim
        n_other = backbone.n_other_features
        fc_input = embed_dim + n_other + n_funding

        self.fc1 = nn.Linear(fc_input, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, funding: torch.Tensor) -> torch.Tensor:
        # Run frozen backbone up to the concat point
        # We need to replicate v6's forward up to pooled + other, then add funding
        backbone = self.backbone
        assert x.shape[1] == backbone.n_days * backbone.bars_per_day

        batch_size = x.shape[0]
        vp_bins = x[:, :, :backbone.N_VP_BINS]
        other = x[:, -1, backbone.N_VP_BINS:]

        # Spatial attention (frozen)
        vp_daily = vp_bins[:, backbone.bars_per_day - 1::backbone.bars_per_day, :]
        flat_vp = vp_daily.reshape(batch_size * backbone.n_days, backbone.N_VP_BINS, 1)
        flat_tokens = backbone.bin_embed(flat_vp)
        cls_spatial = backbone.spatial_cls.expand(batch_size * backbone.n_days, -1, -1)
        flat_tokens = torch.cat([cls_spatial, flat_tokens], dim=1)
        flat_tokens = flat_tokens + backbone.spatial_pos
        flat_attended = backbone.spatial_transformer(flat_tokens)
        vp_spatial_out = flat_attended[:, 0, :].view(batch_size, backbone.n_days, backbone.embed_dim)

        # Enrich day tokens (frozen)
        daily_candles = backbone._aggregate_daily_candles(x)
        daily_vp_struct = backbone._sample_daily_vp_structure(x)
        enriched = torch.cat([vp_spatial_out, daily_candles, daily_vp_struct], dim=2)
        day_tokens = backbone.day_projection(enriched)

        # Temporal attention (frozen)
        cls_temporal = backbone.temporal_cls.expand(batch_size, -1, -1)
        temporal_input = torch.cat([cls_temporal, day_tokens], dim=1)
        temporal_input = temporal_input + backbone.temporal_pos
        temporal_out = backbone.temporal_transformer(temporal_input)
        pooled = temporal_out[:, 0, :]  # CLS token (batch, 32)

        # NEW: concat VP embedding + other features + funding features
        combined = torch.cat([pooled, other, funding], dim=1)

        # New FC head (trainable)
        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit


def finetune_fc(model, train_loader, val_loader, device, use_amp=False):
    """Train only the FC head layers."""
    all_labels = torch.cat([y for _, _, y in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    model = model.to(device)
    # Only optimize FC head params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR_FINETUNE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for x, f, y in train_loader:
            x = x.to(device, non_blocking=True)
            f = f.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x, f)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds = (logits.detach() > 0).float()
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, f, y in val_loader:
                x = x.to(device, non_blocking=True)
                f = f.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(x, f)
                    loss = criterion(logits, y)
                val_loss_sum += loss.item() * len(y)
                preds = (logits > 0).float()
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"      FT Epoch {epoch:3d} | Train {train_acc:.4f} | Val {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FINETUNE_PATIENCE:
                print(f"      FT early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model


# ═══════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════
def main():
    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load data
    t0 = time.time()
    print("\nLoading v1 features...")
    df = build_feature_matrix_v1()
    print(f"  {len(df)} rows, {len(FEATURE_COLS_V1)} features")

    print("Loading funding rate...")
    fr = load_funding_rate(ohlcv_dates=df["date"])
    print(f"  {len(fr)} rows, funding coverage: {(fr['funding_rate'] != 0).sum()}/{len(fr)}")

    print("Precomputing labels...")
    labels = precompute_labels(df["close"].values, df["date"].values)
    n_valid = (~np.isnan(labels)).sum()
    print(f"  {n_valid} valid labels")

    regime = build_regime_array(df["date"].values)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Convert to tensors
    features = torch.from_numpy(df[FEATURE_COLS_V1].values.astype(np.float32))
    funding = torch.from_numpy(
        fr[["funding_rate", "funding_zscore", "funding_trend"]].values.astype(np.float32)
    )
    labels_t = torch.from_numpy(labels)
    dates = pd.to_datetime(df["date"]).values

    # Run walk-forward: baseline (v6 only) + finetuned (v6 + funding)
    all_baseline_preds, all_baseline_labels = [], []
    all_ft_preds, all_ft_labels, all_ft_regimes = [], [], []
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

        print(f"\n  Fold {i + 1} ({FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]})")

        # ── Stage 1: Train backbone ──
        print("    Stage 1: Training backbone (VP-only)...")
        train_ds = FastDataset(features[train_idx[0]:train_idx[-1]+1],
                               labels_t[train_idx[0]:train_idx[-1]+1])
        val_ds = FastDataset(features[val_idx[0]:val_idx[-1]+1],
                             labels_t[val_idx[0]:val_idx[-1]+1])
        test_ds = FastDataset(features[test_idx[0]:test_idx[-1]+1],
                              labels_t[test_idx[0]:test_idx[-1]+1],
                              regime[test_idx[0]:test_idx[-1]+1])

        if len(train_ds) < 100 or len(val_ds) < 50 or len(test_ds) < 50:
            continue

        train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_ds, BATCH_SIZE)
        test_loader = make_loader(test_ds, BATCH_SIZE)

        print(f"    Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

        backbone = build_v6()
        if USE_COMPILE and hasattr(torch, "compile"):
            try:
                backbone = torch.compile(backbone)
            except Exception:
                pass
        backbone = train_backbone(backbone, train_loader, val_loader, device, use_amp)

        # Baseline eval (VP-only)
        backbone.eval()
        base_preds, base_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                logits = backbone(x)
                preds = (logits.cpu() > 0).float()
                base_preds.extend(preds.tolist())
                base_labels.extend(y.tolist())
        base_acc = np.mean(np.array(base_preds) == np.array(base_labels))
        all_baseline_preds.extend(base_preds)
        all_baseline_labels.extend(base_labels)

        # ── Stage 2: Fine-tune with funding ──
        print(f"    Stage 1 acc: {base_acc:.4f}. Stage 2: Fine-tuning with funding...")

        # Unwrap compiled model if needed
        raw_backbone = backbone._orig_mod if hasattr(backbone, "_orig_mod") else backbone

        ft_model = FinetunedV6(raw_backbone)
        ft_trainable = count_params(ft_model, only_trainable=True)
        ft_total = count_params(ft_model, only_trainable=False)
        if i == 0:
            print(f"    Trainable: {ft_trainable:,} / {ft_total:,} total params")

        # Build funding datasets
        ft_train_ds = FundingDataset(
            features[train_idx[0]:train_idx[-1]+1],
            funding[train_idx[0]:train_idx[-1]+1],
            labels_t[train_idx[0]:train_idx[-1]+1],
        )
        ft_val_ds = FundingDataset(
            features[val_idx[0]:val_idx[-1]+1],
            funding[val_idx[0]:val_idx[-1]+1],
            labels_t[val_idx[0]:val_idx[-1]+1],
        )
        ft_test_ds = FundingDataset(
            features[test_idx[0]:test_idx[-1]+1],
            funding[test_idx[0]:test_idx[-1]+1],
            labels_t[test_idx[0]:test_idx[-1]+1],
            regime[test_idx[0]:test_idx[-1]+1],
        )

        ft_train_loader = make_loader(ft_train_ds, BATCH_SIZE, shuffle=True)
        ft_val_loader = make_loader(ft_val_ds, BATCH_SIZE)
        ft_test_loader = make_loader(ft_test_ds, BATCH_SIZE)

        ft_model = finetune_fc(ft_model, ft_train_loader, ft_val_loader, device, use_amp)

        # Eval finetuned model
        ft_model.eval()
        ft_preds, ft_labels, ft_regimes = [], [], []
        with torch.no_grad():
            for x, f, y in ft_test_loader:
                x = x.to(device, non_blocking=True)
                f = f.to(device, non_blocking=True)
                logits = ft_model(x, f)
                preds = (logits.cpu() > 0).float()
                ft_preds.extend(preds.tolist())
                ft_labels.extend(y.tolist())

        for j in range(len(ft_test_ds)):
            ft_regimes.append(ft_test_ds.get_regime(j))

        ft_acc = np.mean(np.array(ft_preds) == np.array(ft_labels))
        all_ft_preds.extend(ft_preds)
        all_ft_labels.extend(ft_labels)
        all_ft_regimes.extend(ft_regimes)

        fold_time = time.time() - fold_start
        delta = ft_acc - base_acc
        print(f"    Finetuned acc: {ft_acc:.4f} (Δ={delta:+.4f}), {fold_time:.0f}s")

        fold_results.append({
            "fold": i + 1,
            "period": f"{FOLD_BOUNDARIES[i + 1]} → {FOLD_BOUNDARIES[i + 2]}",
            "baseline_acc": float(base_acc),
            "finetuned_acc": float(ft_acc),
            "delta": float(delta),
            "n_test": len(ft_preds),
            "time_sec": round(fold_time, 1),
        })

        del backbone, ft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Overall stats ──
    preds = np.array(all_ft_preds)
    labels = np.array(all_ft_labels)
    regimes = np.array(all_ft_regimes)
    base_preds = np.array(all_baseline_preds)
    base_labels = np.array(all_baseline_labels)

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

    base_acc_overall = float((base_preds == base_labels).mean())
    ft_stats = confusion(preds, labels)

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

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: v6 baseline vs v6 + funding fine-tune")
    print(f"{'=' * 70}")
    print(f"  Baseline (VP-only):  {base_acc_overall:.4f}")
    print(f"  Finetuned (+funding): {ft_stats['accuracy']:.4f} (Δ={ft_stats['accuracy'] - base_acc_overall:+.4f})")

    print(f"\n  Per-fold:")
    print(f"  {'Fold':<6} {'Period':<30} {'Baseline':>9} {'Finetuned':>10} {'Delta':>7}")
    for fr in fold_results:
        print(f"  {fr['fold']:<6} {fr['period']:<30} {fr['baseline_acc']*100:8.1f}% {fr['finetuned_acc']*100:9.1f}% {fr['delta']*100:+6.1f}%")

    if bull_stats:
        print(f"\n  BULL ({bull_stats['n']} bars): Prec={bull_stats['precision']:.4f} NPV={bull_stats['npv']:.4f}")
        print(f"    Long EV: {regime_ev.get('bull_long', 0):+.2f}%  Short EV: {regime_ev.get('bull_short', 0):+.2f}%")
    if bear_stats:
        print(f"\n  BEAR ({bear_stats['n']} bars): Prec={bear_stats['precision']:.4f} NPV={bear_stats['npv']:.4f}")
        print(f"    Long EV: {regime_ev.get('bear_long', 0):+.2f}%  Short EV: {regime_ev.get('bear_short', 0):+.2f}%")

    # Save
    output = {
        "config": {"tp": TP, "sl": SL, "batch_size": BATCH_SIZE,
                    "lr_backbone": LR_BACKBONE, "lr_finetune": LR_FINETUNE,
                    "finetune_epochs": FINETUNE_EPOCHS, "n_funding_features": N_FUNDING_FEATURES},
        "baseline_accuracy": base_acc_overall,
        "finetuned": {
            "overall": ft_stats, "bull_stats": bull_stats, "bear_stats": bear_stats,
            "regime_ev": regime_ev,
        },
        "folds": fold_results,
    }
    out_path = EXPERIMENTS_DIR / "eval_finetune_funding_results.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
