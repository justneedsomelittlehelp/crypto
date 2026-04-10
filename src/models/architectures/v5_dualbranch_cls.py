"""FROZEN: DualBranchTransformerV5 — VP + Candle with CLS pooling and volume.

DO NOT MODIFY THIS FILE. It is a frozen snapshot of the architecture under
active investigation. If you iterate further, create a v6_* file next to
this one.

── Provenance ──────────────────────────────────────────────────────────────
Latest iteration of the dual-branch design. Evolution:
    v3 (not frozen): basic dual-branch (7 candle features, mean pool)
    v4 (not frozen): added day_volume_ratio (8 candle features, mean pool)
    v5 (THIS):       added CLS token pooling + per-day VP sampling + candle volume

Best result so far (DualBranch v1 lineage, Colab 2026-04-08):
    - Walk-forward accuracy: 60.5%
    - Strategy 4 daily EV: ~+0.71%
    - Pipeline: v2_scaled (no VP structure features, expanding z-score + tanh)

Known issue: diagnostic sweep showed Temporal-only beats DualBranch by ~3%
on the same pipeline. Candle branch is under investigation.

── Architecture ────────────────────────────────────────────────────────────
VP BRANCH:
  Stage 1 (spatial): Sample 1 VP per day (last hour, end-of-day snapshot).
    Each day's 50 VP bins go through self-attention with a [CLS_spatial]
    token prepended. CLS output is the day's summary. Shared weights across
    all 30 days.
  Stage 2 (temporal): 30 daily embeddings + [CLS_vp_temporal] token go
    through self-attention across days. CLS output is the VP branch summary.

CANDLE BRANCH:
  Aggregates 24 hourly bars into 1 daily candle with 8 features:
    day_open, day_close, day_high, day_low, body, upper_wick, lower_wick,
    day_volume_ratio (mean hourly volume_ratio).
  Project 8 → 16, prepend [CLS_candle], add positional encoding, run
  self-attention across 30 days. CLS output is the candle branch summary.

MERGE: concat(vp_cls, candle_cls, last_other_features) → FC → logit.
── Input contract ──────────────────────────────────────────────────────────
Input tensor shape: (batch, 720, n_total_features)
    where 720 = 30 days × 24 hours
    and   n_total_features = 50 VP bins + n_other_features

The candle branch expects specific feature indices for OHLC ratios, log_return,
and volume_ratio. These indices are passed explicitly at construction.
"""

import torch
import torch.nn as nn


class DualBranchTransformerV5(nn.Module):
    """Frozen dual-branch Transformer with CLS pooling + volume in candles."""

    N_VP_BINS = 50

    def __init__(
        self,
        # Feature indices (required for candle reconstruction)
        ohlc_open_idx: int,
        ohlc_high_idx: int,
        ohlc_low_idx: int,
        log_return_idx: int,
        volume_ratio_idx: int,
        # Model dims
        n_other_features: int,
        vp_embed_dim: int = 32,
        candle_embed_dim: int = 16,
        n_heads: int = 4,
        n_spatial_layers: int = 1,
        n_temporal_layers: int = 1,
        n_candle_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.15,
        n_days: int = 30,
        bars_per_day: int = 24,
    ):
        super().__init__()
        self.n_days = n_days
        self.bars_per_day = bars_per_day
        self.vp_embed_dim = vp_embed_dim
        self.candle_embed_dim = candle_embed_dim
        self.n_other_features = n_other_features

        # Feature indices are absolute (within full feature vector)
        self.ohlc_open_idx = ohlc_open_idx
        self.ohlc_high_idx = ohlc_high_idx
        self.ohlc_low_idx = ohlc_low_idx
        self.log_return_idx = log_return_idx
        self.volume_ratio_idx = volume_ratio_idx

        # ─────────────── VP BRANCH ───────────────
        self.bin_embed = nn.Linear(1, vp_embed_dim)
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, vp_embed_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.N_VP_BINS + 1, vp_embed_dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=vp_embed_dim,
            nhead=n_heads,
            dim_feedforward=vp_embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        self.vp_temporal_cls = nn.Parameter(torch.randn(1, 1, vp_embed_dim) * 0.02)
        self.vp_temporal_pos = nn.Parameter(torch.randn(1, n_days + 1, vp_embed_dim) * 0.02)
        vp_temporal_layer = nn.TransformerEncoderLayer(
            d_model=vp_embed_dim,
            nhead=n_heads,
            dim_feedforward=vp_embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.vp_temporal_transformer = nn.TransformerEncoder(vp_temporal_layer, num_layers=n_temporal_layers)

        # ─────────────── CANDLE BRANCH ───────────────
        self.candle_embed = nn.Linear(8, candle_embed_dim)
        self.candle_cls = nn.Parameter(torch.randn(1, 1, candle_embed_dim) * 0.02)
        self.candle_pos = nn.Parameter(torch.randn(1, n_days + 1, candle_embed_dim) * 0.02)
        candle_layer = nn.TransformerEncoderLayer(
            d_model=candle_embed_dim,
            nhead=2,
            dim_feedforward=candle_embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.candle_transformer = nn.TransformerEncoder(candle_layer, num_layers=n_candle_layers)

        # ─────────────── MERGE ───────────────
        merged_dim = vp_embed_dim + candle_embed_dim + n_other_features
        self.fc1 = nn.Linear(merged_dim, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def _aggregate_daily_candles(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate 24 hourly bars per day into 1 daily candle (8 features per day)."""
        batch_size = x.shape[0]
        open_r = x[:, :, self.ohlc_open_idx]
        high_r = x[:, :, self.ohlc_high_idx]
        low_r = x[:, :, self.ohlc_low_idx]

        open_r = open_r.view(batch_size, self.n_days, self.bars_per_day)
        high_r = high_r.view(batch_size, self.n_days, self.bars_per_day)
        low_r = low_r.view(batch_size, self.n_days, self.bars_per_day)

        log_ret = x[:, :, self.log_return_idx]
        log_ret = log_ret.view(batch_size, self.n_days, self.bars_per_day)

        vol_ratio = x[:, :, self.volume_ratio_idx]
        vol_ratio = vol_ratio.view(batch_size, self.n_days, self.bars_per_day)
        day_volume_ratio = vol_ratio.mean(dim=2)

        day_cumret = log_ret.cumsum(dim=2)
        rel_to_day_close = torch.exp(day_cumret - day_cumret[:, :, -1:])

        day_open = open_r[:, :, 0] * rel_to_day_close[:, :, 0]
        day_close = torch.ones_like(day_open)
        day_high = (high_r * rel_to_day_close).max(dim=2).values
        day_low = (low_r * rel_to_day_close).min(dim=2).values

        bar_height = (day_high - day_low).clamp(min=1e-8)
        body = day_close - day_open
        upper_wick = (day_high - torch.max(day_open, day_close)) / bar_height
        lower_wick = (torch.min(day_open, day_close) - day_low) / bar_height

        candle = torch.stack([
            day_open, day_close, day_high, day_low,
            body, upper_wick, lower_wick, day_volume_ratio,
        ], dim=2)
        candle = torch.nan_to_num(candle, nan=0.0, posinf=0.0, neginf=0.0)
        return candle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape validation
        assert x.shape[1] == self.n_days * self.bars_per_day, (
            f"Expected lookback {self.n_days * self.bars_per_day}, got {x.shape[1]}"
        )
        assert x.shape[2] == self.N_VP_BINS + self.n_other_features, (
            f"Expected {self.N_VP_BINS + self.n_other_features} features, got {x.shape[2]}"
        )

        batch_size = x.shape[0]
        vp_bins = x[:, :, :self.N_VP_BINS]
        other = x[:, -1, self.N_VP_BINS:]

        # VP branch: sample 1 per day, spatial CLS, then temporal CLS
        vp_daily = vp_bins[:, self.bars_per_day - 1::self.bars_per_day, :]  # (batch, 30, 50)

        flat_vp = vp_daily.reshape(batch_size * self.n_days, self.N_VP_BINS, 1)
        flat_tokens = self.bin_embed(flat_vp)
        cls_spatial = self.spatial_cls.expand(batch_size * self.n_days, -1, -1)
        flat_tokens = torch.cat([cls_spatial, flat_tokens], dim=1)
        flat_tokens = flat_tokens + self.spatial_pos
        flat_attended = self.spatial_transformer(flat_tokens)
        flat_cls_out = flat_attended[:, 0, :]
        vp_temporal_input = flat_cls_out.view(batch_size, self.n_days, self.vp_embed_dim)

        cls_vp_temp = self.vp_temporal_cls.expand(batch_size, -1, -1)
        vp_temporal_input = torch.cat([cls_vp_temp, vp_temporal_input], dim=1)
        vp_temporal_input = vp_temporal_input + self.vp_temporal_pos
        vp_temporal_out = self.vp_temporal_transformer(vp_temporal_input)
        vp_pooled = vp_temporal_out[:, 0, :]

        # Candle branch
        candles = self._aggregate_daily_candles(x)
        candle_tokens = self.candle_embed(candles)
        cls_candle = self.candle_cls.expand(batch_size, -1, -1)
        candle_tokens = torch.cat([cls_candle, candle_tokens], dim=1)
        candle_tokens = candle_tokens + self.candle_pos
        candle_attended = self.candle_transformer(candle_tokens)
        candle_pooled = candle_attended[:, 0, :]

        # Merge
        combined = torch.cat([vp_pooled, candle_pooled, other], dim=1)
        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit
