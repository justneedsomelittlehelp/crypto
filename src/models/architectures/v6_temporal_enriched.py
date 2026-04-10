"""FROZEN: TemporalEnrichedV6 — single-path temporal with enriched day tokens.

DO NOT MODIFY THIS FILE once results are committed.

── Provenance ──────────────────────────────────────────────────────────────
Simplification of the DualBranch design. Instead of separate VP + Candle
branches that merge at FC, this model enriches each day token with ALL
available info (VP spatial summary + daily candle + daily volume + VP
structure features) and runs ONE temporal attention over them.

Key insight from DualBranch experiments: the candle branch as a separate
attention pathway added noise and hurt performance. Concatenating candle
features into the day token and letting temporal attention handle
everything in one pass is simpler and more natural.

── Architecture ────────────────────────────────────────────────────────────
Stage 1 (spatial): Same as v2_temporal. For each of 30 days, sample the
    end-of-day VP (last hour), run self-attention across 50 bins with CLS
    pooling. Output: (batch, 30, vp_embed_dim).

Enrichment: For each day, concat:
    - VP spatial CLS output (32 dim)
    - Daily candle features (8: open, close, high, low, body, upper/lower wick, volume_ratio)
    - Daily VP structure features (8: ceiling/floor dist/strength/consistency, num_peaks, mid_range)
    Project (32 + 8 + 8 = 48) → 32 via Linear, so temporal embed_dim stays 32.

Stage 2 (temporal): CLS-pooled self-attention across 30 enriched day tokens.
    Output: (batch, vp_embed_dim).

Final: concat(temporal CLS, last hour's other features) → FC → logit.

── Input contract ──────────────────────────────────────────────────────────
Designed for v1_raw pipeline (68 features = 50 VP + 10 derived + 8 VP structure).
Feature indices for candle reconstruction and VP structure extraction are
passed explicitly at construction.
"""

import torch
import torch.nn as nn


class TemporalEnrichedV6(nn.Module):
    """Single-path temporal Transformer with enriched day tokens."""

    N_VP_BINS = 50

    def __init__(
        self,
        # Feature indices (required for candle + VP structure extraction)
        ohlc_open_idx: int,
        ohlc_high_idx: int,
        ohlc_low_idx: int,
        log_return_idx: int,
        volume_ratio_idx: int,
        vp_structure_start_idx: int,  # first VP structure col index
        n_vp_structure: int = 8,      # number of VP structure features
        n_other_features: int = 18,   # total non-VP features for FC
        # Model dims
        embed_dim: int = 32,
        n_heads: int = 4,
        n_spatial_layers: int = 1,
        n_temporal_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.15,
        n_days: int = 30,
        bars_per_day: int = 24,
        n_candle_features: int = 8,
    ):
        super().__init__()
        self.n_days = n_days
        self.bars_per_day = bars_per_day
        self.embed_dim = embed_dim
        self.n_other_features = n_other_features
        self.n_vp_structure = n_vp_structure
        self.n_candle_features = n_candle_features

        # Feature indices
        self.ohlc_open_idx = ohlc_open_idx
        self.ohlc_high_idx = ohlc_high_idx
        self.ohlc_low_idx = ohlc_low_idx
        self.log_return_idx = log_return_idx
        self.volume_ratio_idx = volume_ratio_idx
        self.vp_structure_start_idx = vp_structure_start_idx

        # ─────────────── SPATIAL ATTENTION ───────────────
        self.bin_embed = nn.Linear(1, embed_dim)
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.N_VP_BINS + 1, embed_dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        # ─────────────── DAY TOKEN ENRICHMENT ───────────────
        # Project enriched token (VP embed + candle + VP structure) back to embed_dim
        enriched_dim = embed_dim + n_candle_features + n_vp_structure
        self.day_projection = nn.Linear(enriched_dim, embed_dim)

        # ─────────────── TEMPORAL ATTENTION ───────────────
        self.temporal_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_days + 1, embed_dim) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=n_temporal_layers)

        # ─────────────── FINAL FC ───────────────
        self.fc1 = nn.Linear(embed_dim + n_other_features, fc_size)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def _aggregate_daily_candles(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate 24 hourly bars per day into 1 daily candle (8 features)."""
        batch_size = x.shape[0]
        open_r = x[:, :, self.ohlc_open_idx].view(batch_size, self.n_days, self.bars_per_day)
        high_r = x[:, :, self.ohlc_high_idx].view(batch_size, self.n_days, self.bars_per_day)
        low_r = x[:, :, self.ohlc_low_idx].view(batch_size, self.n_days, self.bars_per_day)
        log_ret = x[:, :, self.log_return_idx].view(batch_size, self.n_days, self.bars_per_day)
        vol_ratio = x[:, :, self.volume_ratio_idx].view(batch_size, self.n_days, self.bars_per_day)

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
        return torch.nan_to_num(candle, nan=0.0, posinf=0.0, neginf=0.0)

    def _sample_daily_vp_structure(self, x: torch.Tensor) -> torch.Tensor:
        """Sample VP structure features from the last hour of each day."""
        # VP structure features are already 180-day rolling — just pick end-of-day
        # (batch, 30, n_vp_structure)
        vp_struct = x[:, self.bars_per_day - 1::self.bars_per_day,
                       self.vp_structure_start_idx:self.vp_structure_start_idx + self.n_vp_structure]
        return vp_struct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.n_days * self.bars_per_day
        assert x.shape[2] == self.N_VP_BINS + self.n_other_features

        batch_size = x.shape[0]
        vp_bins = x[:, :, :self.N_VP_BINS]
        other = x[:, -1, self.N_VP_BINS:]

        # ─── Stage 1: Spatial attention with CLS ───
        vp_daily = vp_bins[:, self.bars_per_day - 1::self.bars_per_day, :]  # (batch, 30, 50)

        flat_vp = vp_daily.reshape(batch_size * self.n_days, self.N_VP_BINS, 1)
        flat_tokens = self.bin_embed(flat_vp)
        cls_spatial = self.spatial_cls.expand(batch_size * self.n_days, -1, -1)
        flat_tokens = torch.cat([cls_spatial, flat_tokens], dim=1)
        flat_tokens = flat_tokens + self.spatial_pos
        flat_attended = self.spatial_transformer(flat_tokens)
        vp_spatial_out = flat_attended[:, 0, :].view(batch_size, self.n_days, self.embed_dim)
        # (batch, 30, 32)

        # ─── Enrich day tokens ───
        daily_candles = self._aggregate_daily_candles(x)       # (batch, 30, 8)
        daily_vp_struct = self._sample_daily_vp_structure(x)   # (batch, 30, 8)

        enriched = torch.cat([vp_spatial_out, daily_candles, daily_vp_struct], dim=2)
        # (batch, 30, 32 + 8 + 8 = 48)

        day_tokens = self.day_projection(enriched)             # (batch, 30, 32)

        # ─── Stage 2: Temporal attention with CLS ───
        cls_temporal = self.temporal_cls.expand(batch_size, -1, -1)
        temporal_input = torch.cat([cls_temporal, day_tokens], dim=1)  # (batch, 31, 32)
        temporal_input = temporal_input + self.temporal_pos
        temporal_out = self.temporal_transformer(temporal_input)
        pooled = temporal_out[:, 0, :]                         # CLS token (batch, 32)

        # ─── Final FC ───
        combined = torch.cat([pooled, other], dim=1)
        out = self.dropout_layer(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        logit = self.fc2(out).squeeze(-1)
        return logit
