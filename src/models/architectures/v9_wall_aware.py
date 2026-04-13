"""FROZEN: TemporalEnrichedV9WallAware — v6-prime + structure context token.

DO NOT MODIFY THIS FILE once results are committed.

── Provenance ──────────────────────────────────────────────────────────────
Architectural variant of TemporalEnrichedV6Prime for the Stage 2 experiment
from the post-audit work (2026-04-12).

Key change vs v6-prime:
  - The spatial self-attention in v6-prime operates on 51 tokens:
    [CLS + 50 VP bins]. VP structure features (peak_strength,
    ceiling_dist, floor_dist, num_peaks, consistency, …) only enter the
    model via the day-token enrichment bottleneck (Linear 48→32) and the
    last-hour FC skip connection. The spatial attention is blind to which
    bins are thick walls and which are thin spikes.

  - v9 adds ONE extra token to the spatial sequence: a "structure context
    token" projected from the 8 VP structure features of that day.
    Sequence becomes 52 tokens: [CLS + 50 VP bins + structure_context].
    Now every bin can attend to the structure context (asking "how does
    my position relate to ceiling_dist / peak_strength?") and the
    structure context can attend to every bin (integrating which bins
    match the per-day structure summary).

  - Day-token enrichment path and last-hour FC skip path are UNCHANGED.
    v9 is strictly additive over v6-prime, to keep the A/B comparison
    clean: if v9 beats v6-prime on the honest backtest, the delta is
    fully attributable to the structure context token in spatial
    attention.

── Architecture ────────────────────────────────────────────────────────────
Stage 1 (spatial): for each of 30 days, sample the end-of-day VP. Build a
    sequence of CLS + 50 bin tokens + 1 structure context token.
    Spatial self-attention with CLS pooling → (batch, 30, embed_dim).

Enrichment (UNCHANGED from v6-prime): for each day, concat
    [VP spatial CLS (32), daily candle (8), daily VP structure (8)]
    Project 48 → 32 via Linear.

Stage 2 (temporal): CLS-pooled self-attention across 30 enriched day
    tokens → (batch, 32). UNCHANGED from v6-prime.

Final: concat(temporal CLS, last hour's other features) → FC → logit.
    UNCHANGED from v6-prime.

── Input contract ──────────────────────────────────────────────────────────
v1_raw pipeline (68 features = 50 VP + 10 derived + 8 VP structure).

── Parameter overhead vs v6-prime ──────────────────────────────────────────
- structure_embed: Linear(8, 32)           = 288 params
- spatial_pos: 51 → 52 slots × 32 embed    = +32 params
Total: ~320 extra params on top of v6-prime's ~24,737 (≈ +1.3%).
"""

import torch
import torch.nn as nn


class TemporalEnrichedV9WallAware(nn.Module):
    """v6-prime plus a structure context token in the spatial attention layer."""

    N_VP_BINS = 50

    def __init__(
        self,
        # Feature indices (required for candle + VP structure extraction)
        ohlc_open_idx: int,
        ohlc_high_idx: int,
        ohlc_low_idx: int,
        log_return_idx: int,
        volume_ratio_idx: int,
        vp_structure_start_idx: int,
        n_vp_structure: int = 8,
        n_other_features: int = 18,
        # Model dims
        embed_dim: int = 32,
        n_heads: int = 4,
        n_spatial_layers: int = 1,
        n_temporal_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.3,
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

        self.ohlc_open_idx = ohlc_open_idx
        self.ohlc_high_idx = ohlc_high_idx
        self.ohlc_low_idx = ohlc_low_idx
        self.log_return_idx = log_return_idx
        self.volume_ratio_idx = volume_ratio_idx
        self.vp_structure_start_idx = vp_structure_start_idx

        # ─────────────── SPATIAL ATTENTION (v9: +1 structure context token) ───────────────
        self.bin_embed = nn.Linear(1, embed_dim)
        self.structure_embed = nn.Linear(n_vp_structure, embed_dim)  # NEW vs v6-prime

        self.spatial_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Sequence length: CLS (1) + bins (50) + structure context (1) = 52
        self.spatial_pos = nn.Parameter(torch.randn(1, self.N_VP_BINS + 2, embed_dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        # ─────────────── DAY TOKEN ENRICHMENT (unchanged) ───────────────
        enriched_dim = embed_dim + n_candle_features + n_vp_structure
        self.day_projection = nn.Linear(enriched_dim, embed_dim)

        # ─────────────── TEMPORAL ATTENTION (unchanged) ───────────────
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

        # ─────────────── FINAL FC (unchanged) ───────────────
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
        """Sample VP structure features from the last hour of each day. (batch, 30, 8)"""
        vp_struct = x[:, self.bars_per_day - 1::self.bars_per_day,
                       self.vp_structure_start_idx:self.vp_structure_start_idx + self.n_vp_structure]
        return vp_struct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.n_days * self.bars_per_day
        assert x.shape[2] == self.N_VP_BINS + self.n_other_features

        batch_size = x.shape[0]
        vp_bins = x[:, :, :self.N_VP_BINS]
        other = x[:, -1, self.N_VP_BINS:]

        # Sample daily VP structure ONCE — reused for both paths below.
        daily_vp_struct = self._sample_daily_vp_structure(x)  # (batch, 30, 8)

        # ─── Stage 1: Spatial attention with CLS + structure context token ───
        vp_daily = vp_bins[:, self.bars_per_day - 1::self.bars_per_day, :]  # (batch, 30, 50)

        flat_vp = vp_daily.reshape(batch_size * self.n_days, self.N_VP_BINS, 1)
        flat_bin_tokens = self.bin_embed(flat_vp)  # (batch*30, 50, embed_dim)

        # Structure context token: project the 8 per-day structure features to embed_dim.
        # Each day gets its own structure token, carrying that day's ceiling_dist,
        # peak_strength, etc.
        flat_struct = daily_vp_struct.reshape(batch_size * self.n_days, self.n_vp_structure)
        structure_token = self.structure_embed(flat_struct).unsqueeze(1)
        # (batch*30, 1, embed_dim)

        cls_spatial = self.spatial_cls.expand(batch_size * self.n_days, -1, -1)
        # Sequence: [CLS, bin_0, ..., bin_49, structure_context]
        flat_tokens = torch.cat([cls_spatial, flat_bin_tokens, structure_token], dim=1)
        # (batch*30, 52, embed_dim)

        flat_tokens = flat_tokens + self.spatial_pos
        flat_attended = self.spatial_transformer(flat_tokens)
        # CLS output: (batch*30, embed_dim) → reshape to (batch, 30, embed_dim)
        vp_spatial_out = flat_attended[:, 0, :].view(batch_size, self.n_days, self.embed_dim)

        # ─── Enrich day tokens (unchanged from v6-prime) ───
        daily_candles = self._aggregate_daily_candles(x)  # (batch, 30, 8)

        enriched = torch.cat([vp_spatial_out, daily_candles, daily_vp_struct], dim=2)
        # (batch, 30, 32 + 8 + 8 = 48)

        day_tokens = self.day_projection(enriched)  # (batch, 30, 32)

        # ─── Stage 2: Temporal attention with CLS (unchanged) ───
        cls_temporal = self.temporal_cls.expand(batch_size, -1, -1)
        temporal_input = torch.cat([cls_temporal, day_tokens], dim=1)  # (batch, 31, 32)
        temporal_input = temporal_input + self.temporal_pos
        temporal_out = self.temporal_transformer(temporal_input)
        pooled = temporal_out[:, 0, :]  # CLS token (batch, 32)

        # ─── Final FC (unchanged) ───
        combined = torch.cat([pooled, other], dim=1)
        out = self.dropout_layer(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        logit = self.fc2(out).squeeze(-1)
        return logit
