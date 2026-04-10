"""FROZEN: SimpleTemporalV7 — 2 spatial + 1 temporal, no enrichment.

DO NOT MODIFY THIS FILE once results are committed.

── Provenance ──────────────────────────────────────────────────────────────
Direct extension of the spatial-only Transformer (Eval 4, 63.3% acc).
Adds temporal attention across 30 daily VP embeddings on top of 2 spatial
layers. Uses mean pooling (no CLS tokens). Same v2_scaled pipeline (60 feat).

Eval 4 architecture: sum 720 bars → spatial attention across 50 bins (2 layers)
This architecture: per-day VP sum → spatial attention (2 layers) → temporal (1 layer)

── Architecture ────────────────────────────────────────────────────────────
Stage 1 (spatial, 2 layers): For each of 30 days, sum 24 hourly VP bins
    into a daily VP profile, normalize, run 2-layer self-attention across
    50 bins. Mean pool → one embedding per day. Shared weights across days.
    Output: (batch, 30, 32).

Stage 2 (temporal, 1 layer): Self-attention across 30 daily embeddings
    with positional encoding. Learns VP persistence, breakouts, evolving
    support/resistance over time. Mean pool → (batch, 32).

Final: concat(temporal pool, last hour's non-VP features) → FC → logit.

── Input contract ──────────────────────────────────────────────────────────
Designed for v2_scaled pipeline (60 features = 50 VP + 10 derived).
N_VP_BINS = 50, N_OTHER_FEATURES = 10.

── Parameters ──────────────────────────────────────────────────────────────
embed_dim=32, n_heads=4, n_spatial_layers=2, n_temporal_layers=1
fc_size=64, dropout=0.15, n_days=30, bars_per_day=24
Total: 31,073 params
"""

import torch
import torch.nn as nn


N_VP_BINS = 50
N_OTHER_FEATURES = 10


class SimpleTemporalV7(nn.Module):
    """Two-stage Transformer: 2-layer spatial + 1-layer temporal, mean pooling."""

    def __init__(
        self,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_spatial_layers: int = 2,
        n_temporal_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.15,
        n_days: int = 30,
        bars_per_day: int = 24,
    ):
        super().__init__()
        self.n_days = n_days
        self.bars_per_day = bars_per_day
        self.embed_dim = embed_dim

        # Stage 1: Spatial attention across VP bins (shared across all days)
        self.bin_embed = nn.Linear(1, embed_dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, N_VP_BINS, embed_dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        # Stage 2: Temporal attention across days
        self.temporal_pos = nn.Parameter(torch.randn(1, n_days, embed_dim) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=n_temporal_layers)

        # Final FC: temporal output + non-VP features
        self.fc1 = nn.Linear(embed_dim + N_OTHER_FEATURES, fc_size)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback=720, features)
        batch_size = x.shape[0]
        vp_bins = x[:, :, :N_VP_BINS]        # (batch, 720, 50)
        other = x[:, -1, N_VP_BINS:]          # (batch, N_OTHER)

        # Stage 1: spatial attention — batched across all 30 days at once
        # Reshape: (batch, 720, 50) → (batch, 30, 24, 50) → sum hours → (batch, 30, 50)
        vp_daily = vp_bins.view(batch_size, self.n_days, self.bars_per_day, N_VP_BINS)
        vp_daily = vp_daily.sum(dim=2)                             # (batch, 30, 50)

        # Normalize per day per sample
        day_max = vp_daily.max(dim=2, keepdim=True).values.clamp(min=1e-8)
        vp_daily = vp_daily / day_max                              # (batch, 30, 50)

        # Flatten batch and days for one Transformer call: (batch*30, 50, 1)
        flat_vp = vp_daily.reshape(batch_size * self.n_days, N_VP_BINS, 1)
        flat_tokens = self.bin_embed(flat_vp) + self.spatial_pos   # (batch*30, 50, embed_dim)
        flat_attended = self.spatial_transformer(flat_tokens)       # (batch*30, 50, embed_dim)
        flat_pooled = flat_attended.mean(dim=1)                    # (batch*30, embed_dim)

        # Unflatten: (batch, 30, embed_dim)
        temporal_input = flat_pooled.view(batch_size, self.n_days, self.embed_dim)

        # Stage 2: temporal attention across days
        temporal_input = temporal_input + self.temporal_pos
        temporal_out = self.temporal_transformer(temporal_input)    # (batch, 30, embed_dim)

        # Pool across days
        pooled = temporal_out.mean(dim=1)                          # (batch, embed_dim)

        # Combine with non-VP features
        combined = torch.cat([pooled, other], dim=1)

        out = self.dropout_layer(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        logit = self.fc2(out).squeeze(-1)
        return logit
