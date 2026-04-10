"""FROZEN: TemporalTransformerV2 — VP-only with spatial + temporal attention.

DO NOT MODIFY THIS FILE. It is a frozen snapshot of the architecture that
produced our best validated result. If you need to iterate, create a new
v3_*, v4_* file next to this one and leave this untouched.

── Provenance ──────────────────────────────────────────────────────────────
Original name: TemporalTransformerClassifier
Best result (Colab, 2026-04-08):
    - Walk-forward accuracy: 61.9%
    - 0/10 folds below 50%
    - Batch=512, constant LR 5e-4, no grad clipping
    - Pipeline: v1_raw (VP structure features present, no scaling)
── Architecture ────────────────────────────────────────────────────────────
Stage 1 (spatial): Each of 30 daily VP snapshots is processed by
    self-attention across 50 VP bins. Shared weights across days.
    Mean pool → one embedding per day.
Stage 2 (temporal): Self-attention across 30 daily embeddings.
    Mean pool → single 32-dim VP summary.
Final: concat(VP summary, last hour's other features) → FC → logit.
── Input contract ──────────────────────────────────────────────────────────
Input tensor shape: (batch, 720, n_total_features)
    where 720 = 30 days × 24 hours
    and   n_total_features = 50 VP bins + n_other_features
Default: n_other_features = 18 (matches v1_raw pipeline: 10 derived + 8 VP structure)
"""

import torch
import torch.nn as nn


class TemporalTransformerV2(nn.Module):
    """Frozen temporal Transformer. Best validated accuracy model."""

    N_VP_BINS = 50
    DEFAULT_N_OTHER = 18  # v1_raw pipeline: 10 derived + 8 VP structure

    def __init__(
        self,
        n_other_features: int = DEFAULT_N_OTHER,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_spatial_layers: int = 1,
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
        self.n_other_features = n_other_features

        # Stage 1: Spatial attention across VP bins (shared across all days)
        self.bin_embed = nn.Linear(1, embed_dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.N_VP_BINS, embed_dim) * 0.02)
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
        self.fc1 = nn.Linear(embed_dim + n_other_features, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 720, n_total_features)
        # Validate input shape
        assert x.shape[1] == self.n_days * self.bars_per_day, (
            f"Expected lookback {self.n_days * self.bars_per_day}, got {x.shape[1]}"
        )
        assert x.shape[2] == self.N_VP_BINS + self.n_other_features, (
            f"Expected {self.N_VP_BINS + self.n_other_features} features, got {x.shape[2]}"
        )

        batch_size = x.shape[0]
        vp_bins = x[:, :, :self.N_VP_BINS]        # (batch, 720, 50)
        other = x[:, -1, self.N_VP_BINS:]          # (batch, n_other_features)

        # Stage 1: spatial attention — batched across all 30 days at once
        vp_daily = vp_bins.view(batch_size, self.n_days, self.bars_per_day, self.N_VP_BINS)
        vp_daily = vp_daily.sum(dim=2)                             # (batch, 30, 50)

        # Normalize per day per sample
        day_max = vp_daily.max(dim=2, keepdim=True).values.clamp(min=1e-8)
        vp_daily = vp_daily / day_max                              # (batch, 30, 50)

        # Flatten batch and days for one Transformer call
        flat_vp = vp_daily.reshape(batch_size * self.n_days, self.N_VP_BINS, 1)
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

        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit
