"""v11 — AbsVPv11: 2-channel (vp_abs + self) spatial attention over 90 day tokens.

── Provenance ──────────────────────────────────────────────────────────────
Built from scratch for the absolute-range VP representation (2026-04-13).
Structurally mirrors v10 (2+1 transformer, 1 spatial layer + 1 temporal)
but takes pre-aggregated daily features as input instead of raw hourly bars.

── Input contract ──────────────────────────────────────────────────────────
`forward(day_tokens, last_bar)` where:
    day_tokens:  (B, 90, 110) — one row per day, with columns:
        [0:50]   vp_abs_00..49   (absolute-range VP, sums to 1)
        [50:100] self_00..49     (hard one-hot on current-price bin)
        [100:108] candle_0..7    (8 daily candle features, pre-aggregated)
        [108]    price_pos       (continuous ∈ [0, 1])
        [109]    range_pct       (continuous, = (hi - lo) / close)
    last_bar:    (B, 4) — features from the most recent 15m bar:
        [0] log_return
        [1] volume_ratio
        [2] price_pos
        [3] range_pct

── Why pre-aggregated ──────────────────────────────────────────────────────
At 15m, the raw window per sample is 8,640 bars × ~110 features ≈ 3.8 MB.
A batch of 256 → ~1 GB just for the input tensor, and aggregation inside
forward() wastes GPU cycles re-computing the same daily stats across
neighbouring samples. Pre-aggregating once on CPU shrinks per-sample
memory by 96× and removes all recomputation. The day-gather happens on
the GPU as a single index op in the training loop — see eval_v11.py.

── Architecture ────────────────────────────────────────────────────────────
Stage 1 (spatial):
    For each of 90 days, 50 bins carrying (vp_abs, self) as a 2-D token.
    `bin_embed`: Linear(2, 32) — the one real shape change vs v10.
    Positional embed, CLS token, 1 transformer layer (4 heads).
    CLS → (B, 90, 32).

Day enrichment:
    concat(spatial_cls [32], candle [8], scalars [2]) → 42
    `day_projection`: Linear(42, 32)

Stage 2 (temporal):
    CLS + 90 day tokens + positional, 1 transformer layer, CLS pool.
    → (B, 32)

Final FC:
    concat(temporal_cls [32], last_bar [4]) → 36
    Linear(36, 64) → ReLU → Dropout → Linear(64, 1) → logit
"""

import torch
import torch.nn as nn


class AbsVPv11(nn.Module):
    """Absolute-VP transformer with 2-channel spatial attention (v11)."""

    N_BINS = 50
    N_CANDLE = 8
    N_SCALARS = 2       # price_pos, range_pct
    N_LAST_BAR = 4      # log_return, volume_ratio, price_pos, range_pct
    DAY_TOKEN_WIDTH = 2 * N_BINS + N_CANDLE + N_SCALARS  # 110

    def __init__(
        self,
        n_days: int = 90,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_spatial_layers: int = 1,
        n_temporal_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_days = n_days
        self.embed_dim = embed_dim

        # ─────────────── SPATIAL ATTENTION ───────────────
        # Each bin carries (vp_abs, self) as a 2-D token.
        self.bin_embed = nn.Linear(2, embed_dim)
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.N_BINS + 1, embed_dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        # ─────────────── DAY TOKEN ENRICHMENT ───────────────
        enriched_dim = embed_dim + self.N_CANDLE + self.N_SCALARS
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
        self.fc1 = nn.Linear(embed_dim + self.N_LAST_BAR, fc_size)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, day_tokens: torch.Tensor, last_bar: torch.Tensor) -> torch.Tensor:
        """
        day_tokens: (B, n_days, 110)
        last_bar:   (B, 4)
        returns logit: (B,)
        """
        B = day_tokens.shape[0]
        assert day_tokens.shape[1] == self.n_days, (
            f"expected n_days={self.n_days}, got {day_tokens.shape[1]}"
        )
        assert day_tokens.shape[2] == self.DAY_TOKEN_WIDTH, (
            f"expected day_token width {self.DAY_TOKEN_WIDTH}, got {day_tokens.shape[2]}"
        )

        vp = day_tokens[:, :, 0:self.N_BINS]                          # (B, D, 50)
        self_ch = day_tokens[:, :, self.N_BINS:2 * self.N_BINS]       # (B, D, 50)
        candle = day_tokens[:, :, 2 * self.N_BINS:2 * self.N_BINS + self.N_CANDLE]  # (B, D, 8)
        scalars = day_tokens[:, :, 2 * self.N_BINS + self.N_CANDLE:]  # (B, D, 2)

        # ─── Stage 1: spatial attention over 50 bins, 2 channels per bin ───
        # Stack the two channels into the per-bin feature dim.
        two_ch = torch.stack([vp, self_ch], dim=-1)                   # (B, D, 50, 2)
        flat = two_ch.reshape(B * self.n_days, self.N_BINS, 2)

        flat_tokens = self.bin_embed(flat)                            # (B*D, 50, 32)
        cls_spatial = self.spatial_cls.expand(B * self.n_days, -1, -1)
        flat_tokens = torch.cat([cls_spatial, flat_tokens], dim=1)    # (B*D, 51, 32)
        flat_tokens = flat_tokens + self.spatial_pos
        flat_attended = self.spatial_transformer(flat_tokens)
        spatial_cls_out = flat_attended[:, 0, :]                      # (B*D, 32)
        spatial_cls_out = spatial_cls_out.view(B, self.n_days, self.embed_dim)

        # ─── Enrich day tokens ───
        enriched = torch.cat([spatial_cls_out, candle, scalars], dim=2)  # (B, D, 42)
        day_tok = self.day_projection(enriched)                          # (B, D, 32)

        # ─── Stage 2: temporal attention ───
        cls_temporal = self.temporal_cls.expand(B, -1, -1)
        temporal_input = torch.cat([cls_temporal, day_tok], dim=1)    # (B, D+1, 32)
        temporal_input = temporal_input + self.temporal_pos
        temporal_out = self.temporal_transformer(temporal_input)
        pooled = temporal_out[:, 0, :]                                # (B, 32)

        # ─── Final FC ───
        combined = torch.cat([pooled, last_bar], dim=1)               # (B, 36)
        h = self.dropout_layer(combined)
        h = self.relu(self.fc1(h))
        h = self.dropout_layer(h)
        logit = self.fc2(h).squeeze(-1)
        return logit
