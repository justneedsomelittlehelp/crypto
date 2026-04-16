"""v12 — AbsVPv12: v11 + regime-aware encoder for macro context.

── Provenance ──────────────────────────────────────────────────────────────
Built on top of frozen v11 (v11_abs_vp.py). Adds two regime-awareness
channels to address fold-12 regime sensitivity identified in Stage 8
matched-gradient-steps (2026-04-16). The VP pipeline is unchanged.

── What's new vs v11 ──────────────────────────────────────────────────────
1. Day enrichment widened: daily regime scalars (VIX close, DXY close,
   GLD close, USO close, FFR level, 10Y-2Y spread) appended alongside
   candle + scalars before projection. The temporal transformer sees the
   90-day macro trajectory.

2. Separate regime encoder: a 3-layer 1D conv stack processes the last
   72 hours of macro data (VIX, DXY, GLD, USO at 1h resolution) and
   outputs a regime embedding. Conv kernels at 3/6/12 hours detect
   spike patterns at different time scales. Injected at the final FC
   alongside temporal_cls and last_bar.

── Input contract ──────────────────────────────────────────────────────────
`forward(day_tokens, last_bar, regime_hourly)` where:
    day_tokens:     (B, 90, 110 + N_REGIME_DAILY)
        [0:110]     same as v11 (vp_abs, self, candle, price_pos, range_pct)
        [110:110+N]  daily regime scalars (VIX, DXY, GLD, USO, FFR, yield)
    last_bar:       (B, 4) — same as v11
    regime_hourly:  (B, 72, N_REGIME_HOURLY) — last 72h of hourly macro data
        Each row: [VIX, DXY, GLD, USO] at 1h resolution

── Architecture ────────────────────────────────────────────────────────────
Spatial stage:       identical to v11 (50 bins × 2 channels per day)
Day enrichment:      concat(spatial_cls [32], candle [8], scalars [2],
                     regime_daily [N_REGIME_DAILY]) → Linear(42+N, 32)
Temporal stage:      identical to v11 (CLS + 90 day tokens)
Regime encoder:      LSTM(N_REGIME_HOURLY, 16) over 72 hourly tokens → (B, 16)
Final FC:            concat(temporal_cls [32], last_bar [4], regime_emb [16])
                     → Linear(52, 64) → ReLU → Dropout → Linear(64, 1) → logit
"""

import torch
import torch.nn as nn


class RegimeEncoder(nn.Module):
    """1D conv stack over recent hourly macro data → regime embedding.

    Conv layers detect spike/shift patterns (e.g. VIX doubling in 3 hours)
    without the sequential overhead of an LSTM. Three kernel sizes (3, 6, 12)
    capture dynamics at different time scales (~3h, ~6h, ~12h).
    """

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=6, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=12, padding=5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_features) — hourly macro data, T=72 typically
        returns: (B, hidden_dim)
        """
        x = x.transpose(1, 2)          # (B, n_features, T) for Conv1d
        x = self.relu(self.conv1(x))    # (B, 16, T)
        x = self.relu(self.conv2(x))    # (B, 16, T')
        x = self.relu(self.conv3(x))    # (B, 16, T'')
        x = self.pool(x).squeeze(-1)    # (B, 16)
        return self.dropout(x)


class AbsVPv12(nn.Module):
    """Absolute-VP transformer with regime encoder (v12)."""

    N_BINS = 50
    N_CANDLE = 8
    N_SCALARS = 2           # price_pos, range_pct
    N_LAST_BAR = 4          # log_return, volume_ratio, price_pos, range_pct
    N_REGIME_DAILY = 6      # VIX, DXY, GLD, USO, FFR, 10Y-2Y
    N_REGIME_HOURLY = 4     # VIX, DXY, GLD, USO
    REGIME_LOOKBACK = 72    # hours

    DAY_TOKEN_WIDTH_V11 = 2 * N_BINS + N_CANDLE + N_SCALARS     # 110
    DAY_TOKEN_WIDTH = DAY_TOKEN_WIDTH_V11 + N_REGIME_DAILY       # 116

    def __init__(
        self,
        n_days: int = 90,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_spatial_layers: int = 1,
        n_temporal_layers: int = 1,
        fc_size: int = 64,
        dropout: float = 0.3,
        regime_hidden: int = 16,
    ):
        super().__init__()
        self.n_days = n_days
        self.embed_dim = embed_dim
        self.regime_hidden = regime_hidden

        # ─────────────── SPATIAL ATTENTION (unchanged from v11) ───────────────
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

        # ─────────────── DAY TOKEN ENRICHMENT (widened for regime daily) ──────
        enriched_dim = embed_dim + self.N_CANDLE + self.N_SCALARS + self.N_REGIME_DAILY
        self.day_projection = nn.Linear(enriched_dim, embed_dim)

        # ─────────────── TEMPORAL ATTENTION (unchanged from v11) ──────────────
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

        # ─────────────── REGIME ENCODER (new in v12) ─────────────────────────
        self.regime_encoder = RegimeEncoder(
            n_features=self.N_REGIME_HOURLY,
            hidden_dim=regime_hidden,
            dropout=dropout,
        )

        # ─────────────── FINAL FC (widened for regime embedding) ──────────────
        self.fc1 = nn.Linear(embed_dim + self.N_LAST_BAR + regime_hidden, fc_size)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(
        self,
        day_tokens: torch.Tensor,
        last_bar: torch.Tensor,
        regime_hourly: torch.Tensor,
    ) -> torch.Tensor:
        """
        day_tokens:    (B, n_days, 116) — v11 columns [0:110] + regime daily [110:116]
        last_bar:      (B, 4)
        regime_hourly: (B, 72, 4) — last 72h of VIX/DXY/GLD/USO at 1h
        returns logit: (B,)
        """
        B = day_tokens.shape[0]
        assert day_tokens.shape[1] == self.n_days
        assert day_tokens.shape[2] == self.DAY_TOKEN_WIDTH

        # ─── Parse day_tokens ───
        vp = day_tokens[:, :, 0:self.N_BINS]                                        # (B, D, 50)
        self_ch = day_tokens[:, :, self.N_BINS:2 * self.N_BINS]                      # (B, D, 50)
        candle = day_tokens[:, :, 2 * self.N_BINS:2 * self.N_BINS + self.N_CANDLE]   # (B, D, 8)
        scalars = day_tokens[:, :, 2 * self.N_BINS + self.N_CANDLE:
                             2 * self.N_BINS + self.N_CANDLE + self.N_SCALARS]       # (B, D, 2)
        regime_daily = day_tokens[:, :, self.DAY_TOKEN_WIDTH_V11:]                   # (B, D, 6)

        # ─── Stage 1: spatial attention over 50 bins (unchanged from v11) ───
        two_ch = torch.stack([vp, self_ch], dim=-1)                    # (B, D, 50, 2)
        flat = two_ch.reshape(B * self.n_days, self.N_BINS, 2)

        flat_tokens = self.bin_embed(flat)                             # (B*D, 50, 32)
        cls_spatial = self.spatial_cls.expand(B * self.n_days, -1, -1)
        flat_tokens = torch.cat([cls_spatial, flat_tokens], dim=1)     # (B*D, 51, 32)
        flat_tokens = flat_tokens + self.spatial_pos
        flat_attended = self.spatial_transformer(flat_tokens)
        spatial_cls_out = flat_attended[:, 0, :]                       # (B*D, 32)
        spatial_cls_out = spatial_cls_out.view(B, self.n_days, self.embed_dim)

        # ─── Enrich day tokens (now includes regime daily) ───
        enriched = torch.cat([spatial_cls_out, candle, scalars, regime_daily], dim=2)
        day_tok = self.day_projection(enriched)                        # (B, D, 32)

        # ─── Stage 2: temporal attention (unchanged from v11) ───
        cls_temporal = self.temporal_cls.expand(B, -1, -1)
        temporal_input = torch.cat([cls_temporal, day_tok], dim=1)     # (B, D+1, 32)
        temporal_input = temporal_input + self.temporal_pos
        temporal_out = self.temporal_transformer(temporal_input)
        pooled = temporal_out[:, 0, :]                                 # (B, 32)

        # ─── Regime encoder (new in v12) ───
        regime_emb = self.regime_encoder(regime_hourly)                # (B, 16)

        # ─── Final FC ───
        combined = torch.cat([pooled, last_bar, regime_emb], dim=1)    # (B, 52)
        h = self.dropout_layer(combined)
        h = self.relu(self.fc1(h))
        h = self.dropout_layer(h)
        logit = self.fc2(h).squeeze(-1)
        return logit
