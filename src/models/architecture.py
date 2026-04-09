import torch
import torch.nn as nn

from src.config import (
    FEATURE_COLS,
    RNN_HIDDEN_SIZES,
    RNN_DROPOUT,
    RNN_ACTIVATION,
    LSTM_HIDDEN_SIZES,
    LSTM_DROPOUT,
    REL_BIN_COUNT,
    DERIVED_FEATURE_COLS,
    VP_STRUCTURE_COLS,
    CNN_CHANNELS,
    CNN_KERNEL_SIZE,
    CNN_FC_SIZE,
    CNN_DROPOUT,
)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = len(FEATURE_COLS),
        hidden_sizes: list[int] = RNN_HIDDEN_SIZES,
        dropout: float = RNN_DROPOUT,
        activation: str = RNN_ACTIVATION,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.RNN(
                input_size=prev_size,
                hidden_size=h,
                num_layers=1,
                nonlinearity=activation,
                batch_first=True,
            ))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = h

        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback, features]
        out = x
        for rnn, drop in zip(self.layers, self.dropouts):
            out, _ = rnn(out)
            out = drop(out)

        # Take the last timestep's hidden state
        last = out[:, -1, :]
        logit = self.fc(last).squeeze(-1)
        return logit


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = len(FEATURE_COLS),
        hidden_sizes: list[int] = LSTM_HIDDEN_SIZES,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.LSTM(
                input_size=prev_size,
                hidden_size=h,
                num_layers=1,
                batch_first=True,
            ))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = h

        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for lstm, drop in zip(self.layers, self.dropouts):
            out, _ = lstm(out)
            out = drop(out)

        last = out[:, -1, :]
        logit = self.fc(last).squeeze(-1)
        return logit


N_VP_BINS = REL_BIN_COUNT  # 50
N_OTHER_FEATURES = len(DERIVED_FEATURE_COLS) + len(VP_STRUCTURE_COLS)

# OHLC ratio indices within DERIVED_FEATURE_COLS (relative to VP bins)
# Used by dual-branch transformer to extract per-day candle features
OHLC_OPEN_IDX = N_VP_BINS + DERIVED_FEATURE_COLS.index("ohlc_open_ratio")
OHLC_HIGH_IDX = N_VP_BINS + DERIVED_FEATURE_COLS.index("ohlc_high_ratio")
OHLC_LOW_IDX = N_VP_BINS + DERIVED_FEATURE_COLS.index("ohlc_low_ratio")


class TransformerClassifier(nn.Module):
    """Tiny Transformer that uses self-attention across VP bins.

    Each VP bin attends to all other bins to find peak pairs (ceiling/floor).
    Single attention layer, single head, small embedding — keeps params low for ~14k samples.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        fc_size: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Project each VP bin (scalar) to embed_dim
        self.bin_embed = nn.Linear(1, embed_dim)

        # Learnable positional encoding for 50 bins
        self.pos_embed = nn.Parameter(torch.randn(1, N_VP_BINS, embed_dim) * 0.02)

        # Stacked transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool attention output + combine with non-VP features
        self.fc1 = nn.Linear(embed_dim + N_OTHER_FEATURES, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, features)
        vp_bins = x[:, :, :N_VP_BINS]        # (batch, lookback, 50)
        other = x[:, -1, N_VP_BINS:]          # (batch, N_OTHER)

        # Aggregate VP across lookback window
        agg_vp = vp_bins.sum(dim=1)           # (batch, 50)

        # Normalize per sample
        vp_max = agg_vp.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        agg_vp = agg_vp / vp_max

        # Reshape: each bin becomes a token → (batch, 50, 1)
        tokens = agg_vp.unsqueeze(-1)

        # Embed + positional encoding
        tokens = self.bin_embed(tokens) + self.pos_embed  # (batch, 50, embed_dim)

        # Self-attention across bins
        attended = self.transformer(tokens)   # (batch, 50, embed_dim)

        # Global average pool across bins
        pooled = attended.mean(dim=1)         # (batch, embed_dim)

        # Combine with non-VP features
        combined = torch.cat([pooled, other], dim=1)

        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit


class TemporalTransformerClassifier(nn.Module):
    """Two-stage Transformer: spatial attention per day + temporal attention across days.

    Stage 1 (spatial): For each of 30 days, sum 24 hourly VP bins into a daily VP,
        then run self-attention across 50 bins. Produces one embedding per day.
        Shared weights across all days.
    Stage 2 (temporal): Self-attention across 30 daily embeddings.
        Learns VP persistence, breakouts, evolving support/resistance.

    Non-VP features: taken from the last timestep (same as current model).
    """

    def __init__(
        self,
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
        self.dropout = nn.Dropout(dropout)
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

        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit


class DualBranchTransformerClassifier(nn.Module):
    """Dual-branch Transformer: VP branch + Candle branch, merged at FC layer.

    VP branch (same as TemporalTransformerClassifier):
        - Stage 1: spatial attention across 50 VP bins per day (shared weights)
        - Stage 2: temporal attention across 30 daily VP embeddings
        - Output: (batch, vp_embed_dim)

    Candle branch (NEW):
        - Aggregates 24 hourly bars into 1 daily candle: (open, high, low, close)
        - Computes 7 candle features per day: open_ratio, close_ratio, high_ratio,
          low_ratio, body, upper_wick, lower_wick
        - Project 7 → candle_embed_dim (16)
        - Temporal attention across 30 daily candle embeddings
        - Output: (batch, candle_embed_dim)

    Merging:
        concat(vp_pooled, candle_pooled, last_other_features) → FC → logit

    The two branches don't talk to each other until the FC layer, mirroring
    how a trader reads VP and candles separately then combines mentally.
    """

    def __init__(
        self,
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

        # ─────────────── VP BRANCH ───────────────
        # Stage 1: spatial attention across VP bins (50 bins + 1 CLS = 51 tokens)
        self.bin_embed = nn.Linear(1, vp_embed_dim)
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, vp_embed_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, N_VP_BINS + 1, vp_embed_dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=vp_embed_dim,
            nhead=n_heads,
            dim_feedforward=vp_embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=n_spatial_layers)

        # Stage 2: temporal attention across days for VP (30 days + 1 CLS = 31 tokens)
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
        # 7 candle features per day → embed_dim, with CLS for temporal pooling
        self.candle_embed = nn.Linear(7, candle_embed_dim)
        self.candle_cls = nn.Parameter(torch.randn(1, 1, candle_embed_dim) * 0.02)
        self.candle_pos = nn.Parameter(torch.randn(1, n_days + 1, candle_embed_dim) * 0.02)
        candle_layer = nn.TransformerEncoderLayer(
            d_model=candle_embed_dim,
            nhead=2,  # smaller embed_dim, fewer heads
            dim_feedforward=candle_embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.candle_transformer = nn.TransformerEncoder(candle_layer, num_layers=n_candle_layers)

        # ─────────────── MERGE ───────────────
        # Concat: VP pooled + Candle pooled + last bar's non-VP features
        merged_dim = vp_embed_dim + candle_embed_dim + N_OTHER_FEATURES
        self.fc1 = nn.Linear(merged_dim, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, 1)

    def _aggregate_daily_candles(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate 24 hourly bars per day into 1 daily candle (7 features per day).

        Args:
            x: (batch, 720, features) — full input

        Returns:
            (batch, 30, 7) — daily candle features per day
        """
        batch_size = x.shape[0]
        # Extract OHLC ratios (relative to each hour's close)
        # Shape: (batch, 720)
        open_r = x[:, :, OHLC_OPEN_IDX]
        high_r = x[:, :, OHLC_HIGH_IDX]
        low_r = x[:, :, OHLC_LOW_IDX]

        # Reshape to days: (batch, 30, 24)
        open_r = open_r.view(batch_size, self.n_days, self.bars_per_day)
        high_r = high_r.view(batch_size, self.n_days, self.bars_per_day)
        low_r = low_r.view(batch_size, self.n_days, self.bars_per_day)
        # close ratio = 1.0 by definition (each hour's close is the reference)
        # but we need actual price ratios across the day, so reconstruct from close
        # Use log_return to track close changes
        log_ret = x[:, :, N_VP_BINS + DERIVED_FEATURE_COLS.index("log_return")]
        log_ret = log_ret.view(batch_size, self.n_days, self.bars_per_day)

        # For each day, compute candle features RELATIVE to that day's close
        # Day's close = close of last hour of that day
        # Day's open = first hour's open = first_hour_close * first_hour_open_ratio
        # We use the hour-level open_ratio (open_t / close_t) as a proxy

        # Daily aggregation: open from first hour, high/low extremes, close = last hour
        # All ratios are normalized relative to that day's close (last hour's close)
        # Reconstruct: cumulative log returns from first hour of each day
        day_cumret = log_ret.cumsum(dim=2)  # (batch, 30, 24)
        # day's close ratio = exp(0) = 1 for last hour by definition
        # day's first hour close (relative to last hour close) = exp(-day_cumret[:, :, -1])
        # So price at hour h relative to day's close = exp(day_cumret[:, :, h] - day_cumret[:, :, -1])
        rel_to_day_close = torch.exp(day_cumret - day_cumret[:, :, -1:])  # (batch, 30, 24)

        # Day's open: first hour's open
        # first_hour_open / last_hour_close = first_hour_open/first_hour_close * first_hour_close/last_hour_close
        day_open = open_r[:, :, 0] * rel_to_day_close[:, :, 0]  # (batch, 30)
        # Day's close: by construction = 1.0
        day_close = torch.ones_like(day_open)
        # Day's high: max across hours of (high_ratio * rel_to_day_close)
        day_high = (high_r * rel_to_day_close).max(dim=2).values
        # Day's low: min across hours
        day_low = (low_r * rel_to_day_close).min(dim=2).values

        # Candle shape features
        bar_height = (day_high - day_low).clamp(min=1e-8)
        body = day_close - day_open
        upper_wick = (day_high - torch.max(day_open, day_close)) / bar_height
        lower_wick = (torch.min(day_open, day_close) - day_low) / bar_height
        body_dir = torch.sign(body)

        # Stack: (batch, 30, 7)
        candle = torch.stack([
            day_open, day_close, day_high, day_low,
            body, upper_wick, lower_wick,
        ], dim=2)
        # Replace any NaN/inf with 0 (defensive)
        candle = torch.nan_to_num(candle, nan=0.0, posinf=0.0, neginf=0.0)
        return candle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback=720, features)
        batch_size = x.shape[0]
        vp_bins = x[:, :, :N_VP_BINS]        # (batch, 720, 50)
        other = x[:, -1, N_VP_BINS:]          # (batch, N_OTHER) — last bar's other features

        # ─────────────── VP BRANCH ───────────────
        # Reshape and sum hourly VP into daily VP
        vp_daily = vp_bins.view(batch_size, self.n_days, self.bars_per_day, N_VP_BINS)
        vp_daily = vp_daily.sum(dim=2)                             # (batch, 30, 50)

        # Normalize per day
        day_max = vp_daily.max(dim=2, keepdim=True).values.clamp(min=1e-8)
        vp_daily = vp_daily / day_max

        # Stage 1: spatial attention (batched across days) with CLS pooling
        flat_vp = vp_daily.reshape(batch_size * self.n_days, N_VP_BINS, 1)
        flat_tokens = self.bin_embed(flat_vp)                              # (batch*30, 50, embed)
        # Prepend CLS token
        cls_spatial = self.spatial_cls.expand(batch_size * self.n_days, -1, -1)
        flat_tokens = torch.cat([cls_spatial, flat_tokens], dim=1)         # (batch*30, 51, embed)
        flat_tokens = flat_tokens + self.spatial_pos                       # add positional encoding
        flat_attended = self.spatial_transformer(flat_tokens)              # (batch*30, 51, embed)
        # Take CLS output (position 0) as the day's summary
        flat_cls_out = flat_attended[:, 0, :]                              # (batch*30, embed)
        vp_temporal_input = flat_cls_out.view(batch_size, self.n_days, self.vp_embed_dim)

        # Stage 2: temporal attention across VP days with CLS pooling
        cls_vp_temp = self.vp_temporal_cls.expand(batch_size, -1, -1)
        vp_temporal_input = torch.cat([cls_vp_temp, vp_temporal_input], dim=1)  # (batch, 31, embed)
        vp_temporal_input = vp_temporal_input + self.vp_temporal_pos
        vp_temporal_out = self.vp_temporal_transformer(vp_temporal_input)
        vp_pooled = vp_temporal_out[:, 0, :]                               # CLS token (batch, embed)

        # ─────────────── CANDLE BRANCH ───────────────
        candles = self._aggregate_daily_candles(x)                         # (batch, 30, 7)
        candle_tokens = self.candle_embed(candles)                         # (batch, 30, candle_embed)
        # Prepend CLS token
        cls_candle = self.candle_cls.expand(batch_size, -1, -1)
        candle_tokens = torch.cat([cls_candle, candle_tokens], dim=1)      # (batch, 31, candle_embed)
        candle_tokens = candle_tokens + self.candle_pos
        candle_attended = self.candle_transformer(candle_tokens)
        candle_pooled = candle_attended[:, 0, :]                           # CLS token (batch, candle_embed)

        # ─────────────── MERGE ───────────────
        combined = torch.cat([vp_pooled, candle_pooled, other], dim=1)
        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit


class CNNClassifier(nn.Module):
    """1D CNN that treats the aggregated VP profile as a spatial signal.

    Forward pass:
    1. Sum the 50 VP bins across all timesteps → one (batch, 50) aggregated profile
    2. Run 1D conv filters across the 50 bins to detect peaks/shapes
    3. Concatenate conv output with non-VP features from the last timestep
    4. FC layer → logit
    """

    def __init__(
        self,
        channels: list[int] = CNN_CHANNELS,
        kernel_size: int = CNN_KERNEL_SIZE,
        fc_size: int = CNN_FC_SIZE,
        dropout: float = CNN_DROPOUT,
    ):
        super().__init__()

        # Conv layers over the 50 VP bins
        conv_layers = []
        in_ch = 1  # single-channel: the aggregated VP profile
        for out_ch in channels:
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        # After conv: (batch, last_channel, 50) → flatten → (batch, last_channel * 50)
        conv_out_size = channels[-1] * N_VP_BINS + N_OTHER_FEATURES

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_out_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, features) where features = 50 VP + 11 other
        vp_bins = x[:, :, :N_VP_BINS]        # (batch, lookback, 50)
        other = x[:, -1, N_VP_BINS:]          # (batch, 11) — last timestep only

        # Aggregate VP across lookback window
        agg_vp = vp_bins.sum(dim=1)           # (batch, 50)

        # Normalize per sample
        vp_max = agg_vp.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        agg_vp = agg_vp / vp_max

        # Reshape for Conv1d: (batch, 1, 50)
        agg_vp = agg_vp.unsqueeze(1)

        # Conv layers
        conv_out = self.conv(agg_vp)          # (batch, channels[-1], 50)

        # Flatten to preserve spatial position
        flat = conv_out.flatten(1)            # (batch, channels[-1] * 50)

        # Concatenate with non-VP features
        combined = torch.cat([flat, other], dim=1)

        # FC layers
        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logit = self.fc2(out).squeeze(-1)
        return logit
