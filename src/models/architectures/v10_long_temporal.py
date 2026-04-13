"""v10 — TemporalEnrichedV10: v6-prime architecture with 90-day temporal window.

── Provenance ──────────────────────────────────────────────────────────────
Subclass of TemporalEnrichedV6Prime with two defaults changed:
  - n_days: 30 → 90 (temporal transformer attends over 90 day tokens)
  - Intended CSV source: BTC_1h_RELVP_30d.csv (VP lookback shortened 180d → 30d)

No architectural changes. Same single-path design, same embed_dim, same
attention layers. The only new parameters come from the larger temporal
positional embedding (91 × 32 vs 31 × 32): +1,920 params (~8% larger).

── Rationale ───────────────────────────────────────────────────────────────
Matches how the user reads charts: recent-volume zoom (30d VP per row)
paired with longer price-action context (90 days of day tokens). Also
shrinks warmup loss: 180d+30d=210d → 30d+90d=120d, gaining ~2,100 usable
training rows vs the v6-prime configuration.

See arch-ml-model.md §v10 for the full design note.
"""

from src.models.architectures.v6_prime_vp_labels import TemporalEnrichedV6Prime


class TemporalEnrichedV10(TemporalEnrichedV6Prime):
    """v6-prime with n_days=90 by default."""

    def __init__(self, *args, n_days: int = 90, **kwargs):
        super().__init__(*args, n_days=n_days, **kwargs)
