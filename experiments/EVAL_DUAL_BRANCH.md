# Dual-Branch Transformer Evaluations

**Why:** The TemporalTransformer (best previous model) only sees aggregated VP. Candle features come from one bar (last hour) and 1h candles are mostly noise — too small to capture meaningful patterns. The user trades on 1d candle patterns (hammer, inverted hammer, engulfing) but the model never sees these.

**Strategy alignment:** High — mirrors how the user reads charts. VP gives ceiling/floor structure, daily candles give directional confirmation when price is mid-range. The dual-branch design keeps these signals separate (not entangled in one vector) and only merges them at the FC layer, just like a human combining the two readings mentally.

---

## Architecture

```
Input: (batch, 720, 68)   720 = 30 days × 24 hours, 68 features
                                              ↓
        ┌─────────────────────────────────────┴─────────────────────────────────────┐
        ↓                                                                            ↓
  ┌──────────────┐                                                          ┌──────────────────┐
  │  VP BRANCH   │                                                          │  CANDLE BRANCH   │
  └──────────────┘                                                          └──────────────────┘
  Sample end-of-day VP per day → (batch, 30, 50)                            Aggregate hourly OHLC into
       ↓                                                                    daily candle (rolling 24h):
  Stage 1: spatial attention                                                - day_open (first hour open)
  (50 bins, embed=32, 1 layer)                                              - day_close (last hour close)
  Shared weights across all 30 days                                         - day_high (max of 24h)
       ↓                                                                    - day_low (min of 24h)
  Mean pool → (batch, 30, 32)                                               - body, upper_wick, lower_wick
       ↓                                                                          ↓
  + temporal positional encoding                                            (batch, 30, 7)
       ↓                                                                          ↓
  Stage 2: VP temporal attention                                            Linear: 7 → 16
  (30 days, embed=32, 1 layer)                                                    ↓
       ↓                                                                    + temporal positional encoding
  Mean pool → (batch, 32)                                                         ↓
        ↓                                                                   Candle temporal attention
        │                                                                   (30 days, embed=16, 1 layer, 2 heads)
        │                                                                         ↓
        │                                                                   Mean pool → (batch, 16)
        │                                                                         ↓
        └───────────────────────────────────┬───────────────────────────────────┘
                                            ↓
                                Concat: VP(32) + Candle(16) + last bar's other features(18) = 66
                                            ↓
                                     FC → ReLU → Dropout → FC → logit
```

## Key design decisions

**Why two separate branches instead of one:**
The VP captures ceiling/floor structure. The candle captures directional momentum. Mixing them in one attention vector forces the model to entangle these signals. Separating them lets each branch specialize: VP attention learns peak persistence, candle attention learns multi-day pattern sequences. They only meet at the FC layer, mirroring how a trader mentally combines two independent readings.

**Why daily candles, not hourly:**
1h candles are noise. A 0.3% body with a 0.6% wick on 1h tells you nothing. The user only reads 1d candles for pattern detection. The model should see what the user sees.

**Why VP sampling (not aggregation):**
Each hourly row already contains a complete 180-day VP centered on that hour's close, sum=1 normalized. We don't aggregate 24 hourly VPs into a daily VP — we just **sample one per day** (the last hour, which has a VP centered on end-of-day close). This gives us 30 daily snapshots that match exactly what a trader sees when checking VP at end of day. No averaging artifacts, no per-day re-normalization needed, no magnitude distortion.

The 1h data resolution is just for sample generation (24x more training samples via sliding window). The actual VP is always a 180-day window.

**Why rolling 24h windows:**
Each hour, the model gets a "1-day candle ending now." This rolls forward by 1h between consecutive samples (matches the existing 1h prediction frequency). Computed from raw OHLC inside the model's forward pass — no new data needed.

**Why merge at FC layer (not earlier):**
Cross-attention between VP and candles would force premature interaction. Concatenating at FC lets each branch produce a clean summary first, then learns the conditional logic ("when VP says ceiling AND candle says rejection wick → strong sell") in the final layer.

**Why CLS token pooling instead of mean pool:**
Mean pool averages all token positions equally, which destroys position-aware information. For VP spatial attention, mean pool would lose "peak at bin 12 + peak at bin 38" — these become indistinguishable from a flat distribution with the same total mass. For temporal attention, mean pool treats day 1 and day 30 equally, destroying recency.

CLS tokens (the standard BERT/ViT approach) solve this:
1. Prepend a learnable [CLS] token at position 0
2. Self-attention runs as normal — CLS attends to all real tokens, they attend to it
3. Take only the CLS token's output as the summary
4. The CLS token learns to write "what's important about this sequence" into itself

Applied to all 3 attention stages:
- **Spatial CLS:** learns to summarize VP shape (peaks, gaps, distribution)
- **VP temporal CLS:** learns to summarize VP evolution (persistence, breakouts)
- **Candle temporal CLS:** learns to summarize candle pattern sequence (reversals, momentum)

Param cost: ~160 (3 CLS tokens + extended positional encodings). Negligible.

## Parameter budget

| Component | Params |
|-----------|--------|
| VP branch (with CLS pooling) | ~23,200 |
| Candle embed (7→16) | 128 |
| Candle positional encoding (incl. CLS) | 496 |
| Candle Transformer (1 layer, embed=16, 2 heads) | ~2,200 |
| Candle CLS token | 16 |
| Wider FC (concat 16 extra) | 1,048 |
| **Total** | **27,057** |

17% larger than TemporalTransformer. Still within data budget (75k samples / 27k params = 2.8x).

## Pipeline change

Added 3 new features in `compute_derived_features`:
- `ohlc_open_ratio = open / close`
- `ohlc_high_ratio = high / close`
- `ohlc_low_ratio = low / close`

These are scale-invariant (ratios) and let the model reconstruct daily OHLC from hourly bars without needing absolute prices.

## Evals

Pending — script ready, will run on Colab.

Results: `experiments/dualbranch_compare_results.json` (TBD)
Script: `src/models/eval_dualbranch.py`
Architecture: `src/models/architecture.py` → `DualBranchTransformerClassifier`
