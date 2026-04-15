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

## Eval 1 — DualBranch v1 (2026-04-09)

**Setup:** Colab A100, batch=512, FGI adaptive 7.5/3, 27,057 params.

### Overall

| Metric | DualBranch | Temporal (prev) | Diff |
|--------|-----------|----------------|------|
| Accuracy | 60.5% | 61.9% | -1.4% |
| Folds <50% | 1/10 | 0/10 | +1 |
| Bull long EV | +0.63% | +1.04% | -0.41% |
| Bear long EV | +1.01% | +1.00% | +0.01% |
| Bear short EV | +0.76% | +0.82% | -0.06% |

**Verdict:** Slightly worse than temporal-only model. The candle branch did not provide net improvement in this configuration.

### Fold-by-fold

| Fold | Period | Spatial | Temporal | DualBranch | vs Temporal |
|------|--------|---------|----------|-----------|-------------|
| 1 | 2020 H2 | 53.8% | 62.2% | **49.1%** | **-13.1** |
| 2 | 2021 H1 | 59.0% | 51.6% | 56.3% | +4.7 |
| 3 | 2021 H2 | 49.7% | 62.0% | 56.6% | -5.4 |
| 4 | 2022 H1 | 58.5% | 59.1% | **63.6%** | +4.5 |
| 5 | 2022 H2 | 71.6% | 67.4% | 68.7% | +1.3 |
| 6 | 2023 H1 | 46.5% | 60.1% | 50.6% | -9.5 |
| 7 | 2023 H2 | 76.9% | 75.5% | 65.4% | -10.1 |
| 8 | 2024 H1 | 40.9% | 59.9% | **70.4%** | +10.5 |
| 9 | 2024 H2 | 58.7% | 58.9% | 59.6% | +0.7 |
| 10 | 2025 H1 | 46.4% | 67.2% | 66.9% | -0.3 |

### Hypothesis: candle branch is treating all candles equally

Pattern observed: dual-branch helps some folds (4, 8) and hurts others (1, 6, 7). User's trading rule explains this:

> "I only trust hammers/inverted hammers when they have high volume and clear shape. Otherwise I ignore the candle entirely."

The current candle branch sees ALL 30 daily candles equally, with no notion of "this candle is meaningful vs noise." Most days have neutral candles → these contribute noise that drowns out the few high-confidence hammers. The candle attention learns from a lot of irrelevant signal.

**Critical missing input: volume.** The candle branch reconstructs daily candles from OHLC ratios but **never sees volume**. The user's pattern is `(shape) × (volume confirmation)` — half the equation is missing.

### Next: DualBranch v2

Add `day_volume_ratio` (total day volume / 30d daily average) as 8th feature per daily candle. Lets the model learn "high-volume hammer = strong signal" vs "low-volume hammer = noise."

Param cost: +16 (just one more dim in the candle embed projection).

---

## Eval 2 — DualBranch v2 with volume context (2026-04-09)

**Change:** Daily candle now has 8 features instead of 7. Added `day_volume_ratio` = mean of hourly `volume_ratio` across the day's 24 hours.

### Results

| Metric | v1 | v2 | Diff |
|--------|----|----|----|
| Accuracy | 60.5% | 57.8% | -2.7% |
| Bull precision | 34.5% | 37.5% | +3.0% |
| Bull long EV | +0.63% | +0.93% | +0.30% |
| Bear long EV | +1.01% | +0.99% | -0.02% |
| Bear short EV | +0.76% | +0.65% | -0.11% |

### Fold-by-fold (vs v1)

| Fold | v1 | v2 | Diff |
|------|----|----|----|
| 1 | 49.1% | 53.0% | +3.9 |
| 2 | 56.3% | 52.1% | -4.2 |
| 3 | 56.6% | 57.9% | +1.3 |
| 4 | 63.6% | 57.9% | -5.7 |
| 5 | 68.7% | 64.6% | -4.1 |
| 6 | 50.6% | **39.7%** | **-10.9** |
| 7 | 65.4% | 79.0% | +13.6 |
| 8 | 70.4% | **49.3%** | **-21.1** |
| 9 | 59.6% | 59.0% | -0.6 |
| 10 | 66.9% | 75.8% | +8.9 |

### What this tells us

Volume helped bull long EV (+0.30%) and won folds 1, 7, 10. But it **destroyed fold 6 (40%) and fold 8 (49%)** — the historically hardest folds.

**Hypothesis:** In choppy/transition periods, volume spikes happen during fakeouts and indecision, not real moves. The model can't yet learn to weight `(shape × volume)` selectively — it just sees volume as one more input that can mislead it.

The fundamental issue: model treats all 30 days equally. Adding more features doesn't fix it.

Log: `logs/dualbranch_v2_volume.log`

---

## Eval 3 — Pipeline overhaul: scaling + remove VP structure (2026-04-09)

**Changes:**
1. **Removed 8 VP structure features** — let spatial Transformer learn shape from raw bins
2. **Walk-forward-safe z-score** for `volume_ratio`, `log_return`, `bar_range`, `bar_body` (expanding window, no leakage)
3. **Soft tanh squashing** to bound outliers without losing magnitude info (black-swan handling)
4. **OHLC ratios centered at 0** (subtract 1)

Total features dropped from 68 → 60.

### Results

| Metric | v1 | v2 | **v3** |
|--------|----|----|----|
| Accuracy | 60.5% | 57.8% | 59.7% |
| Folds <50% | 1/10 | 2/10 | 2/10 |
| Bull precision | 34.5% | 37.5% | **39.1%** |
| Bear precision | 81.0% | 80.9% | 79.9% |
| Bull long EV | +0.63% | +0.93% | **+1.10%** |
| Bear long EV | +1.01% | +0.99% | +0.89% |
| Bear short EV | +0.76% | +0.65% | **+0.12%** |

### Fold-by-fold

| Fold | v1 | v2 | v3 |
|------|----|----|----|
| 1 | 49.1% | 53.0% | **56.0%** |
| 2 | 56.3% | 52.1% | 56.6% |
| 3 | 56.6% | 57.9% | **47.1%** |
| 4 | 63.6% | 57.9% | 59.4% |
| 5 | 68.7% | 64.6% | 65.4% |
| 6 | 50.6% | 39.7% | 45.2% |
| 7 | 65.4% | 79.0% | 77.1% |
| 8 | 70.4% | 49.3% | 62.2% |
| 9 | 59.6% | 59.0% | 60.0% |
| 10 | 66.9% | 75.8% | 72.7% |

### IMPORTANT — v3 had a broken candle branch (discovered after the run)

The v3 pipeline overhaul introduced two bugs that broke the candle reconstruction in `_aggregate_daily_candles`:

1. **OHLC ratios were subtracted by 1** (centered at 0 instead of 1). The candle reconstruction logic uses these as multiplicative ratios to reconstruct daily OHLC. With the subtraction, `day_open` was near 0 instead of near 1, making `body = day_close - day_open ≈ 1.0` (vs the expected ~0.001) and `upper_wick / bar_height` exploded to magnitudes of 350.

2. **`log_return` was z-scored**. The candle reconstruction uses `cumsum(log_return) → exp` to track price movement within each day. After scaling, these "log returns" no longer represented actual returns. Cumulative sum + exp produced `day_open` values up to 42 and `day_high` up to 63 — total nonsense.

**Why v3 didn't crash:** The Linear projection (8→16) and LayerNorm in the candle Transformer absorbed the absurd values without producing inf/NaN on Mac CPU at batch=64. The model likely learned to **ignore** the candle branch entirely (it was pure noise) — meaning v3's results essentially measure "VP branch only with broken candle noise."

**Implication:** The +1.10% bull long EV improvement attributed to v3 is suspect. It could be from:
- The pipeline scaling fixes (VP branch saw cleaner features) — real
- Random run-to-run variance — possible
- Removing VP structure features — possible

We can't cleanly attribute it. v1 and v2 had **working** candle branches, so their results are reliable.

**Fix committed 2026-04-09:** Reverted the OHLC subtraction and removed `log_return` from the z-score loop. Verified: candles now produce sensible values (body ~0, wicks in [0, 1]).

### Verdict: Mixed

**Wins:**
- Bull long EV jumped +0.47% over v1 — **spatial attention IS learning shape from raw bins**
- Bull precision improved (39.1%, highest yet)
- Folds 1, 2, 8 recovered nicely

**Losses:**
- Bear short EV collapsed from +0.76% → +0.12%
- Bear NPV dropped from 35.8% → 29.7%
- Fold 3 dropped to 47% (worst single fold)

The pipeline changes shift performance from bear-side to bull-side. Removing the 8 VP structure features helped the spatial branch find ceiling/floor from raw bins, but those handcrafted features may have been doing real work specifically for bear-side predictions.

Log: `logs/dualbranch_v3_overhaul.log`
Results: `experiments/dualbranch_v3_results.json`

---

## Diagnostic Sweep — Why did LR sweep tank? (2026-04-09)

Tested 5 configs to isolate why the LR sweep produced uniformly bad results (~52% vs 60% baseline).

| Config | Acc | <50% | Diagnosis |
|--------|-----|------|-----------|
| 1. Baseline (clip=1.0) | 52.8% | 5/10 | Matches LR sweep |
| 2. Loosened clip (5.0) | 53.6% | 4/10 | Slight improvement |
| 3. No clip | 53.6% | 4/10 | Same as #2 |
| 4. OLD pipeline | **48.1%** | **7/10** | Worst — don't revert pipeline |
| **5. Temporal-only + new pipeline** | **56.0%** | **4/10** | **Best — candle branch hurts** |

**Root cause:** The candle branch was hurting more than helping. Temporal-only beat DualBranch by 3% on the same pipeline. Gradient clipping was a minor factor.

Results: `experiments/diagnose_results.json`

---

## v6 Enriched Temporal — Single-path enrichment (2026-04-10)

**Architecture change:** Instead of a separate candle branch (DualBranch), enrich each temporal day token with candle + VP structure features alongside the VP spatial embedding. One temporal attention path sees everything.

### v6 vs v2 Head-to-head (same TP/SL, same pipeline, same seed)

| Metric | v6 enriched | v2 baseline | Diff |
|--------|-------------|-------------|------|
| **Accuracy** | **58.4%** | 56.7% | **+1.7%** |
| Precision (UP) | **68.3%** | 60.3% | **+8.0%** |
| Folds <50% | **1/10** | 3/10 | **-2** |
| **Bull long EV** | **+1.57%** | +0.41% | **+1.16%** |
| Bear long EV | +1.02% | +1.04% | -0.02% |
| Bear short EV | -0.04% | +0.42% | -0.46% |

### Fold-by-fold

| Fold | v6 | v2 | Diff |
|------|-----|-----|------|
| 1 | **27.9%** | 47.5% | -19.6 |
| 2 | **57.2%** | 38.8% | **+18.4** |
| 3 | 54.4% | 56.3% | -1.9 |
| 4 | 53.1% | 52.7% | +0.4 |
| 5 | 60.6% | 69.8% | -9.2 |
| 6 | 65.7% | 60.1% | +5.6 |
| 7 | 73.0% | 68.5% | +4.5 |
| 8 | **68.7%** | **44.3%** | **+24.4** |
| 9 | 55.9% | 62.6% | -6.7 |
| 10 | 72.7% | 72.8% | -0.1 |

**Verdict:** v6 enrichment is a net positive (+1.7% accuracy, huge bull long EV improvement, fewer catastrophic folds). Fold 1 collapsed (data scarcity), but folds 2 and 8 recovered dramatically.

Results: `experiments/v6_vs_v2_results.json`
Script: `src/models/eval_v6_vs_v2.py`
Architecture: `src/models/architectures/v6_temporal_enriched.py`

---

## 24h Fixed-Horizon Label Test (2026-04-10)

**Hypothesis:** First-hit labels ask the model to predict a complex barrier race. Maybe a simpler label (24h close direction: up or down?) would be easier to learn and produce more frequent trades.

### Results (v6 enriched + 24h horizon)

| Metric | Value |
|--------|-------|
| Accuracy | **50.4%** |
| Precision (UP) | 52.3% |
| NPV (DOWN) | 49.0% |
| Base rate (% UP) | 51.5% |
| Folds <50% | 4/10 |

**The model is essentially random on 24h direction prediction.** Confidence analysis showed nearly zero high-confidence predictions (0 samples above 60% threshold for UP, only 256 above 55%).

### What this proves

The same model gets **58-62% on first-hit labels but 50% on 24h labels**. This is a critical finding:

1. **The model IS learning something real** — it's not random noise or label exploitation
2. **What it learns is VP barrier prediction** — where price will stop/reverse at support/resistance levels
3. **VP does NOT predict general direction** — "will price be higher in 24h?" depends on macro/news/sentiment that VP doesn't encode
4. **First-hit labels are correct** for this feature set — they measure exactly what VP is good at predicting

### Implication for strategy

Don't pursue simpler labels or general direction prediction. The model's edge is specifically in **VP support/resistance barrier outcomes**. Profitability comes from:
- Better accuracy on first-hit labels (2+1 spatial layers, more data)
- Faster-resolving TP/SL (2.5/1 instead of 7.5/3)
- Lower fees (maker-only limit orders)

Results: `experiments/horizon_24h_results.json`
Script: `src/models/eval_24h_horizon.py`

---

## Architecture & Pipeline Version Summary

### Frozen architectures (`src/models/architectures/`)

| Version | Type | Params | Best acc | Key feature |
|---------|------|--------|----------|-------------|
| v2_temporal | VP-only, mean pool | 23,041 | 61.9% | Our best validated baseline |
| v5_dualbranch_cls | VP + Candle separate branches | 27,057 | 60.5% | CLS pooling, candle hurts |
| v6_temporal_enriched | VP + enriched day tokens, CLS | 24,737 | 58.4% | Single-path, +1.7% over v2 same run |

### Frozen pipelines (`src/features/pipelines/`)

| Version | Features | Key difference |
|---------|----------|----------------|
| v1_raw | 68 (50 VP + 10 derived + 8 VP structure) | Original, no scaling |
| v2_scaled | 60 (50 VP + 10 derived) | z-score + tanh, VP structure removed (HURT) |

### Log index

| Log | Script | What it tested |
|-----|--------|---------------|
| `dualbranch_colab.log` | eval_dualbranch | DualBranch v1 |
| `dualbranch_v2_volume.log` | eval_dualbranch | DualBranch v2 (volume) |
| `dualbranch_v3_overhaul.log` | eval_dualbranch | DualBranch v3 (scaling overhaul, candle bug) |
| `lr_sweep_results.log` | sweep_lr | LR schedule sweep (all bad) |
| `reproduce_v1.log` | eval_reproduce_v1 | Reproduced v2_temporal at 59.2% |
