# Model Evolution History

Living document tracking every architecture decision, result, and the reasoning that led to each change. Updated continuously.

---

## 1. Vanilla RNN (4 evals, exhausted)

**Why built:** Establish baseline. RNN is the default starting point for time-series classification.

**Architecture:** 3-layer stacked RNN (64→32→16 hidden), tanh activation, 0.2 dropout. Input: 180 timesteps × features.

**Result:** ~51% accuracy (coin flip). Model collapsed to predicting majority class 100% of the time.

**What went wrong:** Vanishing gradients. 180 timesteps is too long for vanilla RNN — gradient signal dies before reaching early timesteps. Changing learning rate (1e-3 to 1e-4) and lookback (42 to 180 bars) made zero difference.

**Decision → LSTM:** Gated architecture should solve vanishing gradients.

---

## 2. LSTM (7 evals, exhausted)

**Why built:** LSTM's forget/input/output gates + cell state bypass solve vanishing gradients by design.

**Architecture:** Stacked LSTM cells. Iterated from 64→32→16 (overparameterized) down to 16→8. Added weighted BCEWithLogitsLoss (pos_weight) to fix majority-class collapse.

**Result:** 53.5% best val accuracy. Adding VP structure features (Gaussian-smoothed peaks, ceiling/floor distance, consistency) didn't help — stuck at 50-52%.

**What went wrong:** VP features are inherently **spatial** (relationships across 50 bins), but LSTM processes **temporally** (one timestep at a time). The model sees bin values one bar at a time — it can never learn "peak at bin 12 relates to peak at bin 38" because it processes each bar's 50 bins as a flat vector, not as a spatial distribution.

**Key insight:** We need spatial reasoning across VP bins, not temporal reasoning across bars.

**Decision → CNN:** Convolutional filters naturally detect spatial patterns (peaks, gaps, shapes) in the VP histogram.

---

## 3. 1D CNN (26 evals across 8 phases, best: 61.5%)

**Why built:** Treat the 30-day aggregated VP as a 1D spatial signal. Conv filters learn smoothing and peak detection across bin positions — directly mirrors how the user reads VP visually.

**Architecture:** Sum VP across all timesteps → (batch, 1, 50). Conv1d(1→4, k=5) → Conv1d(4→4, k=5) → flatten → concat non-VP features → FC → logit.

### Phase progression (key decisions only):

**Phase 1 — Fixed 24h labels:** ~53% accuracy. Labels too noisy — "will price be higher in 24h?" depends on news/sentiment, not VP structure.

**Phase 2 — First-hit TP/SL labels:** Switched to "does price hit +3% before -3%?" → jumped to **57.3%**. This measures barrier prediction, which is what VP actually predicts. **Foundational decision that carried through all future models.**

**Phase 3 — Asymmetric TP/SL:** Found 1:2 ratio (TP=2.5%, SL=5%) → 74.7% val accuracy on single split. But this was misleading — the asymmetry inherently favors UP predictions.

**Phase 4 — Walk-forward validation:** Single-split results were overly optimistic. Walk-forward (10 folds, 6-month test periods) revealed the truth: 54.5% overall. Some folds catastrophic (Fold 5: 27.4% in bear market). The asymmetric TP/SL failed in bear regimes.

**Phase 5 — Regime-adaptive TP/SL:** Flipped TP/SL ratio based on SMA(90d) regime detection. Bull: TP=2.5% SL=5%. Bear: TP=5% SL=2.5%. **Jumped to 61.2%** and eliminated the 27% bear-market disaster. Best single fold: 88%. **This was the second foundational decision.**

**Phases 6-8 — Tuning:** Iterated VP smoothing (sigma 2.0→1.0→0.8), neutral-zone filtering, L2 regularization. Marginal gains. Best CNN: **61.5% accuracy, 65.7% precision** (4h data, sigma=0.8).

**What went wrong at the ceiling:** CNN's fixed-size conv filters (k=5) can only see 5 adjacent bins at a time. Learning that "peak at bin 12 and peak at bin 38 form a ceiling/floor pair" requires stacking many layers. Self-attention can learn this in one layer.

**Decision → Transformer:** Self-attention lets every bin attend to every other bin directly. Better for finding peak pairs (ceiling/floor).

---

## 4. Spatial Transformer (Evals 1-4, best: 63.3%)

**Why built:** Self-attention across VP bins can directly learn "peak at bin 12 relates to peak at bin 38" without stacking conv layers. Matches how the user visually scans VP for related peaks.

**Architecture:** Sum VP across lookback → embed each of 50 bins (Linear 1→embed_dim) → learnable positional encoding → TransformerEncoder → global avg pool → concat non-VP features → FC → logit.

**Label:** Regime-adaptive first-hit TP=2.5%/SL=5%, vol-scaled. FGI regime (from Eval 3 onwards).

### Key experiments:

**Eval 1 (4h, embed=8, 1.8k params):** 56.9%. Too small, underfitting. Wild variance (Fold 8: 25.6%).

**Eval 2 (4h, embed=16, 4.1k params):** 61.3%. Tied CNN accuracy but **higher precision (67.2% vs 65.7%)**. Fold 7 best single fold ever: 88.2%.

**Eval 3 (1h, embed=16, 4.1k params):** 58.7%. Same model on 1h data — worse. Too small for the 4x more data.

**Eval 4 (1h, embed=32, heads=4, 2 layers, 21.9k params): 63.3% accuracy, 65.8% precision.** NEW BEST. Only 3 folds below 50% (was 4). The combination of 6x more training data (1h vs 4h) and a model big enough to use it broke past the 4h ceiling.

**Key insight:** 1h data is better than 4h — more training samples let the Transformer learn more robustly. Two attention layers can learn peak-pair relationships (layer 1 finds peaks, layer 2 finds pairs).

**Limitation:** No temporal context. The 720-bar lookback is collapsed by summation — the model can't see VP evolution over time (persistence, breakouts, shifting support/resistance).

**Decision → Temporal models:** Add temporal attention across daily VP snapshots.

---

## 5. TP/SL Sweep & Strategy Analysis

**Why:** Before changing architecture, optimize the label design. The 2.5/5 TP/SL from Eval 4 had negative EV despite 63.3% accuracy (wins only +2.5%, losses cost -5%).

**Result:** 7.5/3 TP/SL is optimal. Even at lower accuracy (~58%), the EV per trade is strongly positive. Strategy 4 (long in bull + both sides in bear) best overall.

**Key finding:** Accuracy alone is misleading — EV = accuracy × reward - (1-accuracy) × risk. A 58% model with 7.5/3 beats a 63% model with 2.5/5.

**All subsequent evals use 7.5/3 FGI-adaptive labels.**

---

## 6. Dual-Branch Transformer — v5 (3 evals, abandoned)

**Why built:** The temporal model only sees aggregated VP. The user also reads daily candle patterns (hammer, engulfing) for directional confirmation. Dual-branch: separate VP attention + candle attention, merge at FC layer.

**Architecture:** VP branch (spatial → temporal, CLS pooling) + Candle branch (daily candle reconstruction → temporal attention, embed=16). Merge at FC layer. 27k params.

**Result:** 60.5% accuracy — **worse than temporal-only by 3%**.

**What went wrong:** The candle branch treated all 30 daily candles equally. Most days have neutral candles → noise drowns out the few meaningful patterns. Adding volume to candles (v2) helped some folds but destroyed others. Diagnostic confirmed: **temporal-only beats dual-branch by 3% on same pipeline**. The candle signal as a separate attention stream adds more noise than signal.

**Decision → v6 enriched:** Don't give candles their own attention path. Instead, enrich each day token with candle + VP structure features and let the single temporal attention handle everything.

---

## 7. v6 Enriched Temporal (1+1) — Current best EV

**Why built:** Take the lesson from dual-branch failure. Candle features are useful as **context**, not as an independent attention stream. Enrich each day token with candle + VP structure before temporal attention.

**Architecture:** For each of 30 days: sample end-of-day VP → 1 spatial layer (CLS pool, 50 bins) → concat with daily candle (8 features) + VP structure (8 features) → project back to embed_dim → 1 temporal layer (CLS pool, 30 days) → FC. 24,737 params. Uses v1_raw pipeline (68 features).

**Result on 7.5/3 labels:** 58.4% accuracy. **Bull long EV: +1.57% (best ever).** Only 1 fold below 50%.

**Why it works:** Single attention path sees everything — VP shape + candle context + VP structure — without the noise of a separate candle branch. CLS pooling preserves position-aware information (vs mean pooling which destroys it).

**Limitation:** Data/param ratio tight on early folds (~2.5:1). The architecture might benefit from more data. Also: no second spatial layer — VP shape understanding is one-pass only.

---

## 8. v7 Simple 2+1 (Eval 5, underperformed)

**Why built:** Eval 4's spatial-only model (2 layers) had best raw accuracy. Hypothesis: 2 spatial layers capture peak-pair relationships better. Adding temporal on top should combine spatial depth with temporal context.

**Architecture:** Daily VP sum → 2 spatial layers (mean pool) → 1 temporal layer (mean pool). v2 pipeline (60 features). No enrichment. 31,073 params.

**Result on 7.5/3 labels:** 56.2% accuracy. Bull long EV: +0.96%. **Worst of all temporal models.** 4 folds below 50%.

**What went wrong:** No enrichment = candle/VP-structure info missing. Mean pooling loses position information. Data/param ratio dropped to 0.9:1 on fold 1 → overfitting. The model is a stripped-down version of v8 without the features that matter.

**Decision:** v7 is not viable. v8 (with enrichment) is the right 2+1 design.

---

## 9. v8 Enriched 2+1 (Eval 6 on 1h, Eval 7 on 15min — in progress)

**Why built:** Can the enrichment strategy from v6 + the deeper spatial from Eval 4 combine the best of both? v8 = v6's enrichment + 2 spatial layers.

**Architecture:** End-of-day VP → 2 spatial layers (CLS pool, 50 bins) → enrich with daily candle (8) + VP structure (8) → project → 1 temporal layer (CLS pool, 30 days) → FC. 33,281 params. v1_raw pipeline (68 features).

### On 1h data (Eval 6):
58.0% accuracy. Bull long: +1.08%, Bear long: +1.07%, **Bear short: +0.62%** (only model with positive bear short). But bull long dropped vs v6 (+1.08% vs +1.57%). Root cause: data/param ratio 0.9:1 on fold 1 — the extra spatial layer overfits with insufficient data.

### On 15min data (Eval 7, in progress):
**The hypothesis that v8 needs more data is being validated:**

| Fold | Train samples | v8 15min | v8 1h | Trend |
|------|---------------|----------|-------|-------|
| 1 | ~115k | 36.2% | 53.7% | Still data-starved |
| 2 | ~133k | 52.6% | 42.4% | Crossing over |
| 3 | ~150k | 58.6% | 65.8% | Competitive |
| 4 | ~168k | 61.3% | 55.6% | **Beating 1h** |
| 5 | ~186k | 66.8% | 59.9% | **Beating all models** |
| 6 | ~204k | 64.5% | 49.6% | **Beating all models** |
| 7 | ~222k | 77.2% | 64.5% | **Highest fold on 7.5/3** |

Monotonically increasing with data volume. The 2nd spatial layer is finally earning its parameter cost with 4x more training data.

v6 on 15min was worse than v6 on 1h (55.6% vs 58.4%) — early stopping at epoch 16 suggests it converges too fast with the larger dataset. The architecture is too small to benefit from more data.

### Final results (Eval 8):
**59.6% accuracy — new best on 7.5/3 labels.** Bear precision 79.0%. Bear short EV +0.48%.

| Fold | v8 15min | v8 1h |
|------|----------|-------|
| 8 | 51.4% | 64.3% |
| 9 | 60.7% | 57.2% |
| 10 | 68.3% | 66.0% |

Fold 8 (2024 H1) collapsed to 51.4% — broke the upward trend. But folds 9-10 recovered.

**However, EVs are lower than 1h models.** v6 1h still has best bull long EV (+1.57% vs v8 15min's +0.82%). 15min data adds noise that dilutes per-trade EV despite improving raw accuracy. More predictions ≠ more profit per prediction.

**Conclusion:** v8 on 15min validates the "more data helps larger models" hypothesis for accuracy, but the EV story is more nuanced. For deployment, v6 on 1h may still be the better choice for profit per trade.

---

## Key Lessons Across All Models

1. **Labels matter more than architecture.** First-hit TP/SL + regime-adaptive + vol-scaling was worth more than any model change (57% → 61%).

2. **VP predicts barriers, not direction.** 24h fixed-horizon labels = 50% (random). First-hit labels = 58-63%. The model learns where price stops, not where it goes.

3. **Spatial attention > convolution > recurrence** for VP bins. Self-attention directly learns bin-pair relationships.

4. **Enrichment > separate branches.** Candle features as day-token context works. Candle features as independent attention stream adds noise.

5. **CLS pooling > mean pooling.** Position-aware summarization preserves structural information.

6. **Data/param ratio matters.** Models underperform when ratio drops below ~3:1. v8's 2nd spatial layer only works with 15min data (8:1 ratio).

7. **More data helps, but only if the model has capacity to use it.** v6 (small) didn't benefit from 15min. v8 (larger) thrived.

---

## Architecture Lineage

```
RNN (vanishing gradients)
 └→ LSTM (spatial features need spatial reasoning)
     └→ CNN (fixed filters can't learn bin-pair relationships)
         └→ Spatial Transformer (no temporal context)
             ├→ Dual-Branch v5 (candle branch = noise) ✗
             ├→ v6 Enriched 1+1 ✓ (best EV: bull long +1.57%)
             ├→ v7 Simple 2+1 (no enrichment = bad) ✗
             ├→ v8 Enriched 2+1 (needs more data)
             │  └→ v8 on 15min (best acc 59.6%, lower EV) ~
             └→ v6 + funding fine-tune (Eval 9, fold 6 collapse) ✗
                 └→ v6-prime VP-derived labels (Eval 10, acc up 13%, EV worse) ✗
                     └→ v6-prime + asym filter (Eval 11, +3.49%) ✓
                         └→ v6-prime + 3 seeds + SWA + combined filter (Eval 12, +3.98%, Sharpe 0.97) ⭐⭐ CURRENT BEST
```

---

## 10. Funding Rate Fine-Tuning (Eval 9 — mixed results, abandoned)

**Why built:** VP information ceiling ~60%. Hypothesis: funding rate captures market leverage (user's manual "liquidation heatmap" edge) and can add signal on top of VP.

**Approach:**
- Stage 1: Train v6 backbone fresh per fold (VP-only, 24,737 params)
- Stage 2: Freeze backbone, replace FC head with wider one (50→53 inputs), train only FC head (3,521 params) with 3 funding features added

**Data:** Binance GitHub CSV (2020-2024) + Gate.io API (2024-present) = 6,879 records at 8h resolution. Forward-filled to 1h bars. Pre-2020 zeros (no funding data before).

**Features:** `funding_rate`, `funding_zscore` (30d rolling), `funding_trend` (24h change)

### Results (Eval 9)

| Fold | Baseline | Finetuned | Delta |
|------|----------|-----------|-------|
| 1 | 42.4% | 34.4% | **-8.0%** |
| 2 | 59.7% | 59.6% | -0.1% |
| 3 | 61.0% | **63.3%** | **+2.3%** |
| 4 | 66.4% | 65.7% | -0.7% |
| 5 | 68.6% | **70.4%** | **+1.8%** |
| 6 | 59.3% | **40.5%** | **-18.7%** |
| 7 | 77.6% | 77.0% | -0.5% |
| 8 | 61.1% | **69.2%** | **+8.1%** |
| 9 | 58.5% | **59.5%** | +1.0% |
| 10 | 61.0% | **62.4%** | +1.5% |
| **Overall** | **61.4%** | **60.1%** | **-1.3%** |

**Bull long EV:** +1.01% (down from +1.57% VP-only)
**Bear short EV:** +0.32% (up from -0.04%)

**What went wrong:**
- Fold 6 catastrophic collapse (-18.7%). 2023 H1 was a recovery period where funding was positive (crowded longs) but price went up. FC head learned "positive funding → reversal" from earlier data, then over-applied it.
- Folds 1-2 had too little funding training data (funding history starts 2020).
- Funding signal is regime-dependent — adding it as a flat feature causes brittle overfitting in transition periods.

**Decision: Abandon funding fine-tuning.** The signal is real (positive deltas on later folds) but too unstable for production. Moving to TP/SL optimization and regularization instead.

---

## 11. v6-prime: VP-derived TP/SL + Regularization (Eval 10 — worse than baseline)

**Why built:** Two distinct problems to solve at once:
1. Fixed 7.5/3 labels don't align with model's actual task (VP barrier prediction).
2. v6 baseline converges in 1-2 epochs then overfits (weak regularization).

**Changes from v6:**
- **Per-sample adaptive labels:** TP = `ceiling_dist × 0.25 × 0.8 × vol_scale`, SL = `floor_dist × 0.25 × 0.6 × vol_scale`, clipped [1%, 15%]. Labels aim at VP peaks directly.
- **Regularization overhaul:** dropout 0.15 → 0.3, weight_decay 0 → 1e-3, label_smoothing 0 → 0.1, Adam → AdamW.
- Skip bars with 0 peaks in VP structure.
- Frozen as `v6_prime_vp_labels.TemporalEnrichedV6Prime` (same architecture as v6, separate file for provenance).

**Results (Eval 10):**

| Metric | v6 baseline | v6-prime | Delta |
|--------|-------------|----------|-------|
| Overall accuracy | 61.4% | **74.5%** | +13.1% |
| Long real EV | (fixed 7.5% wins) | **+0.48%** | — |
| Bull long EV | +1.57% | +1.23% | **-0.34%** |
| Bear short EV | -0.04% | +0.90% | +0.94% |
| Epochs per fold | 16 (early stop) | 17-47 | ✓ |
| Per-fold variance | ~20% range | ~9% range (but EV wild) | Mixed |

**What worked:**
- Regularization genuinely fixed early stopping. Models train for 17-47 epochs now, not hitting the wasted patience.
- Bull long precision on VP-derived labels slightly higher (~62% vs baseline).
- Per-fold accuracy is more consistent (tighter range than baseline).

**What failed:**
- **Accuracy did not translate to EV.** The 13% accuracy gain is mostly on "easy default" samples (wide TP where label 0 is trivially correct), not on profitable long trades.
- **Bull long EV dropped (-0.34%)** vs baseline — the best regime direction got worse.
- **Fold 4 (2022 H1, LUNA crash) catastrophic at -3.81%** — model over-entered longs during a crash with 1,797 trades.
- **Per-fold real EV variance is huge:** +5.19% on fold 3, -3.81% on fold 4. Not production-ready.

**Root cause:** Per-sample TP/SL introduced variance without reducing it elsewhere. When TP ≈ SL (fold 4: 8.2%/8.2%), the task is hard and mistakes are expensive. When TP >> SL, label 0 is easy but doesn't generate actionable long trades. The label formula encodes a task that's either trivial or brutal — no middle ground.

**Decision:** v6-prime is not the path forward. The concept works directionally (positive overall EV) but execution is worse than v6 baseline for trading purposes. Accuracy became the wrong metric.

---

## 12. ⭐ BREAKTHROUGH: Asymmetry Filter on v6-prime (Eval 11, 2026-04-12)

**The first strategy that meaningfully beats v6 baseline's per-trade EV.**

### Discovery

Added post-hoc filter analysis to v6-prime predictions. Sweep across filters revealed that **filtering by `tp_pct / sl_pct > 2.0`** (only trade when risk/reward is clearly favorable) produces:

- **652 trades over 5 years** (~65/year, ~1 per 2 days)
- **+3.49% real EV per trade** (vs v6 baseline bull long +1.57%)
- 2.2x higher per-trade edge, 40x lower frequency

### Why this matches user's manual strategy

The user manually filters trades the same way: "I don't enter unless the risk/reward is clearly in my favor — close support, far resistance, or vice versa." The asymmetry filter mathematically encodes this human rule.

What the VP structure means for the filter:
- **High tp_pct / sl_pct ratio** = close VP floor (tight support) + far VP ceiling (room to run) = classic breakout setup
- **Low ratio (~1.0)** = equidistant peaks = no structural edge = skip

The model wasn't doing this internally. It was predicting direction on every bar regardless of quality. Filtering post-hoc extracts the setups where structure actually gives us an edge.

### Filter sweep results

| Filter | Trades | EV/trade |
|--------|--------|----------|
| All longs (no filter) | 10,013 | +0.60% |
| Confidence > 0.65 | 6,840 | +1.07% |
| Expected EV > 0.02 | 4,455 | +1.24% |
| Expected EV > 0.05 | 1,934 | +3.01% |
| **Asymmetry > 2.0** | **652** | **+3.49%** ⭐ |
| Asymmetry > 3.0 | 603 | +3.37% |

The asymmetry filter is the standout because it directly captures the structural edge. Expected EV filter gets similar results but requires model calibration to be reliable.

### Logit calibration success

Label smoothing + weight decay + dropout produced well-calibrated logits (mean +0.26, std 1.33, Q90 +2.89). Unlike Eval 4's bimodal extreme logits, v6-prime can now be filtered by confidence smoothly — this is why confidence thresholds also work this time (+1.07% at 0.65).

### Fold 1 failure mode

Fold 1 collapsed to **0 long predictions** (always predict label 0). Small training set (~20k) + strong regularization pushed the model to majority class. Fixable with multi-seed training or adaptive regularization.

### Path forward

1. **Bake asymmetry filter into training.** Train only on bars with tp_pct/sl_pct > 2.0 — specialist model for the trades that matter.
2. **Apply asymmetry filter to v6 baseline.** Does the same filter work on v6 baseline's predictions? Tests whether asymmetry is the universal trick or specific to VP-derived labels.
3. **Multi-seed + SWA** to fix fold 1 collapse and reduce variance.
4. **This is the new benchmark:** +3.49% per trade with asymmetry > 2.0.

---

## 13. ⭐⭐ Multi-seed Ensemble + SWA + Combined Filter (Eval 12, 2026-04-12)

**The current benchmark.** First strategy to combine high per-trade edge with clean compounding characteristics.

### Setup

- v6-prime architecture (TemporalEnrichedV6Prime, 24,737 params)
- VP-derived per-sample TP/SL labels
- Regularization: dropout 0.3, weight_decay 1e-3, label_smoothing 0.1, AdamW
- **3 seeds per fold (42, 43, 44)** — train each seed independently
- **SWA within each seed** — average weights from epoch 15 onwards
- **Ensemble at test time** — average logits across 3 seeds before thresholding
- Cost: 45 min per run on A100 (vs 15 min single-seed)

### Results (winning filter)

**Filter: `confidence > 0.65 AND tp_pct / sl_pct > 1.5`**

| Metric | Value |
|---|---|
| Trades (5 years) | **435** (~87/year, ~1 per 4 days) |
| Precision | **78.4%** |
| EV arithmetic per trade | **+3.98%** |
| EV geometric per trade | +3.89% |
| Compound total | ×16.5M |
| Sharpe per trade | 0.97 |
| Max consecutive losses | 23 |
| Avg win | +5.79% |
| Avg loss | -2.59% |

### Why this works

1. **Ensemble reduces variance.** 3 seeds trained from different random inits, their predictions averaged via logit mean. Reduces "unlucky seed" effect.
2. **SWA smooths within each seed.** Averaging weights from epoch 15+ puts each seed in a flat minimum instead of a sharp local one.
3. **Label smoothing gives calibrated logits** (std 1.67, not bimodal) — confidence filtering now works.
4. **Combined filter demands both signals.**
   - `conf > 0.65` → model is highly certain
   - `asym > 1.5` → VP structure gives tight SL vs wide TP
   - Intersection: high-certainty predictions on structurally favorable setups
   - Neither alone works with the ensemble (conf alone: +1.17%, asym alone: -0.16%)
5. **Max consecutive losses = 23** — this is what drives Sharpe 0.97. Strategy has short drawdowns, quick recoveries.
6. **Geometric vs arithmetic EV gap is only 0.1%** — indicates low variance. Strategy compounds almost perfectly.

### Key insight: "accuracy of ensemble" vs "accuracy of filter"

The ensemble's overall accuracy (68.6%) is actually LOWER than single-seed v6-prime (74.5%). But per-trade EV is HIGHER. Why?

The ensemble is **more conservative** — it predicts long less often on marginal samples. Those marginal samples were "cheap accuracy" (default-correct on label 0) but not profitable. Dropping them drops accuracy but improves precision on actual trades. Filtering then extracts the genuinely tradeable subset.

Accuracy is a bad metric. Precision × magnitude × low-variance is the right metric. This result proves it.

### Known issues

- **Fold 4 still catastrophic** (-4.38% unfiltered, 2022 H1 LUNA crash). The ensemble didn't fix this — it's a fundamental regime detection problem.
- **Fold 1 fixed** (1,180 trades at +2.66%) vs single-seed's 0 trades.
- **Per-fold variance still high** on unfiltered predictions. The combined filter is what extracts reliable returns.

### Path forward

1. **Bake the combined filter into training** — train only on samples where the filter would accept.
2. **Increase seeds to 5 for production** — diminishing returns past 5, but 5 is standard for deployment.
3. **Add more filter variations** — test `conf > 0.70 + asym > 1.5`, etc.
4. **Regime-specific training** — separate models for bull vs bear (fold 4 is a clear regime failure).

**This is now the strategy to beat.** +3.98% EV per trade on 435 trades with 78% precision and Sharpe ~1.0 is the benchmark for all future experiments.

---

## 14. First Realistic Backtest (Eval 13, 2026-04-12)

**Reality check on the +3.98% per-trade edge** — what does it actually look like with capital constraints, fees, slippage, and pyramiding rules?

### Setup

- Starting capital: $5,000
- Reserve: 30% of equity (dynamic)
- Fees: taker entry 0.26%, maker TP exit 0.16%, taker SL exit 0.26%
- Slippage: 0.05% per side
- Max hold: 14 days
- 12 combinations tested: 3 filters × 4 sizing strategies
- Period: 2020-07-01 → 2025-07-01 (5 years), 27,842 hourly bars

### Results — top 3 strategies

| Filter | Sizing | Final $ | Return | CAGR | Max DD | Sharpe | Trades | Win % |
|--------|--------|---------|--------|------|--------|--------|--------|-------|
| combined_60_20 | fixed_100pct | **$6,425** | **+28.5%** | **+7.1%** | -6.5% | 1.83 | 49 | 57.1% |
| combined_60_20 | fixed_50pct | $6,319 | +26.4% | +6.6% | -6.6% | 1.78 | 65 | 55.4% |
| combined_60_20 | dynamic | $6,151 | +23.0% | +5.8% | -6.5% | **5.93** | 129 | **76.7%** |

### Major findings

**1. The "less strict" filter wins.** combined_60_20 (conf > 0.60 + asym > 2.0) beat combined_65_15 (conf > 0.65 + asym > 1.5) across every sizing variant. Per-trade EV from Eval 12 suggested the opposite. Why? Capital constraints select which signals you actually take, and the slightly looser filter has more frequent signals — capital efficiency beats per-trade precision when capital is the bottleneck.

**2. Per-trade EV (eval) ≠ Real CAGR (backtest).** Eval 12 said +3.98% per trade with 78% precision. Backtest gave +28.5% over 3.66 years (~7.1% CAGR) with 57% win rate.
- Of 481 signals fired, only 49 executed (10%)
- 432 skipped due to capital lockup
- The 47 trades that DID execute had only 55% win rate, not 78%
- **Capital constraint biased trade selection toward worse outcomes** — when you're locked in a losing trade, you miss the next great signal

**3. Filter alone cuts drawdown 10x.**
- combined_60_20 max DD: -6.5%
- unfiltered max DD: -68% on full allocation, -41% on 50%
- Even at +16% return, unfiltered's -68% drawdown is untradeable

**4. Dynamic sizing has Sharpe 5.93 but lower returns.** 129 trades vs 49 (split capital allows more concurrent positions), 76.7% win rate (capital flow lets you not get stuck in losing trades during good signals), but smaller per-trade contribution to compounding.

**5. First trade not until 2021-04-19** (despite period starting 2020-07-01). Fold 1 ensemble couldn't find qualifying long signals — dataset scarcity at that fold's training period made the model too conservative.

### The honest comparison

| | Our bot | Passive BTC HODL | S&P 500 |
|---|---|---|---|
| CAGR | +7.1% | ~+15% | ~+10% |
| Max DD | **-6.5%** | -70% | -25% |
| Sharpe | 1.83 | ~0.5 | ~0.6 |

Half the return of HODL but **one-tenth the drawdown** and **3x Sharpe**. Better risk-adjusted but lower headline number than HODL.

**Verdict:** Profitable but not yet compelling vs index funds. CAGR needs to roughly double to be a serious capital allocation choice.

### Path to higher CAGR (next experiments)

1. **Smart pyramiding** — allow multiple concurrent positions with scaled sizing (50/33/25/20/15% as concurrent count grows)
2. **Looser filter** — combined_55_10 to capture more signals
3. **Trade short side** — model's bear NPV is also strong (~78%); leaving money on the table by long-only
4. **Multi-strategy ensemble** — long + short + loose filter strategies sharing capital, smoother utilization
5. **15min data retrain** — 4x more bars = 4x more signals (untested with v6-prime)

Files:
- Eval script: `src/models/eval_v6_prime.py` (predictions cache)
- Backtest engine: `src/backtest/engine.py`
- Backtest driver: `src/models/run_backtest.py`
- Results: `experiments/backtest_results.json`

---

## 15. Sizing Sweep (Eval 14b, 2026-04-12)

After Eval 14 (10% sizing flat) underperformed Eval 13 (100% sizing), tested 6 sizings (10% → 100%) on combined_60_20 with **no reserve buffer**.

### Results

| Sizing | Trades | Return | CAGR | DD | Sharpe |
|--------|--------|--------|------|----|----|
| **100%** | 60 | **+41.7%** | **+10.0%** | -9.3% | 2.44 |
| 50% | 81 | +38.5% | +9.3% | -9.4% | 2.47 |
| 33% | 94 | +34.7% | +8.5% | -9.3% | 2.63 |
| 25% | 98 | +28.5% | +7.1% | -9.0% | 2.73 |
| 20% | 109 | +26.7% | +6.7% | -8.1% | 2.76 |
| 10% | 143 | +22.7% | +5.8% | -7.4% | 2.74 |

### Findings

1. **Removing the reserve added 13% to total return** vs Eval 13 (which had 30% reserve). The reserve was leaving 30% of capital idle.
2. **First time CAGR beat S&P 500** (10.0% vs ~10%). Real psychological milestone.
3. **Monotonic: bigger sizing = more return.** "More frequency" hypothesis officially dead. Each step down loses real return because position size matters more than trade count.
4. **Sharpe peaks at 20% sizing** (2.76). Smaller sizing = smoother equity curve. This matters for leverage — smoother curves leverage better.
5. **Drawdowns barely change** with sizing (-7% to -9%). Risk dominated by filter quality, not position size.

### The pattern
The strategy is bottlenecked by signal frequency, not capital. Splitting capital across many small positions doesn't catch more signals — it just lets you take more signals when many fire close together. Most of the time, only 1-2 signals are firing per week, so a single 100% position captures most of the available edge.

---

## 16. Leverage Sweep (Eval 15, 2026-04-12) ⭐⭐ DEPLOYABLE STRATEGY

**Hypothesis:** With 100% sizing winning at 1x and Sharpe 2.44, leveraging up should multiply returns linearly while keeping drawdowns tradeable. Test 4 sizings × 4 leverages = 16 backtests.

### Setup additions

- Engine now tracks `size_dollars` (margin) and `exposure_dollars` (margin × leverage) separately
- `btc_amount` controls the leveraged exposure → P&L is leveraged
- Added liquidation check (force-close at 95% margin loss)
- Added funding rate cost (0.01%/8h × exposure × hours/8)
- Smoke test verified 3x leverage gives 3x return + 3x DD on synthetic data

### Results — top 8

| Sizing × Lev | Final $ | Return | CAGR | Max DD | Sharpe | Trades | Win % |
|---|---|---|---|---|---|---|---|
| 100% × 5x | $18,170 | +263% | +42.3% | -39.4% | 3.80 | 85 | 68.2% |
| 50% × 5x | $17,390 | +248% | +40.6% | -39.3% | 3.50 | 101 | 65.3% |
| 33% × 5x | $15,972 | +219% | +37.4% | -38.7% | 3.62 | 113 | 64.6% |
| 25% × 5x | $14,148 | +183% | +32.9% | -37.7% | 3.60 | 117 | 65.0% |
| **100% × 3x** ⭐ | **$12,672** | **+153%** | **+28.9%** | **-24.7%** | **3.83** | 82 | **69.5%** |
| 50% × 3x | $12,134 | +143% | +27.4% | -24.9% | 3.38 | 98 | 65.3% |
| 33% × 3x | $11,313 | +126% | +25.0% | -24.6% | 3.56 | 110 | 64.5% |
| 25% × 3x | $10,217 | +104% | +21.6% | -23.9% | 3.52 | 115 | 64.3% |

### Major findings

**1. ZERO liquidations across ALL 16 backtests.** Even at 5x leverage, the strategy never got liquidated. Our SL is tight (2.62% avg loss × 5x = -13% per trade), nowhere near 95% margin loss. **The asymmetry filter that makes the model profitable also keeps it safe at high leverage.**

**2. The deployable winner: 100% × 3x leverage**
- **+28.9% CAGR** (3x S&P 500)
- **-24.7% max DD** (similar to S&P worst periods)
- **Sharpe 3.83** (highest of all 16, hedge-fund tier)
- **69.5% win rate** (highest of all 16)
- 82 trades over 3.66 years (~22/year)
- No liquidations

**3. Drawdown scales linearly with leverage:**
- 1x: -8.6%
- 2x: -16.9% (~2x)
- 3x: -24.7% (~3x)
- 5x: -39.4% (~4.6x — sublinear due to recovery dynamics)

**4. The leverage flywheel.** Higher leverage = MORE trades, not fewer:
- 100% × 1x: 65 trades
- 100% × 3x: 82 trades
- 100% × 5x: 85 trades

Leveraged wins free more cash → more capital → more signals captured. Compounds qualitatively, not just quantitatively.

**5. Concentration > diversification at every leverage level.** 100% sizing > 50% > 33% > 25% always. Same lesson as Eval 14b.

**6. Win rate INCREASES with leverage** (61-66% at 1x → 64-69% at 5x). Same flywheel effect.

### Honest comparison

| Metric | Our bot (3x) | BTC HODL | S&P 500 |
|---|---|---|---|
| CAGR | **+28.9%** | ~+15% | ~+10% |
| Max DD | -24.7% | -70% | -25% |
| Sharpe | **3.83** | ~0.5 | ~0.6 |

**Beats BTC HODL on every metric. 3x the return of S&P 500 with similar drawdown. Sharpe is 6-8x better than either passive strategy.**

### Caveats (don't get euphoric)

1. **Backtest ≠ live.** Real performance always degrades. Expect 70-80% of backtest → realistic +20-22% CAGR.
2. **5 years is one sample path.** Different period could give different results.
3. **Funding rate approximated** at 0.01%/8h fixed. Real funding spikes to 0.1%/8h in euphoric markets → could shave 3-5% CAGR.
4. **Slippage approximated** at 0.05%/side. Flash crashes can hit 1-2% in single fills.
5. **Subtle backtest bias** — model architecture choices were informed by 2020-2025 results.

**Realistic live expectation: +18-22% CAGR with -25% to -35% drawdowns.**

### This is the deployable benchmark
**v6-prime + ensemble + combined_60_20 filter + 100% sizing + 3x leverage**

Files:
- `experiments/backtest_results.json` — full 16-backtest output
- `src/backtest/engine.py` — leverage support
- `src/models/run_backtest.py` — sweep config

### What's next
1. Phase 4: Kraken integration to actually deploy this
2. Test direction="both" (long + short)
3. 15min data with leverage backtest

---

## 17. Circuit Breaker Test (Eval 16, 2026-04-12) — DD Breakers Don't Help

**Hypothesis:** Add drawdown circuit breaker and max-consec-loss killswitch to reduce -24.7% baseline drawdown without sacrificing return.

### Setup

8 variants on the proven config (combined_60_20 + 100% × 3x leverage):
- baseline (no breakers)
- dd_breaker_10/15/20 (time-based pause when dd exceeds threshold)
- killswitch_4L/5L (pause after N consec losses)
- hybrid combinations

DD breaker pause = 30 days (720 bars at 1h). Killswitch pause = 7 days (168 bars).

### Results

| Variant | CAGR | DD | Sharpe | Verdict |
|---|---|---|---|---|
| **baseline** | **+28.9%** | -24.7% | 3.83 | Best return |
| killswitch_4L/5L | +25.8% | -24.7% | **4.05** | Marginal Sharpe gain at 3% CAGR cost |
| dd_breaker_20 | +17.7% | -24.7% | 3.69 | Doesn't fire effectively |
| dd_breaker_15/10 | +13.3% | -15.1% | 3.73 | Cuts DD 38% but cuts return 54% |
| hybrid_10_3L | +10.0% | -15.1% | 2.63 | Worst risk-adjusted |
| hybrid_15_4L | +10.0% | -15.1% | 2.49 | Worst risk-adjusted |

### Why breakers fail

1. **Drawdowns are brief shocks, not slow grinds.** The -24.7% max DD likely happens in 1-2 sharp events (LUNA crash 2022 H1). By the time a breaker triggers (at -10% or -15%), the worst is already done.
2. **Pausing means missing recovery rallies.** Many of the best signals fire during turbulent periods.
3. **Win rate is already 69%.** Filtering out trades that mostly would have won has clear cost.
4. **DD breakers HURT Sharpe** (3.83 → 3.73). Sharpe is best with no breaker.
5. **Killswitch slightly improves Sharpe** (3.83 → 4.05) but at 3% CAGR cost — not worth the trade.

### Engineering note (bug fixed during this experiment)

First DD breaker implementation deadlocked: equity-based reset couldn't trigger because no trades were happening to recover equity. Fixed by switching to time-based pause that resets `peak_equity` when the pause expires, giving the strategy a fresh baseline. Also fixed an immediate-re-fire bug where `update_equity` would re-trigger the breaker on the same bar the pause expired.

### Critical insight

**Max drawdown is concentrated in irreversible early moments.** Time-based pauses can't help because they activate AFTER the worst is already done. If we wanted to cut drawdown further, we'd need to:
- Predict drawdown periods in advance (not feasible)
- Use a regime detector to skip trading during certain market conditions (untested)
- Accept higher friction strategies (lower CAGR)

For now, the -24.7% drawdown is the cost of admission. User's stated tolerance is -30%, so it fits.

### Final decision: deploy the baseline

**No circuit breakers.** The deployable strategy is unchanged from Eval 15:
- v6-prime + 3-seed ensemble + SWA
- combined_60_20 filter
- 100% sizing, no reserve
- 3x leverage
- +28.9% CAGR, -24.7% max DD, Sharpe 3.83

---

## 18. Post-SL Pause (Eval 17) ⭐⭐⭐ NEW DEPLOYABLE STRATEGY (2026-04-12)

**Hypothesis (user's idea):** External events (LUNA collapse, FTX, surprise news) cause sudden adverse moves that VP hasn't yet absorbed. After hitting an SL, pause for 24h to let the immediate shock settle before re-entering. Different from time-based DD breakers because it triggers per-loss, not per-streak.

### Setup

6 variants on combined_60_20 + 100% sizing + 3x leverage:
- baseline (no pause)
- post_sl_12h, _24h, _48h, _72h
- post_sl_24h + killswitch_4L combination

### Results — UNICORN OUTCOME

| Variant | Return | CAGR | DD | Sharpe | Trades | Win % |
|---|---|---|---|---|---|---|
| **post_sl_24h** ⭐⭐⭐ | **+191.6%** | **+34.0%** | **-15.1%** | 2.96 | 50 | **72.0%** |
| baseline | +153.4% | +28.9% | -24.7% | 3.83 | 82 | 69.5% |
| post_sl_24h + kill4L | +146.1% | +27.9% | -15.1% | 2.94 | 48 | 72.9% |
| post_sl_12h | +123.0% | +24.5% | -24.7% | 3.73 | 71 | 71.8% |
| post_sl_48h | +70.4% | +15.7% | -16.8% | 2.69 | 47 | 68.1% |
| post_sl_72h | +46.7% | +11.0% | -15.1% | 1.97 | 32 | 62.5% |

### post_sl_24h DOMINATES baseline

| Metric | post_sl_24h | baseline | Delta |
|---|---|---|---|
| CAGR | **+34.0%** | +28.9% | **+5pp** |
| Max DD | **-15.1%** | -24.7% | **-9.6pp** |
| Win rate | **72.0%** | 69.5% | **+2.5pp** |
| Trades | 50 | 82 | -32 |

This is the unicorn outcome — usually you trade return for DD reduction, but here we get **both better return AND less drawdown**. The post-SL pause filters out specifically the bad trades that come right after a loss (when VP is stale and market is still in shock).

### Why annualized Sharpe is lower

Per-trade Sharpe is identical (~0.80 for both variants). Annualized Sharpe = per-trade × √(trades/year). With fewer trades (50 vs 82), the annualized number scales down purely from the √N factor — not from worse trade quality. **CAGR and DD are what matter for your wallet, and post_sl_24h wins both.**

### Why 24h is the sweet spot

- 12h: too short to filter the immediate shock cluster — barely changes anything
- 24h: optimal — catches the recovery noise window
- 48h: too long — starts missing legitimate recovery signals
- 72h: way too long — most signals get pruned

24h reflects the typical "shock absorption" timescale of crypto markets. After 24h, immediate panic has passed and VP has had time to register the new bars.

### Why this works mechanically

1. **The losing trades right after a previous loss are the worst quality.** They happen during ongoing turbulence when the model is trading against stale VP. The 24h pause filters them out.
2. **The 32 skipped trades had NEGATIVE expected value.** Removing them adds value.
3. **Win rate jumps 2.5pp** (69.5% → 72.0%) because remaining trades are higher quality.
4. **Drawdown drops 40%** (-24.7% → -15.1%). The bad cluster around LUNA crash is what defined max DD, and the post-SL pause prevents re-entering during that event.

### NEW DEPLOYABLE STRATEGY

**v6-prime + 3-seed ensemble + SWA + combined_60_20 + 100% sizing + 3x leverage + 24h post-SL pause**

- **+34.0% CAGR** (vs S&P 500's ~10%)
- **-15.1% max DD** (vs S&P 500's ~-25%)
- 72.0% win rate
- 50 trades over 3.66 years (~14/year)
- Zero liquidations
- Realistic live expectation: +22-27% CAGR with -18% to -22% DD

### Comparison vs passive (now even better)

| | Our bot (NEW) | BTC HODL | S&P 500 |
|---|---|---|---|
| CAGR | **+34.0%** | ~+15% | ~+10% |
| Max DD | **-15.1%** | -70% | -25% |
| Sharpe | 2.96 | ~0.5 | ~0.6 |

**3.4x S&P 500's return with LESS drawdown.** Elite-tier risk-adjusted performance.

This is the strategy to deploy in Phase 4. No more model iteration needed.

---

## 19. Sensitivity Validation of 24h Pause (Eval 18, 2026-04-12)

**Concern:** After Eval 17 found post_sl_24h gave the unicorn outcome (+34% CAGR / -15% DD), we worried 24h might be a lucky single-point optimum overfit to specific events in the test period.

### Coarse sweep (4 values: 12, 24, 48, 72)

Suggested 24h was a sharp peak — but with only 4 sample points, hard to tell.

### Fine-grained sweep around 24h (10 values: 14h-60h)

```
Pause   CAGR     DD       Verdict
─────────────────────────────────
14h    +24.1%  -25.3%   pause too short, no DD benefit
18h    +23.9%  -25.1%   still too short
21h    +30.6%  -15.1%   DD CLIFF — first value to catch the shock
24h    +34.0%  -15.1%   peak
27h    +24.5%  -15.1%   drops 9.5pp
30h    +23.5%  -15.1%   regresses to baseline CAGR (DD still good)
36-42h +23.6%  -15.1%   plateau at baseline-level CAGR
60h     +8.6%  -15.1%   too long, prunes recovery signals
```

This made 24h look like a suspicious sharp spike. So we ran a denser sweep (21-30h at 1h intervals) to see if 24h was alone or part of a plateau.

### Fine-grained 1h sweep (21h-30h)

```
Pause   CAGR     DD       Win%
──────────────────────────────
21h    +30.6%  -15.1%   72.2%
22h    +30.6%  -15.1%   71.2%   ← tied with 21h
23h    +34.0%  -15.1%   70.8%   ← peak
24h    +34.0%  -15.1%   72.0%   ← peak (tied with 23h)
25h    +29.3%  -15.1%   66.7%
26h    +30.5%  -15.1%   69.2%
27h    +24.5%  -15.1%   74.1%
28h    +24.6%  -15.1%   74.1%   ← tied with 27h
29h    +27.4%  -15.1%   69.2%
30h    +23.5%  -15.1%   70.2%
```

### Findings

1. **23h and 24h are TIED at +34.0% CAGR.** It's a 2-hour peak, not a single spike.
2. **22-26h all give CAGR ≥ +29%.** That's a 5-hour window beating baseline.
3. **The drop between 26h and 27h is structural** (-6pp jump), not noise.
4. **DD is robustly -15.1% across all variants 21h-30h** — the drawdown protection is universal in this range.
5. **Consecutive pairs are tied** (21-22, 23-24, 27-28). This suggests structural behavior — specific signal clusters get included/excluded at specific time breakpoints. Not random noise.

### Mechanism interpretation

**Why 23-24h?** Aligns with a typical "shock-then-bounce" pattern in BTC:
- Market shock peaks within minutes
- Immediate panic subsides over hours
- The next legitimate setup emerges ~24h later

**Why 25-26h still works:** Same mechanism, slightly suboptimal.

**Why 27h+ collapses:** Beyond ~26h, the pause starts cutting into legitimate recovery signals from the second day. Strategy shifts from "wait out shock" to "wait out recovery."

### Conclusion: 24h IS deployable

The 2-hour peak width (23-24h) and 5-hour supportive plateau (22-26h) make this a robust effect, not an overfit single-point optimum. The mechanism is real and generalizable.

**Final deployable parameter: post_sl_pause_bars = 24.**

Realistic live expectation (after backtest decay):
- CAGR: +25-30%
- Max DD: -18% to -22%

---

## 20. ⚠ Audit Retraction — Evals 11 / 12 / 17 / 18 (2026-04-12)

**Retraction.** The "deployable strategy" from Eval 17 (+34.0% CAGR / −15.1% DD / 72% win rate) and the asymmetry-filter breakthrough from Evals 11–12 (+3.98% EV/trade, 78% precision) **do not hold up under audit**. A walk-forward label leak and a test-set sweep on the post-SL pause inflated both headlines. With embargo and a pristine out-of-sample holdout, the honest version is:

| Config | CAGR | Max DD | Win % | Holdout |
|---|---|---|---|---|
| Claimed Eval 17 (contaminated) | **+34.0%** | **−15.1%** | 72% | n/a |
| Same config, embargo'd labels | +4.5% | **−49.8%** | 41.5% | −15% (17 trades) |
| Honest best (`conf_70 + guard + 1x/20% + pause24`) | +6.0% | −18.4% | 65% | **−4.8%** |

Mechanisms of the leak:

1. **Walk-forward with no embargo.** First-hit labels look 14 days ahead. Without embargo, training labels for bars near `train_end` used prices inside val/test.
2. **Post-SL pause tuned on test set.** The "24h unicorn" and the 21h–30h sensitivity sweep both selected the pause on the same period whose metrics they reported.
3. **Asymmetry filter rode #1.** Under embargo, `asym ≥ 2.0` drops to 19.8% precision / −0.95% EV. The filter's apparent edge was entirely the leak telling the model which high-ratio setups would actually resolve to TP.
4. **3x leverage + 100% sizing hid a fragile signal.** On clean labels the strategy runs out of capital on 2025-10-28 in full-period backtest.

**Decision.** Phase 3 is not "DONE — ready for Phase 4." It's audit-complete, strategy shelved. +6% CAGR / −18% DD does not beat passive SPY, and the whole point of an automated crypto bot is to beat passive. Project is paused with infrastructure preserved.

Full details, per-fold EV table, config sweep, and what a future attempt would need to clear: `experiments/EVAL_AUDIT.md`.

## 21. What the audit did NOT invalidate

- The v6-prime architecture and training pipeline are sound.
- The embargo'd model still picks 60–75% winners across folds — there's a real (small) edge in the signal itself. What's broken is the deployment wrapper, not the model.
- Per-regime EV breakdown is intact: **bull_long +1.64%**, **bear_short +0.70%**, bull_short and bear_long both negative. The model predicts direction correctly by FGI regime. Long-only in 2025 H2 was fighting the tape. A regime-aware both-sides strategy was never tested on embargo'd data and remains the most credible "unexplored" direction if the project ever restarts.
- The TP/SL guard (`min_asymmetry = 1.0` — reject trades where `tp_pct < sl_pct` at entry) is a legitimate pre-entry filter. It halved DD on honest backtests without using lookahead info. Worth carrying forward.

## 22. Parked directions (if the project ever resumes)

1. **Regime-aware direction switching** (bull→long, bear→short) on embargo'd predictions.
2. **Funding rate / OI as training-time inputs**, not frozen-backbone fine-tuning (Eval 9 failed because of the fine-tune approach, not the feature).
3. **Different label definition** that doesn't produce the payoff asymmetry (avg_loss > avg_win) we saw on holdout.

Deprioritized: more VP-only 1h iterations, tighter TP/SL, architecture tweaks. The ceiling of the current hypothesis is known now.

---

## 23. Post-audit Stage 1 — v9-regression (REJECTED, 2026-04-12)

Attempted Philosophy D: replace binary first-hit labels with continuous realized-P&L labels under the same VP-derived TP/SL exit model, trained with HuberLoss.

**First run collapsed in 5 epochs** (`delta=0.02` too tight — L1 behavior over the entire bimodal label distribution → model parked at the conditional median near 0). Fix: `delta=0.10` to keep L2 over typical label range.

**Second run failed on heterogeneous folds.** Fold 1 (small train): train corr 0.52, val corr ≈ 0, ensemble corr −0.146 (correlated seed errors). Fold 5 (large train, 4 regimes stacked): train corr 0.07 max, `best val_loss at epoch 1` on 2/3 seeds (training actively destroyed generalization), ensemble gave 13 signals out of 3,696 bars with real EV −2.17%.

**Diagnosis.** The labels are nearly bimodal (clustered at `+tp_pct` and `−sl_pct`). Regression loss minimizes squared error across the distribution and its optimum is the conditional mean — near zero for a 60/40 SL-skewed dataset. Classification (BCE) is the structurally correct match for bimodal outcomes; v6-prime's binary version hit 86% accuracy on the same fold 5 window that Stage 1 could barely learn.

**Rejected.** Philosophy D is incompatible with this label distribution at this model scale. Artifacts (`src/models/eval_v9_regression.py`, `src/models/run_backtest_regression.py`) kept in repo for forensic reference.

## 24. Post-audit Stage 2 — v9-wall-aware (NULL RESULT, 2026-04-12)

Attempted Option B: add a "structure context token" to the spatial attention sequence (52 tokens = CLS + 50 VP bins + structure-context projected from the 8 per-day VP structure features). `+320` parameters over v6-prime. All other hyperparameters, labels, training loop, and walk-forward identical to v6-prime.

**Overall metrics (all folds ensembled):**

| Metric | v6-prime | v9 | Delta |
|---|---|---|---|
| Accuracy | 70.52% | 69.88% | −0.64 |
| Precision | 60.69% | 60.11% | −0.58 |
| Long-only real EV/trade | +0.459% | +0.474% | +0.015 |

**Fold-level pattern:** 2 dramatic wins, 5 small losses, 5 ties.

- **Fold 3 (2021 H2):** v6 63.1% acc / +2.44% EV → v9 **85.6% acc / +7.38% EV** (+22 pp acc, 3× EV per trade). Biggest single-fold improvement in the project's history.
- **Fold 8 (2024 H1):** v6 76.3% / +3.36% → v9 **81.6% / +5.01%** (second clean win).
- **Fold 1 (2020 H2, smallest train):** v6 56.1% / +6.30% → v9 **37.5% / +3.32%** (classic overfitting on small heterogeneous train — the added capacity memorized pre-2020 regimes that didn't transfer to post-COVID rally).
- **Fold 11 (2025 H2 holdout):** v6 68.0% / −0.84% → v9 **69.1% / −1.76%** (worse on the metric that matters).

**Backtest head-to-head (`conf_70_guard + 1x/20% + 24h pause`, honest audited config):**

| Metric | v6-prime | v9 |
|---|---|---|
| Full CAGR | +6.0% | +1.9% |
| Max DD | −18.4% | −17.1% |
| Full win rate | 65% | 57% |
| **Holdout return** | **−4.8%** | **−5.7%** |
| Holdout win rate | 13% | **0% (0/10)** |

**None of the three success criteria clear.** V9 did not match v6-prime's full CAGR, did not produce positive holdout return, and fold-11 EV was worse than v6-prime's. Zero winners in 10 holdout trades is not statistically powerful on n=10 but the direction is consistent with the full-window fold-11 EV.

**Interpretation.** The structure context token is not neutral — it's genuinely useful *on some folds*. Fold 3 is the clearest proof-of-concept: the model became more selective (973 trades vs v6's 1,524) and each trade was 3× more profitable. But the same extra capacity hurt on folds with small/heterogeneous training data. Net effect on holdout: slight negative.

**Rejected on holdout criterion.** The architectural hypothesis is falsified for this data/scale: VP structure features reaching the spatial attention layer does NOT produce a strategy that generalizes to post-2025-07 data.

Stage 2 artifacts preserved: `src/models/architectures/v9_wall_aware.py`, `src/models/eval_v9.py`, `src/models/run_backtest_v9.py`, `experiments/v9_test/eval_v9_results.json`, `experiments/v9_test/eval_v9_log.txt`, `experiments/backtest_results_v9.json`.

## 25. Current status (end of Stage 2)

- **Audited v6-prime honest config remains the best available strategy.** +6.0% CAGR / −18.4% DD / holdout ≈ −5% / 65% win rate. Does not beat passive SPY.
- **Two of three post-audit experiments run. Both rejected.** Stage 1 failed for loss-function reasons (regression on bimodal labels). Stage 2 failed on holdout generalization (architecture alone insufficient).
- **Stage 3 (volume-weighted effective-distance labels) is technically available** as the third and final leg. Given the fold-3 and fold-8 wins in Stage 2, the "architecture can use wall structure when conditions are right" hypothesis is partially supported, and better labels might generalize those wins. Honest prior: 20–30% chance Stage 3 clears the holdout bar.

---

## 26. Post-audit Stage 3 — v10 (90d temporal × 30d VP) — REJECTED (2026-04-12)

**Hypothesis.** Reallocate the lookback budget: shrink the VP rolling window from 180 days → 30 days (matches how the user actually reads charts — recent-volume zoom) and grow the temporal transformer window from 30 days → 90 days (preserves the longer price-action context). Same architecture, same labels, same walk-forward. Net history per forward pass drops 210d → 120d and we gain ~2,100 usable training rows from the shorter warmup.

**Setup.** New pipeline `src/data/compute_vp_30d.py` writes `BTC_1h_RELVP_30d.csv` alongside the legacy 180d file. New architecture `src/models/architectures/v10_long_temporal.py` subclasses v6-prime with `n_days=90` and nothing else. New eval wrapper `src/models/eval_v10.py` patches `LOOKBACK_BARS_MODEL` from 720 → 2160 before importing the feature pipeline so `_compute_vp_structure` uses the 90-day window for peak aggregation. Output artifacts: `experiments/eval_v10_results.json`, `experiments/v10_predictions.npz`.

**Results.** Holdout CAGR essentially unchanged vs v6-prime honest (roughly −5% both). A few in-sample folds improved marginally, but fold 11 (holdout) remained negative and the full-period CAGR did not clear v6-prime's +6%. No regime or filter variant produced a meaningful uplift.

**Verdict.** Rejected on the holdout criterion. Reallocating lookback between spatial and temporal dimensions does not unlock the signal — whatever the model is doing on in-sample folds does not generalize past 2025-07. Artifacts preserved.

---

## 27. Post-audit Stage 4 — regime-aware both-sides mirror-short overlay — REJECTED (2026-04-13)

**Hypothesis.** The §20 audit writeup flagged "regime-aware both-sides" as the most credible unexplored direction: bull-regime EV breakdown showed bull_long +1.64%, bear_short +0.70%, with the other two regime/direction combinations negative. A mirror-short overlay should pick up the bear-short EV v6-prime was leaving on the table.

**Setup.** `src/models/backtest_v10_both_sides.py` takes v10 predictions + SMA-regime array and applies a short-side overlay: in bull regimes, use the v10 long signal as-is; in bear regimes, flip the signal to a short and enter at the opposite side. TP/SL ratios mirrored (SL above, TP below). Same 14-day vertical barrier. Ran multiple logit-filter variants and an unconditional-short baseline for comparison.

**Results (holdout, bear-regime subset).**

| Variant | Trades | EV/trade | Notes |
|---|---|---|---|
| Unconditional short (no model) | baseline | **+5.87%** | the ceiling |
| `logit < 0.30` short | fewer | +4.1% | worse than no filter |
| `logit < 0.50` short | more | +3.4% | worse than no filter |
| v10-signal-inverted short | even fewer | +2.8% | worst |

**Diagnosis.** The apparent bear-short edge is a **first-hit mechanics artifact**, not model skill. Under asymmetric TP/SL geometry in a sustained bear trend, *any* short trade barely ever hits SL before the 14-day vertical barrier, so the label is dominated by "price closes lower 14 days later, label = 1." An unconditional short captures this mechanical bias in full. Every logit-filtered variant takes fewer trades and misses some of the easy wins, underperforming the baseline. The v10 model has no actual bear-regime discrimination — it just happened to sit adjacent to a geometry that rewards any short-side exposure.

**Verdict.** Closed as a dead end. The "regime-aware both-sides" direction from the audit is formally **not** an unexplored edge — it's a TP/SL geometry artifact and was already fully captured by the unconditional short baseline. Artifacts: `src/models/backtest_v10_both_sides.py`, `experiments/backtest_v10_both_sides.json`.

---

## 28. Post-audit Stage 5 — v11 absolute-range VP @ 15m — REJECTED, root cause found (2026-04-13)

**Hypothesis.** Three changes stacked for what was billed as the "final iteration":

1. **Representation**: relative log-distance VP → absolute visible-range VP (50 bins spanning the actual trailing 30-day high/low, using wicks not closes) + hard one-hot self-channel marking the bin containing `close_t` + continuous `price_pos` and `range_pct` scalars. Mirrors how the user reads Kraken's VRVP.
2. **Resolution**: 1h → 15m. Training rows jump from ~87k → ~357k. Sample/param ratio improves from 2:1 → 14:1.
3. **Architecture**: new `AbsVPv11` class — 2-channel spatial attention (`Linear(2, 32)` bin embedding instead of `Linear(1, 32)`), otherwise identical to v10's 2+1 shape. 25,601 params.

**Setup.** New pipeline `src/data/compute_absvp_15m_30d.py` writes `BTC_15m_ABSVP_30d.csv` (357,705 rows, 2016-01-31 → 2026-04-14). New architecture `src/models/architectures/v11_abs_vp.py`. New eval script `src/models/eval_v11.py` with on-GPU index-gather batching (no DataLoader, no CPU↔GPU per-batch copies) and 14-day wall-clock embargo (resolution-independent). Labels: long-only first-hit with range-derived per-sample TP/SL. `TP = (window_hi − close)/close × 0.8`, `SL = (close − window_lo)/close × 0.6`, both clipped [1%, 15%], 1344-bar (14d) horizon. Single-seed walk-forward to keep iteration fast.

**Results.** Total walk-forward: 76.8 min on H100. Overall accuracy 64.3%. Fold-by-fold:

| Fold | Period | Acc | EV/trade | Notes |
|---|---|---|---|---|
| 1 | 2020 H2 | 84.5% | +1.75% | bull trend — easy |
| 2 | 2021 H1 | **42.4%** | −4.22% | top of bull → crash |
| 3 | 2021 H2 | 68.4% | +1.41% | |
| 4 | 2022 H1 | 61.1% | −9.13% | bear, few trades |
| 5 | 2022 H2 | 73.4% | +1.45% | |
| 6 | 2023 H1 | 54.1% | +0.26% | |
| 7 | 2023 H2 | **93.8%** | +2.58% | bull run — easy |
| 8 | 2024 H1 | 64.4% | +1.42% | |
| 9 | 2024 H2 | 72.5% | +1.42% | |
| 10 | 2025 H1 | **30.9%** | −2.89% | worse-than-random |
| 11 | 2025 H2 (holdout) | 63.3% | −0.35% | |
| 12 | 2026 Q1 (holdout) | 58.6% | −0.76% | |

Holdout raw long: **−14.1% CAGR, 51.0% win rate.** Fold 10 at 30.9% is worse than random; fold 2 at 42.4% is the same bull-top failure mode v10 had. High-accuracy folds cluster on uniform trends (2020 H2, 2023 H2, 2024 H2), bear/transition folds collapse.

**Filter analysis** (170k predictions, `src/models/analyze_v11_filters.py`). None of the v10-recipe filters rescue it:

| Filter | Holdout CAGR | Win rate |
|---|---|---|
| Raw long (no filter) | −14.1% | 51.0% |
| Pred == 0 (sign flip) | −16.1% | 31.4% |
| `conf ≥ 0.70 + asym ≥ 1 + pause` | **−13.3%** | 35.8% |
| `asym < 1` (inverted) | −12.6% | 63.4% |
| `conf ≥ 0.70 + asym < 1` | −15.5% | 62.0% |

**Key finding: confidence is uncalibrated.** Sweeping the threshold 0.50 → 0.80 does not shift any metric. Normally higher confidence raises win rate; here it does not. The model has no discrimination above what the label geometry provides for free.

### Root cause — label formula leaks feature geometry into labels

Running the asymmetry/label coupling table on the 170k test set exposed a structural bias:

| `asym` band | n | pos_rate |
|---|---|---|
| `[0.0, 0.5)` | 65,359 | **88.5%** |
| `[0.5, 0.8)` | 14,231 | 70.2% |
| `[0.8, 1.2)` | 14,809 | 51.7% |
| `[1.2, 2.0)` | 22,211 | 38.3% |
| `[2.0, ∞)` | 54,386 | **18.6%** |

The label is a monotonic deterministic function of `asym`, and `asym` is itself a deterministic function of `(window_hi, window_lo, close)` — columns the model sees as features. A free classifier predicting "label = 1 iff asym < 0.8" scores ~80% accuracy before any ML. v11's 64.3% overall accuracy is **below the free classifier**, confirming the representation did not help it exploit the geometry. The v10 `asym ≥ 1` filter "worked" on v10 precisely because its peak-derived labels had the same coupling at lower intensity — everything in Phase 3 that looked like edge was partly just label bias.

**Verdict.** Rejected. But this is the most productive rejection in the project's history because it identified the binding constraint. The representation change did not break v11; **v11 exposed a pre-existing problem** that made the earlier experiments unfalsifiable.

### Unlocks the decisive experiment

Triple-barrier labels with volatility-scaled barriers (López de Prado Ch. 3) decouple labels from `(window_hi, window_lo, close)` — the barriers become functions of trailing return volatility, not range distance. Under clean labels, running **v11-full (with VP features) vs v11-nopv (candle features only, VP zeroed)** on the same holdout is the decisive test: does the model get any lift from VP when the labels aren't pre-biased by range geometry?

- v11-full > v11-nopv → VP carries real ML-exploitable signal, v10/v11 failures were label formulas, Phase 3 reopens with direction.
- v11-full ≈ v11-nopv → VP is not ML-exploitable (the user's manual strategy may still work, but the ML angle is falsified), Phase 3 formally closes.

Full design in `experiments/LABEL_REDESIGN.md`. Artifacts preserved: `src/data/compute_absvp_15m_30d.py`, `src/models/architectures/v11_abs_vp.py`, `src/models/eval_v11.py`, `src/models/analyze_v11_filters.py`, `experiments/eval_v11_results.json`, `experiments/v11_predictions.npz`.

---

## 29. Current status (end of Stage 5)

- **Audited v6-prime honest config still nominally best.** +6.0% CAGR / −18.4% DD / holdout ≈ −5% / 65% win rate. Still does not beat SPY. All post-audit experiments have failed to beat it.
- **Five post-audit experiments rejected**: Stage 1 regression (loss function), Stage 2 v9 (holdout generalization), Stage 3 v10 (same), Stage 4 both-sides overlay (TP/SL geometry artifact, not model signal), Stage 5 v11 absolute-VP (label formula leak — the binding constraint for Phase 3 as a whole).
- **Root cause of Phase 3 struggles identified**: every label formula tried shared inputs with the feature tensor, making the "does VP carry signal" hypothesis unfalsifiable. v11's asym/label coupling table was the final confirmation.
- **Next experiment is the first falsifiable VP test in the project's history**: triple-barrier labels + v11-full vs v11-nopv ablation. This is a clean binary outcome with project-defining consequences either way.
- **User has shifted project framing** from "build something profitable" to "experimental playground — use remaining experiments to answer whether VP carries ML-exploitable signal." Phase 3 formally reopens under this framing.

---

*Last updated: 2026-04-13 after Stage 5 (v11). Next: triple-barrier labels + VP ablation per `LABEL_REDESIGN.md`.*
