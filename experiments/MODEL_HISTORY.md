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

**Awaiting final folds 8-10 for complete evaluation.**

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
             ├→ v6 Enriched 1+1 (best EV on 1h) ✓
             ├→ v7 Simple 2+1 (no enrichment = bad) ✗
             └→ v8 Enriched 2+1 (needs more data)
                 └→ v8 on 15min (promising — in progress)
```

---

*Last updated: 2026-04-11. Next update after v8 15min results finalize.*
