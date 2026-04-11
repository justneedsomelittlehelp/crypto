# Transformer Evaluations

**Why:** Self-attention lets each VP bin attend to all other bins — can directly learn "peak at bin 12 relates to peak at bin 38" without stacking conv layers. Better for finding peak pairs (ceiling/floor).
**Strategy alignment:** High — attention mechanism matches how user visually scans VP for related peaks. See [STRATEGY.md](STRATEGY.md).

**Architecture:**
- Sum VP bins across 180 timesteps → (batch, 50)
- Embed each bin: Linear(1→embed_dim) + learnable positional encoding
- Single TransformerEncoderLayer (self-attention across 50 bins)
- Global avg pool → concat with non-VP features → FC → logit
- Weighted BCEWithLogitsLoss

**Label:** Regime-adaptive first-hit (bull: TP=2.5% SL=5%, bear: TP=5% SL=2.5%), vol-scaled
**VP smoothing:** sigma=0.8, prominence=0.05

---

### Eval 1 — Walk-forward (embed=8, heads=2, FC=32, params=1,817)

**Overall:** 56.9% acc, 63.1% precision, F1=0.638

| Fold | Period | Acc |
|------|--------|-----|
| 1 | 2020 H2 | 70.8% |
| 2 | 2021 H1 | 42.7% |
| 3 | 2021 H2 | 47.1% |
| 4 | 2022 H1 | 76.7% |
| 5 | 2022 H2 | 62.6% |
| 6 | 2023 H1 | 56.1% |
| 7 | 2023 H2 | 81.8% |
| 8 | 2024 H1 | 25.6% |
| 9 | 2024 H2 | 70.2% |
| 10 | 2025 H1 | 40.5% |

**Notes:** Too small (1.8k params). Underfitting — some folds good (7: 81.8%) but very unstable (8: 25.6%).

---

### Eval 2 — Walk-forward (embed=16, heads=2, FC=32, params=4,113)

**Overall:** **61.3% acc, 67.2% precision**, F1=0.670

| Fold | Period | Acc |
|------|--------|-----|
| 1 | 2020 H2 | 84.0% |
| 2 | 2021 H1 | **55.8%** |
| 3 | 2021 H2 | 44.3% |
| 4 | 2022 H1 | 73.8% |
| 5 | 2022 H2 | 62.6% |
| 6 | 2023 H1 | 57.4% |
| 7 | 2023 H2 | **88.2%** |
| 8 | 2024 H1 | 40.4% |
| 9 | 2024 H2 | 70.4% |
| 10 | 2025 H1 | 40.5% |

**Notes:** Matches CNN overall (61.3% vs 61.5%) with **higher precision (67.2% vs 65.7%)**. Fold 2 improved significantly over CNN (55.8% vs 48.0%). Fold 7 best single fold ever (88.2%). Still 4 folds below 50% — same regime transition problem.

---

### Summary

| Eval | Embed | Heads | Params | Acc | Precision | F1 | Notes |
|------|-------|-------|--------|-----|-----------|-----|-------|
| 1 | 8 | 2 | 1,817 | 56.9% | 63.1% | 0.638 | Underfitting |
| 2 | 16 | 2 | 4,113 | **61.3%** | **67.2%** | 0.670 | Matches CNN, better precision |

**Comparison with CNN (best):**

| Model | Params | Acc | Precision | F1 | Best fold | Worst fold |
|-------|--------|-----|-----------|-----|-----------|------------|
| CNN (Eval 23) | 3,581 | 61.5% | 65.7% | 0.690 | 88.0% | 41.7% |
| Transformer (Eval 2) | 4,113 | 61.3% | 67.2% | 0.670 | 88.2% | 40.4% |

**Conclusion:** Transformer matches CNN with slightly better precision. Both hit ~61% ceiling on 4h data.

---

## 1h Data Experiments

Switched to 1h timeframe: 84,750 rows (vs 20,832 at 4h). LOOKBACK_BARS_MODEL=720 (30 days). ~50k+ training samples per fold.

### Eval 3 — Walk-forward (1h, embed=16, heads=2, params=4,113)

**Overall:** 58.7% acc, 61.2% precision, F1=0.682

| Fold | Period | Acc |
|------|--------|-----|
| 1 | 2020 H2 | 88.4% |
| 2 | 2021 H1 | 39.9% |
| 3 | 2021 H2 | 45.0% |
| 4 | 2022 H1 | 64.6% |
| 5 | 2022 H2 | 53.2% |
| 6 | 2023 H1 | 58.4% |
| 7 | 2023 H2 | 61.0% |
| 8 | 2024 H1 | 52.6% |
| 9 | 2024 H2 | 64.4% |
| 10 | 2025 H1 | 59.5% |

**Notes:** Small Transformer too small for 1h data complexity. Worse than 4h version.

---

### Eval 4 — Walk-forward (1h, embed=32, heads=4, 2 layers, params=21,889) **NEW BEST**

**Overall:** **63.3% acc, 65.8% precision, F1=0.702**

| Fold | Period | Acc | vs 4h best |
|------|--------|-----|------------|
| 1 | 2020 H2 | 88.4% | +0.4 |
| 2 | 2021 H1 | **57.5%** | **+9.5** |
| 3 | 2021 H2 | 44.4% | +0.8 |
| 4 | 2022 H1 | 69.0% | -8.8 |
| 5 | 2022 H2 | **66.4%** | **+3.8** |
| 6 | 2023 H1 | **58.4%** | **+11.1** |
| 7 | 2023 H2 | **81.0%** | +6.3 |
| 8 | 2024 H1 | **54.8%** | +1.2 |
| 9 | 2024 H2 | 72.9% | -6.3 |
| 10 | 2025 H1 | 42.0% | +0.3 |

**Only 3 folds below 50%** (was 4). Key improvements: Fold 2 (48→57.5%), Fold 6 (47.3→58.4%).

---

### Cross-model comparison

| Model | Data | Params | Acc | Precision | F1 | Folds <50% |
|-------|------|--------|-----|-----------|-----|------------|
| CNN (Eval 23) | 4h | 3,581 | 61.5% | 65.7% | 0.690 | 4 |
| Transformer (Eval 2) | 4h | 4,113 | 61.3% | 67.2% | 0.670 | 4 |
| Transformer (Eval 3) | 1h | 4,113 | 58.7% | 61.2% | 0.682 | 4 |
| **Transformer (Eval 4)** | **1h** | **21,889** | **63.3%** | **65.8%** | **0.702** | **3** |

**Key insight:** 1h data + larger Transformer broke past the 4h ceiling. The combination of 6x more training data and a model big enough to use it pushed accuracy to 63.3% — first time above 62%. The Transformer's self-attention benefits more from data volume than the CNN's fixed filters.

---

## Confidence Threshold Analysis

Collected raw logits from walk-forward (10 folds, 20,435 test samples) and analyzed precision/accuracy at different confidence thresholds.

| Threshold | Trades | Per day | Precision | Accuracy |
|-----------|--------|---------|-----------|----------|
| 0.50 | 20,435 | 24.0 | 61.6% | 59.8% |
| 0.55 | 19,890 | 23.4 | 62.1% | 59.8% |
| 0.60 | 19,372 | 22.8 | 62.8% | 60.1% |
| 0.63 | 18,907 | 22.2 | 63.2% | 60.4% |
| 0.65 | 17,902 | 21.0 | 63.1% | 60.2% |
| 0.70 | 13,306 | 15.6 | 62.1% | 59.1% |
| 0.75 | 7,787 | 9.1 | 59.1% | 60.4% |
| 0.80 | 3,382 | 4.0 | 29.4% | 42.9% |

**Conclusion:** Confidence filtering gives minimal improvement (~1.5% precision gain at 0.63 threshold). The model's logits are too concentrated — it doesn't differentiate between "very sure" and "guessing." An ensemble or calibration approach would be needed for meaningful confidence-based filtering.

---

## 2+1 Spatial-Temporal Experiments (2026-04-10)

Tested adding 1 temporal layer to the 2-layer spatial architecture (Eval 4). Two variants:
- **v7 (simple 2+1):** Daily VP sum → 2 spatial layers → 1 temporal layer. Mean pool. v2 pipeline (60 feat). 31,073 params.
- **v8 (enriched 2+1):** End-of-day VP → 2 spatial layers (CLS pool) → enrich with daily candle + VP structure → 1 temporal layer. v1 pipeline (68 feat). 33,281 params.

Labels: FGI adaptive first-hit, TP=7.5% SL=3%. Colab A100, batch=512, mixed precision bf16.

### Eval 5 — v7 Simple 2+1 (walk-forward, 31,073 params)

**Overall:** 56.2% acc, 57.4% precision, F1=0.626

| Fold | Period | Acc | vs Eval 4 |
|------|--------|-----|-----------|
| 1 | 2020 H2 | 64.0% | -24.4 |
| 2 | 2021 H1 | 40.7% | -16.8 |
| 3 | 2021 H2 | 47.4% | +3.0 |
| 4 | 2022 H1 | 48.6% | -20.4 |
| 5 | 2022 H2 | 71.1% | +4.7 |
| 6 | 2023 H1 | 59.7% | +1.3 |
| 7 | 2023 H2 | 69.2% | -11.8 |
| 8 | 2024 H1 | 45.5% | -9.3 |
| 9 | 2024 H2 | 58.7% | -14.2 |
| 10 | 2025 H1 | 58.5% | +16.5 |

**Regime EV:** Bull long +0.96%, Bull short -0.62%, Bear long +0.57%, Bear short -0.07%
**Best strategy:** S1 Long only +0.79%

**Notes:** Worse than Eval 4 on most folds. Temporal layer adds 9.5k params (data/param ratio drops to 0.9:1 on fold 1). Overfits on early folds with insufficient training data.

---

### Eval 6 — v8 Enriched 2+1 (walk-forward, 33,281 params)

**Overall:** 58.0% acc, 60.4% precision, F1=0.610

| Fold | Period | Acc | vs Eval 4 |
|------|--------|-----|-----------|
| 1 | 2020 H2 | 53.7% | -34.7 |
| 2 | 2021 H1 | 42.4% | -15.1 |
| 3 | 2021 H2 | 65.8% | +21.4 |
| 4 | 2022 H1 | 55.6% | -13.4 |
| 5 | 2022 H2 | 59.9% | -6.5 |
| 6 | 2023 H1 | 49.6% | -8.8 |
| 7 | 2023 H2 | 64.5% | -16.5 |
| 8 | 2024 H1 | 64.3% | +9.5 |
| 9 | 2024 H2 | 57.2% | -15.7 |
| 10 | 2025 H1 | 66.0% | +24.0 |

**Regime EV:** Bull long +1.08%, Bull short -0.55%, Bear long +1.07%, Bear short +0.62%
**Best strategy:** S1 Long only +1.08%

**Notes:** Bear short is positive (+0.62%) — v8 is only model with all-positive bear EVs. Bear precision 81.6% (highest of all models). But bull long EV dropped vs v6 (+1.08% vs +1.57%). Late folds (8, 10) improved significantly — enrichment + temporal helps with more training data.

---

### 2+1 Comparison (all on 7.5/3 FGI labels)

| Model | Spatial | Temporal | Params | Acc | Bull long | Bear long | Bear short |
|-------|---------|----------|--------|-----|-----------|-----------|------------|
| v2 temporal | 1 | 1 | 23,041 | 56.7% | +0.41% | +1.04% | +0.42% |
| v6 enriched | 1 | 1 | 24,737 | 58.4% | **+1.57%** | +1.02% | -0.04% |
| v7 simple | 2 | 1 | 31,073 | 56.2% | +0.96% | +0.57% | -0.07% |
| v8 enriched | 2 | 1 | 33,281 | 58.0% | +1.08% | +1.07% | **+0.62%** |

**Conclusion:** Adding a second spatial layer does not improve over 1-layer variants. v6 (1+1 enriched) remains best for bull long EV. v8 has the broadest positive EV spread but lower peak performance. The extra spatial depth isn't earning its parameter cost at these data volumes.

---

## 15min Data Experiments (2026-04-11)

Tested whether 4x more data (340k rows vs 84k) improves v6 and v8. 15min resolution, bars_per_day=96, lookback=2880 (30 days).

### Eval 7 — v6 enriched 1+1 on 15min (24,737 params)

**Overall:** 55.6% acc, 60.7% precision, F1=0.525

| Fold | Period | Acc |
|------|--------|-----|
| 1 | 2020 H2 | 37.0% |
| 2 | 2021 H1 | 56.9% |
| 3 | 2021 H2 | 57.0% |
| 4 | 2022 H1 | 61.4% |
| 5 | 2022 H2 | 51.4% |
| 6 | 2023 H1 | 48.7% |
| 7 | 2023 H2 | 57.1% |
| 8 | 2024 H1 | 58.3% |
| 9 | 2024 H2 | 58.5% |
| 10 | 2025 H1 | 70.6% |

**Regime EV:** Bull long +0.64%, Bear long +0.69%, Bear short -0.06%
**Best strategy:** S1 Long only +0.66%

**Notes:** Worse than v6 on 1h (55.6% vs 58.4%). Early stopping at epoch 16 every fold — model converges in 1 epoch, remaining epochs just overfit. Architecture too small to benefit from 4x more data. LR=5e-4 is too high for 4x more gradient steps per epoch.

---

### Eval 8 — v8 enriched 2+1 on 15min (33,281 params) **NEW BEST ACCURACY**

**Overall:** **59.6% acc**, 63.0% precision, F1=0.603

| Fold | Period | Acc | vs v8 1h |
|------|--------|-----|----------|
| 1 | 2020 H2 | 36.2% | -17.5 |
| 2 | 2021 H1 | 52.6% | +10.2 |
| 3 | 2021 H2 | 58.6% | -7.2 |
| 4 | 2022 H1 | 61.3% | +5.7 |
| 5 | 2022 H2 | **66.8%** | +6.9 |
| 6 | 2023 H1 | **64.5%** | +14.9 |
| 7 | 2023 H2 | **77.2%** | +12.7 |
| 8 | 2024 H1 | 51.4% | -12.9 |
| 9 | 2024 H2 | 60.7% | +3.5 |
| 10 | 2025 H1 | 68.3% | +2.3 |

**Regime EV:** Bull long +0.82%, Bear long +0.80%, Bear short +0.48%
**Best strategy:** S1 Long only +0.81%
**Bear precision:** 79.0%

**Notes:** Best overall accuracy on 7.5/3 labels (59.6%). Fold 7 at 77.2% is highest single fold on 7.5/3. Clear monotonic improvement with data volume (folds 1→7). Fold 8 collapsed (51.4%) — 2024 H1 transition period. v8's 2nd spatial layer finally earning its param cost with sufficient data (8:1 ratio on fold 1 vs 0.9:1 on 1h).

However, **EVs are lower than 1h models**. v6 1h still best bull long (+1.57% vs +0.82%). 15min data adds noise that dilutes per-trade EV despite improving raw accuracy.

---

### 15min vs 1h full comparison

| Model | Data | Acc | Bull long | Bear long | Bear short | S1 EV |
|-------|------|-----|-----------|-----------|------------|-------|
| v6 enriched | 1h | 58.4% | **+1.57%** | +1.02% | -0.04% | — |
| v8 enriched | 1h | 58.0% | +1.08% | +1.07% | **+0.62%** | +1.08% |
| v6 enriched | 15m | 55.6% | +0.64% | +0.69% | -0.06% | +0.66% |
| **v8 enriched** | **15m** | **59.6%** | +0.82% | +0.80% | +0.48% | +0.81% |

**Conclusion:** 15min data helps v8 accuracy (+1.6%) but not v6 (-2.8%). However, per-trade EV is lower on 15min — the noise from finer resolution dilutes the signal. v6 on 1h remains the most profitable per trade (bull long +1.57%). v8 on 15min is most accurate but less profitable.

---

### Training optimization (2026-04-10)

Identified and fixed 6 bottlenecks in training pipeline:
1. **Batch 64 → 512** on A100 (GPU was ~5% utilized)
2. **Labels precomputed once** (was recomputed per fold, 30× redundant Python loops)
3. **Mixed precision bf16** (~2× A100 throughput)
4. **torch.compile** (10-30% free speedup)
5. **pin_memory + 4 workers** (overlapped CPU→GPU transfer)
6. **Vectorized index construction** (numpy vs Python loop)

Result: **12-19 min per model** (10-fold walk-forward), down from ~70 min.

---

## Infrastructure

### MPS (Metal) GPU acceleration
Enabled Apple MPS backend for training. ~2x speedup over CPU. Full walk-forward: ~25-30 min (was ~50-60 min).

### Pipeline optimization
VP structure feature computation optimized with cumulative sum + peak cache: **2.8 seconds** for 84k rows (was ~30 minutes). 640x speedup. Enables 15min data scaling.
