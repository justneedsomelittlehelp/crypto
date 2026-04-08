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
