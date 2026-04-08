# 1D CNN Evaluations

**Why:** Treats the aggregated 30d VP as a spatial signal — conv filters learn smoothing + peak detection from data. Matches user's strategy of reading VP as a histogram shape.
**Strategy alignment:** High — see [STRATEGY.md](STRATEGY.md).

**Best architecture (used from Eval 4 onward):**
- Sum VP bins across 180 timesteps → (batch, 1, 50)
- Conv1d(1→4, k=5) → ReLU → Conv1d(4→4, k=5) → ReLU
- Flatten → concat with non-VP features → FC(215→16) → ReLU → FC(16→1)
- Weighted BCEWithLogitsLoss

---

## Phase 1: Architecture tuning (24h fixed label)

| Eval | Run | Channels | Pool | Features | LR | Dropout | Val Acc | Train Acc | Notes |
|------|-----|----------|------|----------|-----|---------|---------|-----------|-------|
| 1 | run_1775614321 | 8/16 | avg | 61 | 1e-4 | 0.1 | 48.4% | 52.9% | Underfitting |
| 2 | run_1775614348 | 8/16 | avg | 61 | 1e-3 | 0.1 | 49.0% | 53.4% | LR helped slightly |
| 3 | run_1775614379 | 16/32 | avg | 61 | 1e-3 | 0.0 | 50.1% | 54.5% | Wider channels |
| 4 | run_1775614449 | 4/4 | flat | 61 | 1e-3 | 0.0 | 50.8% | 55.7% | Flatten > avg pool |
| 5 | run_1775614882 | 4/4 | flat | 65 | 1e-3 | 0.0 | 52.9% | 59.3% | +candle features (upper_wick, lower_wick, body_dir, mid_range) |
| 6 | run_1775614924 | 4/4 | flat | 65 | 5e-4 | 0.3 | 50.5% | 54.3% | Regularized, gap closed but val flat |

**Takeaway:** Flatten preserves spatial position (which VP bin activated). Candle features add real signal (59.3% train). But 24h fixed label caps val at ~53%.

---

## Phase 2: Label design (first-hit TP/SL)

| Eval | Run | Label | TP/SL | Vol-scaled | Val Acc | Test Acc | Train Acc | Notes |
|------|-----|-------|-------|------------|---------|----------|-----------|-------|
| 7 | run_1775615624 | ±3% | 1:1 | No | **57.3%** | 45.0% | 61.4% | First-hit labels = biggest single improvement |
| 8 | run_1775615674 | ±3% | 1:1 | No | 49.5% | 42.7% | 57.2% | Avg pool, best val loss (0.6203) |
| 9 | run_1775615703 | ±3% | 1:1 | No | 53.6% | 44.6% | 57.6% | Avg pool, patience=15 |
| 10 | walk-forward | ±3% | 1:1 | No | — | 48.7%* | ~62% | 10 folds, range 40-60%, *overall |
| 11 | run_1775616543 | ±3% | 1:1 | Yes | 55.5% | 49.3% | 63.1% | Vol-scaling improves test generalization |

**Takeaway:** First-hit ±3% jumped val from ~53% to 57.3%. Vol-scaling trades val acc for better test generalization.

---

## Phase 3: Asymmetric TP/SL ratios

| Eval | Run | TP | SL | Ratio | Vol-scaled | Val Acc | Test Acc | Train Acc | Notes |
|------|-----|-----|-----|-------|------------|---------|----------|-----------|-------|
| 12 | run_1775616987 | 3% | 2% | 1.5:1 | Yes | 51.9% | 49.8% | 60.5% | Wide TP hurts — DOWN majority |
| 13 | run_1775617025 | 4% | 2% | 2:1 | Yes | 51.6% | 41.7% | 59.2% | Worse — UP too rare |
| 14 | run_1775617152 | 2.5% | 5% | **1:2** | Yes | **74.7%** | 47.9% | 66.6% | **Best val ever** — tight TP, wide SL |
| 15 | run_1775617201 | 3.3% | 5% | 1:1.5 | Yes | 66.6% | 50.7% | 65.7% | Mid-point, better test than 1:2 |

**Takeaway:** Inverted ratio (tight TP, wide SL) dramatically improves accuracy. 1:2 ratio hit 74.7% val — matches user's manual 70%. The question "will price go up 2.5% before dropping 5%?" is inherently more predictable.

---

## Phase 4: Walk-forward with 1:2 ratio

### Eval 16 — Walk-forward (10 folds, TP=2.5% SL=5% vol-scaled)

**Method:** Expanding window, 6-month folds from 2020-2025.

**Overall:** 54.5% accuracy, 66.3% precision, F1=0.646

| Fold | Test Period | Acc | Notes |
|------|------------|-----|-------|
| 1 | 2020 H2 | **82.4%** | Bull run — tight TP hit easily |
| 2 | 2021 H1 | 58.7% | Bullish |
| 3 | 2021 H2 | 55.8% | Transition |
| 4 | 2022 H1 | 54.8% | Bear start |
| 5 | 2022 H2 | **27.4%** | Deep bear — model predicts UP, price keeps falling |
| 6 | 2023 H1 | 55.4% | Recovery |
| 7 | 2023 H2 | 46.0% | Choppy |
| 8 | 2024 H1 | 58.8% | Bull run, val hit 87% |
| 9 | 2024 H2 | 46.5% | Correction |
| 10 | 2025 H1 | 52.8% | Mixed |

**Takeaway:** 1:2 TP/SL works well in bull markets (82%, 59%) and poorly in bear markets (27%). The asymmetry inherently favors UP predictions. Need regime-adaptive TP/SL: use 1:2 in bull, 2:1 in bear.

---

## Phase 5: Regime-adaptive TP/SL

### Eval 17 — Walk-forward (10 folds, regime-adaptive)

**Label:** Price > SMA(90d) = bull (TP=2.5%, SL=5%), price < SMA = bear (TP=5%, SL=2.5%). Vol-scaled.

**Overall:** **61.2% accuracy**, 65.2% precision, F1=0.690

| Fold | Test Period | Acc | Regime | Notes |
|------|------------|-----|--------|-------|
| 1 | 2020 H2 | **88.0%** | Bull | Strong trend, tight TP hit easily |
| 2 | 2021 H1 | 44.0% | Mixed | Regime transitions hurt |
| 3 | 2021 H2 | 48.2% | Transition | Bear starting |
| 4 | 2022 H1 | **77.8%** | Bear | Flipped ratio works — tight SL catches drops |
| 5 | 2022 H2 | **62.6%** | Bear | Was 27.4% without regime flip — biggest improvement |
| 6 | 2023 H1 | 56.1% | Recovery | Mixed regime |
| 7 | 2023 H2 | **76.4%** | Bull | Strong |
| 8 | 2024 H1 | 43.3% | Bull | Overfit to val |
| 9 | 2024 H2 | **72.6%** | Mixed | Good |
| 10 | 2025 H1 | 45.3% | Mixed | Transition period |

**Comparison across walk-forward approaches:**

| Method | Overall Acc | Worst Fold | Best Fold | F1 |
|--------|-----------|------------|-----------|-----|
| Symmetric ±3% (Eval 10) | 48.7% | 40.1% | 59.6% | 0.492 |
| 1:2 fixed (Eval 16) | 54.5% | 27.4% | 82.4% | 0.646 |
| **Regime-adaptive (Eval 17)** | **61.2%** | 43.3% | **88.0%** | **0.690** |

**Strategy link:** User adjusted TP/SL based on market trend intuitively. SMA(90d) proxy captures this. The regime flip eliminated the catastrophic bear-market failure (Fold 5: 27% → 63%).

---

## Phase 6: Neutral zone filtering (num_peaks=0)

### Eval 18 — Walk-forward (symmetric TP/SL on neutral bars)
**Overall:** 57.5% acc, 61.9% precision — worse than Eval 17

### Eval 19 — Walk-forward (skip neutral bars)
**Overall:** 58.6% acc, 61.1% precision — slightly better but fewer samples (3,991 vs 5,050)

**Conclusion:** Neutral filtering hurts with aggressive smoothing (sigma=2.0, 24% of bars have 0 peaks).

---

## Phase 7: Gaussian filter tuning + neutral retry

Changed VP smoothing from sigma=2.0/prominence=0.15 (aggressive) to sigma=1.0/prominence=0.08 (lighter). Peak distribution improved: 6.6% have 0 peaks (truly structureless), 56.5% have 1 peak, 36.7% have 2+ peaks.

### Eval 20 — Walk-forward (new sigma, no neutral filter)
**Overall:** 60.6% acc, 65.6% precision, F1=0.675

### Eval 21 — Walk-forward (new sigma, skip neutral)
**Overall:** 59.9% acc, 64.0% precision — skipping 6.6% of data didn't help

### Eval 22 — Walk-forward (new sigma, symmetric neutral)
**Overall:** 60.8% acc, 64.6% precision — essentially tied with baseline

**Conclusion:** Better smoothing gives more informative VP features (mean 1.4 peaks vs 0.8), but neutral filtering still doesn't improve results. Only 6.6% of samples are "truly structureless" — too few to matter.

---

## Phase 8: Sigma=0.8 + L2 regularization

### Eval 23 — Walk-forward (sigma=0.8, prom=0.05, no filter)
**Overall:** **61.5% acc**, 65.7% precision, F1=0.690 — **new best** (marginal)
53% of bars now have 2+ peaks. Better VP structure features.
4 folds below 50%: Fold 2 (48.0%), Fold 3 (43.6%), Fold 6 (47.3%), Fold 10 (41.7%)

### Eval 24 — Walk-forward (+ L2 weight_decay=1e-3)
**Overall:** 60.6% acc — L2 too strong, hurt performance

### Eval 25 — Walk-forward (+ L2 weight_decay=1e-4)
**Overall:** 60.3% acc — lighter L2 also didn't help

**Conclusion:** Sigma=0.8/prom=0.05 is marginally best (61.5%). L2 regularization doesn't help — the generalization gap is from regime shifts, not weight magnitude. **Eval 23 is current best. Moving to different model architectures next.**
