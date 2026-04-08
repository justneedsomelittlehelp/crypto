# TP/SL Sweep Evaluations

**Why:** The Transformer (Eval 4, 63.3%) was trained with TP=2.5% SL=5% (1:2). Accuracy alone doesn't determine profitability — expected value depends on the TP/SL ratio AND win rate at that ratio. A 55% win rate with 3:1 reward:risk beats 65% with 1:3.

**Model used:** Transformer (embed=32, heads=4, 2 layers, 21,889 params) on 1h data — same as Eval 4.
**Method:** 10-fold walk-forward (2020 H1 → 2025 H1, 6-month folds), full retrain per fold per config.
**Date:** Overnight 2026-04-07 → 2026-04-08.

---

## Sweep 1: Fixed 3% TP, Varying SL (wider stops)

Tests whether giving trades more room (wider SL) improves accuracy enough to offset bigger losses.

| Config | Ratio | Acc | Precision | EV/trade (acc) | Folds <50% |
|--------|-------|-----|-----------|----------------|------------|
| 3/3 (1:1) | 1:1 | 55.4% | 57.1% | +0.33% | 3/10 |
| 3/4 (1:1.33) | 1:1.33 | 56.3% | 59.2% | -0.06% | 2/10 |
| 3/5 (1:1.67) | 1:1.67 | 59.6% | 61.6% | -0.23% | 4/10 |
| 3/6 (1:2) | 1:2 | 60.5% | 63.2% | -0.56% | 3/10 |
| 3/7.5 (1:2.5) | 1:2.5 | 61.1% | 64.1% | -1.09% | 3/10 |
| 3/9 (1:3) | 1:3 | 67.9% | 71.6% | -0.85% | 1/10 |

**Finding:** Accuracy climbs with wider SL (55% → 68%) but EV goes negative after 1:1. More room = more losses that eat the small TP. Only 3/3 is EV-positive on the long side.

---

## Sweep 2: Fixed 3% SL, Varying TP (wider targets)

Tests whether letting winners run compensates for lower win rate.

| Config | Ratio | Acc | Precision | EV/trade (acc) | Folds <50% | Folds <BE |
|--------|-------|-----|-----------|----------------|------------|-----------|
| 4/3 (1.33:1) | 1.33:1 | 51.4% | 53.6% | +0.60% | 4/10 | 3/10 |
| 5/3 (1.67:1) | 1.67:1 | 59.4% | 59.7% | +1.75% | 2/10 | 0/10 |
| 6/3 (2:1) | 2:1 | 59.0% | 58.5% | +2.31% | 3/10 | 1/10 |
| 7.5/3 (2.5:1) | 2.5:1 | 60.8% | 58.7% | +3.38% | 2/10 | 0/10 |
| 9/3 (3:1) | 3:1 | 57.5% | 57.0% | +3.90% | 5/10 | 0/10 |

**Finding:** Every config from 5/3 onward is strongly EV-positive. Wider TP dominates because the model maintains ~58-60% precision even as the target moves further away.

---

## EV Correction: Precision vs Accuracy

The sweep code computed EV as `acc × TP - (1-acc) × SL`. This is **wrong** — accuracy includes correct "no trade" predictions (true negatives), which don't earn money. Correct EV uses **precision** (win rate when the model predicts "go long"):

| Config | EV (acc-based) | EV (precision-based) | Difference |
|--------|----------------|---------------------|------------|
| 3/3 | +0.33% | +0.43% | +0.10% |
| 5/3 | +1.75% | +1.78% | +0.03% |
| 7.5/3 | +3.38% | +3.16% | -0.22% |
| 9/3 | +3.90% | +3.84% | -0.06% |

Ranking holds, but precision-based numbers are more accurate for real trading.

---

## Trade Frequency & Daily Profit Analysis

Average trade duration (hours until TP or SL hit) and estimated daily EV:

| Config | Avg Duration | Trades/day | Precision EV/trade | Daily EV | Annual (est.) |
|--------|-------------|------------|-------------------|----------|---------------|
| 3/3 (1:1) | 54h | 0.22 | +0.43% | +0.10% | +35% |
| 4/3 (1.33:1) | 64h | 0.19 | +0.75% | +0.14% | +51% |
| 5/3 (1.67:1) | 72h | 0.17 | +1.78% | +0.30% | +108% |
| 6/3 (2:1) | 79h | 0.15 | +2.27% | +0.35% | +127% |
| **7.5/3 (2.5:1)** | **86h** | **0.14** | **+3.16%** | **+0.44%** | **+162%** |
| 9/3 (3:1) | 91h | 0.13 | +3.84% | +0.51% | +186% |

*Trades/day assumes model predicts "long" ~50% of the time, constrained by position overlap (can't enter while in a trade).*

**Key insight:** Wider TP means fewer trades/day, but the EV/trade gap is so large (9x between 3/3 and 9/3) that frequency can't compensate. Wider TP configs dominate daily profit.

---

## Long+Short Analysis

When model predicts 0 (bear), go short instead of sitting out. Short side flips the TP/SL: short wins SL% when price drops, loses TP% when price rises.

**Derived short-side win rate (NPV)** from accuracy, precision, and base rate using Bayes' theorem.

### Short-side EV by config

| Config | NPV (short win rate) | Short EV/trade | Short Daily EV |
|--------|---------------------|----------------|----------------|
| 3/3 | 0.524 | +0.15% | +0.02% |
| 3/9 | 0.622 | **+4.47%** | +0.49% |
| 5/3 | 0.589 | -0.29% | -0.04% |
| 7.5/3 | 0.671 | -0.46% | -0.03% |
| 9/3 | 0.583 | -2.00% | -0.18% |

**Pattern:** Configs with wide SL (sweep 1) have great short-side EV because the ratios flip favorably. Configs with wide TP (sweep 2) have negative short-side EV.

### Combined Long+Short Daily EV

| Config | Long Daily EV | Combined Daily EV | Short helps? |
|--------|--------------|-------------------|-------------|
| 3/3 (1:1) | +0.12% | +0.15% | Slightly |
| 3/9 (1:3) | -0.07% | **+0.42%** | **Yes — short side carries** |
| 5/3 (1.67:1) | +0.38% | +0.34% | No — hurts |
| 6/3 (2:1) | +0.47% | +0.42% | No — hurts |
| **7.5/3 (2.5:1)** | **+0.67%** | **+0.63%** | No — hurts |
| 9/3 (3:1) | +0.67% | +0.49% | No — hurts |

### Conclusion

**For wide TP configs (our best performers), adding shorts hurts total daily EV.** The short side is EV-negative and drags down the average. Better to sit in cash when the model says "don't go long."

The one exception: **3/9** — a short-biased strategy where the long side is barely profitable but the short side is a monster (+4.47%/trade). This is effectively a separate strategy.

---

## Recommendations

### Best single strategy: **7.5/3 (2.5:1), long-only**
- +3.16% EV/trade, +0.44% daily, ~+162% annual
- 0 folds below breakeven, 2/10 below 50%
- Trades roughly every 3.5 days

### Runner-up: **9/3 (3:1), long-only**
- Higher EV/trade (+3.84%) but 5/10 folds below 50% — less reliable
- Better annualized (+186%) if it holds up, but higher variance

### Potential dual-model approach
- **Long model:** trained on 7.5/3 labels, trades long
- **Short model:** trained on 3/9 labels (flipped), trades short
- Each model optimized for its direction. Not yet tested — would require separate sweep.

---

## Regime Detection: FGI vs SMA (2026-04-08)

Tested Fear & Greed Index (FGI >= 50 = bull) vs SMA(90d) for regime-adaptive labeling, both with 7.5/3 TP/SL.

### Overall

| Metric | FGI(50) | SMA(90d) |
|--------|---------|----------|
| Accuracy | **60.1%** | 54.1% |
| Precision | **62.4%** | 47.2% |
| F1 | **0.634** | 0.530 |
| Folds <50% | **1/10** | 4/10 |
| Samples | 31,835 | 31,175 |

### Regime-Split EV (precision-based, accounts for TP/SL flip)

| | FGI | SMA |
|--|-----|-----|
| Bull precision | 39.4% | 40.7% |
| Bear precision | **77.9%** | 72.9% |
| EV bull (7.5/3) | +1.14% | +1.27% |
| EV bear (3/7.5 flipped) | **+0.68%** | +0.15% |
| **EV combined** | **+0.94%** | +0.85% |

### Fold-by-fold

| Fold | Period | FGI | SMA | Diff |
|------|--------|-----|-----|------|
| 1 | 2020 H2 | 51.8% | 56.1% | -4.3% |
| 2 | 2021 H1 | 60.0% | 61.8% | -1.8% |
| 3 | 2021 H2 | 38.3% | 41.5% | -3.3% |
| 4 | 2022 H1 | 59.9% | 48.5% | +11.3% |
| 5 | 2022 H2 | 68.5% | 66.2% | +2.3% |
| 6 | 2023 H1 | 57.0% | 60.3% | -3.4% |
| 7 | 2023 H2 | **76.8%** | 57.5% | +19.3% |
| 8 | 2024 H1 | **67.6%** | 36.5% | +31.1% |
| 9 | 2024 H2 | 56.4% | 47.8% | +8.6% |
| 10 | 2025 H1 | 71.1% | 70.4% | +0.6% |

### Key findings

- **Bear-side longs are EV-positive** with both methods (precision > 71.4% breakeven)
- **FGI eliminates SMA warmup penalty** — no 90-day data loss per split
- **FGI's weakness:** overreacts in choppy regimes (fold 3, 2021 H2: FGI swung 10-84)
- **SMA's weakness:** lags at transitions, causing catastrophic folds (fold 8: 36.5%)
- **Bull precision is ~40% for both** — but EV is positive due to 7.5:3 reward:risk

### Open question

Bear-side longs work but have unfavorable 3:7.5 risk/reward. A dual-model approach (long model for bull, short model for bear) would give both sides favorable ratios.

Results: `experiments/regime_fgi_compare_results.json`, `experiments/regime_compare_results.json`
Scripts: `src/models/eval_regime_fgi.py`, `src/models/eval_regime_compare.py`

---

## Run folders

Sweep 1 (3% TP, vary SL) — 6 configs × 10 folds = 60 models:
- Results: `experiments/sweep_tpsl_results.json`

Sweep 2 (3% SL, vary TP) — 5 configs × 10 folds = 50 models:
- Results: `experiments/sweep_tpsl_wide_results.json`

Combined analysis: `experiments/sweep_analysis_combined.json`
Scripts: `src/models/sweep_tpsl.py`, `src/models/sweep_tpsl_wide.py`, `src/models/analyze_sweep.py`
