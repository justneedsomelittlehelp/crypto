# Eval Audit — Post-Mortem (2026-04-12)

> **Retraction notice.** The Eval 11 / 12 / 17 / 18 headline results were distorted by a walk-forward label leak and a test-set sweep on the post-SL pause. With those bugs fixed, the deployable-strategy claim (+34.0% CAGR / -15.1% DD / 72% win rate / zero liquidations) does **not** reproduce. The strategy as specified is **not deployable**, and Phase 3 is no longer marked "DONE — ready for Phase 4." See below for the full audit, the honest numbers, and the decision.

## 1. What triggered the audit

Before moving to Phase 4 (Kraken integration) we did a bug sweep on the backtest pipeline. Two real issues surfaced:

1. **No embargo between walk-forward folds.** First-hit labels look up to `LABEL_MAX_BARS = 336` bars (14 days) into the future. Without an embargo, training labels computed for bars near `train_end` used prices inside the val/test window — a direct temporal leak.
2. **Post-SL pause was tuned on the test set.** The "24h unicorn fix" in Eval 17 and the Eval 18 fine-grained sensitivity sweep (21h–30h at 1h resolution) both selected the pause duration on the same period whose metrics we were celebrating. Classic in-sample tuning.

Two additional concerns turned out to be false alarms on closer inspection:

- First-hit labeling itself — that's a standard supervised-learning setup; TP/SL come from current-bar VP, not future data.
- Vol-adjusted TP/SL using a 30-bar rolling window — uses past data only, no leak.

## 2. What we changed

**Walk-forward embargo (`src/models/eval_v6_prime.py`):**

```python
EMBARGO_BARS = LABEL_MAX_BARS  # 336 bars = 14 days at 1h
embargo_td = pd.Timedelta(hours=EMBARGO_BARS)

train_mask = dates < (train_end - embargo_td)
val_mask   = (dates >= train_end) & (dates < (val_end - embargo_td))
test_mask  = (dates >= val_end)   & (dates < test_end)
```

Train and val shrink by 14 days at their tails, eliminating the label-horizon overlap into the next fold.

**Extended fold boundaries for an out-of-sample holdout:**

```python
FOLD_BOUNDARIES = [
    "2020-01-01", "2020-07-01",
    ...
    "2024-07-01", "2025-01-01", "2025-07-01",
    "2026-01-01", "2026-04-08",      # NEW — holdout
]
```

Fold 11 (test 2025-07 → 2026-01) and Fold 12 (test 2026-01 → 2026-04) were never touched by any previous walk-forward or sweep. They are the honest verdict.

**Locked pause as a design choice, not a tuned parameter (`src/models/run_backtest.py`):**
Removed the 21h–30h sweep; pause is fixed at whatever we decide and evaluated once.

## 3. Results under the clean protocol

Starting capital $5,000. Commissions and slippage applied (Kraken taker 26bps, maker 16bps, slip 5bps/side). Funding 0.01% / 8h when leveraged.

### Per-fold real long-only EV (unfiltered, 1x, no pause)

| Fold | Period | Acc | Long EV | Notes |
|---|---|---|---|---|
| 1  | 2020 H2 | 56% | +6.30% |  |
| 2  | 2021 H1 | 49% | −3.11% | blow-off top |
| 3  | 2021 H2 | 63% | +2.44% |  |
| 4  | 2022 H1 | 58% | −4.56% | Terra/Luna |
| 5  | 2022 H2 | 86% | −1.50% | FTX |
| 6  | 2023 H1 | 81% | +0.36% |  |
| 7  | 2023 H2 | 66% | +1.23% |  |
| 8  | 2024 H1 | 76% | +3.36% |  |
| 9  | 2024 H2 | 82% | +2.82% |  |
| 10 | 2025 H1 | 90% | +1.25% |  |
| **11** | **2025 H2** | **68%** | **−0.84%** | **holdout** |
| **12** | **2026 Q1** | **92%** | **−0.60%** | **holdout** |

The model still has a measurable edge on most in-sample folds. Holdout EV is mildly negative — a genuine but modest regime degradation, not a collapse.

### Original Eval 17 config under clean labels

Same config as the original "deployable" strategy — combined_60_20 filter + 100% sizing + 3x leverage + 24h post-SL pause — run on embargoed predictions:

| Scope | Return | CAGR | Max DD | Win % | Trades |
|---|---|---|---|---|---|
| **Claimed (Eval 17)** | — | **+34.0%** | **−15.1%** | **72.0%** | 50 |
| Full (embargo) | +25.9% | +4.5% | **−49.8%** | 41.5% | 123 |
| Holdout (embargo) | −15.0% | −81.2% | −18.5% | 5.9% | 17 |

Full-period backtest ran out of capital on 2025-10-28 — the engine logged 636 capital-starved skips. The holdout is a 17-trade stub on fresh capital.

### Why Eval 17 looked so good

1. **Asymmetry filter was riding leakage.** Embargoed numbers: `asym ≥ 2.0` alone drops to **19.8% precision / −0.95% EV per trade**. Combined `conf ≥ 0.60 AND asym ≥ 2.0` drops to **39.0% precision / +0.74% EV**. The Eval 11 / 12 claim of "78% precision, +3.49% EV" was contaminated first-hit labels letting the model retroactively identify which high-asymmetry setups actually resolved to TP.
2. **24h pause tuned on the same contaminated test window.** With clean labels, the window it was "catching" has nothing to catch. It's neither a unicorn nor a regularizer — it's overfitting artifact.
3. **3x leverage + 100% sizing hid a fragile signal.** On leaked labels the precision cluster was tight enough (~78%) that a 3x position could survive the losing streaks. On clean labels (~39% precision on the same filter), leverage amplified normal streaks into capital wipeouts.

## 4. Honest search for a deployable config

After the audit we ran several backtest configurations against the embargoed predictions without retraining, looking for a setup that was both backtest-positive and holdout-survivable.

### Confidence filter precision (clean labels, from eval JSON)

| Filter | Trades | Precision | Arith EV / trade |
|---|---|---|---|
| conf ≥ 0.60 | 10,759 | 72.5% | +1.21% |
| conf ≥ 0.65 | 9,784 | **75.6%** | **+1.36%** |
| conf ≥ 0.70 | 8,930 | 77.0% | +1.41% |
| conf ≥ 0.75 | 7,924 | 76.4% | +1.27% |
| asym ≥ 2.0 alone | 3,473 | 19.8% | **−0.95%** |

The model's raw confidence signal is real. The asymmetry filter actively hurts.

### Backtest sweep — key configs

All runs: $5,000 capital, `combined` filter variants removed, 5.6-year walk-forward (2020-08 → 2026-04).

| Filter | Sizing | Full CAGR | Full DD | Full Win% | Holdout Ret | Holdout DD | Note |
|---|---|---|---|---|---|---|---|
| conf_65 | 1x / 20% | +10.5% | −65.5% | 71.8% | −23.5% | −24.8% | capital starved early |
| conf_65 + pause24 | 1x / 20% | +7.7% | −63.0% | 74.1% | −18.3% | −19.7% | pause helps a bit |
| conf_75 + pause24 | 1x / 20% | +9.2% | −60.9% | 76.4% | −12.5% | −13.4% | still leaking capital |
| conf_65 + **guard** | 1x / 20% | +6.4% | −30.4% | 60.0% | −8.7% | −9.6% | DD halved |
| **conf_70 + guard + pause24** | **1x / 20%** | **+6.0%** | **−18.4%** | **65.0%** | **−4.8%** | **−4.8%** | **best** |
| conf_70 + guard + pause24 | **3x / 100%** | +11.5% | **−58.5%** | 69.7% | −12.4% | −12.1% | leverage destroys MAR |

Two observations that matter:

1. **The tp/sl guard (`min_asymmetry = 1.0`, reject trades where `tp_pct < sl_pct` at entry) is load-bearing.** Without it, the avg-loss / avg-win ratio stays inverted and the DD sits around −60% regardless of other knobs. With it, DD drops to −18% and the payoff structure becomes symmetric (+8.2% / −4.5%).
2. **Leverage strictly worsens risk-adjusted returns.** 1x / 20% → MAR 0.33. 3x / 100% → MAR 0.20. The Eval 17 numbers were a leverage-math fantasy on top of a leaked signal.

### The honest best config

**`conf_70 + min_asymmetry=1.0 + 1x leverage + 20% sizing + 24h post-SL pause`**

```
Period       : 2020-08 → 2026-04 (5.6 years)
CAGR         : +6.0%
Max DD       : −18.4%
Win rate     : 65.0%
Avg win/loss : +8.2% / −4.5%
Trades/year  : 23.5
Sharpe (trade): 0.526

Holdout (9 months, 2025-07 → 2026-04):
Return       : −4.8%
Max DD       : −4.8%
Trades       : 15
```

On paper this is a real strategy — modest CAGR, survivable DD, healthy payoff math, trade frequency you can actually live with. But:

- Holdout is **flat-to-negative**. 9 months, 15 trades, no money made.
- +6% CAGR with −18% DD does not clear passive SPY benchmarks (~10% CAGR, ~−25% DD, no engineering, no babysitting, tax-advantaged account options).
- The strategy wasn't obviously losing, but it wasn't obviously winning either.

## 5. Decision

**Phase 3 → Audit complete. Strategy is shelved, not deployed.** Phase 4 (Kraken integration) is not started on the basis of this model.

Rationale:

1. The previous "deployable" headline was an artifact. The honest version delivers single-digit CAGR, which does not justify the operational overhead, tax disadvantage, and custodial risk of running an automated crypto bot versus passive index exposure.
2. The core model signal **is real** (60–75% win rate on embargoed data, ~75% precision at conf ≥ 0.65). The specific hypothesis that failed is: *VP-only features on 1h BTC, long-only, can produce an edge that beats passive after honest accounting*. That hypothesis is rejected.
3. The infrastructure (walk-forward with embargo, holdout protocol, backtest engine, VP features, transformer architecture) is solid and reusable for any future hypothesis.

Not deleted. Not written off. Parked with documentation.

## 6. What's preserved for future work

- **Embargo + holdout protocol** (`eval_v6_prime.py`): don't regress.
- **Guard flag (`min_asymmetry=1.0`)**: legitimate pre-entry filter for any future TP/SL-aware strategy.
- **`v6_prime_predictions_preembargo.npz`**: the old leaky predictions cache, kept for forensic comparison only. Do not use for any new backtest.
- **`experiments/v6_prime_renewed_test/`**: frozen artifacts from the first post-embargo walk-forward (eval JSON, npz, backtest log). Reference point for any future rerun.

## 7. What a future attempt would need

If the project ever restarts, any new hypothesis should clear a higher bar:

1. **Honest beat over passive.** ≥ +15% CAGR at ≤ −25% DD on an embargoed holdout that the hypothesis has never seen.
2. **Hold up on a fresh holdout window** (another 3–6 months after the current one).
3. **Positive holdout even under realistic execution assumptions.** No leverage sleight-of-hand, no sweeps on the evaluation period.

Candidate directions that were *not* invalidated by this audit:

- **Regime-aware direction switching.** From the embargoed real-EV table: bull_long +1.64% (6,301 trades), bear_short +0.70% (6,056 trades), bull_short and bear_long both negative. The model predicts direction correctly by FGI regime, and we've been running long-only — leaving half the signal on the table. Worth testing if the project ever resumes.
- **Funding rate or OI as inputs**, integrated from training time (not fine-tuned onto a frozen backbone like Eval 9).
- **Different label definition** that isn't VP-derived first-hit, to address the payoff asymmetry at its root.

But none of those are in scope right now. The project is paused at the end of Phase 3 with a clean bill on the infrastructure and an honest verdict on the first hypothesis.

---

**Authoritative numbers to quote going forward:**

- Best honest strategy profile: **+6.0% CAGR / −18.4% DD / 65% win rate / 23 trades per year / holdout ≈ −5%**
- This does not beat passive SPY on either return or risk.
- Phase 3 = audit complete, strategy shelved, infra preserved.

---

## 8. Post-audit experiments (2026-04-12, same day)

The user chose to attempt one more round of experiments before re-shelving, explicitly as "not the type of person that gives up." Two leverage points were identified from the audit and the discussion that followed it:

1. **Architecture**: v6-prime's spatial attention is blind to VP structure features (`peak_strength`, `ceiling_dist`, `floor_dist`, `num_peaks`, etc). Those features are wired into the day-token enrichment bottleneck and the last-hour FC skip, but they never enter the spatial attention layer that's supposed to reason about bin positions. Adding a "structure context token" to the spatial sequence would give attention explicit access.
2. **Labels (Philosophy D)**: replace binary first-hit `{0, 1, NaN}` labels with continuous realized-P&L labels under the same TP/SL exit model. Hypothesis: continuous labels give the model richer information, and a regressor predicting expected return is a cleaner deployment target than a classifier predicting binary barrier hits.

Staged plan was three experiments, each with a clean checkpoint. Labels locked to the existing VP-derived formula with no changes. Walk-forward + 14-day embargo + holdout protocol preserved throughout.

### Stage 1 — Regression labels with v6-prime architecture (REJECTED)

Goal: isolate the effect of Philosophy D before touching architecture.

**First run collapsed in 5 epochs** (`Huber delta = 0.02` was the wrong choice for label distribution). Fix: `delta = 0.10` to put L2 behavior over the typical label range.

**Second run (after fix) failed on heterogeneous folds.** Key observations:

- **Fold 1** (2020 H2, smallest train set): train correlation reached 0.52, val correlation hovered near 0 with occasional negative spikes. Ensemble correlation was **−0.146**, worse than individual seeds — three seeds with correlated errors produced an ensemble more biased than any single member. Overfitting, not collapse.
- **Fold 5** (2022 H2, large heterogeneous train set spanning 6 years and 4 regimes): train correlation peaked at **0.07**. `best val_loss at epoch 1` for seeds 42 and 44 — the model's best generalization came after a single gradient step on random init. Training actively destroyed generalization. Ensemble produced 13 signals out of 3,696 bars (0.35%) with `real_EV@1.5% = −2.17%`.

**Diagnosis.** The VP-derived labels are nearly bimodal (cluster at `+tp_pct` and `−sl_pct` with a thin timeout tail). Regression loss functions assume approximately continuous targets — when given bimodal distributions, they minimize the squared error across the whole distribution and the optimum is the conditional mean, which is near zero. **Classification loss (BCE) is the structurally correct match for bimodal outcomes**; that's why the v6-prime binary version hit 86% accuracy on the exact same fold 5 window that Stage 1 could barely learn at all.

**Conclusion.** Philosophy D is rejected. The experiment taught us something specific and useful: *the magnitude information in P&L labels is not learnable by a ~25k-parameter model on this feature set*. The direction information is — that's what v6-prime was already capturing. Adding continuous magnitude as a target strictly made the optimization landscape worse.

Stage 1 artifacts kept in the repo for forensic reference:

- `src/models/eval_v9_regression.py` — regression eval script (kept but not in use)
- `src/models/run_backtest_regression.py` — regression backtest driver (kept but not in use)
- `src/backtest/engine.py` — `prediction_mode` / `min_predicted_return` config fields remain, inert in binary mode

### Stage 2 — Structure context token with v6-prime binary labels (NULL RESULT)

Goal: isolate the effect of the architectural change on the pipeline we already know works. Binary first-hit labels, BCE loss, label smoothing, AdamW, SWA, multi-seed ensembling — all held constant. **Only** change: spatial attention now operates on 52 tokens instead of 51, with a dedicated "structure context token" that projects the 8 VP structure features per day to `embed_dim`.

New files:

- `src/models/architectures/v9_wall_aware.py` — frozen architecture `TemporalEnrichedV9WallAware`, `+320` parameters over v6-prime (Linear(8, 32) for the structure embedder + 1 extra positional slot).
- `src/models/eval_v9.py` — eval script. Imports all non-model utilities directly from `eval_v6_prime.py` so the Stage 2 experiment is a strict A/B against the v6-prime baseline.
- `src/models/run_backtest_v9.py` — backtest driver. Same locked honest config as the audited best: `conf_65/70/75 + min_asymmetry=1.0 + 1x/20% + 24h post-SL pause`.

#### Overall eval metrics

| Metric | v6-prime (audited) | **v9 wall-aware** | Delta |
|---|---|---|---|
| n (all folds) | 31,243 | 31,243 | — |
| Accuracy | 70.52% | 69.88% | −0.64 |
| Precision | 60.69% | 60.11% | −0.58 |
| NPV | 79.03% | 78.20% | −0.83 |
| Long-only real EV/trade | +0.459% | +0.474% | +0.015 |

Essentially indistinguishable at the aggregate level. The structure context token produces a tiny EV uplift that's lost in the noise.

#### Fold-by-fold (v9 vs v6-prime long-only real EV %)

| Fold | Period | v6-prime acc/EV | **v9 acc/EV** | Verdict |
|---|---|---|---|---|
| 1 | 2020 H2 | 56.1% / +6.30% | **37.5% / +3.32%** | v9 much worse |
| 2 | 2021 H1 | 49.3% / −3.11% | 47.5% / −4.19% | v9 slightly worse |
| 3 | 2021 H2 | 63.1% / +2.44% | **85.6% / +7.38%** | **v9 dramatically better** |
| 4 | 2022 H1 | 58.0% / −4.56% | 60.9% / −4.51% | tied |
| 5 | 2022 H2 | 85.6% / −1.50% | 82.1% / −2.38% | v9 slightly worse |
| 6 | 2023 H1 | 81.4% / +0.36% | 79.8% / +0.67% | tied |
| 7 | 2023 H2 | 65.8% / +1.23% | 63.7% / +1.13% | tied |
| 8 | 2024 H1 | 76.3% / +3.36% | **81.6% / +5.01%** | **v9 better** |
| 9 | 2024 H2 | 81.5% / +2.82% | 76.4% / +1.63% | v9 worse |
| 10 | 2025 H1 | 90.3% / +1.25% | 90.7% / +1.40% | tied |
| **11** | **2025 H2 (holdout)** | 68.0% / −0.84% | **69.1% / −1.76%** | **v9 worse** |
| **12** | **2026 Q1 (holdout)** | 91.7% / −0.60% | 91.7% / −0.60% | tied |

**Two wins (folds 3, 8), five losses (1, 2, 5, 9, 11), five ties.** Fold 3 and fold 8 are dramatic — +4.94pp and +1.65pp of per-trade EV respectively, on folds with rich training data and well-defined VP walls. But those wins don't propagate to the holdout.

#### Backtest (conf_70_guard + 1x/20% + 24h pause — best filter for both)

| Metric | v6-prime | **v9** | Delta |
|---|---|---|---|
| Full return | +33.5% | +6.8% | −26.7 pts |
| CAGR | +6.0% | +1.9% | −4.1 |
| Max DD | −18.4% | −17.1% | +1.3 |
| Win rate | 65% | 57% | −8 |
| Trades | 117 | 100 | −17 |
| **Holdout return** | **−4.8%** | **−5.7%** | **−0.9** |
| Holdout trades | 15 | 10 | −5 |
| Holdout win rate | 13% | **0%** (0/10) | −13 |

**None of the three success criteria clear.** V9 does not beat v6-prime on full-period CAGR, does not produce positive holdout return, and fold-11 EV is worse than v6-prime's.

#### Diagnosis

The structure context token is not neutral — it's genuinely useful *on specific folds*. Fold 3 is the clearest signal: 22 percentage points of accuracy and 3× per-trade EV over v6-prime, with fewer trades (the model became more selective). Fold 8 shows the same pattern at a smaller magnitude. Both are folds with (a) ≥5 years of training data and (b) test windows with well-defined VP consolidation zones.

But the added capacity also hurts on folds where training data is small or heterogeneous. Fold 1 (smallest train) shows classic overfitting: v9 takes 321 more long trades than v6-prime and gets them systematically wrong (accuracy drops from 56% → 38%). The fold-1 damage roughly cancels the fold-3 gain, and the other folds are noise.

**Net effect on holdout: slightly negative.** Zero wins out of 10 trades on fold 11's holdout subset is not statistically powerful on 10 trades, but combined with the fold-11 full-window EV of −1.76% (worse than v6-prime's −0.84%), the direction is clear.

#### Conclusion

**Stage 2 is a null result on the success criteria and a slight net negative on holdout.** The architectural hypothesis — "VP structure features aren't reaching spatial attention, so the model can't reason about walls" — was falsifiable and has been falsified. The model *can* use the information when conditions are right (folds 3, 8), but can't use it consistently enough to improve generalization to post-2025-07 data.

Where this leaves us:

1. **The audited v6-prime configuration remains the best honest strategy.** +6.0% CAGR / −18.4% DD / holdout ≈ −5%. Still does not beat passive SPY.
2. **Two of the three major hypothesis legs have now been tested** (regression labels in Stage 1, structure context token in Stage 2). Both failed to clear the bar.
3. **Stage 3 (volume-weighted effective-distance labels) remains technically available** as the third leg — a label formula rewrite. But the pattern of two successive null-to-negative results lowers the prior that Stage 3 will produce the breakthrough. Honest odds estimate: 20–30% chance of clearing the holdout bar.

Stage 2 artifacts preserved in the repo for forensic reference:

- `src/models/architectures/v9_wall_aware.py` — frozen v9 architecture
- `src/models/eval_v9.py` — eval script
- `src/models/run_backtest_v9.py` — backtest driver
- `experiments/v9_test/eval_v9_results.json` — full fold-by-fold eval output
- `experiments/v9_test/eval_v9_log.txt` — training log
- `experiments/backtest_results_v9.json` — backtest output (full + holdout scopes)

---

**Current project status (end of Stage 2):** Audit-complete, Stage 1 and Stage 2 experiments both rejected, v6-prime honest configuration remains the best available strategy, and it does not beat passive SPY. Project remains paused at end of Phase 3. Stage 3 is the final available experiment but is not currently planned — user's call whether to run it.
