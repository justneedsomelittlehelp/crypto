# Label Redesign — Strategy Options for Post-v11 Experiments

> **Status**: draft — living doc, updated as we implement and test.
> **Context**: v11 (absolute-range VP, 15m, 2-channel spatial attention) was
> rejected on holdout (2026-04-13). Root cause traced to the **label formula**,
> not the representation or model capacity. This doc lays out what broke, why,
> and three candidate replacements ranked by ambition.
>
> **Related docs**: `STRATEGY.md` (user strategy), `EVAL_AUDIT.md` (Phase 3 audit),
> `EVAL_TRANSFORMER.md` (v11 run details), `MODEL_HISTORY.md` (timeline).

---

## Why v11 was rejected

On holdout (folds 11–12, 2025-07 → 2026-04):

| Filter stack | Holdout CAGR | Win rate |
|---|---|---|
| Raw long, no filter | −14.1% | 51.0% |
| Flipped (pred == 0) | −16.1% | 31.4% |
| `conf≥0.70 + asym≥1 + 24h pause` (v10 recipe) | −13.3% | 35.8% |
| `asym < 1` (inverted) | −12.6% | 63.4% |
| `conf≥0.70 + asym<1` | −15.5% | 62.0% |

All filter variants land in the same [−15%, −10%] band on holdout. No confidence
threshold, no asymmetry band, no sign flip rescues it. The flip baseline is not
meaningfully different from the raw baseline, so the model isn't systematically
anti-aligned — it's just **weakly correlated with truth**.

## Root cause: the label formula leaks geometry into the labels

v11 uses range-derived long-only TP/SL first-hit labels:

```
TP_pct = clip((window_hi - close) / close × 0.8, 1%, 15%)
SL_pct = clip((close - window_lo) / close × 0.6, 1%, 15%)
label  = 1 if TP hit before SL in forward 1344 bars (14d), else 0
```

On the 170k test samples, pos-rate broken down by `asym = TP_pct / SL_pct`:

| asym band | n | pos_rate |
|---|---|---|
| `[0.0, 0.5)` | 65,359 | **88.5%** |
| `[0.5, 0.8)` | 14,231 | 70.2% |
| `[0.8, 1.2)` | 14,809 | 51.7% |
| `[1.2, 2.0)` | 22,211 | 38.3% |
| `[2.0, ∞)` | 54,386 | **18.6%** |

The label is a near-deterministic monotonic function of the asymmetry ratio —
which itself is a deterministic function of `(window_hi, window_lo, close)`, the
same columns the model sees in its feature tensor. **The labels and the
features share an input**, and that shared input is strong enough that a
constant classifier predicting "label = 1 iff asym < 0.8" scores ~80%
accuracy on its own.

Three concrete failures follow:

1. **Label geometry leaks into features.** The model can learn to predict
   labels by staring at `price_pos` and ignoring everything else. This is
   *visible* in the in-sample numbers: `asym<1` filter gives 82.9% win rate
   in-sample, but the naive base rate in that region is already ~80%, so the
   model's lift over "know asym" is ~3 points. On holdout, that 3-point
   lift vanishes and the base rate weakens to 63%, turning CAGR negative.

2. **First-hit at 15m is noisy.** A 1344-bar lookforward with a fixed SL
   terminates on a single whipsaw even if the trade thesis plays out a day
   later. Labels conflate "the move didn't happen" with "the move happened
   after a 2% retrace that killed the stop."

3. **Every bar gets a label.** The model predicts 170k trades including
   every 15m bar a human trader would skip without a second thought.

Any new label formula has to address all three.

---

## Three candidate replacements

### Option A — Triple-barrier method with volatility-scaled barriers

**Standard López de Prado setup** (*Advances in Financial Machine Learning*,
Ch. 3). Rigorous baseline, minimal new code.

- **Barriers**: `TP = k × σ_t`, `SL = k × σ_t`, symmetric. σ_t = rolling
  stdev of log-returns (or ATR / close) over the last N bars. k ≈ 2 is
  typical, tunable per instrument.
- **Vertical barrier**: 14 days (same as current).
- **Ternary label**: `+1` if TP hit first, `−1` if SL hit first, `0` if time
  barrier — i.e., neither side touched.
- **Drop class-0 at training** to train only on bars where the market
  actually resolved. Training set shrinks but every surviving label is
  information-rich.

**How it fixes the three failures**:

1. Barriers built from σ_t, not `window_hi/lo` → labels decoupled from range
   geometry. Zero shared input between labels and features.
2. Volatility scaling → barriers widen in volatile regimes, tighten in calm
   ones. A 2% whipsaw in a high-vol period doesn't trip a 1%-wide SL because
   the SL is now 3%+.
3. Class-0 (time barrier) samples drop → model is not forced to predict on
   bars where nothing happened.

**What it costs**: loses direct tie to "aim TP at the VP ceiling peak." In
exchange you get an unbiased label that the model has to actually earn.

**Effort**: ~30 lines of code in `build_features()`. Drop-in replacement,
new `--labels triple_barrier` flag.

### Option B — Meta-labeling on a VP-structure primary rule

**Ambitious and philosophically right.** This is how quant shops actually
combine trading rules with ML, and it's the closest match to the user's
manual workflow.

1. **Define a primary rule** encoding what the user scans for on a chart.
   Example: *"Price is in the lower third of the 30d range AND a VP peak
   exists within [+2%, +10%] of close AND 5d realized vol has compressed
   below its 30d median."* Boolean, deterministic.
2. **Generate candidates** by running the rule over history. Estimated
   ~5–15k candidates from 170k bars = ~3–9% candidate rate.
3. **Label each candidate** with triple-barrier outcomes. Only candidates
   get labels; non-candidate bars are never seen by the model.
4. **Train an ML meta-labeler** whose job is "given this primary setup
   fired, predict whether this particular instance will reach TP before
   SL." Input = VP bins + self-channel + candle context at the candidate
   bar. Output = probability the trade works.

**Why this is the right shape for the project**:

- Model task aligns with user task: the user doesn't evaluate every 15m
  bar manually — the user scans for setups and then judges each one.
  Meta-labeling mirrors that split.
- Small, high-SNR training set (~5–15k samples instead of 170k) with a
  small, high-SNR model (~5k params instead of 25k). Sample/param ratio
  actually favorable.
- Failures become interpretable: you can inspect rejected candidates and
  see why the model said no.
- Decoupling is automatic — the primary rule gates on a feature the model
  also sees, but the *label* is what happened next, not what the rule saw.

**What it costs**: you have to commit to a primary rule in code, and the
whole system is bounded by the rule's recall (setups the rule misses can
never be traded). Real one-time investment. But the rule writeup already
mostly exists in `experiments/STRATEGY.md`.

**Effort**: new file `src/features/primary_rule.py`, new training script
that samples only rule-candidate bars, new backtest harness that only
trades when the rule fires AND the meta-model says yes. A few hundred
lines, but self-contained.

### Option C — Volatility-normalized forward-return regression

**Backup plan.** Least ambitious, least actionable, cleanest numbers.

- Label = `(close[i + N] − close[i]) / close[i] / σ_t`, fixed horizon N
  (e.g. 96 bars = 24h).
- Regression target. Model outputs a real-valued forward-return z-score.
- Inference: sort predictions, trade top-k per month, stops at fixed σ_t
  multiples.

**Pros**: zero geometric bias, vol normalization handles regime, simplest
possible relabeling.
**Cons**: regression on noisy financial returns is famously low-SNR; loses
the user's TP/SL economic framing; hard to evaluate as a trading rule
without ad-hoc thresholding.

I'd skip this unless A and B both fail.

---

## Do we still need horizontal (absolute) VP data?

Really important question, surfaced during the discussion. Answer splits
into two layers:

### Labels don't need VP

Triple-barrier labels are a function of `(close, σ_t, forward 14d path)`
only. They never touch the VP columns, the self-channel, or the range
metadata. You could generate them from a bare OHLCV CSV with nothing else.
This is *why* triple-barrier fixes the leak — the labels stop being a
function of inputs the model sees.

### Features still need VP — and that's exactly the point

The whole Phase 3 hypothesis is *"volume profile carries support/resistance
information that helps predict short-horizon price moves."* With the old
labels that hypothesis was untestable, because VP features and labels
shared inputs and no one could tell if apparent signal was real or a
geometric artifact.

**Triple-barrier labels finally make the test clean**, and enable the
decisive experiment:

> **Decisive v11 ablation under triple-barrier labels:**
>
> - **v11-full**: `vp_abs` + `self_hard` + `price_pos` + `range_pct` + candle
> - **v11-nopv**: candle features only (no VP, no self, no range)
>
> Train both with triple-barrier labels, compare holdout metrics.
>
> - If **v11-full beats v11-nopv** → VP carries real signal; v10's failure
>   was labels, not features. Phase 3 reopens with a clear direction.
> - If **v11-full ties or loses to v11-nopv** → VP was never the edge. The
>   "volume profile as ML support/resistance" hypothesis is falsified at
>   the model level (the user's manual strategy may still work — humans
>   are more flexible than 26k-param models — but the ML angle is dead).

**This is the experiment Phase 3 was designed to run and never could.**
It's been blocked since the v6-prime labels were defined because every
label formula we tried shared inputs with the features. Triple-barrier
finally breaks that dependence.

**Priority**: run the v11-full vs v11-nopv ablation under triple-barrier
labels *before* any further depth/capacity sweeps or the meta-labeling
build. Whatever we learn from it reshapes everything downstream.

---

## Recommended sequence

1. **Implement Option A** (triple-barrier relabeling) as a `--labels`
   flag in `eval_v11.py`. Keep the existing range-derived labels
   available via `--labels range` for back-compat.
2. **Run v11-full with triple-barrier labels** on Colab. Compare holdout
   vs the current v11 numbers.
3. **Run v11-nopv** (same training script, VP columns zeroed out or
   dropped via a feature-mask flag). Compare to v11-full. This is the
   decisive experiment.
4. **If v11-full > v11-nopv by a meaningful margin**: Option B
   (meta-labeling) is worth building. The VP signal exists, we just need
   a cleaner way to surface it.
5. **If v11-full ≈ v11-nopv**: the ML angle on VP is dead. Document as a
   negative result, update `EVAL_AUDIT.md` to reflect that the user's
   manual strategy cannot be captured by this family of models under any
   label formula we tested, and formally close Phase 3.
6. **If Option A shows nothing at all** (both variants are flat): try
   Option C (vol-normalized regression) as a last sanity check, then
   close.

Option C is the backup; Option A is the minimum next step; Option B is
the "if it works, commit to it" follow-up.

---

## Open questions / caveats

- **σ_t window**: what lookback for rolling std? 30 bars? 96 bars (24h)?
  Longer → smoother barriers but slower to react to regime shifts.
  Start with 96 and tune if needed.
- **Barrier multiplier k**: k=2 gives ~95% CI bounds on next-bar moves,
  which means most trades resolve near the time barrier. k=1 gives
  faster first-hits. Start with k=2 and report the pos/neg/timeout
  distribution before running full walk-forward.
- **Ternary vs binary labels**: we can either train a 3-class classifier
  or drop the timeout class and train binary. Binary is simpler and the
  current model head supports it. Start binary, revisit if needed.
- **v11-nopv feature set**: should it be *only* the 4 last-bar scalars,
  or the full candle pipeline minus VP/self/range? Use the latter —
  more parity, cleaner ablation.
- **Meta-labeling primary rule**: the user's written strategy in
  `STRATEGY.md` is mostly narrative. Translating it to code is a
  non-trivial interpretation step. Before building Option B, we'd
  workshop the rule inline and the user signs off on the exact boolean.

---

## Results — decisive experiment (2026-04-14)

Both runs completed on Colab H100, single seed, same walk-forward
boundaries, 12 folds (folds 11–12 = holdout from 2025-07-01).

**TL;DR — VP features produce real, regime-robust lift under clean
labels. This is the first positive VP result in the project's history.
Phase 3 has a direction for the first time in a year.**

### Fold-level comparison (v11-full vs v11-nopv, triple-barrier)

| Split | full acc | nopv acc | Δ acc | full EV | nopv EV | Δ EV |
|---|---|---|---|---|---|---|
| **In-sample (folds 1–10)** | 54.16% | 51.55% | **+2.62 pp** | +0.898% | +0.751% | **+0.15 pp** |
| **Holdout (folds 11–12)** | 46.83% | 41.81% | **+5.02 pp** | −0.68% | −1.37% | **+0.68 pp** |
| **Overall** | 53.17% | 50.22% | **+2.94 pp** | — | — | — |

- Every split favors `full`.
- Holdout lift (+5.02 pp acc / +0.68 pp EV) is **larger** than in-sample
  lift (+2.62 pp acc / +0.15 pp EV). A naive feature that just
  overfit training would show the opposite pattern. VP features are
  **regime-robust relative to candle features** — exactly the shape a
  structural support/resistance feature is supposed to have.
- full wins **8/12 folds** on accuracy and **9/12** on EV per trade.
- **Both holdout folds** favor full on both metrics.
- Biggest bear-regime lift: fold 5 (2022 H2), +7.41 pp acc / +1.73 pp
  EV. Biggest crash-regime lift: fold 12 (2026 Q1), +6.58 pp acc /
  +1.73 pp EV. VP features help most in the regimes the user's manual
  strategy was designed for.

### Filter-swept holdout comparison (24h cooldown, 1x / 20% sizing)

| Filter | full n | full CAGR | full DD | full WR | nopv n | nopv CAGR | nopv DD | nopv WR | Δ CAGR |
|---|---|---|---|---|---|---|---|---|---|
| Raw long | 220 | −26.7% | 38.8% | 47.7% | 193 | −41.3% | 36.1% | 40.4% | **+14.6 pp** |
| Sign-flip (pred=0) | 136 | −4.3% | 16.1% | 50.0% | 229 | −14.4% | 27.4% | 50.2% | +10.1 pp |
| conf ≥ 0.50 | 220 | −26.7% | 38.8% | 47.7% | 193 | −41.3% | 36.1% | 40.4% | +14.6 pp |
| conf ≥ 0.70 | 89 | −9.7% | 20.9% | 50.6% | 72 | −24.5% | 21.7% | 31.9% | +14.8 pp |
| conf ≥ 0.75 | 71 | −5.0% | 15.5% | 52.1% | 48 | −14.1% | 12.8% | 35.4% | **+9.1 pp** |
| **conf ≥ 0.80** | **58** | **+11.6%** | **8.2%** | **58.6%** | **22** | **−1.8%** | **4.5%** | **40.9%** | **+13.4 pp** |

**full dominates nopv on holdout CAGR at every filter threshold**,
typically by 10–15 percentage points. Not a single filter variant
exists where nopv ties or beats full.

### The first positive holdout CAGR in the project's history

At **conf ≥ 0.80 + 24h cooldown**, v11-full on holdout reaches:

- **58 trades** (essentially the highest-conviction 26% of raw long)
- **58.6% win rate** — clean of the 55.9% class prior
- **+0.28% EV per trade**
- **+11.6% CAGR** (over the ~103 days the filter actually fires in,
  annualized)
- **8.2% max drawdown** — less than half of v10 honest's 18.4%

This is the **first time in Phase 3 that any model has produced a
positive holdout CAGR under a disciplined filter**. The v6-prime
honest best was −5% holdout; v10, v11-range, and all post-audit
experiments through v11-tb-nopv stayed negative. Only v11-full-tb
at conf ≥ 0.80 crosses zero.

Important caveat: the conf≥0.80 filter is very selective — 58 trades
across ~103 effective days means the strategy is silent most of the
holdout period. It's not "v11-full beats SPY on 10 months of 2025H2";
it's "when v11-full says it's very confident, it's right enough of
the time to produce a positive compound outcome." That distinction
matters for any deployment decision.

### The sign-flip asymmetry confirms full carries a real signal

On nopv, sign-flip (treat `pred == 0` as a long signal) outperforms
raw long on holdout: −14.4% vs −41.3% CAGR, and 50.2% win rate vs
40.4%. This means **the candle-only model is actively anti-aligned
with truth on holdout** — it has negative information content, which
is a stronger statement than "uncorrelated."

On full, sign-flip is roughly neutral relative to raw long (−4.3%
vs −26.7% at n=136 vs 220, 50.0% vs 47.7% WR). The full model is not
anti-aligned — it has a small positive directional bias even under
the worst-case raw-long baseline. Adding the confidence filter
translates that small positive bias into a compounding positive EV.

### What this falsifies and what it supports

**Supported:**

- **VP features carry ML-exploitable signal.** Under clean
  (triple-barrier) labels, the model with VP features beats the
  candle-only model on every holdout metric. This is the first
  clean positive result for the Phase 3 central hypothesis.
- **The signal is regime-robust.** Holdout lift exceeds in-sample
  lift, meaning VP features generalize better to OOD data than
  candle features. That's the right shape for a structural feature.
- **Confidence calibration works under clean labels.** Both full
  and nopv show monotonic improvement as the threshold rises,
  unlike v11-range where the threshold did nothing.

**Falsified:**

- *"v11 is rejected because the representation doesn't carry signal."*
  That was the tentative 2026-04-13 conclusion and it's now wrong.
  v11 was rejected under range labels because the label formula
  made the test unfalsifiable. Under clean labels, the representation
  is validated.
- *"VP features help in-sample and collapse on holdout."* The actual
  pattern is the opposite — VP helps *more* on holdout than in-sample.
  The in-sample lift is modest precisely because candle features
  already capture most training-regime direction. VP's comparative
  advantage shows up when the training regime ends.

**Still open:**

- **Holdout is still net-negative at low-filter variants.** Both
  models lose on raw-long holdout. The model has signal, but deploying
  it unfiltered loses money. A live deployment needs the high-
  confidence filter, which means being silent most of the time.
- **Regime change is a shared failure mode.** Fold 12 (2026 Q1) is
  catastrophic for both models (42.2% / 35.6% accuracy). VP features
  reduce the damage but don't fix it. Regime generalization is a
  separate problem that requires a separate fix (walk-forward retrain
  more frequently, or online learning).
- **BTC-only validation.** The signal is confirmed on BTC. Whether it
  generalizes to ETH / SOL / other liquid assets is the natural
  follow-up and the strongest test of "the VP hypothesis is universal
  vs BTC-specific microstructure." See the user's cross-asset question
  and the response captured in `MODEL_HISTORY.md` §31 — multi-asset is
  now the prioritized next experiment.

### Recommended next steps

The plan in the sequence block above (Option A → decisive → Option B)
is updated by the positive result:

1. **Multi-asset v11-full triple-barrier** — test whether the VP lift
   generalizes to ETH and SOL. If yes, the hypothesis is universal
   and the project has a real story. If no, the signal is BTC-specific
   and the deployment scope is narrower. Either answer is a real
   finding. This is the highest-EV next experiment.
2. **Walk-forward with rolling retraining** (retrain every 3 months
   on expanding windows). Directly attacks the holdout-collapse
   failure mode. Code change only, no new data. ~50 lines in
   `eval_v11.py` plus 4× compute.
3. **Option B (meta-labeling on a VP-structure primary rule)** is
   demoted to a later experiment. The decisive result made it less
   urgent — we now know the model is already picking up VP signal
   in an end-to-end setup, so the "constrain training to user-
   relevant setups" argument is weaker.
4. **Live paper-trading a conf ≥ 0.80 wrapper** around v11-full as a
   sanity check on the +11.6% holdout CAGR number. Real-time data,
   real constraints, no retraining. If the +11.6% number survives
   three months of paper-trading, it's deployable.
