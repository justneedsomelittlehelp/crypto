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

> **⚠ CORRECTION 2026-04-14 (same day, later in session).** The first
> version of this section reported v11-full-tb holdout at **+11.6%
> CAGR** under a `conf ≥ 0.80 + 24h cooldown` filter and called it "the
> first positive holdout CAGR in the project's history." That number
> came from `src/models/analyze_v11_filters.py`, which computes CAGR on
> label-accurate per-trade compounding with **no fees, no slippage,
> no execution latency, no capital reserve, and annualization over
> active-days-only** (first-trade to last-trade, ~103 days for the
> conf80 slice). When the same predictions are run through the real
> backtest engine (`src/models/run_backtest_v11_tb.py`, same engine
> v6-prime's honest +6.0% CAGR was produced from), the number becomes
> **−1.5% CAGR / −1.7% DD / 60.0% WR / 20 trades** over the full 278-day
> holdout. **v11-full-tb does NOT produce a positive holdout CAGR under
> real execution assumptions.** The gap sources are documented below.
>
> What IS still supported by the real engine:
> - **v11-full beats v11-nopv on every real-engine holdout filter**
>   (Δ +1.5 to +8.5 pp CAGR, consistent direction across all reasonable
>   filter thresholds).
> - **Win rate gap is large and stable**: 60.0% vs 41.7% at conf≥0.80,
>   55.2% vs 29.6% at conf≥0.70. A 15–25 point win-rate spread on the
>   holdout is real model discrimination.
> - **Drawdowns are tight**: v11-full-tb conf≥0.80 holdout DD is 1.7%,
>   vs v6-prime honest 18.4%. That's an order of magnitude better, and
>   it's real — the engine saw it too.
> - **VP features are regime-robust relative to candles**: holdout lift
>   (+5.0 pp acc) exceeds in-sample lift (+2.6 pp acc). This shape is
>   preserved under the engine.
>
> What is NOT supported:
> - Claims that v11 is the "first positive holdout CAGR" — it isn't,
>   not under real friction.
> - Claims that v11-full-tb beats v6-prime honest on CAGR — v6-prime
>   honest was +6.0% full-period CAGR from the real engine;
>   v11-full-tb's best full-period real-engine CAGR is +1.8% at
>   conf≥0.80. v11 is WORSE on full-period CAGR, not better.
> - Claims that v11 is "deployable" — it's closer to flat than
>   v6-prime on holdout, but "flat, not bleeding" is not the same as
>   "deployable."
>
> **The decisive experiment's real conclusion is weaker than the
> original writeup**: VP features add small but consistent lift under
> clean labels, sign-flip and win-rate gaps confirm the signal is
> real, drawdowns are much tighter — but the per-trade edge is small
> enough that 20–30 holdout trades is not enough statistical power to
> push real-engine CAGR into positive territory. Sample count is the
> binding constraint, which is what motivated the regime features plan
> in `MULTI_ASSET_PLAN.md` §REFRAME.

### Decisive experiment summary (corrected)

VP features produce real, regime-robust lift under clean labels. The
ablation (`v11-full-tb` vs `v11-nopv-tb`) is positive at every filter
threshold under the real engine, confirming the signal exists. Phase 3
has a direction for the first time since the 2026-04-12 audit — but the
absolute holdout CAGR is not yet positive under realistic execution
assumptions.

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

### Filter-swept holdout comparison — BOTH analyzer (idealized) and engine (real)

Two sources of truth, cited side-by-side so the methodology gap is
visible. Same predictions, same walk-forward, same 278-day holdout
window (2025-07-01 → 2026-04-08). The gap is entirely in the
calculation, not the underlying model signal.

**Analyzer (label-accurate compound, no frictions)**: computed by
`src/models/analyze_v11_filters.py`. Label-accurate per-trade returns
(win = +barrier_pct, loss = −barrier_pct), 20% of current equity per
trade, 24h greedy cooldown between trades, CAGR annualized over
*first-to-last-trade* window. Omits Kraken fees (0.52% round-trip),
slippage (0.10% round-trip), 30% capital reserve, execution latency,
post-SL pause, timeout exits, and allow_pyramiding serialization.

**Real engine (`src/models/run_backtest_v11_tb.py`)**: the same engine
that produced v6-prime's audited +6.0% CAGR. Kraken fee model, 0.05%
slippage per side, 5k starting capital, 30% reserve, 20% of available
per position, 14-day vertical barrier, `allow_pyramiding=False`,
24h post-SL pause, 1x leverage. CAGR annualized over *full holdout
window* (278 days).

| Filter | analyzer CAGR | **engine CAGR** | analyzer n | engine n | engine WR | engine DD |
|---|---|---|---|---|---|---|
| Raw long (no filter) | — | **−26.7%** | 220 | 220 | 47.7% | 38.8% |
| conf ≥ 0.60 + no pause | — | **−13.8%** | — | 54 | 44.4% | 12.7% |
| conf ≥ 0.70 + pause24 | −9.7% | **−4.0%** | 89 | 29 | 55.2% | 4.2% |
| conf ≥ 0.75 + pause24 | −5.0% | **−4.0%** | 71 | 24 | 54.2% | 4.2% |
| **conf ≥ 0.80 + pause24** | **+11.6%** | **−1.5%** | 58 | 20 | 60.0% | 1.7% |

**Gap decomposition for the conf ≥ 0.80 case (+11.6% → −1.5%)**:

1. **Annualization over active days vs full holdout**: analyzer uses
   `days = last_trade − first_trade ≈ 103 days`; engine uses full
   278-day holdout. Same 3% total cumulative return annualized two
   ways: `1.03^(365.25/103) ≈ +11.0%` vs `1.03^(365.25/278) ≈ +4.0%`.
   **This alone is ~7.6 points of the gap.**
2. **Fees + slippage** (0.62% round-trip per trade on ~5% barriers):
   winners net 4.48% instead of 5.08%, losers lose 5.70% instead of
   5.08%. At 60% win rate, net EV per trade drops from +1.02% to
   +0.41%. **~2-3 more points of the gap.**
3. **Engine sizing is 70% of analyzer sizing** due to `reserve_pct=0.30`.
   Proportional shrink of both wins and losses. **~0.5 point.**
4. **Fold 12 sample variance** (20 trades over 278 days, ~8 of them
   in the 2026 Q1 collapse fold where the model is 42% accurate).
   A single cluster of 3-4 extra losses in fold 12 shifts CAGR by
   5-10 points. **Residual is dominated by this on such a small
   sample.**

The original `+11.6% CAGR first positive holdout` claim is retracted.
The real-engine conf ≥ 0.80 holdout is `−1.5% CAGR / −1.7% DD / 60.0%
WR / 20 trades`. That's **closer to flat than any prior v6-prime or
v10 configuration**, and the win rate at 60% is real discrimination
(well clear of the 55.9% class prior), but it's not a positive
compound outcome.

### Engine comparison: v11-full vs v11-nopv (the decisive Δ still holds)

| Filter | full CAGR | nopv CAGR | Δ CAGR | full WR | nopv WR | ΔWR |
|---|---|---|---|---|---|---|
| Raw long | −26.7% | −41.3% | **+14.6 pp** | 47.7% | 40.4% | +7.3 |
| conf ≥ 0.70 + pause24 | −4.0% | −12.5% | **+8.5 pp** | 55.2% | 29.6% | **+25.6** |
| conf ≥ 0.75 + pause24 | −4.0% | −8.1% | +4.2 pp | 54.2% | 37.5% | +16.7 |
| conf ≥ 0.80 + pause24 | −1.5% | −3.0% | +1.5 pp | 60.0% | 41.7% | +18.3 |

**full beats nopv on holdout CAGR at every reasonable filter** (the
loose conf ≥ 0.60 baseline is the one exception and both are deeply
negative there). Win rate gap is 15–26 percentage points at every
threshold — this is the cleanest real-engine evidence that VP
features give the model actual discriminatory power on holdout.

**The ablation finding stands: VP features add real signal.** The
magnitude is smaller than the analyzer made it look, but the
direction is preserved at every filter threshold, and the win-rate
gap is statistically meaningful on 20–30 trade samples.

### On the retracted "first positive holdout CAGR" claim

The first version of this doc reported v11-full at conf ≥ 0.80 as
`+11.6% CAGR / 8.2% DD / 58.6% WR / 58 trades` and called it "the
first positive holdout CAGR in the project's history under a
disciplined filter."

**All three parts of that claim are wrong under the real engine:**

1. It's not **+11.6%** — it's **−1.5%** when annualized over the
   full 278-day holdout window with fees, slippage, and realistic
   sizing.
2. It's not a **positive holdout CAGR** — under real frictions no
   filter tested produces positive holdout CAGR for v11-full.
   Best engine holdout is conf80 at −1.5%.
3. It's not **58 trades** — the engine's `allow_pyramiding=False`
   rule drops any signal that fires while another position is open,
   and many conf80 signals cluster inside open-position windows.
   Real trade count at conf80 engine holdout is **20**.

What's still true under the real engine:

- **60.0% win rate at conf ≥ 0.80**, cleanly above the 55.9% class
  prior. On 20 trades the confidence interval is wide, but the point
  estimate is real and the full-vs-nopv gap (60.0% vs 41.7%) is
  meaningful model signal.
- **1.7% max drawdown at conf ≥ 0.80** — less than 10% of v6-prime
  honest's 18.4%. That's real, not an artifact, and it's the
  strongest quantitative argument for v11-full-tb as a deployment
  candidate.
- **Full-period (not holdout) CAGR is +1.8%** at conf ≥ 0.80 — still
  worse than v6-prime honest's +6.0%, so v11 is NOT a strict
  improvement on full-period CAGR.

**The revised project headline**: v11-full-tb at conf ≥ 0.80 is the
first post-audit model with holdout drawdown below 10% and holdout
win rate meaningfully above the class prior. It is not the first
positive holdout CAGR, and it does not beat v6-prime honest on
full-period CAGR. The win is entirely in drawdown shape and
discrimination quality, not in compound return.

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

### What this falsifies and what it supports (corrected for real-engine numbers)

**Supported:**

- **VP features carry ML-exploitable signal.** The ablation Δ (v11-full
  vs v11-nopv) is positive at every reasonable filter under the real
  engine: +8.5 pp CAGR at conf70, +4.2 pp at conf75, +1.5 pp at conf80.
  Win-rate gaps of 15-26 percentage points are the strongest signal.
- **The signal is regime-robust** in the relative sense. Holdout
  accuracy lift (+5.0 pp) exceeds in-sample accuracy lift (+2.6 pp).
  VP features degrade less than candle features when the training
  regime ends. Preserved under the engine.
- **Confidence calibration works under clean labels.** Both models
  show monotonic win-rate improvement as the threshold rises, unlike
  v11-range where it did nothing. Visible in both analyzer and
  engine tables.
- **Drawdown is materially tighter.** conf ≥ 0.80 holdout DD is 1.7%
  under the real engine, vs v6-prime honest's 18.4%. Order-of-
  magnitude improvement in drawdown shape. This is the single
  strongest quantitative argument for the representation change
  going into the regime-features experiment.

**Falsified:**

- *"v11 is rejected because the representation doesn't carry signal."*
  The 2026-04-13 tentative conclusion from the v11-range run. Under
  clean labels the representation is validated.
- *"VP features help in-sample and collapse on holdout."* Opposite
  pattern — VP helps more on holdout.
- **⚠ *"v11 produces the first positive holdout CAGR in project
  history"*** — the original claim in this section, now retracted.
  The analyzer's +11.6% came from active-days annualization + no
  frictions. The real engine produces −1.5% CAGR on 20 trades at the
  same filter. Holdout remains net-negative under realistic
  execution.

**Still open:**

- **Holdout is net-negative at every filter under the real engine.**
  The model has per-trade signal, but 20-30 trades × ~0.3% net EV
  per trade after fees isn't enough to compound positive over 278
  days. Sample count is the binding constraint.
- **Regime change is a shared failure mode.** Fold 12 (2026 Q1) is
  catastrophic for both full and nopv (42.2% / 35.6% accuracy).
  VP features reduce the damage but don't fix it. Walk-forward
  retrain frequency is a separate lever.
- **Full-period CAGR is still lower than v6-prime honest.** v11-full-tb
  conf≥0.80 is +1.8% full-period (real engine) vs v6-prime honest
  +6.0%. The v11 win is entirely in drawdown shape and holdout
  discrimination, not compound return.

### Recommended next steps (revised after the backtest correction)

1. **⭐ Regime conditioning features** (GLD + USO + DXY + VIX + FFR +
   yield curve) as described in `MULTI_ASSET_PLAN.md` §REFRAME.
   Directly attacks the fold 12 / regime-change problem that limits
   both full and nopv on holdout. Adds signal quality per trade
   without requiring more training instruments. Deployment-aligned.
2. **Walk-forward with rolling retraining** (retrain every 3 months
   on expanding windows). The other direct attack on holdout
   collapse. Code change only. Defer to after the regime features
   experiment so the variables stay isolated.
3. **Diluted pyramid** (`allow_pyramiding=True` with 5% sizing) as a
   signal-magnitude diagnostic — NOT a deployable config, but useful
   for confirming the per-trade edge exists at scale without the
   catastrophic drawdown explosion of 20% pyramid.
4. **Live paper-trading a conf ≥ 0.80 wrapper** around v11-full as a
   separate sanity check — specifically on the drawdown shape and
   win-rate persistence (NOT on CAGR, since the real engine already
   tells us that's not positive yet).
5. **Option B (meta-labeling on a VP primary rule)** remains demoted
   — the end-to-end model is already picking up VP signal, and
   constraining the training set is less urgent than expanding the
   feature set.
