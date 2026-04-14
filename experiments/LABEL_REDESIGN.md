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
