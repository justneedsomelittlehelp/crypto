# Multi-Asset Plan — Testing the Universality of the VP Signal

> **Status**: planning doc — no code written yet, no runs executed.
> **Context**: the v11 triple-barrier decisive ablation (2026-04-14) gave
> the first clean positive VP result in project history on BTC. Before
> building live-deployment infrastructure around it, we want to know
> whether the lift generalizes beyond BTC — is the VP signal *universal
> market microstructure* or *BTC-specific memorization*? This doc is
> the plan for answering that.
>
> **Related docs**: `LABEL_REDESIGN.md` §Results (the positive v11-tb
> finding), `MODEL_HISTORY.md` §§30–31, `EVAL_AUDIT.md` §9 Stage 6,
> `STRATEGY.md` (user's manual trading philosophy).

---

## Why this is the right next experiment

After the v11-tb decisive ablation came back positive, three possible
next directions were on the table:

1. **More precise label design** (asymmetric barriers, longer vol
   windows, meta-labeling on a VP primary rule).
2. **More data points via multi-asset** (BTC + ETH + SOL, and
   eventually ES / CL / GC).
3. **Walk-forward rolling retraining** (retrain every 3 months to
   address the regime-change failure on fold 12).

**Multi-asset wins on expected value**, and the real-engine backtest
review (2026-04-14) reinforced this. Three reasons:

### 1. The binding constraint is statistical power, not signal quality

The real-engine v11-full-tb holdout is −1.5% CAGR at n=20 trades (conf≥0.80).
v6-prime honest was ≈−5% at n≈15. Both confidence intervals are
enormous. **No label improvement can fix "we only have 20 trades to
judge the holdout window with."** If a better-label variant produces
+2% CAGR on 22 trades, that's statistically indistinguishable from
the current −1.5% on 20 trades. We haven't actually learned anything.
More trades is the only lever that attacks this directly.

Multi-asset with BTC + ETH + SOL approximately triples holdout trade
count at the same filter thresholds without touching the model or
labels.

### 2. Multi-asset is the cleanest falsification of the central claim

The user's strategy isn't "BTC volume profile is special." It's *"volume
profile tells you where support/resistance is."* That's a claim about
market microstructure. Testing it on BTC only is necessary but not
sufficient. Multi-asset tests the claim directly:

- **Full beats nopv on all three assets** → VP is universal, the
  hypothesis holds, and the model validates the user's manual
  intuition across instruments.
- **Full beats nopv on BTC only** → the signal is BTC-specific
  microstructure, not universal. Narrower scope for the strategy,
  still a legitimate finding.
- **Full ties nopv everywhere** → VP is not ML-exploitable at this
  scale. Phase 3 closes cleanly with a strong negative result.

Every outcome is valuable. Label tweaks can't answer universality
because they test the same model on the same asset.

### 3. Label tweaks have diminishing returns after triple-barrier

The range → triple-barrier switch was a qualitative step (decoupling
labels from features). That was the big win and we already took it.
Further label refinements — different `k`, different σ window,
asymmetric barriers, meta-labeling on a VP primary rule — are
quantitative parameter tuning. Expected improvement magnitude is
small compared to the experimental cost, and none address the
sample-count problem.

---

## Asset selection — what to include and what to skip

Evaluated several candidates against five criteria:

1. Continuous or near-continuous trading (gaps break rolling VP semantics)
2. Reliable volume data
3. Liquid enough that the 50-bin histogram has real structure
4. 5+ years of clean 15m history available
5. VP-native practitioner culture (so the market has participants
   leaving volume-at-price footprints)

### Included (staged)

#### Stage 1 (immediate) — BTC + ETH + SOL

- **24/7 trading** → zero gap issue
- **Existing infra**: ccxt covers Bitstamp + Coinbase for all three
- **Data history**:
  - BTC: clean from 2016 (confirmed)
  - ETH: clean from ~2017 (Bitstamp added ETH/USD Aug 2017)
  - SOL: clean from ~2021 at best (Coinbase listed June 2021). Real
    liquid history likely starts late 2021.
- **Liquid enough for real VP structure** (all three top 5 by volume)
- **Minimum viable experiment** — uses existing scraper, existing
  compute_absvp_15m_30d.py, existing model

**Open question: SOL's shorter history.** BTC has ~10 years, ETH has
~8.5, SOL has ~4-5. Options:

- **Option A — truncate all to SOL's start date (~2021)**. Clean,
  equal-history per asset, but throws away half of BTC and ETH data.
  Probably a bad trade-off given our sample-count problem.
- **Option B — per-asset start dates, stratified walk-forward**.
  BTC starts 2016, ETH starts 2017, SOL starts 2021. Each asset's
  walk-forward folds begin at its own first-valid row. Training
  uses whatever assets are available at each time point. More
  complex but preserves data.
- **Option C — defer SOL, run BTC + ETH only**. Two-asset experiment.
  Simpler, slightly fewer holdout trades than three-asset, still
  tests universality.

**Recommended for Stage 1**: Option C → BTC + ETH only. Avoids the
stratification complexity entirely, still approximately doubles
holdout trade count, and keeps all folds at the same calendar
windows as the existing v11 runs. If the two-asset result is
positive and clean, Stage 1b can add SOL with Option B.

#### Stage 2 (if Stage 1 is positive) — add ES futures

- **E-mini S&P 500 continuous futures**: 23h/day trading, 1h daily
  maintenance break, weekend gap only. Much cleaner than SPY cash.
- **VP-native**: Peter Steidlmayer's market profile was literally
  developed on CBOT futures. Decades of practitioner usage.
- **Data**: ccxt does NOT cover ES. Requires one of:
  - Polygon.io (free tier limited, paid ~$50-200/month)
  - Alpha Vantage (free tier rate-limited, historical depth questionable)
  - Databento (paid, high-quality)
  - IBKR API (need broker account, has data limits)
- **Implementation cost**: ~2-3 hours of data-engineering work to
  build an ES scraper + align with the existing CSV schema.

Do this ONLY if Stage 1 returns positive. Otherwise there's no
reason to build the infra.

#### Stage 3 (if Stage 2 is positive) — add CL or GC futures

- **CL** (WTI crude oil continuous futures) or **GC** (gold continuous
  futures). Pick one — they're interchangeable for the
  "does VP work in physical commodities" question.
- **VP is heavily used on oil and gold by professional traders**
  (despite user's initial concern that "VP might not be available" —
  it is, the confusion was about data availability, not VP viability).
- **Data hurdle**: same as ES plus **continuous-contract stitching**.
  Futures roll every 1-3 months. Naive concatenation creates price
  gaps at every roll that would destroy VP. Need back-adjusted
  continuous series (ratio or absolute) from a provider that handles
  it (Polygon, CQG, Databento, iVolatility).
- Tests "VP in physical commodities vs financial instruments" —
  the most different market class we can reasonably include.

### Considered and rejected

#### SPY cash / individual stocks (AAPL, TSLA, NVDA, etc.)
- **Cash-session gap problem**: 6.5h trading day, 17.5h overnight gap
  every day. A "30-day rolling VP" averages only cash-session bars
  and systematically ignores overnight price action. VP semantics
  violated.
- Individual stocks also dominated by idiosyncratic news (earnings,
  events), which is a different signal regime from the user's
  microstructure-based strategy.
- **Skip.** Use ES futures instead if you want US equity exposure.

#### Forex majors (EUR/USD, USD/JPY, etc.)
- **OTC volume is unreliable** — FX is decentralized, no unified
  volume feed. What ccxt returns as "volume" for EUR/USD is one
  exchange's tick count, not market volume. VP would be noise.
- **Skip spot forex.** If we want a currency test later, use CME
  currency futures (6E for EUR, 6J for JPY) which have real volume.

#### Altcoins beyond top 5 (LINK, AVAX, DOGE, etc.)
- Liquid enough for VP structure at top 5, but rapidly thins below.
- Correlated too strongly with BTC to add real signal diversity.
- **Skip for now.** The point of multi-asset isn't more crypto —
  it's different microstructures.

---

## Implementation sketch (Stage 1 only)

When we build Stage 1, the work is:

### 1. Data layer

- Modify `src/data/compute_absvp_15m_30d.py` to accept an asset
  argument and output per-asset CSVs:
  - `BTC_15m_ABSVP_30d.csv` (exists, 459MB)
  - `ETH_15m_ABSVP_30d.csv` (new, similar size)
- Run for ETH via `ccxt` on Bitstamp + Coinbase, starting 2017-08-17
  (first clean Bitstamp ETH/USD data).
- Document the new CSV in `arch-data-pipeline.md`.

### 2. Model input layer

- Feature tensor shape per bar: unchanged (110-dim day token).
- Add an **asset embedding** to the day token. Two options:
  - **Concat a learned 4-dim asset embedding** to the 110-dim day
    token → 114-dim. `day_projection` becomes Linear(46, 32) instead
    of Linear(42, 32). +128 params.
  - **No embedding, pooled training**. Treat all assets as if they
    were the same asset. Simpler, forces the model to learn
    asset-invariant features, which is exactly what we want to test.
    **Recommended for the first Stage 1 run.**

### 3. Eval layer

- Modify `src/models/eval_v11.py` to accept multiple CSVs and
  concatenate their rows into a single training set, stratifying
  the walk-forward folds by **calendar date only** (not by asset).
- Each fold contains bars from both BTC and ETH in that calendar
  window. The model is trained on the union and evaluated per-asset
  on the test slice.
- Output per-asset results in the JSON so we can compute per-asset
  Δ full-vs-nopv.
- New flags: `--assets BTC,ETH` (default `BTC`), `--tag` propagates
  to include asset list.

### 4. Walk-forward decision — unified fold boundaries

Keep `FOLD_BOUNDARIES` calendar-aligned across assets. That means:

- Fold 1 (2020-01 → 2020-07): trains on BTC + ETH data before
  2020-01, tests on BTC + ETH data in 2020-01 → 2020-07.
- Holdout folds 11-12 (2025-07 → 2026-04): trained on all BTC + ETH
  data before 2025-07 minus 14d embargo, tested per-asset.

SOL deferred (Option C) until the two-asset result lands.

### 5. Backtest layer

- `run_backtest_v11_tb.py` needs to handle per-asset predictions.
  Run one backtest per (asset × variant × scope) combination.
- Real engine should treat each asset as an independent portfolio
  (separate capital) OR a pooled portfolio (single capital across
  all trades). **Recommended: separate per-asset portfolios for the
  first run** so per-asset CAGR/DD is directly comparable. Pooled
  comes later if we want a multi-asset portfolio strategy.

---

## Success criteria for Stage 1

The Stage 1 run is considered **positive** if:

1. `full` beats `nopv` on holdout accuracy **on both BTC and ETH
   individually**. A BTC-only win is not enough — that's what we
   already have.
2. The holdout Δ (CAGR or accuracy) is the same sign across both
   assets.
3. Holdout trade count increases by ~2× vs. single-asset BTC,
   giving tighter confidence intervals on the real engine backtest.

**Negative** (univerality falsified) if:

1. full beats nopv on BTC, ties or loses on ETH.
2. This would mean the v11-tb positive was BTC-specific
   microstructure, not universal. Narrower deployment scope, still
   a legitimate finding.

**Null** (signal gone entirely) if:

1. full ties nopv on both assets.
2. Would suggest the BTC-only result was sample-variance luck.
   Phase 3 likely closes.

Report the outcome in `LABEL_REDESIGN.md` as a new §Results subsection
and add `MODEL_HISTORY.md` §32.

---

## What would NOT be done in Stage 1

- No architectural changes beyond accepting multi-asset input.
- No label changes (triple-barrier as-is).
- No new features (no funding rate, no OI, no macro indicators).
- No walk-forward retraining frequency change.
- No ES / CL / GC — those are Stage 2 and Stage 3 only.
- No asset embedding (forces invariance, cleanest test).

One variable at a time. The audit lesson still stands.

---

## Decisions locked in today (2026-04-14)

1. **Next experiment is multi-asset Stage 1 (BTC + ETH), not label tweaks.**
   Sample-count is the binding constraint, and label tweaks can't
   address it.
2. **SOL is deferred** to Stage 1b (after the two-asset result) to
   avoid stratified-walk-forward complexity on the first iteration.
3. **No asset embedding** in the first run — pooled training forces
   the model to learn asset-invariant features, which is exactly
   the hypothesis we're testing.
4. **ES and CL / GC are not in Stage 1.** They require new data
   infrastructure and we only build it if crypto-universality works.
5. **Per-asset backtest portfolios** for Stage 1 reporting (separate
   capital per asset). Pooled portfolio deferred.

---

## Open questions (not blocking, but should be answered before Stage 1)

- **ETH 15m history availability check**: run `ccxt.fetch_ohlcv('ETH/USD', '15m', since=2017-01-01)` against Bitstamp and Coinbase, confirm clean data from the intended start date. Expected: Bitstamp from ~2017-08, Coinbase from ~2016 but with sparse early bars.
- **Asset embedding trade-off**: if the no-embedding version ties nopv, try a 4-dim learned asset embedding as a secondary experiment to see if the model needs asset identity to specialize. Expected prior: no, but worth having the escape hatch.
- **Does the 14-day embargo need to be per-asset?** Yes — label forward windows are per-asset, so each asset's embargo mask is independent. Walk-forward already handles this correctly since embargo is a wall-clock delta applied to each row individually.
- **File size**: two 459MB CSVs are manageable but annoying. Consider writing combined_15m_absvp_30d.csv as a single file with an asset_id column. Decision: use per-asset files for now; merge later if it becomes a pain.

---

*Plan locked in 2026-04-14 at end of session. Next session: run the
ETH 15m data availability check, then implement Stage 1.*
