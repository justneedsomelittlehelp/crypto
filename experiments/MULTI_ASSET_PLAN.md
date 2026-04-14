# Multi-Asset Plan — Reframed to Regime-Conditioning Features

> **Status**: planning doc — no code written yet, no runs executed.
> **Context**: the v11 triple-barrier decisive ablation (2026-04-14) gave
> the first clean positive VP result in project history on BTC. The
> original plan in this doc framed the follow-up as "test VP universality
> on BTC + ETH (+ SOL)," quant-firm-shaped thinking that assumed we'd
> trade multiple assets simultaneously for statistical power. **User
> correction 2026-04-14 (same day, later in session): that framing is
> wrong for this project's shape.** See the `REFRAME` section immediately
> below — the revised plan replaces multi-asset training with macro
> regime conditioning features on a single-asset (BTC) model.
>
> **Related docs**: `LABEL_REDESIGN.md` §Results (the positive v11-tb
> finding), `MODEL_HISTORY.md` §§30–31, `EVAL_AUDIT.md` §9 Stage 6,
> `STRATEGY.md` (user's manual trading philosophy).

---

## ⭐ REFRAME (2026-04-14) — Regime features, not multi-asset training

The original plan below was structured around *"train on BTC + ETH (+ SOL)
jointly so the model sees more microstructures and produces more holdout
trades for statistical power."* That's the correct framing for a prop
firm running simultaneous positions across many instruments. It's the
**wrong** framing for this project because:

1. **The user is an individual swing trader with capital constraints.**
   They cannot realistically hold simultaneous positions across multiple
   instruments the way a portfolio strategy would. The model's job is
   not to pick the best-signal asset from a basket and trade it — that's
   a fundamentally harder ML problem (cross-asset ranking, capital
   allocation, volatility normalization across asset classes) that
   requires much more capacity and infrastructure than what we've built.
2. **The strategy is hours-to-weeks swing trading with filtered entries,
   not minutes-scale portfolio execution.** The filter layer already
   decides "is this a good BTC entry right now?" — the model's role is
   to condition that answer on the best available information about
   current market state, not to propose alternative trade destinations.
3. **In that framing, other assets become** *extra market state
   information* — **context features that tell the BTC model "we're in
   a risk-off regime" or "the dollar is ripping" — not additional
   training instruments.** This is how discretionary macro traders
   actually use gold, oil, DXY, VIX, yields — they don't trade those
   instruments; they use them to condition their primary trade thesis.

### The revised Stage 1

Replace the BTC + ETH training experiment with a **single-asset BTC model
augmented with macro regime features**. The model still predicts BTC
triple-barrier outcomes. It still runs on `BTC_15m_ABSVP_30d.csv`. The
change is that each day token (and/or the final-FC input) gains a small
set of scalar features describing the broader market state at that
moment:

| Category | Features | Source | n |
|---|---|---|---|
| **Commodities** | GLD 5d & 30d log return | yfinance | 2 |
| | USO 5d & 30d log return | yfinance | 2 |
| **FX** | DXY 5d & 30d log return | yfinance | 2 |
| **Vol** | VIX level, VIX 7d change | yfinance / FRED | 2 |
| **Rates** | Fed funds level, 90d change | FRED | 2 |
| **Term structure** | 10Y − 2Y yield curve slope | FRED | 1 |
| **Total** | — | — | **11** |

All eleven are continuous scalars (confirmed 2026-04-14 — the user's
initial categorical framing for FFR was discussed and continuous
explicitly chosen because it preserves information at bucket
boundaries and doesn't require an embedding layer for a field that's
constant 99% of bars anyway).

### Why this is better than the original plan

1. **Deployment-aligned.** When the model goes live, checking daily
   GLD/USO/DXY/VIX/FFR/yield closes is exactly what a macro-aware
   discretionary trader already does. The model learns the same mental
   model the human would apply.
2. **Directly attacks the fold 12 problem.** The holdout collapse in
   2026 Q1 is almost certainly regime-driven — some specific macro
   configuration that the model has never seen. Regime features give
   the model a way to detect "this is a different regime, proceed
   with caution" that it currently lacks.
3. **Much lower implementation cost than multi-asset.** No per-asset
   CSVs, no asset embedding, no pooled training plumbing, no
   stratified walk-forward. Just two new scrapers (yfinance + FRED),
   one merge column layer, and a slightly wider day token.
4. **Single variable at a time.** Adding regime features is one
   isolated change on top of the validated v11-full-tb baseline.
   Multi-asset + architecture + asset embedding was stacking three
   changes into one experiment.
5. **Preserves all the sample-count benefit we wanted from multi-asset
   *differently*.** The original plan's sample-count argument was
   "more trades let us distinguish edge from noise." Regime features
   don't add trade count, but they do add **signal quality per trade**,
   which moves the same needle (detectable edge per unit data)
   without the infrastructure cost. Combined with walk-forward
   retraining in a later experiment, we can get both.
6. **The "does VP generalize" question can still be answered later** —
   but as a separate experiment with a different purpose (pure science,
   not deployment prep). If we build a case for publication later, we
   revisit multi-asset then.

### Revised implementation sketch

1. **Scrape daily macro series.**
   - `yfinance`: `GLD`, `USO`, `^VIX`, `^DXY` (or `DX-Y.NYB`)
   - `pandas_datareader` / `fredapi` (FRED): `FEDFUNDS`, `DGS2`,
     `DGS10`, `VIXCLS` (backup for VIX if yfinance is noisy)
   - Daily resolution, start 2016-01-01 to current, forward-fill
     weekends and holidays.
2. **Compute derived features.**
   - For each of GLD, USO, DXY: 5-day and 30-day log returns.
   - VIX: level + 7-day change.
   - Fed funds: current level + 90-day change.
   - Yield curve: `DGS10 − DGS2`.
   - All 11 features are float32 scalars, one row per calendar day.
3. **Merge to the BTC 15m CSV.**
   - New columns on `BTC_15m_ABSVP_30d.csv` (or a parallel
     `BTC_15m_ABSVP_30d_regime.csv` to avoid touching the existing
     file).
   - Forward-fill from daily to 15m: every 15m bar carries the most
     recent realized daily macro values. Always look backward — never
     let the 15m bar see a macro value that wasn't published yet.
4. **Add `--regime {none, all}` flag to `eval_v11.py`.**
   - `none`: current behavior, no regime features (baseline for the
     ablation).
   - `all`: day token width grows from 110 → 121. `day_projection`
     becomes `Linear(53, 32)` instead of `Linear(42, 32)`. ~360 new
     params. Model class unchanged beyond the layer dimension.
5. **Run the decisive comparison.**
   - `v11-full-tb + regime=none` (replicate of the existing result)
   - `v11-full-tb + regime=all` (the new experiment)
   - Same triple-barrier labels, same walk-forward, same holdout,
     same backtest engine (run_backtest_v11_tb.py).
6. **Report**: does regime lift v11-full-tb holdout numbers? Fold
   12 specifically? Real-engine CAGR at conf≥0.70 / 0.75 / 0.80?

### Where the features attach in the day token

Three options, from simplest to most expressive:

1. **Append to `last_bar`** (currently 4-dim) → becomes 15-dim.
   Model sees regime only at the final FC head. Minimal change.
   Loses temporal structure (regime evolution over 90 days is
   invisible).
2. **Append to each `day_token`** (currently 110-dim) → becomes
   121-dim. Temporal transformer sees regime history day-by-day.
   Lets model learn "regime started shifting 7 days ago" reasoning.
   **Recommended first try.**
3. **Separate "regime branch"** with its own small transformer over
   the 90-day regime sequence, then concatenate with v11's output.
   Maximum capacity, maximum overfitting risk. Not for the first run.

**Decision locked**: Option 2 (append to each day token). Cleanest
middle ground, preserves temporal information, minimal parameter
growth.

### Updated success criteria

Stage 1 (regime features) is **positive** if:

1. `v11-full-tb + regime=all` beats `v11-full-tb + regime=none` on
   holdout accuracy or holdout real-engine CAGR by a meaningful
   margin (≥ 3 pp acc or ≥ 2 pp CAGR at the conf≥0.70 filter).
2. Fold 12 (2026 Q1) specifically improves — that's the regime-
   change failure mode the feature set is designed to address.
3. Ideally: the sign-flip baseline gap narrows (less anti-alignment
   on holdout), which would indicate the model is no longer
   memorizing pre-2025 direction.

**Negative** (regime features don't help):

1. `regime=all` ties or loses to `regime=none` on holdout.
2. Possible interpretations: daily macro data is too slow for 14-day
   swing trades, the existing v11 features already encode what's
   useful, or fold 12's specific regime isn't detectable from any
   of the 11 features chosen.
3. Still a legitimate finding — would falsify "macro regime features
   help short-horizon crypto prediction" which is a meaningful
   negative result.

**Null** (no difference either way): regime features are free to
include but don't move the needle. We'd probably keep them on a
priori grounds (they're cheap and deployment-relevant) but not rely
on them.

---

## Original (superseded) multi-asset plan

*Everything below is the pre-reframe content, preserved because the
rejected-candidates section (SPY cash, individual stocks, spot forex)
still stands as a record of assets we considered and why we skipped
them. The BTC + ETH Stage 1 is no longer the plan — the reframe above
supersedes it.*

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
