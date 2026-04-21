# HMM Macro-Regime Detector — Iteration Log

> Parallel track inside `stat_test/`: build an HMM that labels macro regimes
> so v12 can condition on them. Input candidates: gold, copper, corn, soy, oil,
> FFR, 2Y yield, yield-curve slope, DXY, credit spread, global M2, Fed net liquidity.
> Target: informative label (or posterior) for BTC fwd-return prediction.

## ⭐⭐ Frozen model v2 (2026-04-20, global-outer redesign)

**Hierarchical HMM: outer K=3 (global macro) × inner K=[5, 5, 5] → 15 composite states.**

Supersedes the morning's 2×[3,2] US-rate model — same day, because the redesign
motivation (US-rate outer is too US-centric) was strong and the swap was clean.

- **Outer** (4 features, 3y rolling rank → inverse-normal [Van der Waerden]):
  - `dxy` (global USD strength)
  - `credit_spread` = Moody's Baa − 10Y (FRED `BAA10Y`; replaces HY OAS, which
    FRED truncated in 2023 due to ICE licensing. Correlation ~0.9 with HY OAS
    during stress; same regime shape.)
  - `net_fed_liq_yoy` = YoY %Δ of (WALCL − RRPONTSYD − WTREGEN)
  - `global_m2_yoy` = equal-weight avg of YoY %Δ of US/EU/JP/CN M2 (local
    currency — no FX conversion to avoid DXY collinearity)
- **Inner** (4 commodity CS ranks: gld/uso/cper/corn — unchanged from v1): K=5 each.
- **Outer regime BTC fwd20d** (2012-08-07 → 2026-04-17, n=5002):
  - `S0` (n=2687, ~biggest state): BTC fwd20d +0.76%
  - `S1` (n=1183): BTC fwd20d **+4.83%** (strongest bull macro)
  - `S2` (n=1112): BTC fwd20d **+3.37%** (secondary bull)
- **Composite signal**: per-regime fwd20d spread max−min = 12.1% (vs 5.9% for v1).

**Canonical artifacts:**
- Script: `hmm_hierarchical_global.py` (use `--k-outer 3 --k-inner 5,5,5`)
- Results: `results/hierarchical_global_K3_555/` — `timeline.csv`,
  `outer_K3_{btc,nq,es}.png`, `composite_{btc,nq,es}.png`, `stats.txt`
- K-selection: `results/k_selection_hier_global/`
- Stability + VR comparison: `results/compare_global/summary.txt`
- Head-to-head vs v1: `results/compare_frozen_vs_global/summary.csv`
- Data prep: `python3 -m src.data.fetch_regime --global-outer --fred-key $KEY`
  writes `data/regime_global_outer.csv`. Publication-lag safety applied
  (30d for monthly M2, 7d for WALCL, 1d for daily series).
- Quality diagnostic tool: `hmm_quality.py --timeline <csv> --col <regime-col>`

**Head-to-head vs v1 (same 5002-day panel):**

| metric                        | v1: 2×[3,2] (5 st) | v2: 3×[5,5,5] (15 st) |
|-------------------------------|-------------------:|----------------------:|
| Outer-only BTC VR             |              0.74% |             **1.19%** |
| Composite BTC VR              |              1.20% |             **2.57%** |
| Per-regime fwd20d spread      |              5.94% |            **12.11%** |
| Mean ARI (seed stability)     |          **0.491** |                 0.456 |
| Min  ARI (seed stability)     |              0.291 |             **0.366** |
| Mean composite dwell          |            **8.9d** |                  4.7d |
| States w/ mean dwell <4d      |                **0/5** |                 7/15 |

**Interpretation**: v2 doubles total BTC VR and return-spread, with min-ARI
*higher* (worst-case seed disagreement actually improves). Cost is 7/15
short-dwell composite states — transition fragments the HMM uses to bridge
between persistent regimes. Not a bug for v12 (regime label feeds an
embedding); only a readability cost for humans.

**For v12 integration (still not yet wired):**
- Option A: hard composite label ∈ {0..14} → 15-dim one-hot or small embedding
- Option B: outer posterior (3-dim simplex) + inner-given-outer label (richer)
- Still outstanding: walk-forward refit before treating labels as v12 features,
  else we leak future info on the rank-window normalization.

## 2026-04-21 — Walk-forward experiment + regime-vocabulary problem identified

**Experiment** (`stat_test/hmm_walk_forward.py`, results in
`results/walk_forward_global/timeline_walk_forward.csv`):
- Protocol: `MIN_TRAIN_DAYS=1260` (5y), `REFIT_INTERVAL=63` (quarterly), 61
  refits over the 5002-day panel. At each refit R: 3y rolling rank on [0, R],
  outer HMM fit on [0, R], inner HMMs fit per outer state on [0, R]. Label
  days (prev_R, R] with those parameters. Outer aligned by sorting states by
  mean dxy_rn ascending; inner by gld_z.

**Results (walk-forward vs fit-once, 3804 common days, 2015-11 → 2026-04):**

| metric                     | value  |
|----------------------------|-------:|
| Outer BTC VR (walk-fwd)    |  1.20% |
| Outer BTC VR (fit-once)    |  1.19% |
| Composite BTC VR (walk-fwd)|  5.11% |
| Composite BTC VR (fit-once)|  3.41% |
| Outer ARI vs fit-once      |  0.134 |
| Composite ARI vs fit-once  |  0.067 |
| Outer exact match          |  19.7% |
| Composite exact match      |   5.2% |

**Interpretation**: clustering *structure* (BTC signal content) is preserved
— walk-forward VR matches or slightly exceeds fit-once, so fit-once does not
enjoy a leakage advantage. But **label IDs drift heavily** between refits:
composite labels only match in 5.2% of days. Deterministic sort-by-mean
fails when two cluster centroids have near-equal sort keys at some refit;
inner IDs cascade any outer swap.

**The deeper problem (user-surfaced)**: a per-fold refit does not just shift
IDs — **the regime definitions themselves change**. "State 2 in fold 3" and
"state 2 in fold 8" are not the same regime; they may not even be nearby in
feature space. For a *feature fed to v12*, this is incoherent: the embedding
layer can't learn a stable mapping from a moving vocabulary. Refit-every-fold
is valid for descriptive analysis of history but not for forward prediction.

**Resolved path forward (design only, not yet implemented — deferred to next
session):** Option A — **fit-once-early, predict-forward** with a frozen
reference CDF.
1. Burn-in window: fit outer + inner HMM on early panel (candidate:
   2012-08 → 2018-12, ~6.4 years). Covers post-crisis ZIRP, taper, 2015-16
   slowdown, 2018 hikes. Misses COVID/2020-21 liquidity surge and 2022
   inflation — those will be labeled as "nearest existing regime," which is
   honest about what the model knew at the time.
2. **Freeze the rolling-rank reference.** The critical detail: 3y rolling
   rank silently re-scales every day (a 2020 value ranks against 2017-2020),
   which is itself a form of drift. Replace with a fixed empirical CDF built
   from the burn-in window — any future day's feature is ranked against the
   frozen 2012-2018 distribution.
3. Freeze HMM parameters (transitions, emission means/covariances, state
   IDs). For every later day (train/val/test in every fold): apply
   frozen-CDF rank → frozen HMM `predict`. No leakage, universal regime
   vocabulary, no refits.

**Trade-offs acknowledged:**
- Pro: stable regime semantics, defensible (zero leakage), matches how macro
  regime models are deployed in practice.
- Con: regimes outside the burn-in support will be labeled as nearest
  neighbor, not as novel states (e.g., 2020-21 liquidity surge gets mapped
  to some existing state). Arguably a feature (avoids overfitting novelty)
  but worth flagging.
- Alt: fit-once-on-full-history (Option C). Leakage is weak (categorical
  feature, not target) and walk-forward VR was higher than fit-once VR, so
  fit-once is not meaningfully over-optimistic. But not defensible if asked.

**Status end of 2026-04-21**: walk-forward experiment is descriptive and
complete. Next session will implement frozen-CDF + frozen-HMM (Option A)
as the actual v12 integration path. `hmm_walk_forward.py` remains as the
tool for future drift diagnostics.

## v1 frozen model (2026-04-20, US-rate outer) — superseded

Kept here as historical record. Scripts: `hmm_hierarchical.py`,
`hmm_k_selection_hierarchical.py`. Results: `results/hierarchical/`.

- **Outer** (rate levels, FFR/DGS2/yield-curve, expanding z-score 1990+, full cov):
  - `S0` = easy money. Dwell ~1300d. BTC fwd20d +8.12%.
  - `S1` = tight money. Dwell ~1200d. BTC fwd20d −0.19%.
- **Inner**: S0 at K=3 (gold-weak/mid/strong, 2.98% VR), S1 at K=2 (0.56% VR).
- **Quality**: F=64.5, VR=6.77%, 9/10 pairwise Bonferroni.
- **Why superseded**: US-only outer misses global liquidity and credit-risk
  appetite — both materially affect BTC beyond the FFR/yield-curve channel.
  The v2 redesign (DXY + credit spread + net Fed liquidity + global M2)
  captures those dimensions while keeping the hierarchical factorization.

## Iteration trail

Chronological. Each row is one experiment; the "kept?" column records whether it still informs the frozen model or was rejected.

| # | Script | Feature recipe | Result | Kept? |
|---|---|---|---|---|
| 1 | `hmm_regime.py` | 20d log-ret → CS rank across {gld, uso, cper, corn, soyb} → inv-normal, drop last | Baseline K=2/3/4 | Superseded by #7 |
| 2 | `hmm_seed1.py` | Same as #1 at K=3, seed=1 diag | Stability check — seed 42 was suboptimal | Closed |
| 3 | `hmm_seed_stability.py` | 5-seed fit + Hungarian state alignment | Confirmed seed instability for raw CS-rank formulation | Closed |
| 4 | `hmm_timeseries_rank.py` | Per-commodity expanding-%ile-rank → inv-normal | K=3 — bull and bear both landed in r0 | Rejected |
| 5 | `hmm_zscore_cs_rank.py` | 14d log-ret → 252d rolling z → CS rank → inv-normal, drop last | Cleaner than #1 | Superseded by #8 |
| 6 | `hmm_k_selection.py` | BIC / AIC / BIC_neff sweep on #5 features | Neff-adjusted BIC flat across K=2..7 | Informed #11 |
| 7 | `hmm_validate_equities.py` | Overlay #5 regimes on SPY + QQQ | Confirms macro signal (not BTC-specific) | Replaced by `hmm_validate_nq.py` (futures instead of ETFs) |
| 8 | `hmm_cs_plus_rates.py` `--rate-mode level` | #5 commodity + raw-level rank of FFR/DGS2/yc | Rate levels dominated HMM — degenerate | Rejected |
| 9 | `hmm_cs_plus_rates.py` `--rate-mode change` | #5 commodity + 6m-change rank of FFR/DGS2/yc | r1 dwell 177d — lumped 2017-18 bull and 2022-23 bear together | Rejected |
| 10 | `hmm_validate_nq.py` | NQ=F overlay of #9 | Same regime ordering on NQ/SPY/QQQ/BTC — macro is real, just poorly stratified | Diagnostic only |
| 11 | `hmm_cs_plus_rates_no_ffr.py` | Drop FFR from #9, keep DGS2 + yc | Still long dwells (DGS2 also persistent) — not enough | Rejected |
| 12 | `hmm_hierarchical.py` (K_outer=2, K_inner=3) | Outer=rate-levels-zscore, Inner=commodity-CS-ranks, separable factorization | **Works.** Outer gives 5.4% BTC VR, long dwells are a feature not a bug | Kept |
| 13 | `hmm_hierarchical.py` (K_outer=3) | Same structure, K=3 outer | Prettier for equities, but K=3 outer BTC VR drops to 0.8% | Rejected |
| 14 | `hmm_k_selection_hierarchical.py` | BIC/AIC/BIC_neff/CV-LL/BTC-VR per layer | BIC pathological (→ K=5 with singular covs); CV-LL + BTC-VR both pick K_outer=2, K_inner_S0=3, K_inner_S1=2 | Frozen |
| 15 | `hmm_hierarchical.py --k-outer 2 --k-inner 3,2` | Asymmetric inner | v1 frozen model. BTC VR 6.77%, F=64.5 | Superseded (same day) |
| 16 | `fetch_regime.py --global-outer` | New outer feature set: DXY, BAA10Y credit spread, net Fed liquidity YoY, G4 M2 YoY (equal-weight local-currency). Publication-lag safety. | Data prep | Kept |
| 17 | `hmm_hierarchical_global.py` | Outer=4 global features with 3y rolling rank→inv-normal; inner unchanged (commodity CS ranks) | K=3×[5,5,5] wins on CV-LL | Kept |
| 18 | `hmm_k_selection_hierarchical_global.py` | BIC / CV-LL / BTC-VR sweep on new outer, then inner per outer state | K_outer=3 (CV-LL); K_inner=5 unanimous across all three outer states (CV-LL) | Kept |
| 19 | `hmm_compare_global.py` | [3,3,3] vs [5,5,5] head-to-head: BTC-VR + cross-seed ARI (8 seeds, 28 pairs) | [5,5,5] wins on VR (+74%), min-ARI (+0.13), ties on mean-ARI; cost is 6/15 short-dwell states | Kept |
| 20 | `hmm_compare_frozen_vs_global.py` | Apples-to-apples on shared 5002-day panel: v1 2×[3,2] vs v2 3×[5,5,5] | v2 doubles composite BTC-VR (2.57% vs 1.20%) and return spread (12.1% vs 5.9%), min-ARI also better | Kept |
| 21 | `hmm_hierarchical_global.py --k-outer 3 --k-inner 5,5,5` | Frozen global-outer fit | **v2 frozen model.** | ⭐ Current |
| — | `hmm_quality.py` | Diagnostic tool: ANOVA / VR / bootstrap CI / pairwise Welch / naive-strategy Sharpe on any timeline | Generic eval | Kept |

## Key lessons (write these in stone)

1. **BIC/AIC fail for this family.** Gaussian HMM with full covariance finds
   singular-covariance solutions at high K — LL climbs monotonically, BIC
   rewards it. CV-LL (date-split) and downstream-target variance-reduction
   are the honest selection criteria. `hmmlearn` emits "Model is not
   converging" warnings when this happens — treat those as "BIC is lying."
2. **Commodity block (ρ₂₁ ≈ 0) and rate block (ρ₂₁ ≈ 0.9) live on different
   timescales.** Mixing them in one HMM forces the model to match the slow
   timescale, which silences the fast one. Fix: factorize (outer/inner).
3. **Rate *levels* are the right transform for US rates**; rate *changes*
   lump regimes that happen to share "rates moving" regardless of direction
   (2017-18 hiking-in-a-bull vs 2022-23 hiking-in-a-bear). **For non-US or
   non-rate macro features, 3y rolling rank → inverse-normal** handles
   non-stationarity and fat tails while matching Gaussian-HMM assumptions.
4. **Long outer dwells are a feature, not a bug.** Rate cycles really do
   last 2–4 years. The only reason it was a bug in flat (non-hierarchical)
   HMMs is that it pulled the commodity dimensions with it.
5. **Short-dwell composite states (<4d) are transition fragments, not
   regimes.** Unavoidable cost of richer inner models. Acceptable when the
   label feeds a downstream model (embedding absorbs the fragmentation);
   problematic only if we need human-interpretable regimes.
6. **US-only macro features are not enough for a BTC regime detector.**
   DXY, credit spread (BAA10Y), net Fed liquidity YoY, and G4 M2 YoY (equal-
   weight local-currency) materially improve outer-layer BTC VR (+60%) over
   FFR/DGS2/yield-curve. The US-rate outer captures the monetary-policy
   dimension but misses global liquidity and credit-risk appetite.
7. **FRED truncated the BAML HY indexes (e.g., BAMLH0A0HYM2) in 2023** due
   to ICE Data Indices licensing changes. Only ~3y of data is available
   publicly. Use Moody's Baa corporate spreads (`BAA10Y`) as a substitute —
   daily, free, back to 1986, correlation ~0.9 with HY OAS during stress.
8. **For global M2, avoid FX conversion.** G4 M2 in USD has DXY-collinearity
   that double-counts your DXY feature. Use equal-weight of local-currency
   YoY% growth instead.
9. **Validating on NQ=F / ES=F (futures, not ETFs) going forward.** Done in
   `hmm_quality.py` and `hmm_validate_nq.py`.

## Open loose ends

- Walk-forward refit of v2 HMM — full-history fit would leak the 3y rolling
  rank normalization if used as-is in v12. Must refit walk-forward before
  integration.
- Posterior calibration check for outer-posterior continuous features.
- `hmm_quality.py` ANOVA/bootstrap/Bonferroni run on v2 composite — v1 had
  VR=6.77% / F=64.5 on a narrower panel; need comparable numbers for v2.
- v12 integration: hard composite (15-dim one-hot / small embedding) vs
  outer posterior + inner-given-outer label. Decide after walk-forward.
