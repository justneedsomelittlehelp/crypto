# HMM Macro-Regime Detector — Iteration Log

> Parallel track inside `stat_test/`: build an HMM that labels macro regimes
> so v12 can condition on them. Input candidates: gold, copper, corn, soy, oil,
> FFR, 2Y yield, yield-curve slope. Target: informative label (or posterior)
> for BTC fwd-return prediction.

## ⭐ Frozen model (2026-04-20)

**Hierarchical HMM: outer K=2 × inner K=[3, 2] → 5 composite states.**

- **Outer** (rate levels, FFR/DGS2/yield-curve, expanding z-score 1990+ baseline, full cov):
  - `S0` = easy money (FFR below hist-avg, normal curve). Dwell ~1300 d. BTC fwd20d +8.12%.
  - `S1` = tight money (FFR/DGS2 elevated, curve inverted). Dwell ~1200 d. BTC fwd20d −0.19%.
- **Inner within S0** (commodity CS ranks, K=3): gold-weak / gold-mid / gold-strong.
  Meaningful sub-signal for BTC (2.98% extra VR).
- **Inner within S1** (K=2): weak sub-signal only (0.56% VR) — low informational value.
- **Quality (BTC fwd20d)**: ANOVA F=64.5, p=9e-53, variance reduction **6.77%**,
  9 of 10 pairwise comparisons survive Bonferroni.

**Canonical artifacts:**
- Script: `hmm_hierarchical.py` (use `--k-outer 2 --k-inner 3,2`)
- Results: `results/hierarchical/` — `timeline.csv`, `outer_K2_btc/nq/es.png`, `composite_btc/nq/es.png`, `quality_outer.txt`, `quality_composite.txt`
- K-selection rationale: `results/k_selection_hier/`
- Quality diagnostic tool: `hmm_quality.py --timeline <csv> --col <regime-col>`

**For v12 integration (not yet wired):**
- Option A: hard composite label ∈ {0..4}
- Option B: outer posterior `p(S_easy | x_{1:t})` (continuous) + inner-S0 label (richer)

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
| 15 | `hmm_hierarchical.py --k-outer 2 --k-inner 3,2` | Asymmetric inner | **Frozen model.** BTC VR 6.77%, F=64.5 | ⭐ Current |
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
3. **Rate *levels* are the right outer feature**; rate *changes* lump
   regimes that happen to share "rates moving" regardless of direction
   (2017-18 hiking-in-a-bull vs 2022-23 hiking-in-a-bear).
4. **Long outer dwells are a feature, not a bug.** Rate cycles really do
   last 2–4 years. The only reason it was a bug in flat (non-hierarchical)
   HMMs is that it pulled the commodity dimensions with it.
5. **Inside tight money, commodity rotation does nothing for BTC.** The
   macro headwind dominates tactical flow. Inner layer should be shallower
   (K=2 or skip) in S1.
6. **Validating on NQ=F / ES=F (futures, not ETFs) going forward.** Done in
   `hmm_quality.py` and `hmm_validate_nq.py`.

## Open loose ends

- Outer label not yet fed into v12 (the integration is the point; this is pending a v12 rerun slot).
- No walk-forward test of the HMM (fit-once on full history). Needed before treating the label as a v12 feature — otherwise we leak future.
- Cross-seed ARI on the frozen hierarchical model — build confirms dwell stability but not label-by-label stability.
- Posterior calibration check for the continuous `p(S_easy|x)` feature.
