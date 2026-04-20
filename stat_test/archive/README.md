# stat_test/archive/

Superseded HMM exploration scripts (2026-04-17 → 2026-04-20).
See `../HMM_LOG.md` for the full iteration trail and what replaced each.

Quick map (§ refers to `HMM_LOG.md`):

| File | Status | Replaced by |
|---|---|---|
| `hmm_seed1.py` | one-off seed inspection (§2) | `hmm_seed_stability.py` |
| `hmm_seed_stability.py` | seed diagnostic for flat CS-rank HMM (§3) | `fit_best()` in `hmm_hierarchical.py` |
| `hmm_timeseries_rank.py` | TS-rank alternative — rejected (§4) | n/a |
| `hmm_zscore_cs_rank.py` | z-score CS-rank flat HMM (§5) | inner layer of `hmm_hierarchical.py` |
| `hmm_k_selection.py` | single-layer BIC/AIC sweep (§6) | `hmm_k_selection_hierarchical.py` |
| `hmm_validate_equities.py` | SPY/QQQ overlay (§7) | `hmm_quality.py` (adds NQ=F, ES=F) |
| `hmm_cs_plus_rates.py` | commodity + rate-level/change HMM (§8-9) | `hmm_hierarchical.py` |
| `hmm_cs_plus_rates_no_ffr.py` | drop-FFR variant (§11) | `hmm_hierarchical.py` |
| `hmm_validate_nq.py` | NQ=F overlay for rate-change HMM (§10) | `hmm_quality.py` |

**Utility functions** from `hmm_cs_plus_rates.py` (`load_long_history`,
`build_commodity_features`) were migrated to `../hmm_regime.py` so live
scripts no longer depend on this folder. Nothing here is imported by live
code.

Scripts here will still run if CWD is `stat_test/` — they reference the
data folder and shared `hmm_regime.py` via the parent's Python path when
invoked as `python3 archive/<script>.py`. Do NOT delete; kept for
reproducibility and the chance we want to revisit a rejected variant.
