# stat_test/results/archive/

Superseded HMM result folders. See `../../HMM_LOG.md` for the frozen model.
Canonical live results live one level up in `hierarchical/` and
`k_selection_hier/`.

| Folder | What it was | Why archived |
|---|---|---|
| `cs_rank_k2/`, `cs_rank_k3/`, `cs_rank_k4/` | Flat CS-rank HMM at K=2/3/4 | Replaced by hierarchical composite |
| `cs_rank_seed1/` | K=3 run with seed 1 diagnostic | Seed-stability experiment |
| `cs_rank_stability/` | 5-seed Hungarian-matched fit | Seed-stability experiment |
| `cs_rank_equities/` | SPY/QQQ overlay of flat CS-rank | Now done with NQ=F/ES=F in `hmm_quality.py` |
| `ts_rank/` | Time-series rank HMM output | Rejected — failed bull/bear separation |
| `zscore_cs/`, `zscore_cs_k4/` | z-score-then-CS-rank flat HMM | Used as inner layer of hierarchical; flat version retired |
| `cs_plus_rates/` | Rate-level augmented flat HMM | Rejected — rate levels dominated |
| `cs_plus_rate_changes/` | Rate 6m-change flat HMM | Rejected — r1 dwell 177d lumped bulls and bears |
| `cs_plus_rate_changes_no_ffr/` | Drop-FFR variant | Rejected — DGS2 still too persistent |
| `cs_plus_rate_changes_nq/` | NQ=F overlay of rate-change HMM | Diagnostic — subsumed by `hmm_quality.py` |
| `k_selection/` | Single-layer K sweep BIC/AIC | Superseded by `k_selection_hier/` |
