# Statistical Feature Testing — Workflow & Status

> Cheap-to-expensive feature screening funnel for BTC ML model candidates.
> **Read this first** if you are continuing feature-testing work in a new session.

## Why this folder exists

Full-model ablations are O(N_features × K_seeds × 12_folds) and burn compute.
This folder filters candidates *before* they touch the real model.

The methodology is documented in [`how_to_stat_test.md`](how_to_stat_test.md) — read it before running new tests.

Parallel HMM macro-regime track: see [`HMM_LOG.md`](HMM_LOG.md) for the iteration trail and current frozen model. Frozen: hierarchical HMM (K_outer=2 × K_inner=[3,2], 5 composite states). Script: `hmm_hierarchical.py --k-outer 2 --k-inner 3,2`. BTC fwd20d variance reduction = 6.77%.

## The funnel (4 stages)

| Stage | What | Cost | Artifact |
|---|---|---|---|
| 0 | Economic-mechanism check | free (thinking) | entry in candidate log |
| 1 | Univariate IC (monthly Spearman + t-stat) | seconds-minutes | `stage1_*.py` + `.csv` |
| 2 | Tiny-model ablation (logistic / MLP ~10k params) | hours | `stage2_*.py` (not yet built) |
| 3 | Permutation / SHAP on the full model | medium | n/a yet |
| 4 | Full ablation in the real model (all seeds, walk-forward) | days | lives in `experiments/` |

## Current status (last updated 2026-04-17)

### Stage 1 — DONE for these features

Script: [`stage1_univariate_ic.py`](stage1_univariate_ic.py)
Results: [`stage1_univariate_ic_results.csv`](stage1_univariate_ic_results.csv)

Tested (BTC forward returns at 4h / 24h / 96h / 168h, monthly IC pre-2025-10):

- **Price/return features**: ret_1h/4h/24h/72h/168h, rvol_24h/72h/168h
- **Candle geometry**: body_ratio, upper_wick, lower_wick
- **Volume**: logvol, logvol_z_168, vol_surge_24
- **VP-derived**: vp_cur_weight, vp_max_weight, vp_peak_offset, vp_above_sum, vp_below_sum, vp_skew, vp_hhi, ceiling_dist, floor_dist
- **Sentiment**: fgi, fgi_delta_7d
- **Regime (from `data/regime_hourly.csv` + `regime_daily.csv`)**: vix, vix_chg_24h, vix_chg_168h, vix_z_720h, dxy_ret_24h/168h/720h, gld_ret_168h/720h, uso_ret_168h/720h, btc_dxy_corr_720h, ffr, ffr_chg_30d, yield_curve, yield_curve_chg_30d
- **Time-of-day**: hour_sin/cos, dow_sin/cos

### Stage 1 — KEY FINDINGS

**Strongest signals (consistent across horizons):**
- `fgi` (|t|=13 at 168h, IC=-0.36) — fear/greed mean-reversion. Single strongest feature.
- `vp_skew`, `vp_above_sum`, `vp_below_sum` (|t|≈10 at 168h, IC≈0.32) — VP structural shape dominates.
- `ret_168h`, `ret_72h` (|t|≈11 at 168h, IC=-0.29) — multi-day mean reversion.
- `vp_peak_offset`, `ceiling_dist` (|t|≈7-8) — VP geometry.

**New regime features (high-value adds):**
- `vix_chg_24h` (|t|=4.4 at 4h, |t|=2.5 at 24h) — VIX spikes → BTC drops short-term.
- `dxy_ret_24h` (|t|=3.4 at 4h) — dollar up → BTC down short-term.
- `vix` level (|t|≈2.5 at 96h/168h, *positive* IC) — high VIX → BTC up longer-horizon (contrarian).

**Weak univariate (do not drop yet — may work as regime gates):**
- `ffr`, `ffr_chg_30d`, `yield_curve`, `yield_curve_chg_30d` — all |t|<2.
- These are regime variables; their value may be in *conditioning* other features, not direct prediction. Test via regime-conditioned IC (not yet done) or via architecture (explicit regime classifier / gating head).

**Drop candidates (no mechanism + no signal):**
- `gld_ret_*`, `uso_ret_*` beyond marginal — no univariate signal and weak BTC mechanism.
- Time-of-day features (`hour_*`, `dow_*`) — dead on BTC (24/7 market). **Keep in mind for NQ work.**

### Stage 1 — NOT YET TESTED

- Funding rate features (script has it but CSV join failed — tz issue; fix is ~3 lines). `data/funding_rate_merged.csv` available, 2020-present.
- Per-regime-conditioned IC (e.g., IC of `vp_skew` in high-VIX vs low-VIX buckets). **This is the recommended next step** — it tests whether regime features have interaction value, which raw univariate IC misses.
- 15m-cadence features (script is currently 1h only). Relevant if cadence debate reopens.
- NQ futures features — separate future effort, data not yet pulled.

### Stage 2+ — NOT YET STARTED

No tiny-model ablation run yet. Suggested scope for first Stage 2:
- Baseline = existing VP + candle + returns features
- Candidate groups to test as bundles:
  1. VIX group (vix, vix_chg_24h, vix_z_720h)
  2. DXY group (dxy_ret_24h/168h)
  3. FGI group (fgi, fgi_delta_7d)
  4. Macro-slow group (ffr, ffr_chg_30d, yield_curve, yield_curve_chg_30d) — test via regime-conditioned, not direct
- Compare log-loss at 24h and 168h horizons, 3 seeds.

## How to run Stage 1

From project root:

```bash
python3 stat_test/stage1_univariate_ic.py
```

Outputs:
- Terminal: top-30 features per horizon, sorted by |t-stat|
- `stat_test/stage1_univariate_ic_results.csv`: full table

**Inputs required** (pull before running):
- `BTC_1h_RELVP.csv` (always present)
- `data/funding_rate_merged.csv` (optional; tz join currently broken — low priority)
- `data/fear_greed_index.csv` (present)
- `data/regime_hourly.csv`, `data/regime_daily.csv` (pull via `python3 -m src.data.fetch_regime --fred-key KEY`)

## Adding new features to Stage 1

Edit `build_features()` in `stage1_univariate_ic.py`. Add lines like:

```python
f["my_new_feature"] = df["some_col"].rolling(168).mean() / df["some_col"]
```

Rules:
- Feature must be computable from data available *at decision time* (no lookahead).
- Keep horizon-agnostic — don't window a feature with the forward-return horizon.
- Use `.rolling(N)` windows in **hours** (this is 1h data). 168 = 1 week, 720 = 30 days.

Then rerun — results append to the same CSV.

## Coordination with other agents

- **Training agent is running concurrently** on v12 and may modify `experiments/` and `src/models/`. Do **not** touch those.
- This folder (`stat_test/`) is owned by the feature-testing track. Keep artifacts here.
- Update this README's **Status** section whenever you:
  - Run Stage 1 on new features
  - Promote features to Stage 2
  - Change recommendations based on new results
- If findings change what features should feed v12, flag it explicitly in the MODEL_HISTORY next entry and in MEMORY.md — don't silently update without a pointer.

## Pointers to adjacent docs

- `experiments/MODEL_HISTORY.md` — living log of model iterations; cite Stage 1 findings when they drive decisions
- `experiments/MULTI_ASSET_PLAN.md` — the regime features plan these tests are validating
- `architecture_docs/arch-ml-model.md` — model architecture; Stage 1 feeds feature-set choices here
- `CLAUDE.md` — routing table has an entry pointing here
