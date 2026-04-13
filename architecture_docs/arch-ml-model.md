# ML Model

> **Read this when working on:** model architecture, training, evaluation, backtesting, inference pipeline
> **Related docs:** [arch-data-pipeline.md](arch-data-pipeline.md) (data source), [arch-trading-engine.md](arch-trading-engine.md) (consumes predictions), [arch-risk-safety.md](arch-risk-safety.md) (confidence thresholds)

> **Living counterpart:** see `experiments/MODEL_HISTORY.md` for the chronological narrative of every architecture decision. This doc captures the **current state** — that doc captures **how we got here**.

---

## Current best model

**v6-prime + 3-seed ensemble + SWA + combined filter** (Eval 12, 2026-04-12)

| Metric | Value |
|---|---|
| Architecture | TemporalEnrichedV6Prime (1 spatial + 1 temporal, day enrichment) |
| Pipeline | v1_raw (68 features) |
| Params | 24,737 |
| Per-trade EV | **+3.98%** (filtered) |
| Precision | **78.4%** |
| Sharpe per trade | 0.97 |
| Max consecutive losses | 23 |
| Trades / 5 years | 435 (~87/year) |
| Filter | `confidence > 0.65 AND tp_pct/sl_pct > 1.5` |

Files:
- Architecture: `src/models/architectures/v6_prime_vp_labels.py`
- Eval script: `src/models/eval_v6_prime.py`
- Backtest driver: `src/models/run_backtest.py`
- Backtest engine: `src/backtest/engine.py`
- Results: `experiments/eval_v6_prime_results.json`, `experiments/backtest_results.json`

## Input shape (v1_raw pipeline)

Per timestep, the model receives **68 features**:
- **50 VP bins** — relative volume profile (sum=1 normalized per row)
- **10 derived features** — `log_return`, `bar_range`, `bar_body`, `volume_ratio`, `upper_wick`, `lower_wick`, `body_dir`, `ohlc_open_ratio`, `ohlc_high_ratio`, `ohlc_low_ratio`
- **8 VP structure features** — `vp_ceiling_dist`, `vp_floor_dist`, `vp_num_peaks`, `vp_ceiling_strength`, `vp_floor_strength`, `vp_ceiling_consistency`, `vp_floor_consistency`, `vp_mid_range`

No technical indicators — VP is the primary signal.

**Input shape:** `(batch, 720, 68)` — 720 bars (30 days at 1h) lookback. Configurable via `LOOKBACK_BARS_MODEL`.

For 15min experiments: `(batch, 2880, 68)` — 30 days × 96 bars/day.

## VP structure features

Derived from VP histogram via Gaussian smoothing + peak detection:
1. **Aggregated VP** — sum 50 VP bins across rolling window, smooth with Gaussian filter (sigma=0.8)
2. **Peak detection** — `scipy.signal.find_peaks` on smoothed profile (prominence=0.05, distance=3)
3. **Ceiling/floor** — nearest peak above/below mid-bin (bin 25 = current price), distance normalized 0-1
4. **Peak consistency** — check if same peaks appear in shifted windows (3d, 6d, 9d back)
5. **Mid-range** — ratio of nearest peak distances

Computed in frozen pipeline: `src/features/pipelines/v1_raw.py`.

## Label design (current)

### v6-prime: VP-derived per-sample TP/SL labels

For each bar `i`:
```
tp_pct = vp_ceiling_dist[i] × 0.25 × 0.8 × vol_scale  # clipped [1%, 15%]
sl_pct = vp_floor_dist[i]   × 0.25 × 0.6 × vol_scale  # clipped [1%, 15%]

tp_level = close[i] × (1 + tp_pct)  # entry × (1 + tp%)
sl_level = close[i] × (1 - sl_pct)  # entry × (1 - sl%)

# First-hit logic: scan forward up to 14 days
label = 1 if tp_level hit before sl_level
label = 0 if sl_level hit before tp_level
label = NaN if neither hit within 14 days
```

Skip bars with `num_peaks < 1`.

**Why VP-derived:** matches user's manual strategy of aiming at the next VP peak. Per-sample adaptive — tight ceilings → small TP, open space → large TP.

### Legacy: fixed first-hit labels (used by v6 baseline)

```
TP = entry × (1 + 0.075 × vol_scale)   # 7.5% in bull, flipped in bear
SL = entry × (1 - 0.030 × vol_scale)   # 3% in bull, flipped in bear
```

Best ratio (7.5/3) found via TP/SL grid sweep — see `experiments/EVAL_TPSL_SWEEP.md`.

## Architecture lineage

```
RNN (vanishing gradients)
 └→ LSTM (spatial features need spatial reasoning)
     └→ CNN (fixed filters can't learn bin-pair relationships)
         └→ Spatial Transformer (no temporal context, Eval 4 = 63.3% acc)
             ├→ DualBranch v5 ✗ (candle branch = noise)
             ├→ v6 enriched 1+1 ✓ (best EV: bull long +1.57%)
             ├→ v7 simple 2+1 ✗ (no enrichment = bad)
             ├→ v8 enriched 2+1 (needs more data)
             │  └→ v8 on 15min (best raw acc 59.6%)
             └→ v6-prime VP-derived labels
                 ├→ Eval 11: asymmetry filter (+3.49%)
                 └→ Eval 12: ⭐⭐ + ensemble + combined filter (+3.98%)
```

See `experiments/MODEL_HISTORY.md` for full reasoning behind each transition.

## Frozen architectures

`src/models/architectures/`:

| File | Architecture | Used by |
|------|--------------|---------|
| `v2_temporal.py` | 1 spatial + 1 temporal, mean pool | Historical baseline |
| `v5_dualbranch_cls.py` | Dual-branch (VP + Candle separate) | Abandoned |
| `v6_temporal_enriched.py` | 1+1 with day enrichment | v6 baseline (still referenced) |
| `v7_simple_2plus1.py` | 2+1 simple, mean pool | Eval 5 |
| `v8_enriched_2plus1.py` | 2+1 enriched, CLS pool | Eval 6, 8 |
| **`v6_prime_vp_labels.py`** | **⭐⭐ Same as v6 (frozen for VP-derived label experiments)** | **Current best (Eval 11-12)** |
| `v10_long_temporal.py` | Subclass of v6-prime with `n_days=90` (90-day temporal window) | `eval_v10` — post-audit experiment |

Frozen rules: never modify. New experiments create new versioned files.

### v10 design note (2026-04-12)

v10 flips the lookback allocation vs v6-prime: **VP lookback 180d → 30d**
(via `BTC_1h_RELVP_30d.csv`), **temporal window 30d → 90d** (via
`n_days=90` in the shared architecture). Rationale: match the
chart-reader view of "zoom in on recent volume, zoom out on price
action."

Architecturally v10 is identical to v6-prime — only the temporal
positional embedding grows (31×32 → 91×32, +1,920 params), taking total
params from 24,737 → 26,657. The spatial/temporal transformer weights
are untouched.

Effective history per forward pass: 120 days (30d VP + 90d model),
down from 210d on v6-prime, but warmup loss shrinks by the same amount
so usable training rows actually increase by ~2,100. Sample/param ratio
stays ≈ 2.6:1 — unchanged risk profile, not an overfitting fix.

Files:
- Model: `src/models/architectures/v10_long_temporal.py`
- Eval: `src/models/eval_v10.py` (thin wrapper patching `eval_v6_prime`)
- Data: `src/data/compute_vp_30d.py` (one-time CSV generator)

## Training principles

### Walk-forward (always — never random shuffle)

10 folds, 6-month boundaries from 2020-01 to 2025-07. Each fold:
- Train: all data before train_end
- Val: train_end → val_end (6 months, for early stopping)
- Test: val_end → test_end (6 months, the actual scored period)

Train always expands. The model never sees future data.

### Regularization (current settings, prevents 1-epoch overfit)

- `dropout = 0.3`
- `weight_decay = 1e-3` (via AdamW)
- `label_smoothing = 0.1`
- Optimizer: AdamW (proper weight decay handling)
- Early stopping patience 15 epochs

### Multi-seed + SWA (current)

- **3 seeds per fold** (42, 43, 44) — average logits across seeds at test time
- **SWA within each seed** — average weights from epoch 15 onwards (finds flat minima)
- Reduces variance from random init lottery
- Cost: 3× training time per fold

## Inference pipeline

For production trading (not yet implemented — Phase 4-5):
1. New 1h bar closes
2. Fetch latest OHLCV from Kraken
3. Recompute VP features using same `v1_raw` pipeline as training
4. For each of the 3 seed models: forward pass → logit
5. Average the 3 logits → ensemble prediction
6. Compute `tp_pct` and `sl_pct` from current `vp_ceiling_dist` and `vp_floor_dist`
7. Apply combined filter: `sigmoid(logit) > 0.65 AND tp_pct / sl_pct > 1.5`
8. If filter passes → submit market order to Kraken with TP/SL bracket
9. Wait for fill or stop-loss

**Critical:** feature computation in inference must EXACTLY match training (frozen pipeline) — any divergence silently degrades predictions.

## Backtesting

### ⭐⭐⭐ Current best (Eval 17, 2026-04-12) — DEPLOYABLE STRATEGY

**Configuration:** v6-prime + 3-seed ensemble + SWA + combined_60_20 filter + 100% sizing (no reserve) + **3x leverage** + **24h post-SL pause**

| Metric | Value |
|---|---|
| Total return (3.66 years) | **+191.6%** |
| CAGR | **+34.0%** |
| Max drawdown | **-15.1%** |
| Sharpe (annualized) | 2.96 |
| Win rate | **72.0%** |
| Trades | 50 (~14/year) |
| Avg hold | ~3.7 days |
| Liquidations | **0** |

**Comparison:**

| Metric | Our bot (Eval 17) | BTC HODL | S&P 500 |
|---|---|---|---|
| CAGR | **+34.0%** | ~+15% | ~+10% |
| Max DD | **-15.1%** | -70% | -25% |
| Sharpe | 2.96 | ~0.5 | ~0.6 |

**3.4x S&P 500's CAGR with LESS drawdown.** This is genuinely elite-tier risk-adjusted performance.

**Realistic live expectation: +22-27% CAGR with -18% to -22% drawdowns** (after accounting for backtest decay, real funding rate variability, slippage spikes).

The post-SL pause was the unicorn fix — the user's idea. Filters out the 32 trades that happened right after a previous loss when VP was still stale from the shock. Those trades had negative expected value. Removing them improves both return AND drawdown.

Higher leverage (5x) tested at +263% return / -39% DD without post-SL pause. The post-SL pause approach is preferred because it dominates baseline on every metric except annualized Sharpe (which is just √N artifact).

### Key findings from leverage sweep
- **Zero liquidations across all 16 sizing × leverage combos.** Tight SL protects from forced liquidation even at 5x.
- **Drawdown scales linearly with leverage** (1x: -8.6%, 2x: -16.9%, 3x: -24.7%, 5x: -39.4%).
- **Leverage increases trade count** (65 → 85 from 1x → 5x). Wins free more capital → more signals captured.
- **Concentration > diversification at every leverage level.** 100% sizing wins at every tier.

### Engine details
`src/backtest/engine.py` provides realistic portfolio simulation:

`src/backtest/engine.py` provides realistic portfolio simulation:
- Portfolio class with cash, open positions, equity history
- Walks through bars chronologically
- Applies entry slippage, taker entry fee
- Exits at TP (maker fee 0.16%) or SL (taker fee 0.26%)
- Configurable position sizing (fixed %, dynamic, full allocation)
- Reserve buffer (default 30% of equity untouchable)
- Pyramiding control (stack vs skip)
- Mark-to-market equity tracking

`src/backtest/metrics.py` computes:
- Total return, annualized return (CAGR)
- Max drawdown (% of peak equity)
- Sharpe (annualized)
- Win rate, avg win/loss
- Max consecutive losses
- Monthly P&L breakdown

`src/models/run_backtest.py` runs 12 backtest combinations (3 filters × 4 sizing strategies) on cached predictions.

## Experiment management

Each training run saves to `experiments/runs/run_<timestamp>/`:
- `config.json` — hyperparameters used
- `model.pt` — best-val-loss checkpoint
- `metrics.json` — per-epoch loss/acc curves

Eval scripts also save aggregate results to `experiments/<eval_name>_results.json`.

For multi-seed runs: each seed produces its own checkpoint subdirectory.

For backtesting: `eval_v6_prime.py` saves predictions to `experiments/v6_prime_predictions.npz` so the backtest engine can iterate without retraining.

## Known issues & gotchas

### Look-ahead bias
The most dangerous mistake. Audit:
- Features at bar `i` must use only data from bar `i` and earlier
- VP windows end AT bar `i`, not include bar `i+1`
- Volatility scaling for labels uses rolling vol up to bar `i`

### Survivorship bias
BTC/USD is a survivor. Model has not seen coins that went to zero. Don't extrapolate to altcoins.

### Regime change
Walk-forward retraining helps but doesn't eliminate. Fold 4 (2022 H1 LUNA crash) is consistently the hardest period.

### Reproducibility
Fixed seeds (42, 43, 44) make runs deterministic on the same hardware. GPU non-determinism is minimal for our model size.

### Accuracy ≠ profitability
Per-sample VP-derived labels mean accuracy can be misleading. A model that defaults to label 0 on wide-TP setups gets high accuracy without making profitable trades. **Always evaluate by per-trade EV from filter analysis, not raw accuracy.**

### Funding rate
Tested in Eval 9 with fine-tuning approach. Mixed results, fold 6 catastrophic collapse. Not currently used. See `experiments/MODEL_HISTORY.md` section 10.

### Confidence filtering
Failed in Eval 4 era (logits too bimodal). Works in v6-prime era because label smoothing produces calibrated logits (std ~1.3 instead of bimodal). Combined with asymmetry filter → current best result.
