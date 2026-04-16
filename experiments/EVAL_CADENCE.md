# EVAL — Prediction Cadence Ablation (v11-full-tb, 15m vs 1h)

> **STATUS 2026-04-16 — Matched-gradient-steps (200 epochs) confirms undertraining hypothesis.** Confidence ceiling rises to 0.94 (was 0.77), threshold trade counts stabilize across seeds (conf75 holdout: 135/116/123 vs 51/0/6 on match-epochs), fold 4 mode collapse resolved. **All 3 seeds produce positive holdout CAGR at conf75_pyr** (+20.0/+6.2/+4.0%, mean +10.1%) — first time any config has achieved this. Non-pyramid conf75 shows a reproducible but tiny edge (+0.7% mean, 60.7% WR, 20 trades, 1pp spread). **Red flag**: fold 12 (2026 Q1) raw accuracy crashes to ~30% on all seeds — but confidence-filtered behavior is seed-dependent (59.1% precision at conf80 on seed42, 0% on seed43). Regime features now the highest-priority next experiment. See §10 for the full matched-gradient-steps writeup.
>
> **Prior status (2026-04-15 evening)**: Stage 7 single-seed "+6.1% holdout CAGR" retracted after 3-seed match-epochs replication. See §9.
>
> **Read this when**: continuing the cadence experiment, evaluating matched-gradient-steps results, or planning regime features.
>
> **Related**: `LABEL_REDESIGN.md` (v11-full-tb baseline), `EVAL_TRANSFORMER.md §Cadence`, `MODEL_HISTORY.md §32–33`.

---

## 1. Why this experiment

Post-v11-tb the binding constraint on Phase 3 was **sample count**: 20–30 trades × 0.3% net EV/trade can't compound into positive holdout CAGR after fees over 278 days. One interpretation was "we need more data." Another interpretation, raised 2026-04-14 late-session, was: **we're training at 15m cadence on noise.**

Key observation:
- **Data resolution** (15m): how finely the model *sees* the market
- **Prediction cadence** (originally 15m): how often we emit a prediction and realize a label

These had been collapsed to the same thing in v11 because it was the obvious thing. But they're separable: the model's input window already contains every 15m bar regardless of how often it's called. Striding the anchor grid from 15m → 1h means:

- The model still sees every 15m bar in its 90-day input window (no information loss)
- Training anchors and evaluation anchors are placed every 4 bars (every hour, wall-clock aligned to `:00`)
- Labels at non-anchor bars are never computed
- Training set shrinks ~4× but samples are far less correlated (consecutive 15m bars are near-duplicates)
- Labels at 1h cadence are less dominated by microstructure noise
- Deployment alignment improves: a swing trader doesn't act every 15 minutes

This is not a resampling. The 15m CSV is untouched. Only the anchor grid changes.

## 2. Implementation

**Commit**: `e8e4681 feat(v11): decouple prediction cadence from 15m data resolution`

- `src/models/eval_v11.py`: added `STRIDE_BARS_DEFAULT=1`, `--stride` CLI flag, wall-clock-aligned `cadence_mask` in the eligibility block (`~line 710`). Tag auto-appends `_c{cadence_minutes}m` when stride ≠ 1. `stride_bars` / `cadence_minutes` persisted to both the results JSON and the predictions NPZ.
- `src/models/run_backtest_v11_tb.py`: added `--tags` / `--out` CLI flags, auto-reads cadence from each NPZ, emits a cadence ablation comparison table when ≥2 cadences are present, guards against zero-trade variants (previously crashed with `KeyError: 'annualized_return_pct'`).

**Training command** (Colab, same config as 15m baseline):
```bash
python3 -m src.models.eval_v11 --labels triple_barrier --features full --stride 4
# → experiments/v11_11_tb_full_c60m_predictions.npz
#   experiments/eval_v11_11_tb_full_c60m_results.json
```

**Backtest command** (local):
```bash
python3 -m src.models.run_backtest_v11_tb \
    --tags 11_tb_full,11_tb_nopv,11_tb_full_c60m \
    --out backtest_results_v11_tb_cadence_ablation.json
```

**Config invariants** (held constant between 15m and 1h runs):
- Model: `AbsVPv11`, 1 spatial × 1 temporal transformer layer
- Labels: triple-barrier (`TB_VOL_WINDOW=288`, `TB_HORIZON_BARS=96`, `TB_BARRIER_K=2.0`, 1–15% clip)
- Feature pipeline: `BTC_15m_ABSVP_30d.csv`, untouched
- Epochs: 50, early-stop patience 15, SWA from epoch 15
- Optimizer: AdamW, LR 5e-4, weight decay 1e-3, dropout 0.3, label smoothing 0.1
- Seeds: **1** (⚠ single seed — not statistically replicated)
- Fold boundaries: identical, 14-day wall-clock embargo
- Walk-forward: identical
- **Training budget: matched by epochs, not by gradient steps** — the 1h run sees ~4× fewer gradient updates. This is intentional but has consequences (see §5).

## 3. Per-fold comparison

Training time on Colab A100: **26.7 min** for the 1h run (vs ~1.5h typical for 15m, ~4× speedup as expected from the sample count reduction).

| Fold | Period | 15m n | 15m acc | 15m long_rate | 15m long_EV | 1h n | 1h acc | 1h long_rate | 1h long_EV |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2020-07 → 2021-01 | 16,278 | 0.7134 | 0.795 | +3.585% | 4,078 | **0.7815** | **1.000** | +3.110% |
| 2 | 2021-01 → 2021-07 | 16,106 | 0.4179 | 0.735 | −1.088% | 4,024 | 0.4187 | 0.626 | −1.464% |
| 3 | 2021-07 → 2022-01 | 16,197 | 0.5729 | 0.636 | +1.660% | 4,051 | 0.5376 | 0.578 | +1.310% |
| 4 | 2022-01 → 2022-07 | 16,253 | 0.5267 | 0.406 | −0.950% | 4,061 | 0.5976 | **0.000** | 0.000% |
| 5 | 2022-07 → 2023-01 | 14,858 | 0.4669 | 0.853 | −0.303% | 3,718 | 0.4938 | **0.021** | −1.466% |
| 6 | 2023-01 → 2023-07 | 15,032 | 0.5738 | 0.491 | +1.231% | 3,757 | **0.5997** | 0.153 | **+3.703%** |
| 7 | 2023-07 → 2024-01 | 15,636 | 0.5118 | 0.515 | +1.222% | 3,910 | 0.5494 | 0.431 | +1.558% |
| 8 | 2024-01 → 2024-07 | 16,321 | **0.6171** | 0.587 | +1.174% | 4,079 | 0.4891 | 0.474 | +0.438% |
| 9 | 2024-07 → 2025-01 | 16,592 | 0.4942 | 0.553 | +1.445% | 4,146 | 0.4701 | 0.230 | **+3.106%** |
| 10 | 2025-01 → 2025-07 | 15,976 | 0.5158 | 0.405 | +0.218% | 3,994 | 0.4659 | 0.201 | −0.471% |
| **11** | **2025-07 → 2026-01 (holdout)** | 16,349 | 0.4930 | 0.755 | −0.457% | 4,095 | 0.4608 | 0.384 | −0.656% |
| **12** | **2026-01 → 2026-04 (holdout)** | 8,750 | 0.4221 | 0.565 | −1.245% | 2,192 | 0.5082 | 0.162 | −2.000% |
| **Overall** |  | 184,348 | **0.5317** |  |  | 46,105 | **0.5318** |  |  |

**Observations from the fold grid:**

- **Overall walk-forward accuracy is identical (0.5317 vs 0.5318).** The two models are equivalent on raw classification quality; the difference is entirely in *where* they place their confidence.
- **Fold 1 (2020 bull) sanity check**: 1h acc 0.7815 > 15m 0.7134. The stride mask is correctly aligned.
- **Concentrated-confidence pattern**: the 1h model is dramatically more selective on several folds (long_rate 0.153, 0.230, 0.201 on folds 6/9/10 vs 0.491/0.553/0.405 on 15m). On folds 6 and 9, this concentration produces **far higher per-trade EV** (+3.70% vs +1.23%, +3.11% vs +1.45%). Fewer trades, bigger edge — the "quality over quantity" pattern.
- **Fold 4 mode collapse**: the 1h model produces **zero longs** on the entire fold (4,061 predictions, all 0). 15m was 40.6% long with −0.95% EV. This is the single-seed risk showing up — at 0 longs, long_EV = 0 by definition, so it's not a *loss*, but it's a sign that with 1 seed and ~4× less training data per fold, the model can degenerate to mode prediction on difficult regimes. Worth watching in the 3-seed follow-up.
- **Folds 5 also near-degenerate** (77/3718 = 2.1% long rate). Same concern.
- **Holdout acc drops slightly** on 1h: fold 11 0.4608 (vs 0.4930), fold 12 0.5082 (vs 0.4221). Note fold 12 went *up* — but both are class-base-rate regions, so raw acc on holdout is not the metric to read. Backtest is.

## 4. Backtest results — real engine (fees, slippage, latency)

Run: `run_backtest_v11_tb.py --tags 11_tb_full,11_tb_nopv,11_tb_full_c60m`

### 4.1 Non-pyramid variants (cleaner story, smaller trade counts)

| Config | Cadence | Full CAGR | Full DD | Full n | Holdout CAGR | Holdout DD | Holdout n | Holdout WR |
|---|---|---|---|---|---|---|---|---|
| conf70_pause24 | 15m | +0.3% | −14.9% | 271 | −4.0% | −4.2% | 29 | 55.2% |
| conf70_pause24 | **1h** | **+2.4%** | **−3.4%** | 57 | −5.4% | −1.2% | 13 | 46.2% |
| conf75_pause24 | 15m | +1.1% | −9.2% | 246 | −4.0% | −4.2% | 24 | 54.2% |
| **conf75_pause24** | **1h** | **+2.4%** | **−2.4%** | 44 | **+0.9%** | **−0.5%** | **8** | **62.5%** |
| conf80_pause24 | 15m | +1.8% | −8.8% | 193 | **−1.5%** | **−1.7%** | 20 | 60.0% |
| conf80_pause24 | 1h | −2.8% | −3.6% | 21 | — | — | **0** | — |
| conf60_nopause | 15m | −0.4% | −21.9% | 360 | −13.8% | −12.7% | 54 | 44.4% |
| conf60_nopause | **1h** | **+3.6%** | −10.1% | 165 | −2.6% | −3.3% | 22 | 45.5% |

### 4.2 Pyramid variants (where the deployment story actually lives)

| Config | Cadence | Full CAGR | Full DD | Full n | Holdout CAGR | Holdout DD | Holdout n | Holdout WR |
|---|---|---|---|---|---|---|---|---|
| conf70_pause24_pyr | 15m | +3.5% | −55.9% | 2,568 | −22.0% | −21.8% | 196 | 51.5% |
| conf70_pause24_pyr | **1h** | **+10.2%** | **−16.9%** | 422 | −10.6% | −6.0% | 82 | 58.5% |
| conf75_pause24_pyr | 15m | +6.5% | −40.6% | 2,354 | −18.4% | −19.3% | 164 | 54.9% |
| **conf75_pause24_pyr** | **1h** | **+6.6%** | **−14.2%** | 316 | **+6.1%** | **−2.7%** | **51** | **60.8%** |
| conf80_pause24_pyr | 15m | +9.7% | −40.7% | 1,931 | −15.0% | −10.4% | 138 | 56.5% |
| conf80_pause24_pyr | 1h | −14.9% | −19.6% | 125 | — | — | 0 | — |
| conf60_nopause_pyr | 15m | −7.1% | −78.1% | 4,175 | −55.7% | −50.9% | 452 | 44.2% |
| conf60_nopause_pyr | **1h** | **+15.9%** | −41.8% | 1,647 | −17.1% | −19.3% | 152 | 51.3% |

### 4.3 ⚠ RETRACTED — single-seed "first positive holdout CAGR" — see §9 for 3-seed results

**`conf75_pause24_pyr @ 1h cadence`** (pyramid allowed, conf ≥ 0.75, 24h post-SL pause):
- **Holdout: +6.1% CAGR / −2.7% DD / 51 trades / 60.8% WR**
- **Full period: +6.6% CAGR / −14.2% DD / 316 trades / 57.9% WR**

For comparison with the prior best:

| | v6-prime honest | v11-tb-full 15m (conf80) | **v11-tb-full 1h conf75_pyr** |
|---|---|---|---|
| Full CAGR | +6.0% | +1.8% | **+6.6%** |
| Full DD | −18.4% | −8.8% | **−14.2%** |
| Full n | ~130 | 193 | 316 |
| Holdout CAGR | ~−5% | **−1.5%** | **+6.1%** |
| Holdout DD | ~18.4% | −1.7% | **−2.7%** |
| Holdout n | ~25 | 20 | 51 |
| Holdout WR | ~65% (leaked labels) | 60.0% | **60.8%** |

The 1h conf75-pyr config matches v6-prime's nominal full-period CAGR, **beats it on holdout CAGR by ~11pp**, and cuts holdout drawdown by a factor of ~7. It's also the only v11 config whose holdout trade count (51) is large enough to not be a statistical punchline.

The non-pyramid conf75 at 1h is also positive on holdout (+0.9% CAGR, −0.5% DD, 62.5% WR) but on only 8 trades — too thin to stand alone.

## 5. Three honest caveats (read before declaring victory)

### 5.1 Single seed

Match-epochs, 1 seed, 1 training run. We have no replication. On 51 holdout trades with 60.8% WR the 95% CI on win rate is roughly ±13pp. The CAGR number has a wide confidence interval by construction. **This is the #1 reason not to deploy.**

Required follow-up: **3-seed re-run of 1h-full-tb, same config.** If the conf75-pyr holdout result stays above ~0% CAGR and ~55% WR across all three seeds, it's a real signal. If it spreads wildly, we learned the single run was on the lucky side of seed variance.

### 5.2 Confidence-scale compression (undertraining signature)

The 1h model's probability distribution on holdout is compressed toward zero:
- **Max prob: 0.773** (vs 15m runs where conf80 is reachable)
- **p99: 0.765**
- **Predictions ≥ 0.80: zero**

This is why `conf80_pause24` and `conf80_pause24_pyr` return 0 trades on the 1h holdout. The model's logits haven't spread out as much as 15m's — consistent with the match-epochs tradeoff (4× fewer gradient steps → less-converged bias parameters → logits clustered near zero).

**Implication for threshold comparisons**: conf thresholds are NOT directly comparable across cadences. A conf75 cut on the 1h model sits at approximately the same prediction quantile as a conf80 cut on the 15m model. When reading the tables, compare **conf75 @ 1h to conf80 @ 15m** for a fair apples-to-apples quantile match.

Under that apples-to-apples framing:

|  | 15m conf80 holdout | 1h conf75 holdout |
|---|---|---|
| CAGR | −1.5% | **+0.9%** |
| DD | −1.7% | **−0.5%** |
| WR | 60.0% | **62.5%** |
| n | 20 | 8 |

The 1h model still wins on every metric except trade count. But 8 trades is very thin.

Required follow-up: **matched-gradient-steps re-run** (200 epochs on 1h instead of 50) to test whether conf80 becomes reachable when the bias is fully converged. If so, that's the cleaner deployable config. If not, the confidence compression is a genuine property of 1h prediction on this task, and conf75 is the correct threshold.

### 5.3 Fold 4 and fold 5 mode collapse

The 1h model produced 0 longs on fold 4 and 77/3718 longs on fold 5. With one seed and 4× less data per fold, the optimizer can land in degenerate regions for some regimes. Overall walk-forward accuracy survived (0.5318 ≈ 15m's 0.5317), but the fact that the degenerate behavior is *possible* at 1h and *not* at 15m is noteworthy.

Required follow-up: same 3-seed re-run covers this. If all 3 seeds degenerate on fold 4, that's the model learning "no long signal in this regime" — defensible. If only 1 of 3 degenerates, it's seed variance and we should report the median.

## 6. Interpretation

The **accuracy story** is boring and comforting: raw walk-forward acc is identical across cadences. This tells us the cadence change didn't damage the model's underlying classification quality.

The **real-engine backtest story** is genuinely surprising: 1h dominates 15m on nearly every metric, most dramatically on drawdown (3–5× smaller) and holdout CAGR (first positive result in project history at conf75-pyr). The most parsimonious explanation is the one we sketched when we designed the experiment:

- At 15m cadence, labels are dominated by microstructure noise; consecutive anchors are near-duplicates; the model learns a noisy classifier that makes many weak-confidence calls
- At 1h cadence, labels reflect moves big enough to be signal; anchors are more independent; the model learns a cleaner classifier that makes fewer strong-confidence calls
- The confidence filter + backtest engine turns "fewer strong calls" into smaller drawdowns and better holdout discrimination
- Fewer trades × higher edge per trade can compound into positive CAGR when 15m's many-trades × lower-edge couldn't

This is consistent with the user's original intuition ("we might have been training on noise") — not proof of it, but the strongest evidence we've seen that the intuition had signal. The fact that overall accuracy didn't move while backtest metrics moved a lot supports "the model had signal, the cadence was costing us in execution."

Two other interpretations worth holding:

- **Confidence-scale compression** could be doing more of the work than cadence. Under undertraining, the 1h model's confidence is noisier on the top end — maybe the apparent "strong calls" are partly artifacts of where 0.75/0.80 sits in the compressed distribution. The matched-gradient-steps follow-up separates this from true cadence effects.
- **Single-seed variance**. The whole result could be one lucky draw. 3-seed re-run separates this.

## 7. What to do tomorrow

### Priority 1 — replicate before celebrating

**3-seed re-run of 1h-full-tb** at match-epochs (current config, just bump `--seeds 3`):
```bash
python3 -m src.models.eval_v11 --labels triple_barrier --features full --stride 4 --seeds 3
```
Then rerun the backtest with the same `--tags` flags. Time budget: ~80 min on Colab A100 (3× 27 min + overhead).

**Read as decisive IF**:
- Conf75-pyr holdout CAGR stays positive across all 3 seeds (doesn't need to be +6.1%, just > 0%)
- Holdout WR stays ≥ 55% across all 3 seeds
- Full-period CAGR stays comparable to 15m or better

**Read as noise IF**:
- Any seed flips conf75-pyr holdout to negative
- Holdout WR drops below 50% on any seed
- Fold 4/5 mode collapse persists across all seeds in the same direction (that's the model, not noise — defensible but limits the story)

### Priority 2 — separate cadence from undertraining

**Matched-gradient-steps re-run** of 1h-full-tb: same config, 200 epochs instead of 50. This gives the 1h model the same total optimizer updates as the 15m baseline. Only worth running if Priority 1 passes — otherwise we're polishing noise.

```bash
# Would need a new --epochs CLI flag, or temporarily edit EPOCHS=200
python3 -m src.models.eval_v11 --labels triple_barrier --features full --stride 4 --seeds 3
```

**Read as "cadence is the real lever" IF**:
- Conf80 becomes reachable (max prob > 0.80)
- Conf80-pyr holdout performance matches or exceeds conf75-pyr at 50 epochs

**Read as "it was undertraining, not cadence" IF**:
- Matched-steps results shift back toward 15m's trade distribution
- Drawdown widens
- Conf75-pyr holdout CAGR drops toward zero

### Priority 3 — ablate at other cadences

Only if Priority 1 and 2 both support the cadence story, run:
- **4h cadence** (`--stride 16`). Even fewer samples (~11,500), but matches the v6-prime-era cadence. If 4h also wins, the lesson is "match your training cadence to your deployment cadence, period."
- **15m + 1h joint training** (hybrid stride, e.g., every bar during training, every 4 bars at eval). Ablates whether the 1h win is about label quality or training density.

### Priority 4 — other open questions from earlier in the session

Two items that came up during the design discussion and should not be forgotten:

- **Absolute price-range change features** (envelope dynamics): `Δlog(window_hi)`, `Δlog(window_lo)`, `Δrange_pct`, `bars_since_new_high/low`. The model currently sees a snapshot of the envelope but not its velocity. Orthogonal to regime features. Cheap to add.
- **Regime conditioning features** per `MULTI_ASSET_PLAN.md §REFRAME` (GLD/USO/DXY/VIX + FFR + 10Y-2Y yield curve). Attacks the fold 12 / regime-change problem directly. Still the next-next experiment.

Order recommendation: cadence replication → matched-steps → envelope dynamics → regime features. Each one is a single-variable change on top of a validated predecessor, which keeps the attribution clean.

## 9. ⭐ 3-seed independent replication (2026-04-15 evening) — retracts §4.3

Run: 3 independent training runs at `--stride 4 --seeds 1 --base-seed {42,43,44}` (commit `647cc24` added `--base-seed` arg; previous `--seeds N` did intra-run ensembling which is **not** replication). All three run the identical config to Stage 7; only the base seed differs. Backtested locally with `run_backtest_v11_tb.py` per seed (each seed's output saved before the next overwrote it).

### 9.1 `conf75_pause24_pyr` — the Stage 7 headline config

| Metric | seed42 (= Stage 7) | seed43 | seed44 | mean | spread |
|---|---|---|---|---|---|
| **Full CAGR** | +6.6% | +5.2% | +6.5% | **+6.1%** | 1.4pp |
| Full DD | −14.2% | −13.3% | −10.1% | −12.5% | |
| Full n_trades | 316 | 250 | 221 | 262 | |
| Full WR | 57.9% | 58.8% | 57.0% | 57.9% | |
| **Holdout CAGR** | **+6.1%** | **0.0%** | **−3.0%** | **+1.0%** | **9.1 pp** |
| Holdout DD | −2.7% | 0.0% | −1.8% | | |
| **Holdout n_trades** | **51** | **0** | **6** | **19** | |
| Holdout WR | 60.8% | N/A | 50.0% | | |

Seed43 produced **zero holdout trades** at conf75. Seed44 produced **six**. Only seed42 crossed the threshold meaningfully. The +6.1% headline is one draw from a high-variance distribution, not a reproducible signal.

### 9.2 `conf70_pause24_pyr` — lower threshold, stable trade counts

| Metric | seed42 | seed43 | seed44 | mean | spread |
|---|---|---|---|---|---|
| **Full CAGR** | +10.2% | +11.5% | +16.6% | **+12.8%** | 6.4pp |
| Full DD | −16.9% | −12.3% | −12.8% | −14.0% | |
| Full n | 422 | 424 | 417 | 421 | |
| Full WR | 60.7% | 63.4% | 63.3% | 62.5% | |
| **Holdout CAGR** | **−10.6%** | **+9.8%** | **−6.3%** | **−2.4%** | **20.4 pp** |
| Holdout n | 82 | 56 | 56 | 65 | |
| Holdout WR | 58.5% | 69.6% | 62.5% | 63.5% | |

At conf70_pyr the threshold is low enough that trade count is stable (~60 holdout trades across all 3 seeds), so the holdout CAGR is comparing like-with-like — and the mean is **−2.4%** with a 20pp spread. Under this framing the holdout is net negative.

### 9.3 What's stable and what's not

**Stable across seeds:**
- Overall walk-forward acc: 0.5318 / 0.5224 / 0.5286 (mean 0.5276, spread 0.9pp)
- Full-period CAGR at every variant (spreads 1.4–6.4pp)
- Full-period win rate (~58% at conf75_pyr, ~63% at conf70_pyr)
- **Fold 4 mode collapse** — all three seeds produced `n_long=0` on fold 4 (2022 bear), 4061 samples each. This is a deterministic model failure on that regime, not seed noise. Defensible as "no long signal here" but the model is not *choosing* not to trade — it's saturating to zero.

**Unstable across seeds:**
- Holdout CAGR (the headline metric) — 9–20pp spreads
- Confidence-threshold trade counts — conf75 holdout went 51 / 0 / 6 across seeds
- Per-fold `n_long` on active folds — e.g. fold 2: 2519 / 2708 / 840; fold 5: 77 / 1485 / 239 (up to 20× ratio)
- Prediction probability distributions: p50 = 0.392 / 0.425 / 0.390

The p50 spread alone explains the conf75 trade-count collapse: seed43's median prob is higher but its *tail* is compressed closer to the median, pushing most holdout samples below 0.75.

### 9.4 Why this matters

The whole Stage 7 framing was "1h cadence produces positive holdout CAGR where 15m couldn't." With the 3-seed replication we now know:

1. **Classification quality is the same.** Overall acc 0.53 on both cadences, stable across seeds. The cadence change did not damage or improve classification.
2. **Full-period CAGR is real-ish.** +5.2 to +6.6% at conf75_pyr across seeds, roughly matching v6-prime's +6.0%. Not a regression, not an improvement, not SPY-beating.
3. **Holdout CAGR is dominated by confidence-calibration noise.** 51 → 0 → 6 trades across seeds at conf75 is not a threshold — it's a coin flip. Any claim about holdout CAGR at a fixed threshold is fragile unless the threshold is robust across seeds.
4. **Fold 4 is a model failure, not noise.** Zero longs on all three seeds on the same 2022 bear fold. Now a known architectural limit.
5. **Seed42's +6.1% was cherry-picked by the seed lottery** — it was simultaneously the best walk-forward acc (0.5318 > 0.5286 > 0.5224), the best conf75_pyr holdout CAGR (+6.1 > 0 > −3.0), and the *worst* conf70_pyr holdout CAGR (−10.6 < −6.3 < +9.8). "Best" flipping to "worst" depending on which confidence cut you look at is exactly the signature of seed noise dominating real signal.

### 9.5 What we keep and what we retract

**Keep** (still defensible):
- Walk-forward accuracy equivalence between 15m and 1h (0.5317 vs 0.5318, now 0.5276 mean across 3 seeds)
- The methodological framing that cadence and data resolution are separable
- Full-period drawdown reduction at 1h vs 15m (holds across seeds: −10 to −14% at 1h vs ~−40% at 15m pyr)
- The observation that 1h training is 3–4× faster on Colab and operationally more aligned with swing-trading deployment
- Fold 4 mode collapse as a known failure mode (reproduced on all seeds)

**Retract:**
- "First positive real-engine holdout CAGR in project history" — NOT reproduced across seeds
- "Beats v6-prime on holdout by ~11 pp" — that comparison used seed42's +6.1%, which is the top of a distribution whose mean is +1.0% and whose spread is 9pp
- Any direct threshold comparison between 15m and 1h at a fixed conf value — the logit distributions shift across seeds, so "conf75 @ 1h" is not a fixed object
- The §5.1 claim that 3-seed replication at match-epochs would be "decisive IF CAGR stays positive across all 3 seeds" — it didn't, and we now have the answer

### 9.6 Revised next step

**Matched-gradient-steps re-run at 1h cadence, 3 independent seeds from the start.**

Rationale: the match-epochs config is visibly undertrained (max prob ~0.77, p50 ~0.39–0.43, conf80 unreachable on all 3 seeds). The hypothesis is now "under-convergence is producing the calibration instability — full training will both raise the confidence ceiling and stabilize the threshold trade counts across seeds." If this hypothesis is right, matched-grad-steps will produce tight conf75/conf80 holdout trade counts across seeds (not 51/0/6 but e.g. 45/48/52). If it's wrong, conf75_pyr is just unstable and we should abandon the threshold-filter framing entirely.

Envelope dynamics and regime features are deferred **until we have a reproducible baseline**. Feature engineering on top of a baseline whose holdout CAGR is dominated by seed variance is wasted effort — any improvement we measure will be swamped by the noise floor we already know about.

### 9.7 Artifacts (3-seed replication)

- `experiments/v11_11_tb_full_c60m_seed42_predictions.npz` + `eval_v11_11_tb_full_c60m_seed42_results.json`
- `experiments/v11_11_tb_full_c60m_seed43_predictions.npz` + `eval_v11_11_tb_full_c60m_seed43_results.json`
- `experiments/v11_11_tb_full_c60m_seed44_predictions.npz` + `eval_v11_11_tb_full_c60m_seed44_results.json`
- Code: commit `647cc24 feat(v11): add --base-seed for independent-run replication`
- Raw backtest tables from the 3 per-seed runs are in the session transcript; not persisted to a single JSON file because the backtest script overwrites its output (next time: write per-seed output files explicitly).

---

## 10. ⭐ Matched-gradient-steps (200 epochs, 3 independent seeds) — 2026-04-16

Run: 3 independent training runs at `--stride 4 --seeds 1 --base-seed {42,43,44} --epochs 200 --patience 60 --swa-start 60` (commit `2b3d7c3` added `--epochs/--patience/--swa-start` flags). This gives 4× more gradient updates than match-epochs (50 epochs), matching the 15m baseline's total optimizer budget. Patience and SWA-start scaled proportionally (15→60). ~71 min per seed on Colab A100.

### 10.1 Undertraining hypothesis — CONFIRMED

The three testable predictions from §9.6:

**1. Does max prob rise above 0.80?** YES.

| Seed | match-epochs max | mgs max | match-epochs p50 | mgs p50 |
|---|---|---|---|---|
| 42 | 0.773 | **0.940** | 0.392 | **0.547** |
| 43 | 0.777 | **0.939** | 0.425 | **0.533** |
| 44 | 0.770 | **0.945** | 0.390 | **0.512** |

Confidence ceiling nearly doubled in logit-space. conf80 is now reachable on all seeds.

**2. Do conf75/80 holdout trade counts stabilize across seeds?** YES.

| Threshold | match-epochs holdout n | mgs holdout n |
|---|---|---|
| conf75_pyr | 51 / 0 / 6 | **135 / 116 / 123** |
| conf80_pyr | 0 / 0 / 0 | **114 / 94 / 96** |

The 51→0→6 collapse at conf75 is gone. Trade counts are now within ~15% of each other.

**3. Does holdout CAGR mean ≥ 0%?** YES at every pyramid variant.

### 10.2 Walk-forward accuracy — slight degradation

| Seed | match-epochs | mgs | Δ |
|---|---|---|---|
| 42 | 0.5318 | 0.5007 | −3.1pp |
| 43 | 0.5224 | 0.5074 | −1.5pp |
| 44 | 0.5286 | 0.5323 | +0.4pp |
| **mean** | **0.5276** | **0.5135** | **−1.4pp** |

More training didn't improve raw classification — slight overfitting. But walk-forward acc was never the metric that mattered for backtest performance; confidence calibration was.

### 10.3 Per-fold highlights

**Fold 4 mode collapse — RESOLVED.** Match-epochs produced n_long=0 on all 3 seeds (known model failure). Matched-gradient-steps produces 1681/667/665 longs — the model now makes predictions. But the EV is heavily negative (−1.6%/−4.9%/−3.4% long_EV), so the model went from "correctly abstaining" to "confidently wrong" on the 2022 bear.

**Fold 12 (2026 Q1) raw accuracy — CRASHED.**

| Seed | match-epochs acc | mgs acc |
|---|---|---|
| 42 | 0.508 | **0.309** |
| 43 | 0.493 | **0.333** |
| 44 | 0.546 | **0.286** |

All seeds at ~30% — worse than coin flip. See §10.5 for confidence-filtered analysis (the raw number is misleading).

**Fold 11 (2025 H2) accuracy — improved on 2 of 3 seeds.**

| Seed | match-epochs acc | mgs acc |
|---|---|---|
| 42 | 0.461 | 0.487 |
| 43 | 0.457 | **0.524** |
| 44 | 0.456 | **0.514** |

Fold 11 went from uniformly sub-50% to slightly above. The positive holdout CAGR is mostly driven by fold 11.

### 10.4 Backtest results — real engine

#### `conf75_pause24_pyr` (the headline comparison)

| | match-epochs | | mgs | |
|---|---|---|---|---|
| Seed | CAGR (holdout) | n | CAGR (holdout) | n |
| 42 | +6.1% | 51 | **+20.0%** | 135 |
| 43 | 0.0% | 0 | **+6.2%** | 116 |
| 44 | −3.0% | 6 | **+4.0%** | 123 |
| **mean** | **+1.0%** | **19** | **+10.1%** | **125** |
| spread | 9.1pp | | 16.0pp | |

All 3 seeds positive. Trade counts stable. Mean +10.1%. But seed42 at +20.0% looks like another lucky draw (16pp spread).

Full-period at conf75_pyr: +6.5/+7.3/+12.4% CAGR (mean +8.7%), DD −43.8/−29.8/−24.3%, WR 60.5/60.6/62.9%.

#### `conf80_pause24_pyr` (newly reachable — was 0 trades on match-epochs)

| Seed | Holdout CAGR | Holdout DD | Holdout n | Holdout WR |
|---|---|---|---|---|
| 42 | **+9.5%** | −7.1% | 114 | 65.8% |
| 43 | +0.5% | −5.2% | 94 | 59.6% |
| 44 | +3.7% | −7.0% | 96 | 62.5% |
| **mean** | **+4.6%** | **−6.4%** | **101** | **62.6%** |

Trade counts stable (114/94/96). Tighter spread than conf75_pyr (9pp vs 16pp). Mean +4.6% with 62.6% WR is a reasonable signal if the regime concern (§10.5) can be addressed.

#### `conf75_pause24` (non-pyramid — cleanest signal, no pyramiding amplification)

| Seed | Holdout CAGR | Holdout DD | Holdout n | Holdout WR |
|---|---|---|---|---|
| 42 | +1.2% | −1.4% | 23 | 60.9% |
| 43 | +0.2% | −1.4% | 20 | 60.0% |
| 44 | +0.7% | −2.0% | 18 | 61.1% |
| **mean** | **+0.7%** | **−1.6%** | **20** | **60.7%** |

**Tightest results in project history.** 1pp spread on CAGR, 20 trades per seed, 60–61% WR uniformly, −1.4 to −2.0% DD. This is what a real but tiny edge looks like — reproducible across seeds, just not deployment-scale yet.

#### `conf80_pause24` (non-pyramid, higher threshold)

| Seed | Holdout CAGR | Holdout DD | Holdout n | Holdout WR |
|---|---|---|---|---|
| 42 | +0.4% | −2.4% | 20 | 60.0% |
| 43 | −0.3% | −1.7% | 17 | 58.8% |
| 44 | +0.2% | −2.0% | 18 | 61.1% |
| **mean** | **+0.1%** | **−2.0%** | **18** | **60.0%** |

Basically zero. The signal exists at conf75 but gets margined away by conf80's higher bar.

### 10.5 Fold 12 under confidence filtering — the regime question

Raw fold 12 accuracy is ~30%, but the backtest uses confidence-filtered predictions, not raw predictions. The question: does the model know it's uncertain on fold 12?

**Fold 12 vs fold 11 confidence distribution (sigmoid of logits):**

| | fold 11 p50 | fold 11 max | fold 12 p50 | fold 12 max |
|---|---|---|---|---|
| seed42 | 0.631 | 0.927 | 0.561 | 0.864 |
| seed43 | 0.627 | 0.895 | 0.529 | 0.858 |
| seed44 | 0.576 | 0.926 | **0.430** | 0.887 |

The model IS more uncertain on fold 12 — lower p50, lower max. Seed44's fold-12 p50 is 0.430 (mostly predicting short), which is directionally appropriate for the Jan-Mar 2026 drawdown driven by tariff-related geopolitical events. This supports the interpretation that fold 12 isn't "model collapse" but "novel macro regime the model hasn't seen."

**But: high-confidence precision on fold 12 splits catastrophically across seeds:**

| conf≥ | seed42 prec | seed43 prec | seed44 prec |
|---|---|---|---|
| 0.60 | 34.2% | 56.0% | 51.1% |
| 0.70 | 44.5% | **6.0%** | 49.2% |
| 0.75 | 47.9% | **0.0%** | 37.1% |
| 0.80 | **59.1%** | **0.0%** | **18.3%** |

- **seed42**: conf80 precision is **59.1%** on fold 12 — better than fold 11's 58.3%. The filter works as designed.
- **seed43**: 42 high-confidence longs, **ALL wrong** (0% precision at conf80). Confident and wrong.
- **seed44**: 60 high-confidence longs, **18.3% correct** (82% wrong). Confident and wrong.

**Interpretation:** the model reduces its confidence on fold 12 (fewer high-conf predictions overall), which is appropriate for a novel regime. But when it IS confident, the signal is unreliable — whether the confidence is correct or hallucinated depends on which training seed you got. The model can't distinguish "I see a pattern I've trained on" from "I'm extrapolating into a regime I haven't seen."

This is the exact problem **regime features** are designed to solve. If the model had VIX/DXY/yield-curve as inputs, it could learn to be explicitly uncertain when the macro state is novel. Currently it's flying blind — VP patterns may look superficially similar across regimes even when the underlying driver is entirely different.

### 10.6 What matched-gradient-steps proved

**Confirmed:**
- Undertraining was real: confidence ceiling rises from 0.77 to 0.94, p50 from ~0.40 to ~0.53, conf80 becomes reachable
- Threshold trade counts stabilize: conf75_pyr holdout 135/116/123 vs 51/0/6
- All 3 seeds positive at conf75_pyr holdout (first time any config achieves this)
- Non-pyramid conf75 shows a **reproducible tiny edge**: +0.7% mean CAGR, 60.7% WR, 1pp seed spread
- Fold 4 mode collapse resolves (model now makes predictions, though they're bad)

**New concerns:**
- Walk-forward acc drops ~1.4pp (mild overfitting)
- Fold 12 raw accuracy crashes to ~30% (regime sensitivity, not calibration artifact)
- Fold 12 high-confidence precision is seed-dependent (59%/0%/18% at conf80)
- Pyramid holdout CAGR spread is still 16pp at conf75_pyr — magnitude remains seed-dominated
- Full-period drawdown worsens significantly (−24 to −44% at conf75_pyr vs −10 to −14% on match-epochs)

**The binding constraint has shifted.** It's no longer calibration instability (solved) or undertraining (solved). It's **regime awareness** — the model produces confident predictions in novel macro environments without knowing they're novel.

### 10.7 Revised priority sequence

1. **Regime features** (promoted from #4 to #1). VIX/DXY/yield-curve/FFR give the model a chance to learn "this macro state is novel → reduce confidence." Directly attacks the fold 12 problem. See `MULTI_ASSET_PLAN.md §REFRAME`.
2. **Rolling retraining** (walk-forward retrain every 3 months). The fold 12 failure is partly staleness — training data ends months before holdout. Complementary to regime features, not a substitute.
3. **Envelope dynamics** (Δwindow_hi, Δwindow_lo, Δrange_pct). Still orthogonal, still cheap. Run after regime features.

Deployment recommendation: if deploying before regime features, use **non-pyramid conf75_pause24** (the +0.7% mean CAGR config). It's the only variant where the signal is reproducible, the drawdown is tiny (−1.6% mean), and pyramiding doesn't amplify regime-noise. Add a VIX-based circuit breaker (pause trading when VIX > N) as a manual regime filter until the model can do it natively.

### 10.8 Artifacts (matched-gradient-steps)

- `experiments/v11_11_tb_full_c60m_mgs_seed{42,43,44}_predictions.npz`
- `experiments/eval_v11_11_tb_full_c60m_mgs_seed{42,43,44}_results.json`
- `experiments/backtest_results_v11_tb_mgs_seed{42,43,44}.json`
- Code: commit `2b3d7c3 feat(v11): add --epochs / --patience / --swa-start for matched-gradient-steps`

---

## 8. Artifacts

**Code** (commit `e8e4681`):
- `src/models/eval_v11.py` — stride knob
- `src/models/run_backtest_v11_tb.py` — cadence-aware backtest

**1h cadence artifacts (NEW)**:
- `experiments/v11_11_tb_full_c60m_predictions.npz`
- `experiments/eval_v11_11_tb_full_c60m_results.json`
- `experiments/backtest_results_v11_tb_cadence_ablation.json`

**15m baseline (unchanged)**:
- `experiments/v11_11_tb_full_predictions.npz`
- `experiments/v11_11_tb_nopv_predictions.npz`
- `experiments/eval_v11_11_tb_full_results.json`
- `experiments/eval_v11_11_tb_nopv_results.json`
- `experiments/backtest_results_v11_tb.json`

**Walk-forward config**: fold boundaries `2020-01-01 → 2026-04-08`, 14-day wall-clock embargo, holdout from `2025-07-01`.
