# EVAL — Prediction Cadence Ablation (v11-full-tb, 15m vs 1h)

> **Status**: 2026-04-15 — first 1h cadence run complete (single seed, match-epochs). Produces the **first positive real-engine holdout CAGR in project history**. Not yet replicated across seeds. DO NOT deploy on this evidence alone.
>
> **Read this when**: continuing the cadence experiment, deciding whether to run the 3-seed or matched-gradient-steps follow-up, or debugging the undertraining hypothesis.
>
> **Related**: `LABEL_REDESIGN.md` (the v11-full-tb baseline), `EVAL_TRANSFORMER.md §Cadence`, `MODEL_HISTORY.md §32`.

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

### 4.3 Key result — **first positive real-engine holdout CAGR in project history**

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
