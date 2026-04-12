# Run Index

Maps each `run_*` folder to the eval that produced it. Walk-forward evals generate 10 runs (one per fold).

## Vanilla RNN (EVAL_VANILLA_RNN.md)

| Run | Eval | Notes |
|-----|------|-------|
| run_1775545695 | Eval 1 | Baseline, lr=1e-3, lookback=42 |
| run_1775546043 | Eval 2 | Reproducibility check |
| run_1775610561 | Eval 3 | lr=1e-4 |
| run_1775610681 | Eval 4 | lookback=180 |

## LSTM (EVAL_LSTM.md)

| Run | Eval | Notes |
|-----|------|-------|
| run_1775610927 | Eval 1 | 64/32/16, no weighted loss |
| run_1775611349 | Eval 2 | hidden=[8], no weighted loss |
| run_1775611515 | Eval 3 | hidden=[8], weighted loss |
| run_1775611621 | Eval 4 | hidden=[16,8], weighted loss |
| run_1775612667 | Eval 5 | +VP features (buggy scale) |
| run_1775612780 | Eval 6 | VP features fixed |
| run_1775613483 | Eval 7 | +Gaussian smoothing + consistency |

## CNN — Single evals (EVAL_CNN.md)

| Run | Eval | Phase | Notes |
|-----|------|-------|-------|
| run_1775614321 | Eval 1 | P1 | 8/16 avg pool, lr=1e-4 |
| run_1775614348 | Eval 2 | P1 | 8/16 avg pool, lr=1e-3 |
| run_1775614379 | Eval 3 | P1 | 16/32 avg pool |
| run_1775614449 | Eval 4 | P1 | 4/4 flatten |
| run_1775614882 | Eval 5 | P1 | +candle features |
| run_1775614924 | Eval 6 | P1 | dropout=0.3 |
| run_1775615624 | Eval 7 | P2 | **First-hit ±3%** |
| run_1775615674 | Eval 8 | P2 | Avg pool, best val loss |
| run_1775615703 | Eval 9 | P2 | patience=15 |
| run_1775616543 | Eval 11 | P2 | Vol-scaled ±3% |
| run_1775616987 | Eval 12 | P3 | TP=3% SL=2% (1.5:1) |
| run_1775617025 | Eval 13 | P3 | TP=4% SL=2% (2:1) |
| run_1775617152 | Eval 14 | P3 | **TP=2.5% SL=5% (1:2)** |
| run_1775617201 | Eval 15 | P3 | TP=3.3% SL=5% (1:1.5) |

## CNN — Walk-forward evals (10 folds each)

### Eval 10 — Symmetric ±3%, fixed (P4)
run_1775615934, run_1775615940, run_1775615948, run_1775615960, run_1775615981, run_1775615995, run_1775616007, run_1775616017, run_1775616049, run_1775616061

### Eval 16 — 1:2 fixed, vol-scaled (P4)
run_1775617546, run_1775617551, run_1775617558, run_1775617573, run_1775617589, run_1775617602, run_1775617625, run_1775617635, run_1775617668, run_1775617679

### Eval 17 — **Regime-adaptive (BEST MODEL)** (P5)
run_1775618102, run_1775618108, run_1775618118, run_1775618125, run_1775618132, run_1775618157, run_1775618168, run_1775618177, run_1775618202, run_1775618214

### Eval 18 — Neutral=symmetric, old sigma (P6)
run_1775619316, run_1775619323, run_1775619329, run_1775619339, run_1775619347, run_1775619371, run_1775619381, run_1775619390, run_1775619422, run_1775619435

### Eval 19 — Neutral=skip, old sigma (P6)
run_1775619480, run_1775619484, run_1775619489, run_1775619496, run_1775619501, run_1775619511, run_1775619520, run_1775619527, run_1775619549, run_1775619556

### Eval 20 — New sigma, no filter (P7)
run_1775620009, run_1775620014, run_1775620020, run_1775620028, run_1775620035, run_1775620057, run_1775620067, run_1775620076, run_1775620108, run_1775620120

### Eval 21 — New sigma, skip neutral (P7)
run_1775620171, run_1775620176, run_1775620181, run_1775620189, run_1775620210, run_1775620224, run_1775620233, run_1775620241, run_1775620271, run_1775620283

### Eval 22 — New sigma, symmetric neutral (P7)
run_1775620318, run_1775620323, run_1775620330, run_1775620341, run_1775620353, run_1775620368, run_1775620383, run_1775620392, run_1775620423, run_1775620438

### Eval 23 — sigma=0.8/prom=0.05, no filter **(BEST CNN)** (P8)
run_1775622368, run_1775622373, run_1775622379, run_1775622392, run_1775622399, run_1775622424, run_1775622436, run_1775622447, run_1775622480, run_1775622492

### Eval 24 — + L2 weight_decay=1e-3 (P8)
run_1775623136, run_1775623145, run_1775623154, run_1775623166, run_1775623177, run_1775623199, run_1775623213, run_1775623226, run_1775623264, run_1775623281

### Eval 25 — + L2 weight_decay=1e-4 (P8)
run_1775623819, run_1775623827, run_1775623856, run_1775623878, run_1775623890, run_1775623919, run_1775623939, run_1775623955, run_1775623990, run_1775624009

### CNN Eval 26 — 8ch, k=7, FC=32 (P8)
run_1775624055, run_1775624066, run_1775624098, run_1775624112, run_1775624128, run_1775624160, run_1775624180, run_1775624200, run_1775624269, run_1775624292

## Transformer (EVAL_TRANSFORMER.md)

### Eval 1 — embed=8, heads=2 (walk-forward)
run_1775624339, run_1775624344, run_1775624349, run_1775624355, run_1775624360, run_1775624371, run_1775624380, run_1775624389, run_1775624419, run_1775624430

### Eval 2 — embed=16, heads=2 **(BEST PRECISION)** (walk-forward)
run_1775624483, run_1775624490, run_1775624499, run_1775624511, run_1775624521, run_1775624548, run_1775624563, run_1775624578, run_1775624613, run_1775624631

### Eval 3 — 1h data, embed=16, heads=2 (walk-forward)
*(runs not individually tracked — Colab A100)*

### Eval 4 — 1h data, embed=32, heads=4, 2 layers **(BEST OVERALL 63.3%)** (walk-forward)
*(runs not individually tracked — Colab A100, TP=2.5%/SL=5%)*

### Eval 5 — v7 simple 2+1 (walk-forward, TP=7.5%/SL=3%)
*(eval_2plus1.py — Colab A100, batch=512, bf16. Results: experiments/eval_2plus1_results.json)*

### Eval 6 — v8 enriched 2+1 (walk-forward, TP=7.5%/SL=3%)
*(eval_2plus1.py — Colab A100, batch=512, bf16. Results: experiments/eval_2plus1_results.json)*

### Eval 7 — v6 enriched 1+1 on 15min (walk-forward, TP=7.5%/SL=3%)
*(eval_15min.py — Colab A100, batch=256, bf16. Results: experiments/eval_15min_results.json)*

### Eval 8 — v8 enriched 2+1 on 15min **(BEST ACCURACY 59.6%)** (walk-forward, TP=7.5%/SL=3%)
*(eval_15min.py — Colab A100, batch=256, bf16. Results: experiments/eval_15min_results.json)*

### Eval 9 — v6 baseline + funding rate fine-tune (walk-forward, TP=7.5%/SL=3%)
*(eval_finetune_funding.py — Colab T4. Mixed results, fold 6 collapse. Abandoned. Results: experiments/eval_finetune_funding_results.json)*

### Eval 10 — v6-prime with VP-derived TP/SL + regularization overhaul (walk-forward)
*(eval_v6_prime.py — Colab A100. 74.5% acc but +0.48% real EV. Per-fold variance too high.)*

### Eval 11 — v6-prime + filter analysis (asymmetry filter, single seed, superseded)
*(eval_v6_prime.py with filter analysis — post-hoc selection by tp_pct/sl_pct > 2.0 → +3.49% EV per trade on 652 trades.)*

### Eval 12 — v6-prime + 3 seeds + SWA + combined filter **⭐⭐ NEW BEST**
*(eval_v6_prime.py with N_SEEDS=3, SWA_START=15. Combined filter (conf>0.65 + asym>1.5) → **+3.98% EV/trade, 78.4% precision, 435 trades, Sharpe 0.97**. Fold 1 collapse fixed by ensembling. Results: experiments/eval_v6_prime_results.json)*
