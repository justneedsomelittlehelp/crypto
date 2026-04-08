# Vanilla RNN Evaluations

**Why:** Simplest recurrent baseline to establish a performance floor.
**Strategy alignment:** None — raw VP bins fed as time series, no aggregation, no peak detection. See [STRATEGY.md](STRATEGY.md).

### Eval 1 — run_1775545695

**Layers:**
- RNN(54→64, tanh) → Dropout(0.2)
- RNN(64→32, tanh) → Dropout(0.2)
- RNN(32→16, tanh) → Dropout(0.2)
- Linear(16→1), BCEWithLogitsLoss

**Hyperparameters:** Adam, lr=1e-3, batch=64, patience=10, lookback=42 bars, label=6 bars

**Results:** 14 epochs (early stop), val loss=0.6935, val acc=48.5%, train acc=54.2%

**Notes:** Near random. Train loss moved slightly, val loss flat from epoch 1. Funnel (64→32→16) may be too aggressive.

---

### Eval 2 — run_1775546043

**Layers:** Same as Eval 1

**Hyperparameters:** Same as Eval 1

**Results:** 14 epochs (early stop), val loss=0.6921, val acc=50.9%, train acc=53.3%

**Notes:** Reproducibility check. ~2% val acc difference is noise. Confirms vanilla RNN with these params learns no meaningful signal.

---

### Eval 3 — run_1775610561

**Layers:** Same as Eval 1

**Hyperparameters:** Adam, lr=1e-4, batch=64, patience=10, lookback=42 bars, label=6 bars

**Results:** 11 epochs (early stop), val loss=0.6930, val acc=51.7%, train acc=53.4%

**Notes:** Lower lr didn't help. Val loss nearly identical to Eval 1-2. Model still near random — the issue is not lr, it's the architecture's inability to capture the signal.

---

### Eval 4 — run_1775610681

**Layers:** Same as Eval 1

**Hyperparameters:** Adam, lr=1e-4, batch=64, patience=10, lookback=180 bars (30d), label=6 bars

**Results:** 11 epochs (early stop), val loss=0.6939, val acc=50.9%, train acc=54.2%

**Notes:** Longer lookback made no difference. Confusion matrix shows model predicts UP for 100% of samples — just learning majority class. Vanilla RNN can't propagate signal through 180 timesteps.

---

### Summary

| Eval | LR   | Lookback | Val Loss | Val Acc | Train Acc | Change |
|------|------|----------|----------|---------|-----------|--------|
| 1    | 1e-3 | 42 (1w)  | 0.6935   | 48.5%   | 54.2%     | Baseline |
| 2    | 1e-3 | 42 (1w)  | 0.6921   | 50.9%   | 53.3%     | Reproducibility |
| 3    | 1e-4 | 42 (1w)  | 0.6930   | 51.7%   | 53.4%     | LR sweep — no effect |
| 4    | 1e-4 | 180 (30d)| 0.6939   | 50.9%   | 54.2%     | Longer lookback — no effect |

**Conclusion:** Vanilla RNN learns no meaningful signal across lr and lookback variations. Predicts majority class only. Architecture change needed — LSTM or Transformer to handle longer sequences.
