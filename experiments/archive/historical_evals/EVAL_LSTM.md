# LSTM Evaluations

**Why:** Gated architecture to address vanishing gradients. Cell state highway allows learning long-range dependencies across 180-bar sequences where vanilla RNN failed.
**Strategy alignment:** Evals 1-4 none. Evals 5-7 added VP structure features (ceiling/floor, smoothing, consistency) but LSTM processes temporally, not spatially. See [STRATEGY.md](STRATEGY.md).

### Eval 1 — run_1775610927

**Layers:**
- LSTM(54→64) → Dropout(0.2)
- LSTM(64→32) → Dropout(0.2)
- LSTM(32→16) → Dropout(0.2)
- Linear(16→1), BCEWithLogitsLoss

**Hyperparameters:** Adam, lr=1e-4, batch=64, patience=10, lookback=180 bars (30d), label=6 bars

**Results:** 11 epochs (early stop), val loss=0.6931, val acc=50.9%, train acc=53.7%, params=46,481

**Notes:** Same majority-class collapse as vanilla RNN (predicts all UP). 46k params with 14k samples is over-parameterized. Architecture alone didn't fix the issue — the problem may be upstream (features, label, or class balance).

---

### Eval 2 — run_1775611349

**Layers:**
- LSTM(54→8) → no dropout
- Linear(8→1), BCEWithLogitsLoss

**Hyperparameters:** Adam, lr=1e-4, batch=64, patience=10, lookback=180 bars (30d), label=6 bars

**Results:** 15 epochs (early stop), val loss=0.6938, val acc=50.9%, train acc=53.0%, params=2,057

**Notes:** Right-sized params (~2k vs 14k samples). Still majority-class collapse. Train acc stuck at 53% — model can't even learn the training set. Problem is not architecture or param count.

---

### Eval 3 — run_1775611515

**Layers:** Same as Eval 2

**Hyperparameters:** Same as Eval 2, + **weighted BCEWithLogitsLoss** (pos_weight=0.89)

**Results:** 17 epochs (early stop), val loss=0.6528, val acc=52.7%, train acc=53.1%, params=2,057

**Notes:** No longer majority-class collapse — predicts both UP and DOWN. Test precision=54.0%, recall=47.5%. Weighted loss fixed the collapse. Signal is still weak but the model is now actually learning.

---

### Summary

| Eval | Hidden | Dropout | Params | Weighted | Val Loss | Val Acc | Notes |
|------|--------|---------|--------|----------|----------|---------|-------|
| 1    | 64/32/16 | 0.2   | 46,481 | No       | 0.6931   | 50.9%   | Over-parameterized, majority-class |
| 2    | 8      | 0.0     | 2,057  | No       | 0.6938   | 50.9%   | Right-sized, still majority-class |
| 3    | 8      | 0.0     | 2,057  | Yes      | 0.6528   | 52.7%   | Weighted loss fixed collapse |

### Eval 4 — run_1775611621

**Layers:**
- LSTM(54→16) → no dropout
- LSTM(16→8) → no dropout
- Linear(8→1), weighted BCEWithLogitsLoss

**Hyperparameters:** Adam, lr=1e-4, batch=64, patience=10, lookback=180 bars (30d), label=6 bars

**Results:** 11 epochs (early stop), val loss=0.6519, val acc=53.5%, train acc=53.3%, params=5,449

**Notes:** Best val loss yet (0.6519) and val acc (53.5% at epoch 6). But test acc dropped to 50% with heavy DOWN bias (predicts DOWN 79% of the time). Model may be overfitting the val period's class distribution. More params (5.4k) didn't help generalization.

---

### Summary

| Eval | Hidden | Dropout | Params | Weighted | Val Loss | Val Acc | Notes |
|------|--------|---------|--------|----------|----------|---------|-------|
| 1    | 64/32/16 | 0.2   | 46,481 | No       | 0.6931   | 50.9%   | Over-parameterized, majority-class |
| 2    | 8      | 0.0     | 2,057  | No       | 0.6938   | 50.9%   | Right-sized, still majority-class |
| 3    | 8      | 0.0     | 2,057  | Yes      | 0.6528   | 52.7%   | Weighted loss fixed collapse |
| 4    | 16/8   | 0.0     | 5,449  | Yes      | 0.6519   | 53.5%   | Best val loss, but test=50% (DOWN bias) |

### Eval 5 — run_1775612667

**Layers:** Same as Eval 4 (LSTM 16/8), input=59 features (+ 5 VP structure features, buggy scale)

**Hyperparameters:** Same as Eval 4

**Results:** 14 epochs (early stop), val loss=0.6543, val acc=50.4%, params=5,769

**Notes:** VP structure features added but ceiling/floor dist had scale bug (0-25 instead of 0-1). Worse than without the features.

---

### Eval 6 — run_1775612780

**Layers:** Same as Eval 5 (LSTM 16/8), input=59 features (VP structure scale fixed)

**Hyperparameters:** Same as Eval 4

**Results:** 11 epochs (early stop), val loss=0.6527, val acc=52.7%, params=5,769

**Notes:** Scale fix helped slightly. Val loss best yet (0.6527) but test still near majority-class (94.6% recall = predicting mostly UP). VP structure features not being utilized effectively by the sequential LSTM.

---

### Summary

| Eval | Hidden | Features | Params | Weighted | Val Loss | Val Acc | Notes |
|------|--------|----------|--------|----------|----------|---------|-------|
| 1    | 64/32/16 | 54     | 46,481 | No       | 0.6931   | 50.9%   | Over-parameterized, majority-class |
| 2    | 8      | 54       | 2,057  | No       | 0.6938   | 50.9%   | Right-sized, still majority-class |
| 3    | 8      | 54       | 2,057  | Yes      | 0.6528   | 52.7%   | Weighted loss fixed collapse |
| 4    | 16/8   | 54       | 5,449  | Yes      | 0.6519   | 53.5%   | Best val loss, test=50% (DOWN bias) |
| 5    | 16/8   | 59       | 5,769  | Yes      | 0.6543   | 50.4%   | VP features (buggy scale) |
| 6    | 16/8   | 59       | 5,769  | Yes      | 0.6527   | 52.7%   | VP features (fixed), still weak |

### Eval 7 — run_1775613483

**Layers:**
- LSTM(61→8) → no dropout
- Linear(8→1), weighted BCEWithLogitsLoss

**Hyperparameters:** Adam, lr=1e-4, batch=64, patience=10, lookback=180 bars (30d), label=6 bars

**Features:** 61 total — 50 VP bins + 4 derived + 7 VP structure (Gaussian-smoothed peaks, ceiling/floor consistency across shifted 30d windows)

**Results:** 16 epochs (early stop), val loss=0.6555, val acc=50.0%, test acc=47.0%, params=2,281

**Notes:** Gaussian smoothing + peak consistency features added. Still no improvement. The LSTM processes features sequentially per timestep — VP structure features (which summarize the whole window) get diluted across 180 steps. An LSTM is the wrong tool for spatial pattern recognition on VP profiles.

---

### Summary

| Eval | Hidden | Features | Params | Weighted | Val Loss | Val Acc | Notes |
|------|--------|----------|--------|----------|----------|---------|-------|
| 1    | 64/32/16 | 54     | 46,481 | No       | 0.6931   | 50.9%   | Over-parameterized, majority-class |
| 2    | 8      | 54       | 2,057  | No       | 0.6938   | 50.9%   | Right-sized, still majority-class |
| 3    | 8      | 54       | 2,057  | Yes      | 0.6528   | 52.7%   | Weighted loss fixed collapse |
| 4    | 16/8   | 54       | 5,449  | Yes      | 0.6519   | 53.5%   | Best val loss, test=50% (DOWN bias) |
| 5    | 16/8   | 59       | 5,769  | Yes      | 0.6543   | 50.4%   | VP structure features (buggy scale) |
| 6    | 16/8   | 59       | 5,769  | Yes      | 0.6527   | 52.7%   | VP structure features (fixed) |
| 7    | 8      | 61       | 2,281  | Yes      | 0.6555   | 50.0%   | + Gaussian smoothing + consistency |

**Conclusion:** LSTM cannot leverage VP structure features. The sequential architecture is wrong for spatial VP pattern recognition. Moving to 1D CNN next.
