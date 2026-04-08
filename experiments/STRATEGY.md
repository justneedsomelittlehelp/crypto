# Trading Strategy

How the user reads volume profile to make BTC/USD trading decisions. This is the ground truth that the model should learn to replicate.

---

## Core Strategy: VP Support/Resistance

1. **Aggregate** 30 days of 4h horizontal VP into one stacked profile
2. **Identify local maxima** — price levels with highest volume concentration. These are support/resistance nodes
3. **Two nearest nodes form a range** — the node above current price is the **ceiling**, the node below is the **floor**
4. **Mean-reversion within range** — price near ceiling → likely to go down. Price near floor → likely to go up
5. **Breakout = role reversal** — if price breaks through ceiling, old ceiling becomes new floor, next node above becomes new ceiling. Reverse for downward break

## VP Clarity

- VP is not always cleanly structured. Sometimes the histogram is smooth/normal-distributed with no clear local maxima
- When this happens, **shift the 30d window backward in time** (same window size, earlier start date) to find a period where VP had clearer structure
- If the same peaks appear across multiple shifted windows → **robust levels**. If peaks only exist in the most recent window → **weak/unreliable levels**

## VP Noise

- Raw VP histograms are noisy — small bin-to-bin fluctuations create many false local maxima
- The user mentally **smooths the histogram** to see the true underlying shape (2-3 real peaks, not 10+ noise peaks)
- Model needs to either learn this smoothing or have it done in preprocessing

## Candlestick Patterns + Volume Confirmation

The VP structure gives the range (ceiling/floor). When price is **between** the local maxima (not near either end), candlestick patterns provide the directional hint, and volume confirms reliability.

### Candle patterns
- **Red inverted hammer** (bearish): open > close, high significantly above close, low near open → likely next candle goes down
- **Green hammer** (bullish): close > open, high near close, low significantly below open → likely next candle goes up
- These are most useful when price is **mid-range** between VP ceiling and floor — at the extremes, the VP level itself dominates

### Volume as confidence
- Volume acts as a "review count" for the candle signal
- Same candle pattern with high volume = trustworthy signal
- Same candle pattern with low volume = weak/unreliable signal
- Analogy: 4.5 stars with 1,000 reviews vs 4.5 stars with 15 reviews

### Timeframes
- User looks at **4h and 1d candles** for candle pattern reading
- Currently only 4h data in model — 1d aggregation may be needed

---

## Strategy → Model Translation

| Strategy concept | Model requirement |
|---|---|
| 30d stacked VP | Sum VP bins across lookback window |
| Local maxima identification | Peak detection on aggregated VP |
| Ceiling/floor distance | Distance from mid-bin to nearest peak above/below |
| Peak robustness via shifted windows | Peak consistency score across time-shifted 30d windows |
| Mental smoothing of noisy VP | Gaussian filter on aggregated VP, or let CNN learn filters |
| VP as a spatial shape | 1D CNN (not RNN/LSTM which process temporally) |
| Candle patterns (hammer, inverted hammer) | Features: upper_wick, lower_wick, body_ratio — encode candle shape |
| Volume as confidence multiplier | volume_ratio already exists — combine with candle features |
| Mid-range context | Only trust candle signals when price is between ceiling/floor (not at extremes) |
| 1d candle aggregation | Aggregate 6 bars (1 day) of 4h candles into 1d OHLCV features |

---

## Iteration Log: Strategy-Driven Decisions

### Vanilla RNN (Evals 1-4)
- **No strategy alignment.** Fed raw 50-bin VP per bar as a time series. Model had no concept of aggregated VP, peaks, or support/resistance. Established that raw VP bins + sequential model = random performance.

### LSTM Eval 1-3: Architecture & loss tuning
- **Still no strategy alignment.** Switched to LSTM for long-range dependencies, tuned params and loss weighting. Weighted loss fixed majority-class collapse but model had no access to VP structure. Best val acc: 52.7%.

### LSTM Eval 4: Two-layer depth
- **Strategy link: none.** Tested if more model capacity helped. It didn't — overfitted val period without generalizing.

### LSTM Eval 5-6: VP structure features
- **Strategy link: ceiling/floor distance, peak count, peak strength.** First attempt to encode the user's actual strategy into features. Aggregated 30d VP, found peaks, computed distance to nearest ceiling/floor. Result: no improvement. The LSTM sees these features repeated per timestep — can't distinguish them from the raw VP noise.

### LSTM Eval 7: Gaussian smoothing + peak consistency
- **Strategy link: mental smoothing + shifted window robustness.** Added Gaussian filter (sigma=2) to mimic the user's mental smoothing of noisy VP. Added peak consistency across 3d/6d/9d shifted windows to encode peak robustness. Result: still no improvement. The right features, but the wrong model — LSTM processes sequentially, not spatially.

### 1D CNN Evals 1-4: Spatial VP processing
- **Strategy link: VP as a spatial shape.** CNN treats aggregated VP as histogram, conv filters detect peaks. First architecture to match user's approach. Train acc reached 55.7% (best yet) but val stuck at ~50%. The VP shape alone isn't enough — user also uses candle patterns + volume to decide direction when price is mid-range.

### CNN Evals 5-6: Candle + volume features
- **Strategy link: candlestick patterns as directional hint, volume as confidence.** Added upper_wick, lower_wick, body_dir, vp_mid_range. Train acc jumped to 59.3% — model can learn the candle patterns. But val stuck at ~53%, overfitting to training-period candle patterns.

### CNN Eval 7: First-hit labels
- **Strategy link: holding through drawdowns.** User doesn't exit after exactly 24h — holds until TP or SL is hit. Changed label from fixed-horizon to first-hit ±3%. Val jumped from 53% to 57.3%. The label now matches actual trading behavior.

### CNN Evals 11-15: TP/SL ratio tuning
- **Strategy link: user adjusted TP/SL based on volatility and intuition.** Added vol-scaling (rolling 30-bar std). Tested symmetric and asymmetric ratios. Key finding: **inverted ratio (tight TP=2.5%, wide SL=5%) hit 74.7% val acc**. This matches how the user traded — patient with drawdowns, taking profit on smaller moves. Wide TP / tight SL (traditional risk/reward) performed worst.

### CNN Eval 16: Walk-forward with 1:2 ratio
- **Strategy link: regime dependence.** Walk-forward showed 82% in bull markets, 27% in bear. The 1:2 ratio inherently favors long trades. User confirmed adjusting TP/SL based on market trend — tight TP in bull, tight SL in bear.

### CNN Eval 17: Regime-adaptive TP/SL (walk-forward)
- **Strategy link: user's intuitive regime adjustment.** SMA(90d) as bull/bear proxy. Bull: TP=2.5%, SL=5%. Bear: TP=5%, SL=2.5%. Walk-forward overall **61.2% accuracy, F1=0.690**. Eliminated catastrophic bear-market failures (Fold 5: 27% → 63%). Best folds hit 78-88%. Weakest folds (43-45%) occur during regime transitions where SMA lags.

### CNN Evals 18-22: Neutral zone filtering
- **Strategy link: user skips trading when VP has no structure (normal distribution shape).** Tried filtering bars with num_peaks=0 (symmetric TP/SL or skip). Didn't help — too few bars affected (3-7%) and regime signal dominates.

### CNN Eval 23: Gaussian filter tuning
- **Strategy link: matching user's visual smoothing.** Reduced sigma from 2.0 to 0.8, prominence from 0.15 to 0.05. Now 53% of bars have 2+ peaks (clear ceiling/floor) vs 9% before. Walk-forward: **61.5% — new best.**

### CNN Evals 24-25: L2 regularization
- No improvement. The generalization gap is from regime shifts, not weight magnitude.

### Transformer Evals 1-2: Self-attention across VP bins
- **Strategy link: finding related VP peaks.** Each bin attends to all other bins — can directly learn ceiling/floor pairs regardless of distance. Walk-forward: **61.3% acc, 67.2% precision** (best precision). Matches CNN despite being a first attempt. Data-limited at 4h resolution — 1h data (6x samples) could unlock more.
