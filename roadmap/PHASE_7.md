# Phase 7: Optimization & Hardening

## Goal
Improve reliability, add model retraining, and harden the system for long-term unattended operation.

## What This Phase Builds On
Phase 6 — bot running live with monitoring.

## Implementation Steps

1. **Model retraining pipeline**
   - Schedule periodic retraining (e.g., monthly) on expanded data
   - Walk-forward approach: retrain on all data up to now, validate on recent window
   - Auto-compare new model vs current model on validation set
   - Only deploy new model if it outperforms (avoid regression)
   - Keep model version history

2. **Error recovery**
   - Handle Kraken API outages gracefully (retry with backoff, skip candle if needed)
   - Handle data feed gaps (interpolate or skip, log the gap)
   - Watchdog: if bot hasn't processed a candle in 2x the expected interval, alert and attempt restart

3. **Circuit breakers**
   - Auto-pause trading if:
     - Daily loss exceeds threshold
     - Consecutive losing trades exceed limit
     - Model predictions become erratic (high entropy / low confidence)
     - Market volatility spikes beyond historical norms
   - Require manual intervention to resume after circuit breaker trips

4. **Performance analysis tools**
   - Monthly performance reports (automated)
   - Attribution: which signals drove profits/losses
   - Feature importance analysis: are VP features still predictive?
   - Regime detection: is the model performing differently in trending vs ranging markets?

5. **Operational hardening**
   - Database for trade log instead of flat files (SQLite is sufficient)
   - Backup strategy: daily backup of trade log and model weights
   - Security audit: ensure API keys are properly scoped and rotated periodically

## Test When Done
- [ ] Retraining pipeline produces and validates a new model without manual intervention
- [ ] Circuit breakers trigger correctly under simulated adverse conditions
- [ ] Bot recovers gracefully from simulated API outage
- [ ] Monthly report generates automatically
- [ ] Trade history is backed up and restorable
