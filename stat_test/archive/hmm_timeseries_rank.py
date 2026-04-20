# STATUS: ARCHIVED (2026-04-20) — time-series rank alternative to CS rank.
# Rejected: failed to separate bull vs bear (both landed in r0). See HMM_LOG.md §4.
"""HMM K=3 with TIME-SERIES ranking (expanding window, per commodity).

Contrast to cross-sectional ranking in hmm_regime.py.

For each commodity independently:
  raw 20d log return → expanding percentile rank (min 252d warmup)
                     → inverse-normal (probit) → std normal

Then fit K=3 Gaussian HMM under 5 seeds, pick best-LL, write timeline + plot.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from hmm_regime import (
    load_regime_daily, build_features, summarize, plot_regimes,
    load_btc_daily,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "ts_rank"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
WARMUP = 252
SEEDS = [42, 0, 1, 2, 7]


def expanding_rank_to_normal(s: pd.Series, warmup: int = WARMUP) -> pd.Series:
    n = len(s)
    out = np.full(n, np.nan)
    vals = s.values
    for i in range(warmup, n):
        window = vals[: i + 1]
        rank = np.sum(window <= vals[i]) / (i + 1)
        rank = min(max(rank, 1.0 / (i + 2)), 1.0 - 1.0 / (i + 2))
        out[i] = norm.ppf(rank)
    return pd.Series(out, index=s.index, name=s.name)


def fit(X, seed):
    m = GaussianHMM(n_components=K, covariance_type="full",
                    n_iter=500, tol=1e-4, random_state=seed)
    m.fit(X)
    return m


def main():
    raw = load_regime_daily()
    feats_raw = build_features(raw)
    print(f"raw features: {list(feats_raw.columns)}, rows: {len(feats_raw)}")

    feats = pd.concat(
        {c: expanding_rank_to_normal(feats_raw[c]) for c in feats_raw.columns},
        axis=1,
    ).dropna()
    X = feats.values
    print(f"post-warmup rows: {len(feats)}, dims: {X.shape[1]}")

    # Multi-seed fit
    results = []
    for s in SEEDS:
        m = fit(X, s)
        ll = m.score(X)
        labels = m.predict(X)
        freq = np.array([(labels == k).sum() for k in range(K)])
        results.append((s, ll, m, labels, freq))
        print(f"  seed {s}: LL = {ll:,.2f}, freq = {100*freq/freq.sum()}")

    # Pick best-LL
    best_seed, best_ll, best_m, best_labels, best_freq = max(results, key=lambda r: r[1])
    print(f"\nBest: seed {best_seed}, LL = {best_ll:,.2f}")

    summary = summarize(best_m, feats, best_labels, feats_raw)
    print(summary)
    (OUT / "summary.txt").write_text(summary)

    posts = best_m.predict_proba(X)
    timeline = feats_raw.loc[feats.index].copy()
    timeline["regime"] = best_labels
    for k in range(K):
        timeline[f"p_r{k}"] = posts[:, k]
    timeline.to_csv(OUT / "timeline.csv")

    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv",
                         feats.index[0], feats.index[-1])
    plot_regimes(best_labels, feats.index, btc, K, OUT / "plot.png")
    print(f"Wrote timeline, summary, plot (best seed = {best_seed})")


if __name__ == "__main__":
    main()
