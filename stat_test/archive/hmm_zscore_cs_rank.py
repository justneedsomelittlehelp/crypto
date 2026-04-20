# STATUS: ARCHIVED (2026-04-20) — z-score-then-CS-rank commodity-only HMM.
# Cleaner than raw CS rank but superseded by the hierarchical model which
# uses this same pipeline as its inner layer. See HMM_LOG.md §5.
"""HMM K=3 with vol-normalized cross-sectional ranking.

Feature pipeline (per commodity):
  1. ret14(t) = log(P_t / P_{t-14})
  2. z(t) = (ret14(t) - mean_252d(ret14)) / std_252d(ret14)     # vol-normalized
  3. cross-sectional rank of z across 5 commodities per day
  4. inverse-normal of rank → Gaussian-shaped HMM input
  5. drop last column (rank-sum constraint → 4 DoF)

Fits K=3 Gaussian HMM across 5 seeds, picks best-LL, outputs timeline + plot.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from hmm_regime import (
    load_regime_daily, summarize, plot_regimes, load_btc_daily, COMMODITIES,
)

HERE = Path(__file__).parent
DATA = HERE / "data"

DELTA = 14
ZBASE = 252
SEEDS = [42, 0, 1, 2, 7]


def build_z_features(df: pd.DataFrame) -> pd.DataFrame:
    """14d log return → 252d rolling z-score, per commodity."""
    out = pd.DataFrame(index=df.index)
    for c in COMMODITIES:
        r = np.log(df[c] / df[c].shift(DELTA))
        mu = r.rolling(ZBASE, min_periods=ZBASE).mean()
        sd = r.rolling(ZBASE, min_periods=ZBASE).std(ddof=0)
        out[f"{c}_z"] = (r - mu) / sd
    return out.dropna()


def cs_rank_to_normal(f: pd.DataFrame) -> pd.DataFrame:
    n = len(f.columns)
    ranks = f.rank(axis=1, method="average")
    pct = ranks / (n + 1)
    return pd.DataFrame(norm.ppf(pct.values), index=f.index, columns=f.columns)


def fit(X, seed, k):
    return GaussianHMM(n_components=k, covariance_type="full",
                       n_iter=500, tol=1e-4, random_state=seed).fit(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    K = args.k
    OUT = HERE / "results" / (f"zscore_cs" if K == 3 else f"zscore_cs_k{K}")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading + building features (K={K})...")
    raw = load_regime_daily()
    z = build_z_features(raw)
    print(f"  z-feature rows: {len(z)}, range: {z.index[0].date()} → {z.index[-1].date()}")

    feats_full = cs_rank_to_normal(z)
    feats = feats_full.iloc[:, :-1]  # drop last (rank-sum constraint)
    X = feats.values
    print(f"  HMM dims: {X.shape[1]} (dropped {feats_full.columns[-1]})")

    # Multi-seed
    results = []
    for s in SEEDS:
        m = fit(X, s, K)
        ll = m.score(X)
        labels = m.predict(X)
        freq = np.array([(labels == k).sum() for k in range(K)])
        results.append((s, ll, m, labels, freq))
        print(f"  seed {s}: LL = {ll:,.2f}, freq % = "
              f"{np.round(100*freq/freq.sum(), 1).tolist()}")

    best_seed, best_ll, best_m, best_labels, _ = max(results, key=lambda r: r[1])
    print(f"\nBest: seed {best_seed}, LL = {best_ll:,.2f}")

    summary = summarize(best_m, feats, best_labels, z)
    print(summary)
    (OUT / "summary.txt").write_text(summary)

    posts = best_m.predict_proba(X)
    timeline = z.loc[feats.index].copy()
    timeline["regime"] = best_labels
    for k in range(K):
        timeline[f"p_r{k}"] = posts[:, k]
    timeline.to_csv(OUT / "timeline.csv")

    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv",
                         feats.index[0], feats.index[-1])
    plot_regimes(best_labels, feats.index, btc, K, OUT / "plot.png")
    print("Wrote timeline, summary, plot")


if __name__ == "__main__":
    main()
