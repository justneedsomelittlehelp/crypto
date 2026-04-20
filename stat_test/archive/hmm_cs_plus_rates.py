# STATUS: ARCHIVED (2026-04-20) — rate-change HMM superseded by hmm_hierarchical.py.
# load_long_history + build_commodity_features were migrated to hmm_regime.py.
# Kept for the --rate-mode level/change experiments; rejected (see HMM_LOG.md §8-9).
"""HMM with commodity CS ranks + monetary-block expanding ranks.

Commodity block (4 dims):
  per commodity: 14d log return → 252d rolling z → CS rank across 5
  → inv-normal. Drop last (rank-sum constraint).

Monetary block (3 dims):
  per rate series {ffr, dgs2, yield_curve}: expanding percentile rank using
  1990-onward history as baseline → inv-normal.

HMM input = 7 dims. Fit K=3 Gaussian HMM across 5 seeds, pick best-LL.

The monetary block's wider baseline (1990+) matters: ranking FFR=0.4% in 2016
against only 2016+ data is uninformative (tiny sample); against 1990+ data
it sits near the 20th percentile of a full rate cycle.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from hmm_regime import summarize, plot_regimes, load_btc_daily, COMMODITIES

HERE = Path(__file__).parent
DATA = HERE / "data"

DELTA = 14
ZBASE = 252
RATE_WARMUP = 252  # days of 1990+ baseline before rate rank is trusted
RATE_CHANGE_WINDOW = 126  # ~6 months for rate differences
SEEDS = [42, 0, 1, 2, 7]
RATE_COLS = ["ffr", "dgs2", "yield_curve"]


def load_long_history() -> pd.DataFrame:
    df = pd.read_csv(DATA / "regime_daily.csv", parse_dates=["date_utc"])
    return df.set_index("date_utc").sort_index()


def build_commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    """14d log ret → 252d rolling z → CS rank → inv-normal. Drop last col."""
    # forward-fill commodities within their valid range only
    c = df[COMMODITIES].copy()
    c = c.ffill()
    z = pd.DataFrame(index=c.index)
    for col in COMMODITIES:
        r = np.log(c[col] / c[col].shift(DELTA))
        mu = r.rolling(ZBASE, min_periods=ZBASE).mean()
        sd = r.rolling(ZBASE, min_periods=ZBASE).std(ddof=0)
        z[f"{col}_z"] = (r - mu) / sd
    z = z.dropna()
    # cross-sectional rank
    ranks = z.rank(axis=1, method="average")
    pct = ranks / (len(COMMODITIES) + 1)
    cs = pd.DataFrame(norm.ppf(pct.values), index=z.index, columns=z.columns)
    return cs.iloc[:, :-1]  # drop soyb (implied)


def expanding_rank_to_normal(s: pd.Series, warmup: int = RATE_WARMUP) -> pd.Series:
    """Expanding percentile rank → inv-normal. Uses all history up to t."""
    vals = s.values
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(warmup, n):
        window = vals[: i + 1]
        rank = np.sum(window <= vals[i]) / (i + 1)
        rank = min(max(rank, 1.0 / (i + 2)), 1.0 - 1.0 / (i + 2))
        out[i] = norm.ppf(rank)
    return pd.Series(out, index=s.index, name=s.name)


def build_rate_features(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Expanding rank of each rate series using 1990+ baseline.

    mode='level': rank the raw level.
    mode='change': rank the 6m absolute change (t - t-126).
    """
    r = df[RATE_COLS].ffill().copy()
    out = pd.DataFrame(index=r.index)
    for col in RATE_COLS:
        if mode == "level":
            series = r[col]
        elif mode == "change":
            series = r[col] - r[col].shift(RATE_CHANGE_WINDOW)
        else:
            raise ValueError(mode)
        out[f"{col}_{mode[0]}r"] = expanding_rank_to_normal(series)
    return out.dropna()


def fit(X, seed, k):
    return GaussianHMM(n_components=k, covariance_type="full",
                       n_iter=500, tol=1e-4, random_state=seed).fit(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rate-mode", choices=["level", "change"], default="level")
    args = p.parse_args()
    K = args.k
    base = "cs_plus_rates" if args.rate_mode == "level" else "cs_plus_rate_changes"
    OUT = HERE / "results" / (base if K == 3 else f"{base}_k{K}")
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_long_history()
    print(f"Loaded {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")

    com = build_commodity_features(df)
    rat = build_rate_features(df, args.rate_mode)
    print(f"  commodity block: {com.shape}, starts {com.index[0].date()}")
    print(f"  rate block:      {rat.shape}, starts {rat.index[0].date()}")

    feats = com.join(rat, how="inner").dropna()
    print(f"  joined HMM input: {feats.shape}, "
          f"{feats.index[0].date()} → {feats.index[-1].date()}")
    X = feats.values

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

    summary = summarize(best_m, feats, best_labels, feats)
    print(summary)
    (OUT / "summary.txt").write_text(summary)

    posts = best_m.predict_proba(X)
    timeline = feats.copy()
    timeline["regime"] = best_labels
    for k in range(K):
        timeline[f"p_r{k}"] = posts[:, k]
    timeline.to_csv(OUT / "timeline.csv")

    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv",
                         feats.index[0], feats.index[-1])
    plot_regimes(best_labels, feats.index, btc, K, OUT / "plot.png")
    print(f"Wrote {OUT}/")


if __name__ == "__main__":
    main()
