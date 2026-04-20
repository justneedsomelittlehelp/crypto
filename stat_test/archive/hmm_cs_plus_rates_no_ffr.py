# STATUS: ARCHIVED (2026-04-20) — dropping FFR did not fix long dwell (DGS2
# still too persistent). Superseded by hmm_hierarchical.py. See HMM_LOG.md §11.
"""Same as hmm_cs_plus_rates --rate-mode change, but DROP FFR.

FFR is a monthly staircase → ρ₂₁ ≈ 0.96, which dominates HMM dwell.
Keep only market-traded rate-change features: DGS2, yield-curve slope.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

import hmm_cs_plus_rates as base
from hmm_cs_plus_rates import (
    load_long_history, build_commodity_features, expanding_rank_to_normal,
    RATE_CHANGE_WINDOW,
)
from hmm_regime import summarize, plot_regimes, load_btc_daily

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "cs_plus_rate_changes_no_ffr"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
SEEDS = [42, 0, 1, 2, 7]
RATE_COLS = ["dgs2", "yield_curve"]


def build_rate_features_no_ffr(df: pd.DataFrame) -> pd.DataFrame:
    r = df[RATE_COLS].ffill().copy()
    out = pd.DataFrame(index=r.index)
    for col in RATE_COLS:
        series = r[col] - r[col].shift(RATE_CHANGE_WINDOW)
        out[f"{col}_cr"] = expanding_rank_to_normal(series)
    return out.dropna()


def main():
    df = load_long_history()
    com = build_commodity_features(df)
    rat = build_rate_features_no_ffr(df)
    feats = com.join(rat, how="inner").dropna()
    X = feats.values
    print(f"HMM input: {feats.shape}, "
          f"{feats.index[0].date()} → {feats.index[-1].date()}")
    print(f"  features: {list(feats.columns)}")

    results = []
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=s).fit(X)
        ll = m.score(X)
        labels = m.predict(X)
        freq = np.array([(labels == k).sum() for k in range(K)])
        results.append((s, ll, m, labels, freq))
        print(f"  seed {s}: LL = {ll:,.2f}, freq % = "
              f"{np.round(100*freq/freq.sum(), 1).tolist()}")

    best_seed, best_ll, best_m, best_labels, _ = max(results, key=lambda r: r[1])
    print(f"\nBest: seed {best_seed}, LL = {best_ll:,.2f}")

    # Mean dwell per regime
    lengths = {k: [] for k in range(K)}
    cur_k, cur_n = best_labels[0], 1
    for lab in best_labels[1:]:
        if lab == cur_k:
            cur_n += 1
        else:
            lengths[cur_k].append(cur_n)
            cur_k, cur_n = lab, 1
    lengths[cur_k].append(cur_n)
    print("\nMean dwell (days):")
    for k in range(K):
        print(f"  r{k}: {np.mean(lengths[k]):.1f}  (n_runs={len(lengths[k])})")

    summary = summarize(best_m, feats, best_labels, feats)
    (OUT / "summary.txt").write_text(summary)
    print(summary)

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
