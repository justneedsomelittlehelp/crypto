# STATUS: ARCHIVED (2026-04-20) — seed inspection. Superseded by
# hmm_seed_stability.py. See HMM_LOG.md §2.
"""Fit K=3 HMM with seed=1 (highest LL from stability test) and produce
timeline CSV + summary + plot. Compare against seed=42 run."""

from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM

from hmm_regime import (
    load_regime_daily, build_features, cross_sectional_rank_to_normal,
    summarize, plot_regimes, load_btc_daily,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "cs_rank_seed1"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
SEED = 1


def main():
    raw = load_regime_daily()
    feats_raw = build_features(raw)
    feats_full = cross_sectional_rank_to_normal(feats_raw)
    feats = feats_full.iloc[:, :-1]
    X = feats.values

    m = GaussianHMM(n_components=K, covariance_type="full",
                    n_iter=500, tol=1e-4, random_state=SEED)
    m.fit(X)
    labels = m.predict(X)
    posts = m.predict_proba(X)
    print(f"seed={SEED}, LL={m.score(X):.2f}, iters={m.monitor_.iter}")

    summary = summarize(m, feats, labels, feats_raw)
    print(summary)
    (OUT / "summary.txt").write_text(summary)

    timeline = feats_raw.loc[feats.index].copy()
    timeline["regime"] = labels
    for k in range(K):
        timeline[f"p_r{k}"] = posts[:, k]
    timeline.to_csv(OUT / "timeline.csv")

    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv",
                         feats.index[0], feats.index[-1])
    plot_regimes(labels, feats.index, btc, K, OUT / "plot.png")


if __name__ == "__main__":
    main()
