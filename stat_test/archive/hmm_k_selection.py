# STATUS: ARCHIVED (2026-04-20) — single-layer K selection. Superseded by
# hmm_k_selection_hierarchical.py (per-layer, with CV-LL and target-VR).
# See HMM_LOG.md §6.
"""Pick K via BIC / AIC on the z-score CS rank feature set.

For each K in {2..7}:
  - Fit HMM under 5 seeds, keep best-LL.
  - Compute BIC = -2*LL + p*log(n), AIC = -2*LL + 2*p.

Parameter count for Gaussian HMM with K states, D dims:
  p = K*(K-1)        transition matrix (K rows, each sums to 1)
    + (K-1)           initial distribution (sums to 1)
    + K*D             means
    + K*D*(D+1)/2  if full covariance (symmetric DxD)
    or  K*D        if diagonal covariance

Supports --cov {full,diag} via CLI.

Lower BIC/AIC is better. Report best K by each.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from hmm_regime import load_regime_daily
from hmm_zscore_cs_rank import build_z_features, cs_rank_to_normal

HERE = Path(__file__).parent
OUT = HERE / "results" / "k_selection"
OUT.mkdir(parents=True, exist_ok=True)

K_LIST = [2, 3, 4, 5, 6, 7]
SEEDS = [42, 0, 1, 2, 7]


def n_params(k: int, d: int, cov: str) -> int:
    base = k * (k - 1) + (k - 1) + k * d
    if cov == "full":
        return base + k * d * (d + 1) // 2
    elif cov == "diag":
        return base + k * d
    raise ValueError(cov)


def fit_best_ll(X: np.ndarray, k: int, seeds, cov: str):
    best = None
    for s in seeds:
        m = GaussianHMM(n_components=k, covariance_type=cov,
                        n_iter=500, tol=1e-4, random_state=s).fit(X)
        ll = m.score(X)
        if best is None or ll > best[0]:
            best = (ll, m, s)
    return best


def estimate_n_eff(X: np.ndarray, max_lag: int = 40) -> float:
    """Effective sample size under AR(1) approximation per feature, averaged.

    n_eff = n / (1 + 2 * sum_k rho_k), truncated where rho crosses below 0.05.
    """
    n = len(X)
    ratios = []
    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean()
        c0 = (x * x).mean()
        s = 0.0
        for k in range(1, max_lag + 1):
            rho = (x[:-k] * x[k:]).mean() / c0
            if rho < 0.05:
                break
            s += rho
        ratios.append(1.0 + 2.0 * s)
    return n / np.mean(ratios)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cov", choices=["full", "diag"], default="diag")
    args = p.parse_args()

    raw = load_regime_daily()
    z = build_z_features(raw)
    feats_full = cs_rank_to_normal(z)
    feats = feats_full.iloc[:, :-1]
    X = feats.values
    n, d = X.shape
    n_eff = estimate_n_eff(X)
    print(f"n = {n}, d = {d}, covariance = {args.cov}, n_eff ≈ {n_eff:.1f}")

    rows = []
    for k in K_LIST:
        ll, m, best_seed = fit_best_ll(X, k, SEEDS, args.cov)
        nparam = n_params(k, d, args.cov)
        bic = -2 * ll + nparam * np.log(n)
        bic_eff = -2 * ll + nparam * np.log(n_eff)
        aic = -2 * ll + 2 * nparam
        rows.append({"K": k, "best_seed": best_seed, "n_params": nparam,
                     "logL": ll, "BIC": bic, "BIC_neff": bic_eff, "AIC": aic})
        print(f"K={k}: seed={best_seed}, p={nparam}, LL={ll:,.2f}, "
              f"BIC={bic:,.0f}, BIC_neff={bic_eff:,.0f}, AIC={aic:,.0f}")

    df = pd.DataFrame(rows)

    print(f"\nBest by BIC:       K = {int(df.loc[df['BIC'].idxmin(), 'K'])}")
    print(f"Best by BIC_neff:  K = {int(df.loc[df['BIC_neff'].idxmin(), 'K'])}")
    print(f"Best by AIC:       K = {int(df.loc[df['AIC'].idxmin(), 'K'])}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["K"], df["BIC"], "o-", label="BIC")
    ax.plot(df["K"], df["BIC_neff"], "^-", label=f"BIC (n_eff≈{n_eff:.0f})")
    ax.plot(df["K"], df["AIC"], "s-", label="AIC")
    ax.set_xlabel("K (number of regimes)")
    ax.set_ylabel("Criterion (lower = better)")
    ax.set_title(f"HMM K selection — cov={args.cov}, z-score CS ranks")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"k_selection_{args.cov}.png", dpi=120)
    df.to_csv(OUT / f"k_selection_{args.cov}.csv", index=False)
    print(f"\nWrote {OUT}/k_selection.csv, k_selection.png")


if __name__ == "__main__":
    main()
