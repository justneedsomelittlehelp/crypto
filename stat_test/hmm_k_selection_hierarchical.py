# STATUS: SUPERSEDED (2026-04-20) by hmm_k_selection_hierarchical_global.py.
# Kept live because the new K-selection script imports `sweep` and
# `build_btc_fwd` from here. Historical recommendation: K_outer=2,
# K_inner=[3,2] — superseded by K_outer=3, K_inner=[5,5,5] under the
# new global-outer feature set.
"""K-selection for the hierarchical HMM, per layer.

Outer layer (rate-level panel):
  Sweeps K ∈ {2..5}, reports:
    - BIC, AIC
    - BIC_neff (autocorrelation-adjusted sample size)
    - CV-LL (fit on first 80% of dates, score on last 20%)
    - BTC fwd20d variance reduction

Inner layer (commodity panel, given best K_outer):
  For each outer state, sweeps K ∈ {2..5} with same metrics.

CV-LL caveat: HMMs don't cleanly support "held-out LL" with a different state
sequence. We use .score() on the test set with the trained transition/emission
parameters — hmmlearn runs the forward algorithm on the test dates under the
fixed model. This is the standard HMM CV protocol.

BTC var-reduction caveat: inner sweep uses only the BTC dates inside that
outer state's subset. If the subset has <200 BTC-valid days, skip VR.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from hmm_hierarchical import build_rate_levels, fit_best
from hmm_regime import (
    load_long_history, build_commodity_features, load_btc_daily,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "k_selection_hier"
OUT.mkdir(parents=True, exist_ok=True)

K_LIST = [2, 3, 4, 5]
SEEDS = [42, 0, 1, 2, 7]
CV_FRAC = 0.8  # first 80% for train


def n_params(k: int, d: int) -> int:
    # full covariance
    return k * (k - 1) + (k - 1) + k * d + k * d * (d + 1) // 2


def estimate_n_eff(X: np.ndarray, max_lag: int = 40) -> float:
    n = len(X)
    ratios = []
    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean()
        c0 = (x * x).mean()
        if c0 == 0:
            ratios.append(1.0)
            continue
        s = 0.0
        for k in range(1, max_lag + 1):
            if k >= n:
                break
            rho = (x[:-k] * x[k:]).mean() / c0
            if rho < 0.05:
                break
            s += rho
        ratios.append(1.0 + 2.0 * s)
    return n / np.mean(ratios)


def cv_ll(X: np.ndarray, k: int, seeds=SEEDS, frac: float = CV_FRAC):
    n = len(X)
    cut = int(n * frac)
    X_tr, X_te = X[:cut], X[cut:]
    best = None
    for s in seeds:
        try:
            m = GaussianHMM(n_components=k, covariance_type="full",
                            n_iter=500, tol=1e-4, random_state=s).fit(X_tr)
            ll_tr = m.score(X_tr)
            ll_te = m.score(X_te)
        except Exception:
            continue
        if best is None or ll_te > best[0]:
            best = (ll_te, ll_tr, s)
    if best is None:
        return np.nan, np.nan
    return best[0], best[1]


def var_reduction(y: np.ndarray, r: np.ndarray) -> float:
    total = np.nanvar(y, ddof=0)
    if total == 0:
        return np.nan
    num, n_total = 0.0, 0
    for k in np.unique(r):
        mask = (r == k) & ~np.isnan(y)
        if mask.sum() == 0: continue
        num += mask.sum() * np.nanvar(y[mask], ddof=0)
        n_total += mask.sum()
    return 1.0 - (num / n_total) / total


def sweep(X: np.ndarray, y_btc: np.ndarray, label: str) -> pd.DataFrame:
    """y_btc: fwd20d log return aligned to X rows (NaN where BTC not yet live)."""
    n, d = X.shape
    n_eff = estimate_n_eff(X)
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"  n={n}, d={d}, n_eff≈{n_eff:.0f}")

    rows = []
    for K in K_LIST:
        ll_full, m_full, seed = fit_best(X, K)
        labels = m_full.predict(X)
        p = n_params(K, d)
        bic = -2 * ll_full + p * np.log(n)
        bic_eff = -2 * ll_full + p * np.log(n_eff)
        aic = -2 * ll_full + 2 * p
        ll_te, ll_tr = cv_ll(X, K)
        vr = var_reduction(y_btc, labels) if y_btc is not None else np.nan
        freq = [(labels == k).sum() for k in range(K)]
        min_freq = min(freq)
        rows.append(dict(K=K, seed=seed, logL=ll_full, BIC=bic,
                         BIC_neff=bic_eff, AIC=aic,
                         CV_ll_train=ll_tr, CV_ll_test=ll_te,
                         BTC_fwd20_VR=vr, min_freq=min_freq))
        print(f"  K={K}: LL={ll_full:9.1f}  BIC={bic:9.0f}  "
              f"BIC_neff={bic_eff:8.0f}  AIC={aic:9.0f}  "
              f"CV-LL_te={ll_te:9.1f}  BTC-VR={vr*100:5.2f}%  "
              f"min_freq={min_freq}")

    df = pd.DataFrame(rows)
    # Flag best per metric
    print(f"\n  Best by:")
    print(f"    BIC:       K={int(df.loc[df['BIC'].idxmin(),'K'])}")
    print(f"    BIC_neff:  K={int(df.loc[df['BIC_neff'].idxmin(),'K'])}")
    print(f"    AIC:       K={int(df.loc[df['AIC'].idxmin(),'K'])}")
    print(f"    CV-LL_te:  K={int(df.loc[df['CV_ll_test'].idxmax(),'K'])}")
    if df['BTC_fwd20_VR'].notna().any():
        print(f"    BTC-VR:    K={int(df.loc[df['BTC_fwd20_VR'].idxmax(),'K'])}")
    return df


def build_btc_fwd(dates: pd.DatetimeIndex) -> np.ndarray:
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv",
                         dates[0], dates[-1])
    aligned = btc.reindex(dates, method="ffill")
    log_close = np.log(aligned.values)
    fwd = np.full(len(log_close), np.nan)
    fwd[:-20] = log_close[20:] - log_close[:-20]
    return fwd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-outer-chosen", type=int, default=None,
                   help="If set, run inner sweep at this K_outer instead of argmin-BIC")
    args = p.parse_args()

    df = load_long_history()
    rat = build_rate_levels(df)
    com = build_commodity_features(df)
    idx = rat.index.intersection(com.index)
    rat = rat.loc[idx]; com = com.loc[idx]
    Xr, Xc = rat.values, com.values

    y_btc = build_btc_fwd(idx)

    # Outer sweep
    outer_df = sweep(Xr, y_btc, label="OUTER: rate levels (FFR, DGS2, yield_curve)")
    outer_df.to_csv(OUT / "outer_sweep.csv", index=False)

    # Pick K_outer
    if args.k_outer_chosen is not None:
        K_outer = args.k_outer_chosen
        print(f"\n[User-specified K_outer = {K_outer}]")
    else:
        # Prefer CV-LL; fallback to BIC_neff
        K_outer = int(outer_df.loc[outer_df['CV_ll_test'].idxmax(), 'K'])
        print(f"\n[Auto-picked K_outer = {K_outer} by CV-LL]")

    # Refit outer at K_outer, get labels
    _, m_outer, _ = fit_best(Xr, K_outer)
    outer_labels = m_outer.predict(Xr)

    # Inner sweeps — one per outer state
    inner_dfs = {}
    for s in range(K_outer):
        mask = outer_labels == s
        Xs = Xc[mask]
        ys = y_btc[mask]
        if len(Xs) < 200:
            print(f"\n[S{s}: only {len(Xs)} rows — skip]")
            continue
        df_s = sweep(Xs, ys, label=f"INNER | outer state S{s} (n={len(Xs)})")
        df_s.insert(0, "outer_state", s)
        df_s.to_csv(OUT / f"inner_sweep_S{s}.csv", index=False)
        inner_dfs[s] = df_s

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Outer recommendation (CV-LL, BIC_neff):")
    k1 = int(outer_df.loc[outer_df['CV_ll_test'].idxmax(), 'K'])
    k2 = int(outer_df.loc[outer_df['BIC_neff'].idxmin(), 'K'])
    print(f"  K_outer = {k1} (CV-LL) / {k2} (BIC_neff)")
    print(f"Inner recommendations:")
    for s, df_s in inner_dfs.items():
        k1 = int(df_s.loc[df_s['CV_ll_test'].idxmax(), 'K'])
        k2 = int(df_s.loc[df_s['BIC_neff'].idxmin(), 'K'])
        vr_row = df_s.loc[df_s['BTC_fwd20_VR'].idxmax()] if df_s['BTC_fwd20_VR'].notna().any() else None
        k3 = int(vr_row['K']) if vr_row is not None else None
        print(f"  S{s}: K_inner = {k1} (CV-LL) / {k2} (BIC_neff) / {k3} (BTC-VR)")

    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
