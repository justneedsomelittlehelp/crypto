"""Head-to-head: frozen US-rate K=2×[3,2] vs global K=3×[5,5,5].

Both are fit on their own outer feature sets but restricted to the same
date range so metrics are apples-to-apples.

Metrics:
  - BTC fwd20d variance reduction (outer-only and composite)
  - Seed stability: pairwise ARI on composite labels across 8 seeds
  - Dwell-time summary
  - Per-regime BTC fwd20d return spread (max − min) as signal magnitude

Usage: python3 stat_test/hmm_compare_frozen_vs_global.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

from hmm_hierarchical import build_rate_levels
from hmm_hierarchical_global import build_outer_global, fit_best
from hmm_regime import load_long_history, build_commodity_features, load_btc_daily

HERE = Path(__file__).parent
DATA = HERE / "data"
PROJECT_ROOT = HERE.parent
OUT = HERE / "results" / "compare_frozen_vs_global"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1, 2, 7, 11, 23, 99]


def fit_inner_seed(X, k, seed):
    try:
        m = GaussianHMM(n_components=k, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=seed).fit(X)
        lab = m.predict(X)
        order = np.argsort(m.means_[:, 0])
        relabel = {old: new for new, old in enumerate(order)}
        return np.array([relabel[x] for x in lab])
    except Exception:
        return None


def var_reduction(y, labels):
    valid = ~np.isnan(y)
    y_v, l_v = y[valid], labels[valid]
    total = np.nanvar(y_v, ddof=0)
    if total == 0:
        return np.nan
    num, n_total = 0.0, 0
    for k in np.unique(l_v):
        mask = l_v == k
        n_k = mask.sum()
        if n_k < 2:
            continue
        num += n_k * np.nanvar(y_v[mask], ddof=0)
        n_total += n_k
    return 1.0 - (num / n_total) / total


def per_regime_returns(y, labels):
    valid = ~np.isnan(y)
    y_v, l_v = y[valid], labels[valid]
    out = {}
    for k in np.unique(l_v):
        mask = l_v == k
        if mask.sum() < 5:
            continue
        out[int(k)] = dict(n=int(mask.sum()), mean=float(y_v[mask].mean()))
    return out


def build_composite(outer, inner, Ki_list):
    offsets = np.cumsum([0] + Ki_list[:-1])
    return np.array([offsets[outer[i]] + inner[i] for i in range(len(outer))])


def dwell_summary(labels):
    runs = {}
    cur, n = labels[0], 1
    for x in labels[1:]:
        if x == cur:
            n += 1
        else:
            runs.setdefault(cur, []).append(n)
            cur, n = x, 1
    runs.setdefault(cur, []).append(n)
    means = [np.mean(rs) for rs in runs.values()]
    return dict(mean=float(np.mean(means)),
                min=float(np.min(means)),
                short_n=int(sum(1 for m in means if m < 4)))


def fit_hierarchical(Xr, Xc, K_outer, Ki_list, y_btc, label):
    """Fit one hierarchical config; return metrics dict."""
    # Best-LL outer across seeds
    _, m_outer, seed_o = fit_best(Xr, K_outer, seeds=SEEDS)
    outer = m_outer.predict(Xr)

    # Best-LL inner per outer state
    inner_best = np.full(len(outer), -1, dtype=int)
    for s in range(K_outer):
        mask = outer == s
        _, m_inner, _ = fit_best(Xc[mask], Ki_list[s], seeds=SEEDS)
        lab = m_inner.predict(Xc[mask])
        order = np.argsort(m_inner.means_[:, 0])
        relabel = {old: new for new, old in enumerate(order)}
        inner_best[mask] = np.array([relabel[x] for x in lab])
    composite_best = build_composite(outer, inner_best, Ki_list)

    # Seed stability on composite
    composite_by_seed = {}
    for seed in SEEDS:
        inner = np.full(len(outer), -1, dtype=int)
        ok = True
        for s in range(K_outer):
            lab = fit_inner_seed(Xc[outer == s], Ki_list[s], seed)
            if lab is None:
                ok = False
                break
            inner[outer == s] = lab
        if ok:
            composite_by_seed[seed] = build_composite(outer, inner, Ki_list)
    aris = [adjusted_rand_score(a, b)
            for a, b in combinations(composite_by_seed.values(), 2)]
    ari_mean = float(np.mean(aris)) if aris else float("nan")
    ari_min = float(np.min(aris)) if aris else float("nan")

    # Metrics
    vr_outer = var_reduction(y_btc, outer)
    vr_comp  = var_reduction(y_btc, composite_best)
    n_states = sum(Ki_list)
    vr_per_state = vr_comp / n_states
    dw = dwell_summary(composite_best)
    per_reg = per_regime_returns(y_btc, composite_best)
    means = [r["mean"] for r in per_reg.values()]
    spread = float(max(means) - min(means)) if means else float("nan")

    print(f"\n--- {label} ---")
    print(f"  n_states={n_states}  outer_seed={seed_o}")
    print(f"  BTC VR: outer={vr_outer*100:.2f}%  composite={vr_comp*100:.2f}%  per_state={vr_per_state*100:.3f}%")
    print(f"  Seed ARI: mean={ari_mean:.3f}  min={ari_min:.3f}")
    print(f"  Dwell: mean={dw['mean']:.1f}d  min={dw['min']:.1f}d  short(<4d)={dw['short_n']}/{n_states}")
    print(f"  Per-regime BTC fwd20d spread (max−min): {spread*100:.2f}%")

    return dict(n_states=n_states, vr_outer=vr_outer, vr_comp=vr_comp,
                vr_per_state=vr_per_state, ari_mean=ari_mean, ari_min=ari_min,
                dwell_mean=dw['mean'], dwell_min=dw['min'], short_dwell=dw['short_n'],
                return_spread=spread)


def build_btc_fwd(idx):
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", idx[0], idx[-1])
    aligned = btc.reindex(idx, method="ffill")
    log_close = np.log(aligned.values)
    y = np.full(len(log_close), np.nan)
    y[:-20] = log_close[20:] - log_close[:-20]
    return y


def main():
    # Build both outer panels and commodity inner
    df = load_long_history()
    com = build_commodity_features(df)

    rat_frozen = build_rate_levels(df)
    rat_global = build_outer_global(PROJECT_ROOT / "data" / "regime_global_outer.csv")

    # Common date range
    idx = rat_frozen.index.intersection(rat_global.index).intersection(com.index)
    rat_frozen = rat_frozen.loc[idx]
    rat_global = rat_global.loc[idx]
    com = com.loc[idx]
    Xc = com.values
    y_btc = build_btc_fwd(idx)
    print(f"Common panel: {len(idx)} days, {idx[0].date()} → {idx[-1].date()}")
    print(f"  BTC-valid rows: {np.sum(~np.isnan(y_btc))}")

    results = {}
    results["frozen_K2_32"] = fit_hierarchical(
        rat_frozen.values, Xc, K_outer=2, Ki_list=[3, 2], y_btc=y_btc,
        label="FROZEN: outer(ffr,dgs2,yc) K=2 × inner [3,2]")
    results["global_K3_555"] = fit_hierarchical(
        rat_global.values, Xc, K_outer=3, Ki_list=[5, 5, 5], y_btc=y_btc,
        label="GLOBAL: outer(dxy,cs,netliq,m2) K=3 × inner [5,5,5]")

    # Head-to-head table
    print(f"\n{'='*70}\nHEAD-TO-HEAD (same panel, same BTC fwd20d)\n{'='*70}")
    f = results["frozen_K2_32"]
    g = results["global_K3_555"]
    print(f"{'metric':<34} {'frozen 2×[3,2]':>16} {'global 3×[5,5,5]':>18}")
    for key, label, fmt in [
        ("n_states",       "# composite states",        "d"),
        ("vr_outer",       "BTC VR: outer only",        "p"),
        ("vr_comp",        "BTC VR: composite",         "p"),
        ("vr_per_state",   "BTC VR per state",          "p3"),
        ("ari_mean",       "mean ARI (seed stab.)",     "f3"),
        ("ari_min",        "min  ARI (seed stab.)",     "f3"),
        ("dwell_mean",     "mean dwell (days)",         "f1"),
        ("dwell_min",      "min dwell (days)",          "f1"),
        ("short_dwell",    "states w/ dwell<4d",        "d"),
        ("return_spread",  "per-regime fwd20d spread",  "p"),
    ]:
        v1, v2 = f[key], g[key]
        if fmt == "d":
            print(f"{label:<34} {int(v1):>16d} {int(v2):>18d}")
        elif fmt == "p":
            print(f"{label:<34} {v1*100:>15.2f}% {v2*100:>17.2f}%")
        elif fmt == "p3":
            print(f"{label:<34} {v1*100:>15.3f}% {v2*100:>17.3f}%")
        elif fmt == "f3":
            print(f"{label:<34} {v1:>16.3f} {v2:>18.3f}")
        elif fmt == "f1":
            print(f"{label:<34} {v1:>16.1f} {v2:>18.1f}")

    # Save
    df_out = pd.DataFrame(results).T
    df_out.to_csv(OUT / "summary.csv")
    print(f"\nWrote {OUT}/summary.csv")


if __name__ == "__main__":
    main()
