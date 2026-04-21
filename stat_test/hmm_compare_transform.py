"""Head-to-head: rank→inv-normal vs raw z-score for outer transform.

Same panel, same K_outer=3, same K_inner=[5,5,5], same commodity inner.
Only the outer feature transform differs.

Metrics: BTC VR (outer, composite), per-regime return spread, seed ARI, dwell.

Usage: python3 stat_test/hmm_compare_transform.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

from hmm_hierarchical_global import build_outer_global, fit_best
from hmm_regime import load_long_history, build_commodity_features, load_btc_daily

HERE = Path(__file__).parent
DATA = HERE / "data"
PROJECT_ROOT = HERE.parent
OUT = HERE / "results" / "compare_transform"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1, 2, 7, 11, 23, 99]
K_OUTER = 3
KI = [5, 5, 5]


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


def per_regime_spread(y, labels):
    valid = ~np.isnan(y)
    y_v, l_v = y[valid], labels[valid]
    means = []
    for k in np.unique(l_v):
        mask = l_v == k
        if mask.sum() < 5:
            continue
        means.append(y_v[mask].mean())
    return float(max(means) - min(means)) if means else float("nan")


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


def build_btc_fwd(idx):
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", idx[0], idx[-1])
    aligned = btc.reindex(idx, method="ffill")
    log_close = np.log(aligned.values)
    y = np.full(len(log_close), np.nan)
    y[:-20] = log_close[20:] - log_close[:-20]
    return y


def run_one(rat, com, y_btc, label):
    idx = rat.index.intersection(com.index)
    rat_a, com_a = rat.loc[idx], com.loc[idx]
    Xr, Xc = rat_a.values, com_a.values
    y = y_btc.reindex(idx).values if isinstance(y_btc, pd.Series) else y_btc

    _, m_outer, seed_o = fit_best(Xr, K_OUTER, seeds=SEEDS)
    outer = m_outer.predict(Xr)

    # Best-LL inner
    inner_best = np.full(len(outer), -1, dtype=int)
    for s in range(K_OUTER):
        mask = outer == s
        _, m_inner, _ = fit_best(Xc[mask], KI[s], seeds=SEEDS)
        lab = m_inner.predict(Xc[mask])
        order = np.argsort(m_inner.means_[:, 0])
        relabel = {old: new for new, old in enumerate(order)}
        inner_best[mask] = np.array([relabel[x] for x in lab])
    comp = build_composite(outer, inner_best, KI)

    # Cross-seed ARI on composite
    comp_by_seed = {}
    for seed in SEEDS:
        inner = np.full(len(outer), -1, dtype=int)
        ok = True
        for s in range(K_OUTER):
            lab = fit_inner_seed(Xc[outer == s], KI[s], seed)
            if lab is None:
                ok = False
                break
            inner[outer == s] = lab
        if ok:
            comp_by_seed[seed] = build_composite(outer, inner, KI)
    aris = [adjusted_rand_score(a, b)
            for a, b in combinations(comp_by_seed.values(), 2)]

    out = dict(
        outer_seed=seed_o,
        n_outer=K_OUTER, n_comp=sum(KI),
        vr_outer=var_reduction(y, outer),
        vr_comp=var_reduction(y, comp),
        spread_outer=per_regime_spread(y, outer),
        spread_comp=per_regime_spread(y, comp),
        ari_mean=float(np.mean(aris)),
        ari_min=float(np.min(aris)),
        **dwell_summary(comp),
    )
    print(f"\n--- {label} ---")
    print(f"  BTC VR: outer={out['vr_outer']*100:.2f}%  composite={out['vr_comp']*100:.2f}%")
    print(f"  BTC fwd20d spread: outer={out['spread_outer']*100:.2f}%  composite={out['spread_comp']*100:.2f}%")
    print(f"  Seed ARI: mean={out['ari_mean']:.3f}  min={out['ari_min']:.3f}")
    print(f"  Dwell: mean={out['mean']:.1f}d  min={out['min']:.1f}d  short<4d={out['short_n']}/{out['n_comp']}")
    return out


def main():
    outer_csv = PROJECT_ROOT / "data" / "regime_global_outer.csv"
    df = load_long_history()
    com = build_commodity_features(df)

    rat_rank = build_outer_global(outer_csv, transform="rank")
    rat_z    = build_outer_global(outer_csv, transform="zscore")

    # Shared panel
    idx = rat_rank.index.intersection(rat_z.index).intersection(com.index)
    rat_rank = rat_rank.loc[idx]
    rat_z    = rat_z.loc[idx]
    com      = com.loc[idx]
    y_btc = build_btc_fwd(idx)
    print(f"Shared panel: {len(idx)} days, {idx[0].date()} → {idx[-1].date()}")

    r_rank = run_one(rat_rank, com, y_btc, "RANK → inv-normal (frozen)")
    r_z    = run_one(rat_z,    com, y_btc, "Raw Z-SCORE")

    print(f"\n{'='*70}\nHEAD-TO-HEAD\n{'='*70}")
    print(f"{'metric':<34} {'rank→inv-normal':>18} {'zscore':>12}")
    for key, label, fmt in [
        ("vr_outer",     "BTC VR: outer",          "p"),
        ("vr_comp",      "BTC VR: composite",      "p"),
        ("spread_outer", "Outer fwd20d spread",    "p"),
        ("spread_comp",  "Composite fwd20d spread","p"),
        ("ari_mean",     "mean ARI",               "f3"),
        ("ari_min",      "min  ARI",               "f3"),
        ("mean",         "mean dwell (days)",      "f1"),
        ("min",          "min dwell (days)",       "f1"),
        ("short_n",      "states w/ dwell<4d",     "d"),
    ]:
        v1, v2 = r_rank[key], r_z[key]
        if fmt == "d":
            print(f"{label:<34} {int(v1):>18d} {int(v2):>12d}")
        elif fmt == "p":
            print(f"{label:<34} {v1*100:>17.2f}% {v2*100:>11.2f}%")
        elif fmt == "f3":
            print(f"{label:<34} {v1:>18.3f} {v2:>12.3f}")
        elif fmt == "f1":
            print(f"{label:<34} {v1:>18.1f} {v2:>12.1f}")

    pd.DataFrame({"rank": r_rank, "zscore": r_z}).to_csv(OUT / "summary.csv")
    print(f"\nWrote {OUT}/summary.csv")


if __name__ == "__main__":
    main()
