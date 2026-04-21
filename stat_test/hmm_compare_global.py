"""Compare K_inner=[3,3,3] vs [5,5,5] on the global-outer hierarchical HMM.

Two tests:
  (2) BTC fwd20d variance reduction
      - total VR, VR per composite state, VR normalized by #states
  (5) Seed stability
      - refit each inner HMM under several seeds
      - compute Adjusted Rand Index (ARI) between seed pairs
      - report mean ARI (1.0 = perfectly stable, 0.0 = random)

Outer is fit once per run (K=3, best-of-seeds LL) — held fixed while the
inner sweep varies seeds, so we isolate inner stability.

Usage:
  python3 stat_test/hmm_compare_global.py
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
OUT = HERE / "results" / "compare_global"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1, 2, 7, 11, 23, 99]  # 8 seeds for stability
CONFIGS = {
    "K3_333": [3, 3, 3],
    "K3_555": [5, 5, 5],
}
K_OUTER = 3


def fit_inner_seed(X: np.ndarray, k: int, seed: int) -> np.ndarray | None:
    """Fit inner HMM at given K, given seed. Returns label array or None on fail."""
    try:
        m = GaussianHMM(n_components=k, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=seed).fit(X)
        lab = m.predict(X)
        # Sort by gld mean for deterministic labeling across seeds
        order = np.argsort(m.means_[:, 0])
        relabel = {old: new for new, old in enumerate(order)}
        return np.array([relabel[x] for x in lab])
    except Exception as e:
        print(f"    seed={seed} failed: {e}")
        return None


def var_reduction(y: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Compute total VR and per-state contribution on BTC-valid rows."""
    valid = ~np.isnan(y)
    y_v, l_v = y[valid], labels[valid]
    total = np.nanvar(y_v, ddof=0)
    if total == 0:
        return np.nan, {}
    num, n_total = 0.0, 0
    per_state = {}
    for k in np.unique(l_v):
        mask = l_v == k
        n_k = mask.sum()
        if n_k < 2:
            continue
        v_k = np.nanvar(y_v[mask], ddof=0)
        num += n_k * v_k
        n_total += n_k
        per_state[int(k)] = dict(n=int(n_k), mean=float(y_v[mask].mean()),
                                 var=float(v_k))
    vr = 1.0 - (num / n_total) / total
    return vr, per_state


def build_composite(outer: np.ndarray, inner: np.ndarray, Ki_list: list[int]) -> np.ndarray:
    offsets = np.cumsum([0] + Ki_list[:-1])
    return np.array([offsets[outer[i]] + inner[i] for i in range(len(outer))])


def mean_dwell_all(labels: np.ndarray) -> dict:
    """Dwell stats per label: mean, 10th percentile."""
    runs = {}
    cur, n = labels[0], 1
    for x in labels[1:]:
        if x == cur:
            n += 1
        else:
            runs.setdefault(cur, []).append(n)
            cur, n = x, 1
    runs.setdefault(cur, []).append(n)
    out = {}
    for k, rs in runs.items():
        out[int(k)] = dict(mean=float(np.mean(rs)),
                           p10=float(np.percentile(rs, 10)),
                           n_runs=len(rs))
    return out


def main():
    # --- Load data once ---
    outer_path = PROJECT_ROOT / "data" / "regime_global_outer.csv"
    rat = build_outer_global(outer_path)
    df = load_long_history()
    com = build_commodity_features(df)
    idx = rat.index.intersection(com.index)
    rat = rat.loc[idx]
    com = com.loc[idx]
    Xr, Xc = rat.values, com.values
    print(f"Panel: {len(idx)} days, {idx[0].date()} → {idx[-1].date()}")

    # --- Fit outer once (best-of-seeds LL); freeze labels ---
    print(f"\nFitting outer (K={K_OUTER}, best-of-{len(SEEDS)} seeds)...")
    _, m_outer, seed_outer = fit_best(Xr, K_OUTER, seeds=SEEDS)
    outer = m_outer.predict(Xr)
    print(f"  outer seed={seed_outer}, freqs={[int((outer==k).sum()) for k in range(K_OUTER)]}")

    # --- BTC fwd20d ---
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", idx[0], idx[-1])
    btc_aligned = btc.reindex(idx, method="ffill")
    log_close = np.log(btc_aligned.values)
    y_btc = np.full(len(log_close), np.nan)
    y_btc[:-20] = log_close[20:] - log_close[:-20]

    # --- Per-config: seed-stability + BTC VR ---
    results = {}
    for cfg_name, Ki_list in CONFIGS.items():
        print(f"\n{'='*60}\n{cfg_name}: K_inner={Ki_list}\n{'='*60}")

        # (A) Seed stability: for each seed, fit inner per outer state,
        # build composite label, then compute pairwise ARI.
        composite_by_seed = {}
        for seed in SEEDS:
            inner = np.full(len(idx), -1, dtype=int)
            ok = True
            for s in range(K_OUTER):
                mask = outer == s
                lab = fit_inner_seed(Xc[mask], Ki_list[s], seed)
                if lab is None:
                    ok = False
                    break
                inner[mask] = lab
            if ok:
                composite_by_seed[seed] = build_composite(outer, inner, Ki_list)

        # Pairwise ARI on composite labels
        seeds_fit = list(composite_by_seed.keys())
        aris = []
        for s1, s2 in combinations(seeds_fit, 2):
            aris.append(adjusted_rand_score(composite_by_seed[s1],
                                            composite_by_seed[s2]))
        ari_mean = float(np.mean(aris)) if aris else float("nan")
        ari_min = float(np.min(aris)) if aris else float("nan")
        print(f"  Seed stability ({len(seeds_fit)} seeds, {len(aris)} pairs):")
        print(f"    mean ARI = {ari_mean:.3f}   min ARI = {ari_min:.3f}")

        # (B) BTC VR using the best-LL composite (same seed-picking as fit script)
        # Refit inner via best-of-seeds per outer state, then compute VR
        inner_best = np.full(len(idx), -1, dtype=int)
        for s in range(K_OUTER):
            mask = outer == s
            _, m_inner, seed_inner = fit_best(Xc[mask], Ki_list[s], seeds=SEEDS)
            lab = m_inner.predict(Xc[mask])
            order = np.argsort(m_inner.means_[:, 0])
            relabel = {old: new for new, old in enumerate(order)}
            inner_best[mask] = np.array([relabel[x] for x in lab])
        composite_best = build_composite(outer, inner_best, Ki_list)

        # BTC VR by outer label only
        vr_outer, _ = var_reduction(y_btc, outer)
        # BTC VR by composite
        vr_comp, per_state = var_reduction(y_btc, composite_best)
        n_states = sum(Ki_list)
        vr_per_state = vr_comp / n_states

        # Dwell stats on composite
        dwell = mean_dwell_all(composite_best)
        short_dwell = sum(1 for d in dwell.values() if d["mean"] < 4)

        print(f"  BTC fwd20d variance reduction:")
        print(f"    by outer only:         {vr_outer*100:.2f}%")
        print(f"    by composite ({n_states} states): {vr_comp*100:.2f}%")
        print(f"    VR per state:          {vr_per_state*100:.3f}%")
        print(f"  Composite dwell times:")
        print(f"    states with mean dwell < 4d: {short_dwell} / {n_states}")
        print(f"    state dwell means: {sorted([round(d['mean'],1) for d in dwell.values()])}")

        results[cfg_name] = dict(
            Ki=Ki_list,
            n_states=n_states,
            ari_mean=ari_mean,
            ari_min=ari_min,
            vr_outer=vr_outer,
            vr_composite=vr_comp,
            vr_per_state=vr_per_state,
            short_dwell=short_dwell,
            dwells=dwell,
            per_state=per_state,
        )

    # --- Head-to-head summary ---
    print(f"\n{'='*60}\nHEAD-TO-HEAD\n{'='*60}")
    print(f"{'metric':<30} {'K3_333':>12} {'K3_555':>12}")
    for key, label in [("n_states", "# composite states"),
                       ("ari_mean", "mean ARI (seed stab.)"),
                       ("ari_min",  "min  ARI (seed stab.)"),
                       ("vr_composite", "BTC VR (composite)"),
                       ("vr_per_state", "BTC VR per state"),
                       ("short_dwell", "states w/ dwell<4d")]:
        r1 = results["K3_333"][key]
        r2 = results["K3_555"][key]
        if isinstance(r1, float):
            if "vr" in key or "ari" in key:
                print(f"{label:<30} {r1*100 if 'vr' in key else r1:>12.3f} {r2*100 if 'vr' in key else r2:>12.3f}")
            else:
                print(f"{label:<30} {r1:>12.3f} {r2:>12.3f}")
        else:
            print(f"{label:<30} {r1:>12d} {r2:>12d}")

    # Save
    out_lines = []
    for cfg_name, r in results.items():
        out_lines.append(f"\n=== {cfg_name} (K_inner={r['Ki']}) ===")
        out_lines.append(f"  # states: {r['n_states']}")
        out_lines.append(f"  mean ARI: {r['ari_mean']:.3f}")
        out_lines.append(f"  min ARI:  {r['ari_min']:.3f}")
        out_lines.append(f"  BTC VR (composite): {r['vr_composite']*100:.2f}%")
        out_lines.append(f"  BTC VR per state:   {r['vr_per_state']*100:.3f}%")
        out_lines.append(f"  Short-dwell (<4d): {r['short_dwell']} / {r['n_states']}")
    (OUT / "summary.txt").write_text("\n".join(out_lines))
    print(f"\nWrote {OUT}/summary.txt")


if __name__ == "__main__":
    main()
