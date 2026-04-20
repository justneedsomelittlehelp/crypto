# STATUS: ARCHIVED (2026-04-20) — seed-stability check for raw CS-rank HMM.
# Informed the "seed 42 is not best" finding. Superseded: hierarchical model
# uses fit_best across 5 seeds built in. See HMM_LOG.md §3.
"""HMM K=3 seed stability test.

Refits the same K=3 Gaussian HMM under 5 different random seeds and reports:
  - Log-likelihood per seed
  - Transition matrices
  - Emission means (after aligning states to seed 0 via Hungarian matching)
  - Regime frequency per seed
  - Pairwise label agreement % (post alignment)

If the three regimes are real structure, aligned means/frequencies will look
similar across seeds and pairwise agreement will be high (>80%). If they're
seed-42 artifacts, seeds will diverge.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

from hmm_regime import (
    load_regime_daily, build_features, cross_sectional_rank_to_normal,
)

HERE = Path(__file__).parent
OUT = HERE / "results" / "cs_rank_stability"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
SEEDS = [42, 0, 1, 2, 7]


def fit(X: np.ndarray, seed: int) -> GaussianHMM:
    m = GaussianHMM(n_components=K, covariance_type="full",
                    n_iter=500, tol=1e-4, random_state=seed)
    m.fit(X)
    return m


def align_to_ref(ref_means: np.ndarray, other_means: np.ndarray):
    """Hungarian match other's states to reference. Returns perm: other_state -> ref_state."""
    # cost[i, j] = dist from other state i to ref state j; we want to match
    # each other-state i to ref-state perm[i] minimizing total cost.
    cost = np.linalg.norm(
        other_means[:, None, :] - ref_means[None, :, :], axis=-1
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty(K, dtype=int)
    perm[row_ind] = col_ind
    return perm  # perm[i] = ref-state that other-state i maps to


def remap(labels: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return perm[labels]


def main():
    print("Loading + building features...")
    raw = load_regime_daily()
    feats_raw = build_features(raw)
    feats_full = cross_sectional_rank_to_normal(feats_raw)
    feats = feats_full.iloc[:, :-1]
    X = feats.values
    print(f"  obs: {len(feats)}, dims: {feats.shape[1]}")

    # Fit seed 0 (reference)
    ref_seed = SEEDS[0]
    print(f"\nFitting reference seed {ref_seed}...")
    ref_model = fit(X, ref_seed)
    ref_ll = ref_model.score(X)
    ref_labels_raw = ref_model.predict(X)
    # Canonicalize reference state order by frequency (state 0 = most common)
    freq = np.array([(ref_labels_raw == k).sum() for k in range(K)])
    ref_order = np.argsort(-freq)  # descending freq
    ref_perm = np.empty(K, dtype=int)
    ref_perm[ref_order] = np.arange(K)  # maps original -> freq-rank
    ref_means = ref_model.means_[ref_order]
    ref_labels = remap(ref_labels_raw, ref_perm)

    all_labels = {ref_seed: ref_labels}
    all_means = {ref_seed: ref_means}
    all_ll = {ref_seed: ref_ll}
    all_freq = {ref_seed: np.array([(ref_labels == k).sum() for k in range(K)])}
    all_trans = {ref_seed: None}  # fill below
    # Reorder transition matrix too
    T = ref_model.transmat_
    all_trans[ref_seed] = T[np.ix_(ref_order, ref_order)]

    for s in SEEDS[1:]:
        print(f"Fitting seed {s}...")
        m = fit(X, s)
        labels_raw = m.predict(X)
        perm = align_to_ref(ref_means, m.means_)  # perm[i] = ref-state for other-state i
        labels = remap(labels_raw, perm)
        means = np.empty_like(m.means_)
        means[perm] = m.means_
        T = m.transmat_
        # reorder T: new T[i,j] where i,j are ref-state indices
        inv_perm = np.argsort(perm)
        T_aligned = T[np.ix_(inv_perm, inv_perm)]
        all_labels[s] = labels
        all_means[s] = means
        all_ll[s] = m.score(X)
        all_freq[s] = np.array([(labels == k).sum() for k in range(K)])
        all_trans[s] = T_aligned

    # Report
    cols = list(feats.columns)
    lines = []
    lines.append(f"K={K}, seeds={SEEDS}, obs={len(feats)}")
    lines.append("\n=== Log-likelihood per seed ===")
    for s in SEEDS:
        lines.append(f"  seed {s}: LL = {all_ll[s]:,.2f}")

    lines.append("\n=== Regime frequency (% of days) per seed, aligned ===")
    header = "  seed   " + "  ".join([f"  r{k}  " for k in range(K)])
    lines.append(header)
    for s in SEEDS:
        pct = 100.0 * all_freq[s] / all_freq[s].sum()
        lines.append(f"  {s:>4}   " + "  ".join([f"{v:5.1f}%" for v in pct]))

    lines.append("\n=== Aligned emission means per regime, per seed ===")
    for k in range(K):
        lines.append(f"\n  -- r{k} --")
        lines.append("  seed   " + "  ".join([f"{c:>13s}" for c in cols]))
        for s in SEEDS:
            mu = all_means[s][k]
            lines.append(f"  {s:>4}   " + "  ".join([f"{v:>13.3f}" for v in mu]))

    lines.append("\n=== Transition diagonal (persistence) per seed ===")
    lines.append("  seed   " + "  ".join([f"p(r{k}->r{k})" for k in range(K)]))
    for s in SEEDS:
        diag = np.diag(all_trans[s])
        lines.append(f"  {s:>4}   " + "  ".join([f"{v:>11.3f}" for v in diag]))

    lines.append("\n=== Pairwise label agreement % (post Hungarian alignment) ===")
    header = "         " + "  ".join([f"s{s:>3}" for s in SEEDS])
    lines.append(header)
    for s1 in SEEDS:
        row = [f"  s{s1:>3}   "]
        for s2 in SEEDS:
            agree = 100.0 * (all_labels[s1] == all_labels[s2]).mean()
            row.append(f"{agree:5.1f}")
        lines.append("  ".join(row))

    summary = "\n".join(lines)
    print(summary)
    (OUT / "stability.txt").write_text(summary)
    print(f"\nWrote {OUT / 'stability.txt'}")


if __name__ == "__main__":
    main()
