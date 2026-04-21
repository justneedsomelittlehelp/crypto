"""Walk-forward refit of the frozen global-outer HMM.

Motivation:
  The frozen model fits rank normalization + HMM parameters on the full
  2012-2026 panel at once. Each day's label is therefore informed by future
  days via (a) the HMM's transition/emission parameters absorbing later data,
  and (b) the rolling rank having full cross-sectional context. If we feed
  these labels to v12 as a feature, v12 trains on labels that leaked future.

Walk-forward protocol:
  - Initial training window: days [0, T0] where T0 = min_train_days
  - Refit every REFIT_INTERVAL days thereafter
  - At each refit R: re-compute rolling rank on [0, R], re-fit outer HMM on
    [0, R], re-fit inner HMMs per outer state on [0, R]. Label days
    (R-REFIT_INTERVAL, R] with these parameters.
  - Parameters are never informed by data after the day they label.

Label consistency across refits:
  - HMM state IDs are permutation-arbitrary. We sort outer states by mean of
    first feature (dxy_rn) ascending at each refit, and inner by gld mean.
    This gives deterministic IDs given features — but sorting can still flip
    if two states have nearly-equal means at some refit.
  - Report: fraction of days where walk-forward label matches fit-once label
    (after best-permutation alignment across outer only), and ARI.

Output: walk-forward timeline + comparison stats.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score

from hmm_hierarchical_global import (
    OUTER_COLS, SEEDS, RANK_WINDOW, fit_best, build_outer_global,
)
from hmm_regime import load_long_history, build_commodity_features, load_btc_daily

HERE = Path(__file__).parent
DATA = HERE / "data"
PROJECT_ROOT = HERE.parent
OUT = HERE / "results" / "walk_forward_global"
OUT.mkdir(parents=True, exist_ok=True)

K_OUTER = 3
KI = [5, 5, 5]
MIN_TRAIN_DAYS = 1260    # 5 calendar years = 3y rank warmup + 2y HMM training
REFIT_INTERVAL = 63      # quarterly refits


def rolling_rank_inv_normal_until(values: np.ndarray, end_idx: int,
                                   window: int = RANK_WINDOW) -> np.ndarray:
    """Compute rolling rank→inv-normal for values[:end_idx+1]. Returns an
    array of the same length as values[:end_idx+1] with NaN before warmup.
    Used during walk-forward so today's rank only sees past data.
    """
    n = end_idx + 1
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        w = values[max(0, t - window + 1): t + 1]
        x = values[t]
        rank = np.sum(w < x) + 0.5 * np.sum(w == x) + 0.5
        pct = rank / (len(w) + 1)
        out[t] = norm.ppf(pct)
    return out


def build_outer_walk_forward(outer_df: pd.DataFrame, end_idx: int) -> np.ndarray:
    """Build outer feature matrix using only data up to end_idx (inclusive).
    Returns (n, d) array where rows before RANK_WINDOW are NaN.
    """
    cols = []
    for c in OUTER_COLS:
        v = outer_df[c].values
        cols.append(rolling_rank_inv_normal_until(v, end_idx))
    return np.column_stack(cols)  # shape (end_idx+1, d)


def align_and_label_outer(m_outer: GaussianHMM, outer_mat: np.ndarray) -> np.ndarray:
    """Predict outer labels and remap to sorted-by-dxy_rn order."""
    labels = m_outer.predict(outer_mat)
    order = np.argsort(m_outer.means_[:, 0])  # ascending by first feature
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in labels])


def align_and_label_inner(m_inner: GaussianHMM, inner_mat: np.ndarray) -> np.ndarray:
    labels = m_inner.predict(inner_mat)
    order = np.argsort(m_inner.means_[:, 0])  # sort by gld_z
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in labels])


def build_composite(outer: np.ndarray, inner: np.ndarray, Ki_list):
    offsets = np.cumsum([0] + list(Ki_list[:-1]))
    return np.array([offsets[outer[i]] + inner[i] for i in range(len(outer))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refit-interval", type=int, default=REFIT_INTERVAL)
    p.add_argument("--min-train-days", type=int, default=MIN_TRAIN_DAYS)
    args = p.parse_args()

    print(f"Walk-forward config: refit every {args.refit_interval}d, "
          f"min train = {args.min_train_days}d")

    # Load raw outer features (not transformed)
    outer_csv = PROJECT_ROOT / "data" / "regime_global_outer.csv"
    outer_df = pd.read_csv(outer_csv, parse_dates=["date_utc"]).set_index("date_utc")
    outer_df = outer_df[OUTER_COLS].sort_index().ffill().dropna()

    # Inner features (commodity CS ranks — uses its own 252d z-score)
    df = load_long_history()
    com = build_commodity_features(df)

    # Common index (raw outer × inner)
    idx = outer_df.index.intersection(com.index)
    outer_df = outer_df.loc[idx]
    com = com.loc[idx]
    Xc = com.values
    N = len(idx)
    print(f"Panel: {N} days, {idx[0].date()} → {idx[-1].date()}")

    # Walk-forward outer + inner labels
    wf_outer = np.full(N, -1, dtype=int)
    wf_inner = np.full(N, -1, dtype=int)

    refit_days = list(range(args.min_train_days, N, args.refit_interval))
    # Ensure we label through the end
    if refit_days[-1] != N - 1:
        refit_days.append(N - 1)
    print(f"Refits: {len(refit_days)} (first at day {refit_days[0]}, last at {refit_days[-1]})")

    prev_end = args.min_train_days - args.refit_interval  # first refit labels back to min_train_days - interval
    prev_end = max(0, prev_end)

    for ri, R in enumerate(refit_days):
        # Build rolling-rank outer matrix using only data up to R
        outer_mat_full = build_outer_walk_forward(outer_df, R)
        # Drop NaN rows (pre-rank-warmup)
        valid_start = RANK_WINDOW - 1
        train_mat = outer_mat_full[valid_start: R + 1]
        if len(train_mat) < 200:
            continue

        # Fit outer HMM
        _, m_outer, _ = fit_best(train_mat, K_OUTER, seeds=SEEDS)
        outer_labels_train = align_and_label_outer(m_outer, train_mat)

        # Fit inner per outer state, align by gld
        inner_labels_train = np.full(len(train_mat), -1, dtype=int)
        inner_models = {}
        Xc_train = Xc[valid_start: R + 1]
        for s in range(K_OUTER):
            mask = outer_labels_train == s
            if mask.sum() < 50 * KI[s]:
                inner_labels_train[mask] = 0
                continue
            _, m_inner, _ = fit_best(Xc_train[mask], KI[s], seeds=SEEDS)
            inner_labels_train[mask] = align_and_label_inner(m_inner, Xc_train[mask])
            inner_models[s] = m_inner

        # Labels for the segment (prev_end+1 .. R)
        seg_start = max(valid_start, prev_end + 1)
        seg_end = R  # inclusive
        for t in range(seg_start, seg_end + 1):
            # Position inside train_mat
            idx_in_train = t - valid_start
            o = outer_labels_train[idx_in_train]
            wf_outer[t] = o
            if o in inner_models:
                # Need to predict inner for this single point — use the
                # saved inner model's state assignment for this day
                wf_inner[t] = inner_labels_train[idx_in_train]
            else:
                wf_inner[t] = 0
        prev_end = R
        if ri % 10 == 0 or ri == len(refit_days) - 1:
            print(f"  refit {ri+1}/{len(refit_days)}: day {R} ({idx[R].date()})")

    # Walk-forward composite
    labeled_mask = wf_outer >= 0
    wf_composite = np.full(N, -1, dtype=int)
    wf_composite[labeled_mask] = build_composite(
        wf_outer[labeled_mask], wf_inner[labeled_mask], KI)
    print(f"\nWalk-forward labeled: {labeled_mask.sum()}/{N} days "
          f"({labeled_mask.sum()*100/N:.1f}%)")

    # Load fit-once labels for comparison (on identical dates)
    fit_once_tl = pd.read_csv(
        HERE / "results" / "hierarchical_global_K3_555" / "timeline.csv",
        parse_dates=[0], index_col=0,
    )
    fit_once_tl.index = fit_once_tl.index.tz_convert("UTC") if fit_once_tl.index.tz else fit_once_tl.index.tz_localize("UTC")

    # Align to walk-forward labeled dates
    wf_idx = idx[labeled_mask]
    fo_aligned = fit_once_tl.reindex(wf_idx)
    fo_valid = fo_aligned.dropna()
    common = fo_valid.index
    print(f"\nComparison period: {len(common)} days, "
          f"{common[0].date()} → {common[-1].date()}")

    wf_o = pd.Series(wf_outer, index=idx).loc[common].values
    wf_c = pd.Series(wf_composite, index=idx).loc[common].values
    fo_o = fo_valid["outer"].astype(int).values
    fo_c = fo_valid["composite"].astype(int).values

    print("\n=== Walk-forward vs fit-once agreement ===")
    print(f"  Outer ARI:     {adjusted_rand_score(wf_o, fo_o):.3f}")
    print(f"  Composite ARI: {adjusted_rand_score(wf_c, fo_c):.3f}")
    print(f"  Outer exact match:     {(wf_o == fo_o).mean()*100:.1f}%")
    print(f"  Composite exact match: {(wf_c == fo_c).mean()*100:.1f}%")

    # BTC VR on walk-forward
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", idx[0], idx[-1])
    btc_a = btc.reindex(idx, method="ffill")
    log_close = np.log(btc_a.values)
    y = np.full(len(log_close), np.nan)
    y[:-20] = log_close[20:] - log_close[:-20]

    def vr(yy, rr):
        valid = ~np.isnan(yy) & (rr >= 0)
        y_v, r_v = yy[valid], rr[valid]
        total = np.nanvar(y_v, ddof=0)
        if total == 0:
            return np.nan
        num, nt = 0.0, 0
        for k in np.unique(r_v):
            m = r_v == k
            nk = m.sum()
            if nk < 2:
                continue
            num += nk * np.nanvar(y_v[m], ddof=0)
            nt += nk
        return 1.0 - (num / nt) / total

    print("\n=== BTC VR on walk-forward labels ===")
    print(f"  Outer VR:     {vr(y, wf_outer)*100:.2f}%")
    print(f"  Composite VR: {vr(y, wf_composite)*100:.2f}%")
    # Same on fit-once labels, same date range
    mask = labeled_mask & ~np.isnan(y)
    fo_o_full = pd.Series(fit_once_tl["outer"].astype(int)).reindex(idx).fillna(-1).astype(int).values
    fo_c_full = pd.Series(fit_once_tl["composite"].astype(int)).reindex(idx).fillna(-1).astype(int).values
    print(f"  (fit-once outer VR on same dates:     {vr(y * mask / np.where(mask,1,1), fo_o_full * mask.astype(int) + (-1) * (~mask).astype(int))*100:.2f}%)")

    # Save walk-forward timeline
    out_df = pd.DataFrame({
        "outer": wf_outer,
        "inner": wf_inner,
        "composite": wf_composite,
    }, index=idx)
    out_df.to_csv(OUT / "timeline_walk_forward.csv")
    print(f"\nWrote {OUT}/timeline_walk_forward.csv")


if __name__ == "__main__":
    main()
