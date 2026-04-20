# STATUS: LIVE — generic quality diagnostic tool (any timeline.csv + regime column).
# Used against results/hierarchical/timeline.csv to produce quality_{outer,composite}.txt.
"""Quality diagnostics for a regime timeline.

Input: a timeline.csv with at least a date index + one regime-label column.
Evaluates how informative the labels are for predicting forward returns on
BTC / NQ=F / ES=F.

Metrics:
  - One-way ANOVA F / p on fwd20d return by regime
  - Variance reduction: 1 - E[Var(y|R)] / Var(y)
  - Bootstrap 95% CI on per-regime mean fwd20d (resample-by-label, B=2000)
  - Naive strategy: long when regime == r, else flat.
    Reports per-regime Sharpe + CAGR vs buy-and-hold.
  - Regime separation: pairwise Welch t-tests (Bonferroni-adjusted)

Usage:
  python3 hmm_quality.py --timeline results/hierarchical/timeline.csv --col outer
  python3 hmm_quality.py --timeline results/hierarchical/timeline.csv --col composite
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hmm_hierarchical import fetch_daily
from hmm_regime import load_btc_daily

HERE = Path(__file__).parent
DATA = HERE / "data"

FWD = 20  # forward horizon, trading days of log return


def bootstrap_ci(vals: np.ndarray, B: int = 2000, alpha: float = 0.05,
                 rng=None) -> tuple:
    rng = rng or np.random.default_rng(0)
    n = len(vals)
    idx = rng.integers(0, n, size=(B, n))
    means = vals[idx].mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return lo, hi


def variance_reduction(y: np.ndarray, r: np.ndarray) -> float:
    total = np.nanvar(y, ddof=0)
    if total == 0:
        return 0.0
    num = 0.0
    n_total = 0
    for k in np.unique(r):
        mask = (r == k) & ~np.isnan(y)
        if mask.sum() == 0:
            continue
        num += mask.sum() * np.nanvar(y[mask], ddof=0)
        n_total += mask.sum()
    within = num / n_total
    return 1.0 - within / total


def naive_strategy(rets: np.ndarray, mask: np.ndarray) -> dict:
    """Long when mask True, flat otherwise. rets = daily log returns."""
    strat = np.where(mask, rets, 0.0)
    n = len(strat)
    if n == 0:
        return dict(cagr=np.nan, sharpe=np.nan, exposure=0.0)
    mean_d = strat.mean()
    std_d = strat.std(ddof=0)
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else np.nan
    cagr = np.exp(mean_d * 252) - 1.0
    return dict(cagr=cagr, sharpe=sharpe, exposure=mask.mean())


def evaluate(name: str, series: pd.Series, dates, regime: np.ndarray,
             lines: list):
    """Full evaluation for one asset."""
    aligned = series.reindex(dates, method="ffill")
    log_close = np.log(aligned.values)
    # Forward 20d log return target
    fwd = np.full(len(log_close), np.nan)
    fwd[:-FWD] = log_close[FWD:] - log_close[:-FWD]
    # Daily log return (for strategy sim)
    daily = np.diff(log_close, prepend=log_close[0])
    # NaN-safe: where close is NaN (asset not yet listed), skip that day
    daily = np.nan_to_num(daily, nan=0.0, posinf=0.0, neginf=0.0)
    valid_asset = ~np.isnan(log_close)

    valid = ~np.isnan(fwd)
    y = fwd[valid]
    r = regime[valid]

    lines.append(f"\n{'='*60}\n{name}\n{'='*60}")
    # ANOVA
    groups = [y[r == k] for k in np.unique(r) if (r == k).sum() > 1]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        lines.append(f"ANOVA F = {f:.2f}, p = {p:.2e}")
    # Variance reduction
    vr = variance_reduction(y, r)
    lines.append(f"Variance reduction: {vr*100:.2f}%   "
                 f"(total var y = {y.var():.5f})")

    # Per-regime bootstrap CI + bootstrap-of-zero test
    lines.append(f"\nPer-regime fwd{FWD}d mean (bootstrap 95% CI, B=2000):")
    rng = np.random.default_rng(42)
    regime_stats = {}
    for k in sorted(np.unique(r)):
        vals = y[r == k]
        if len(vals) < 30:
            lines.append(f"  r{int(k)}: n={len(vals)} — skip")
            continue
        mu = vals.mean()
        lo, hi = bootstrap_ci(vals, rng=rng)
        sig = "" if (lo < 0 < hi) else " ★"
        lines.append(f"  r{int(k)}: mean={mu:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]"
                     f"  n={len(vals)}{sig}")
        regime_stats[int(k)] = (mu, lo, hi, len(vals))

    # Pairwise Welch (Bonferroni)
    unique_ks = sorted(regime_stats.keys())
    n_pairs = len(unique_ks) * (len(unique_ks) - 1) // 2
    if n_pairs > 0:
        lines.append(f"\nPairwise Welch t-test on fwd{FWD}d means "
                     f"(Bonferroni α=0.05/{n_pairs}):")
        alpha_bf = 0.05 / n_pairs
        for i, ki in enumerate(unique_ks):
            for kj in unique_ks[i + 1:]:
                yi, yj = y[r == ki], y[r == kj]
                t, p = stats.ttest_ind(yi, yj, equal_var=False)
                sig = "★" if p < alpha_bf else " "
                lines.append(f"  r{ki} vs r{kj}: t={t:+.2f}  p={p:.2e}  {sig}")

    # Naive strategy (long when regime == k, else flat)
    lines.append(f"\nStrategy: long when regime == r, else flat:")
    lines.append(f"  (buy-and-hold reference)")
    bh = naive_strategy(daily[valid_asset], np.ones(valid_asset.sum(), dtype=bool))
    lines.append(f"  B&H:   CAGR={bh['cagr']*100:+6.2f}%  "
                 f"Sharpe={bh['sharpe']:+.2f}  exp=100%  "
                 f"(asset valid days={valid_asset.sum()})")
    for k in sorted(np.unique(regime)):
        mask = (regime == k) & valid_asset
        stats_k = naive_strategy(daily[valid_asset], mask[valid_asset])
        lines.append(f"  r{int(k):<2d}:   CAGR={stats_k['cagr']*100:+6.2f}%  "
                     f"Sharpe={stats_k['sharpe']:+.2f}  "
                     f"exp={stats_k['exposure']*100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeline", type=Path, required=True)
    p.add_argument("--col", type=str, required=True,
                   help="regime column name in timeline.csv")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tl = pd.read_csv(args.timeline, index_col=0, parse_dates=True)
    if tl.index.tz is None:
        tl.index = tl.index.tz_localize("UTC")
    regime = tl[args.col].values
    dates = tl.index
    n_regimes = int(np.unique(regime[~np.isnan(regime)]).max()) + 1

    lines = [f"Quality diagnostics for timeline: {args.timeline}",
             f"Regime column: {args.col}  "
             f"(unique values: {sorted(np.unique(regime))[:10]}{'...' if n_regimes > 10 else ''})",
             f"Panel: {dates[0].date()} → {dates[-1].date()}, n={len(dates)}",
             f"Forward horizon: {FWD} trading days, log return",
             "\n★ = CI excludes 0 / pairwise significantly different at Bonferroni-adjusted α"]

    start, end = dates[0], dates[-1]
    nq  = fetch_daily("NQ=F", start, end).resample("D").last().ffill().loc[start:end]
    es  = fetch_daily("ES=F", start, end).resample("D").last().ffill().loc[start:end]
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", start, end)

    for name, series in [("NQ=F", nq), ("ES=F", es), ("BTC", btc)]:
        evaluate(name, series, dates, regime, lines)

    out = "\n".join(lines)
    print(out)
    out_path = args.out or args.timeline.parent / f"quality_{args.col}.txt"
    Path(out_path).write_text(out)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
