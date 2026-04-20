# STATUS: LIVE — shared utilities module. DO NOT archive. Hosts:
#   COMMODITIES, load_regime_daily, load_long_history, build_features,
#   build_commodity_features, cross_sectional_rank_to_normal, summarize,
#   plot_regimes, load_btc_daily.
# Also contains the original flat-K=3 CS-rank HMM runner (superseded by
# hmm_hierarchical.py), kept for backwards compat with archived scripts.
"""HMM regime detector — Phase 1 (validation only, no leakage concern).

Fits a single K=3 Gaussian HMM on the full 2016-2026 macro feature history
and outputs regime labels + diagnostics so we can judge whether the learned
regimes are interpretable before integrating into v12.

Features (5) — commodity relative strength:
  - 20-day log returns for {gld, uso, cper, corn, soyb}
  - Cross-sectionally ranked each day (1=worst, 5=best among 5 commodities)

Transform pipeline:
  raw 20d log return → cross-sectional rank within the 5 commodities per day
                     → percentile [0,1] → inverse-normal (probit) → std normal

Note: ranks of 5 assets per day sum to a constant (15), so 5 features have
4 degrees of freedom. This is fine for HMM fit, noted for interpretation.

Why single-HMM here:
  HMM is unsupervised and nothing downstream predicts forward BTC returns
  using these labels yet. No leakage. Walk-forward comes in Phase 2.

Outputs (to stat_test/):
  hmm_regime_timeline.csv  — date, regime, P(regime=k), raw features
  hmm_regime_summary.txt   — transition matrix, per-regime means, durations
  hmm_regime_plot.png      — regime bands + BTC price overlay
"""

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

K_LIST = [2, 3, 4]
RETURN_WINDOW = 20  # 20 trading days for commodity log returns
COMMODITIES = ["gld", "uso", "cper", "corn", "soyb"]
SEED = 42


def load_regime_daily() -> pd.DataFrame:
    df = pd.read_csv(DATA / "regime_daily.csv", parse_dates=["date_utc"])
    df = df.set_index("date_utc").sort_index()
    df = df.ffill().dropna()
    return df


def load_long_history() -> pd.DataFrame:
    """Like load_regime_daily but does NOT drop NaN — needed for the extended
    1990+ rate history where commodities are NaN early on."""
    df = pd.read_csv(DATA / "regime_daily.csv", parse_dates=["date_utc"])
    return df.set_index("date_utc").sort_index()


def build_commodity_features(df: pd.DataFrame,
                             delta: int = 14,
                             zbase: int = 252) -> pd.DataFrame:
    """14d log ret → 252d rolling z → CS rank → inv-normal, drop last col.

    Used by the hierarchical HMM as the inner-layer feature block.
    """
    c = df[COMMODITIES].copy().ffill()
    z = pd.DataFrame(index=c.index)
    for col in COMMODITIES:
        r = np.log(c[col] / c[col].shift(delta))
        mu = r.rolling(zbase, min_periods=zbase).mean()
        sd = r.rolling(zbase, min_periods=zbase).std(ddof=0)
        z[f"{col}_z"] = (r - mu) / sd
    z = z.dropna()
    ranks = z.rank(axis=1, method="average")
    pct = ranks / (len(COMMODITIES) + 1)
    cs = pd.DataFrame(norm.ppf(pct.values), index=z.index, columns=z.columns)
    return cs.iloc[:, :-1]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """20d log returns for each commodity."""
    f = pd.DataFrame(index=df.index)
    for c in COMMODITIES:
        f[f"{c}_ret20d"] = np.log(df[c] / df[c].shift(RETURN_WINDOW))
    return f.dropna()


def cross_sectional_rank_to_normal(f: pd.DataFrame) -> pd.DataFrame:
    """Rank commodities against each other each day, map to inv-normal.

    Ranks are in {1..N}; percentile = rank/(N+1) keeps values strictly in (0,1)
    so probit is finite.
    """
    n = len(f.columns)  # 5
    # pandas rank axis=1 handles ties with 'average' by default
    ranks = f.rank(axis=1, method="average")
    pct = ranks / (n + 1)  # maps {1..5} → {1/6..5/6}, all in (0,1)
    out = pd.DataFrame(norm.ppf(pct.values), index=f.index, columns=f.columns)
    return out


def fit_hmm(X: np.ndarray, k: int) -> GaussianHMM:
    model = GaussianHMM(n_components=k, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=SEED)
    model.fit(X)
    return model


def summarize(model: GaussianHMM, feats: pd.DataFrame, labels: np.ndarray,
              raw: pd.DataFrame) -> str:
    K = model.n_components
    lines = []
    lines.append(f"K = {K}, features = {list(feats.columns)}")
    lines.append(f"obs = {len(feats)}, range = {feats.index[0].date()} → {feats.index[-1].date()}")
    lines.append(f"converged = {model.monitor_.converged}, iters = {model.monitor_.iter}")

    lines.append("\n=== Transition matrix (row = from, col = to) ===")
    tm = model.transmat_
    lines.append("       " + "  ".join([f"  r{k}  " for k in range(K)]))
    for i in range(K):
        lines.append(f"  r{i}   " + "  ".join([f"{tm[i, j]:.3f}" for j in range(K)]))

    lines.append("\n=== Expected duration per regime (days) ===")
    for k in range(K):
        p_stay = tm[k, k]
        dur = 1.0 / max(1e-9, 1.0 - p_stay)
        lines.append(f"  r{k}: {dur:.1f} days (p_stay = {p_stay:.3f})")

    lines.append("\n=== Count / frequency per regime ===")
    for k in range(K):
        n = int((labels == k).sum())
        lines.append(f"  r{k}: {n} days ({100.0 * n / len(labels):.1f}%)")

    lines.append("\n=== Mean of TRANSFORMED features per regime (Gaussian emission means) ===")
    cols = list(feats.columns)
    lines.append("       " + "  ".join([f"{c:>14s}" for c in cols]))
    for k in range(K):
        mu = model.means_[k]
        lines.append(f"  r{k}   " + "  ".join([f"{v:>14.3f}" for v in mu]))

    lines.append("\n=== Mean of RAW features per regime (for interpretability) ===")
    raw_aligned = raw.loc[feats.index]
    lines.append("       " + "  ".join([f"{c:>14s}" for c in raw_aligned.columns]))
    for k in range(K):
        mask = labels == k
        mu = raw_aligned[mask].mean()
        lines.append(f"  r{k}   " + "  ".join([f"{v:>14.4f}" for v in mu.values]))

    return "\n".join(lines)


def plot_regimes(labels: np.ndarray, dates: pd.DatetimeIndex,
                 btc_daily: pd.Series, k_val: int, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(btc_daily.index, btc_daily.values, color="black", lw=0.8, label="BTC")
    ax.set_yscale("log")
    ax.set_ylabel("BTC (log)")

    base_palette = ["#5EA0DB", "#E8C547", "#D9534F", "#6CBF6C", "#B68DDB",
                    "#FF8C42", "#7FC8A9", "#C97B84", "#4C6EF5", "#868E96"]
    if k_val <= len(base_palette):
        palette = base_palette[:k_val]
    else:
        import matplotlib.pyplot as _plt
        cmap = _plt.get_cmap("tab20")
        palette = [cmap(i % 20) for i in range(k_val)]
    label_series = pd.Series(labels, index=dates)
    for k in range(k_val):
        mask = label_series == k
        ymin, ymax = ax.get_ylim()
        ax.fill_between(label_series.index, ymin, ymax,
                        where=mask.values, color=palette[k], alpha=0.20,
                        label=f"r{k}", step="post")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(which="major", axis="x", alpha=0.5, lw=0.8)
    ax.grid(which="minor", axis="x", alpha=0.2, lw=0.4)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3)

    ax.legend(loc="upper left")
    ax.set_title(f"HMM K={k_val} regimes over BTC price")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_btc_daily(btc_path: Path, start, end) -> pd.Series:
    btc = pd.read_csv(btc_path, parse_dates=["date"], usecols=["date", "close"])
    btc = btc.set_index("date").sort_index()
    if btc.index.tz is None:
        btc.index = btc.index.tz_localize("UTC")
    daily = btc["close"].resample("D").last().ffill()
    return daily.loc[start:end]


def main():
    print("Loading regime_daily.csv...")
    raw = load_regime_daily()
    print(f"  rows: {len(raw)}, range: {raw.index[0].date()} → {raw.index[-1].date()}")

    print("Building raw features...")
    feats_raw = build_features(raw)
    print(f"  features: {list(feats_raw.columns)}, rows: {len(feats_raw)}")

    print("Cross-sectional rank → inverse-normal...")
    feats_full = cross_sectional_rank_to_normal(feats_raw)
    # Ranks are a permutation → features sum to 0 exactly → full covariance
    # is rank-deficient. Drop the last column (implied by the others) for HMM
    # input. No info loss; the dropped column is visible in the summary.
    feats = feats_full.iloc[:, :-1]
    print(f"  rows: {len(feats)}, HMM input dims: {feats.shape[1]} "
          f"(dropped {feats_full.columns[-1]} — implied by rank sum)")

    X = feats.values
    try:
        btc_daily = load_btc_daily(DATA / "BTC_1h_RELVP.csv",
                                   feats.index[0], feats.index[-1])
    except Exception as e:
        btc_daily = None
        print(f"BTC load failed ({e}); plots will be skipped.")

    for k in K_LIST:
        print(f"\n==== Fitting GaussianHMM (K={k}, full cov, seed={SEED}) ====")
        model = fit_hmm(X, k)
        labels = model.predict(X)
        posts = model.predict_proba(X)

        out_dir = RESULTS / f"cs_rank_k{k}"
        out_dir.mkdir(parents=True, exist_ok=True)

        timeline = feats_raw.loc[feats.index].copy()
        timeline["regime"] = labels
        for j in range(k):
            timeline[f"p_r{j}"] = posts[:, j]
        timeline.to_csv(out_dir / "timeline.csv")

        summary = summarize(model, feats, labels, feats_raw)
        (out_dir / "summary.txt").write_text(summary)
        print(summary)

        if btc_daily is not None:
            plot_regimes(labels, feats.index, btc_daily, k, out_dir / "plot.png")
        print(f"Wrote {out_dir}/")


if __name__ == "__main__":
    main()
