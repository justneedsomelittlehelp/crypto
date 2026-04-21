# STATUS: LIVE — canonical HMM macro-regime model (frozen 2026-04-20).
# Frozen config: --k-outer 3 --k-inner 5,5,5. Results: results/hierarchical_global_K3_555/.
# Supersedes hmm_hierarchical.py (frozen 2026-04-20 morning, superseded same day).
# See HMM_LOG.md for the redesign motivation and head-to-head metrics.
"""Hierarchical HMM — global outer × commodity-CS-rank inner.

Outer (slow, structural global macro) — NEW:
  features: dxy, credit_spread, net_fed_liq_yoy, global_m2_yoy
  transform: 3y rolling rank → inverse-normal (Van der Waerden scores)
  source: data/regime_global_outer.csv (built by src/data/fetch_regime.py
          with --global-outer flag)

  Note: credit_spread = BAA10Y (Moody's Baa − 10Y Treasury) is a substitute
  for HY OAS — FRED truncated the BAML HY series in 2023 due to ICE
  licensing. BAA10Y captures the same credit-risk-premium regime signal
  with full 1986+ daily history.

Inner (fast, tactical commodity flows) — UNCHANGED from hmm_hierarchical.py:
  features: gld, uso, cper, corn  (soyb dropped by rank-sum)
  transform: 14d log ret → 252d rolling z → CS rank → inv-normal
  K_inner = list, fit independently inside each outer state.

Rationale for outer redesign:
  Previous outer (ffr, dgs2, yield_curve) is US-only and captures US
  monetary policy but misses global liquidity and risk appetite. This
  variant uses DXY (global USD), HY OAS (global risk-on/off),
  net Fed liquidity YoY (stationary proxy for effective US liquidity —
  note: still US-centric but represents the world's largest CB balance
  sheet), and G4 M2 YoY (equal-weight local-currency, captures collective
  CB expansion without FX-collinearity with DXY).

Window choice:
  3y rolling rank (756 trading days). Shorter than 5y so the current
  regime is less absorbed into the normalization; longer than 1-2y so
  within-regime adaptation doesn't flatten the signal.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from hmm_regime import (
    build_commodity_features, plot_regimes, load_btc_daily, load_long_history,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
PROJECT_ROOT = HERE.parent
DATA_PROJECT = PROJECT_ROOT / "data"
OUT = HERE / "results" / "hierarchical_global"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1, 2, 7]
RANK_WINDOW = 756  # 3 trading years
OUTER_COLS = ["dxy", "credit_spread", "net_fed_liq_yoy", "global_m2_yoy"]


def rolling_rank_inv_normal(s: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    """3y rolling percentile → inverse-normal (Van der Waerden scores).

    Uses rolling window; returns NaN until `window` observations available.
    Percentile is in (0, 1) by construction: for a window of size n, ranks
    are {1..n}, we divide by (n+1) to keep strictly inside (0,1) so ppf is
    finite even at extremes.
    """
    def _pct(arr):
        # Percentile of the last element within the window, using (rank)/(n+1)
        x = arr[-1]
        n = len(arr)
        # Rank with ties as average (matches scipy/pandas default behaviour)
        rank = (np.sum(arr < x) + 0.5 * np.sum(arr == x) + 0.5)
        return rank / (n + 1)

    pct = s.rolling(window, min_periods=window).apply(_pct, raw=True)
    return pd.Series(norm.ppf(pct.values), index=s.index, name=s.name)


def rolling_zscore(s: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    """3y rolling z-score. Preserves tail magnitudes (vs rank which flattens
    extremes to ~99th percentile). Better for detecting black-swan regimes.
    Does NOT clip — the whole point is to let extreme events stand out.
    """
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return ((s - mu) / sd).rename(s.name)


def build_outer_global(outer_path: Path, transform: str = "rank") -> pd.DataFrame:
    """Load regime_global_outer.csv and apply 3y rolling transform.

    transform:
      "rank"   — rank → inverse-normal (Van der Waerden); frozen default.
                 Robust to outliers, uniform distribution → Gaussian.
      "zscore" — raw rolling z-score. Preserves tail magnitudes; better for
                 black-swan regime detection but sensitive to outliers that
                 stay in the window (e.g., 2008 crisis distorts std for ~3y).
    """
    df = pd.read_csv(outer_path, parse_dates=["date_utc"]).set_index("date_utc")
    missing = [c for c in OUTER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"regime_global_outer.csv missing columns: {missing}")
    df = df[OUTER_COLS].sort_index().ffill()
    out = pd.DataFrame(index=df.index)
    if transform == "rank":
        for c in OUTER_COLS:
            out[f"{c}_rn"] = rolling_rank_inv_normal(df[c])
    elif transform == "zscore":
        for c in OUTER_COLS:
            out[f"{c}_z"] = rolling_zscore(df[c])
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return out.dropna()


def fit_best(X: np.ndarray, k: int, seeds=SEEDS) -> tuple:
    best = None
    for s in seeds:
        m = GaussianHMM(n_components=k, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=s).fit(X)
        ll = m.score(X)
        if best is None or ll > best[0]:
            best = (ll, m, s)
    return best


def mean_dwell(labels: np.ndarray, k: int):
    runs = {i: [] for i in range(k)}
    cur, n = labels[0], 1
    for x in labels[1:]:
        if x == cur:
            n += 1
        else:
            runs[cur].append(n)
            cur, n = x, 1
    runs[cur].append(n)
    return {i: np.mean(runs[i]) if runs[i] else 0 for i in range(k)}


def fetch_daily(ticker: str, start, end) -> pd.Series:
    import yfinance as yf
    safe = ticker.lower().replace("^", "").replace("=", "")
    cache = DATA / f"{safe}_daily.csv"
    if cache.exists():
        s = pd.read_csv(cache, parse_dates=["date"]).set_index("date")["close"]
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s.loc[start:end]
    print(f"  fetching {ticker}...")
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    s = df["Close"].rename("close")
    s.to_frame().reset_index().rename(columns={"Date": "date", "index": "date"}).to_csv(cache, index=False)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-outer", type=int, default=2)
    p.add_argument("--k-inner", type=str, default="3,2",
                   help="int (same K per outer state) or comma list "
                        "e.g. '3,2' for K_inner_S0=3, K_inner_S1=2")
    p.add_argument("--outer-csv", type=str,
                   default=str(DATA_PROJECT / "regime_global_outer.csv"),
                   help="Path to regime_global_outer.csv")
    p.add_argument("--outer-transform", choices=["rank", "zscore"], default="rank",
                   help="3y rolling transform for outer features: 'rank' "
                        "(rank→inv-normal, frozen default, robust) or "
                        "'zscore' (preserves tail magnitudes)")
    args = p.parse_args()

    Ko = args.k_outer
    if "," in args.k_inner:
        Ki_list = [int(x) for x in args.k_inner.split(",")]
        assert len(Ki_list) == Ko, f"Need {Ko} K_inner values"
    else:
        Ki_list = [int(args.k_inner)] * Ko

    # ---- Build outer (global macro, 3y rank→inv-normal) ----
    outer_path = Path(args.outer_csv)
    if not outer_path.exists():
        raise FileNotFoundError(
            f"{outer_path} not found. Run:\n"
            f"  python3 -m src.data.fetch_regime --global-outer --fred-key $FRED_KEY"
        )
    rat = build_outer_global(outer_path, transform=args.outer_transform)

    # ---- Build inner (commodity CS ranks, unchanged) ----
    df = load_long_history()
    com = build_commodity_features(df)

    # Align both to common dates
    idx = rat.index.intersection(com.index)
    rat = rat.loc[idx]
    com = com.loc[idx]
    Xr = rat.values
    Xc = com.values
    print(f"Panel: {len(idx)} days, {idx[0].date()} → {idx[-1].date()}")
    print(f"  outer feats: {list(rat.columns)}")
    print(f"  inner feats: {list(com.columns)}")

    print(f"\n=== Outer HMM (global macro rank→inv-normal, K={Ko}) ===")
    ll_o, m_o, seed_o = fit_best(Xr, Ko)
    outer = m_o.predict(Xr)
    dwell_o = mean_dwell(outer, Ko)
    print(f"  seed={seed_o}, LL={ll_o:.1f}")
    print(f"  means ({', '.join(rat.columns)}):")
    for k in range(Ko):
        mu = m_o.means_[k]
        mu_str = "  ".join([f"{c.replace('_rn',''):>10s}={mu[i]:+.2f}"
                            for i, c in enumerate(rat.columns)])
        print(f"    S{k}: {mu_str}  freq={int((outer==k).sum())}  dwell={dwell_o[k]:.0f}d")

    # ---- Inner (one HMM per outer state) ----
    print(f"\n=== Inner HMMs (commodity CS ranks, K_inner={Ki_list}) ===")
    inner = np.full(len(idx), -1, dtype=int)
    for s in range(Ko):
        Ks = Ki_list[s]
        mask = outer == s
        Xs = Xc[mask]
        if len(Xs) < 50 * Ks:
            print(f"  S{s}: only {len(Xs)} rows — skipping")
            inner[mask] = 0
            continue
        ll_i, m_i, seed_i = fit_best(Xs, Ks)
        lab = m_i.predict(Xs)
        order = np.argsort(m_i.means_[:, 0])
        relabel = {old: new for new, old in enumerate(order)}
        lab = np.array([relabel[x] for x in lab])
        inner[mask] = lab
        dwell_i = mean_dwell(lab, Ks)
        print(f"  S{s} (K={Ks}): seed={seed_i}, LL={ll_i:.1f}, n={len(Xs)}, "
              f"means(gld): {[f'{m_i.means_[order[k],0]:+.2f}' for k in range(Ks)]}, "
              f"dwell: {[f'{dwell_i[k]:.0f}' for k in range(Ks)]}")

    # Composite label: concat unique across outer states.
    offsets = np.cumsum([0] + Ki_list[:-1])
    composite = np.array([offsets[outer[i]] + inner[i] for i in range(len(idx))])
    Kc = sum(Ki_list)

    # ---- Validation overlays ----
    dates = idx
    start, end = dates[0], dates[-1]
    print("\n=== Loading futures + BTC ===")
    nq  = fetch_daily("NQ=F", start, end).resample("D").last().ffill().loc[start:end]
    es  = fetch_daily("ES=F", start, end).resample("D").last().ffill().loc[start:end]
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", start, end)

    plot_regimes(outer, dates, nq,  Ko, OUT / f"outer_K{Ko}_nq.png")
    plot_regimes(outer, dates, es,  Ko, OUT / f"outer_K{Ko}_es.png")
    plot_regimes(outer, dates, btc, Ko, OUT / f"outer_K{Ko}_btc.png")
    plot_regimes(composite, dates, nq,  Kc, OUT / f"composite_nq.png")
    plot_regimes(composite, dates, es,  Kc, OUT / f"composite_es.png")
    plot_regimes(composite, dates, btc, Kc, OUT / f"composite_btc.png")
    print(f"  wrote outer + composite plots to {OUT}/")

    # Per-regime fwd 20d log return
    print("\n=== Per-regime forward 20d log return ===")
    lines = [f"Hierarchical HMM (global outer): outer(global macro, K={Ko}) × "
             f"inner(commodities, K={Ki_list})\n"
             f"Outer features: {list(rat.columns)}\n"
             f"Rank window: {RANK_WINDOW} trading days (~3y)\n"
             f"Panel: {start.date()} → {end.date()}, n={len(dates)}\n"]

    for name, series in [("NQ=F", nq), ("ES=F", es), ("BTC", btc)]:
        aligned = series.reindex(dates, method="ffill")
        fwd = np.log(aligned.shift(-20) / aligned)
        lines.append(f"\n{name} — by OUTER state S:")
        for s in range(Ko):
            mask = (outer == s) & fwd.notna()
            mu = fwd[mask].mean()
            lines.append(f"  S{s}: mean fwd20d = {mu:+.4f}  (n={int(mask.sum())})")
        lines.append(f"{name} — by COMPOSITE (S, z):")
        for s in range(Ko):
            for z in range(Ki_list[s]):
                mask = (outer == s) & (inner == z) & fwd.notna()
                n = int(mask.sum())
                if n == 0:
                    continue
                mu = fwd[mask].mean()
                lines.append(f"  S{s},z{z}: fwd20d = {mu:+.4f}  (n={n})")

    out = "\n".join(lines)
    print(out)
    (OUT / "stats.txt").write_text(out)

    tl = pd.DataFrame({
        "outer": outer,
        "inner": inner,
        "composite": composite,
    }, index=dates)
    tl.to_csv(OUT / "timeline.csv")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
