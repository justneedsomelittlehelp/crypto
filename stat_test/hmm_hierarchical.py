# STATUS: SUPERSEDED (2026-04-20) — see hmm_hierarchical_global.py for the new
# canonical HMM. This script is kept live because (a) the previous frozen run
# lives at results/hierarchical/, (b) hmm_compare_frozen_vs_global.py imports
# build_rate_levels from here for apples-to-apples comparison.
# Previous frozen config: --k-outer 2 --k-inner 3,2. See HMM_LOG.md.
"""Hierarchical HMM: slow rate-level outer × fast commodity-CS-rank inner.

Outer (slow, structural macro):
  features: FFR, DGS2, yield_curve  (levels)
  transform: expanding z-score with 1990+ baseline, 252d warmup
  K_outer: fit both 2 and 3, report

Inner (fast, tactical commodity flows):
  features: gld, uso, cper, corn  (dropped soyb; rank-sum)
  transform: 14d log ret → 252d rolling z → CS rank → inv-normal
  K_inner = 3, fit independently inside each outer state

Composite regime at time t = (S_t, z_t).

Validation: overlay outer label on NQ=F, ES=F, BTC; per-regime fwd 20d ret
for composite labels.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from hmm_regime import (
    load_long_history, build_commodity_features, plot_regimes, load_btc_daily,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "hierarchical"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1, 2, 7]
WARMUP = 252
RATE_COLS = ["ffr", "dgs2", "yield_curve"]


def expanding_zscore(s: pd.Series, warmup: int = WARMUP) -> pd.Series:
    mu = s.expanding(min_periods=warmup).mean()
    sd = s.expanding(min_periods=warmup).std(ddof=0)
    return (s - mu) / sd


def build_rate_levels(df: pd.DataFrame) -> pd.DataFrame:
    r = df[RATE_COLS].ffill().copy()
    out = pd.DataFrame(index=r.index)
    for c in RATE_COLS:
        out[f"{c}_zl"] = expanding_zscore(r[c])
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
        if x == cur: n += 1
        else:
            runs[cur].append(n); cur, n = x, 1
    runs[cur].append(n)
    return {i: np.mean(runs[i]) if runs[i] else 0 for i in range(k)}


def fetch_daily(ticker: str, start, end) -> pd.Series:
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
    p.add_argument("--k-inner", type=str, default="3",
                   help="int (same K per outer state) or comma list "
                        "e.g. '3,2' for K_inner_S0=3, K_inner_S1=2")
    args = p.parse_args()
    Ko = args.k_outer
    if "," in args.k_inner:
        Ki_list = [int(x) for x in args.k_inner.split(",")]
        assert len(Ki_list) == Ko, f"Need {Ko} K_inner values"
    else:
        Ki_list = [int(args.k_inner)] * Ko
    Ki = Ki_list  # list for downstream


    df = load_long_history()

    # ---- Outer ----
    rat = build_rate_levels(df)
    com = build_commodity_features(df)
    # Align both to common dates
    idx = rat.index.intersection(com.index)
    rat = rat.loc[idx]; com = com.loc[idx]
    Xr = rat.values
    Xc = com.values
    print(f"Panel: {len(idx)} days, {idx[0].date()} → {idx[-1].date()}")
    print(f"  outer feats: {list(rat.columns)}")
    print(f"  inner feats: {list(com.columns)}")

    print(f"\n=== Outer HMM (rate levels, K={Ko}) ===")
    ll_o, m_o, seed_o = fit_best(Xr, Ko)
    outer = m_o.predict(Xr)
    dwell_o = mean_dwell(outer, Ko)
    print(f"  seed={seed_o}, LL={ll_o:.1f}")
    print(f"  means (ffr_zl, dgs2_zl, yc_zl):")
    for k in range(Ko):
        mu = m_o.means_[k]
        print(f"    S{k}: ffr={mu[0]:+.2f} dgs2={mu[1]:+.2f} yc={mu[2]:+.2f}  "
              f"freq={int((outer==k).sum())}  dwell={dwell_o[k]:.0f}d")

    # ---- Inner (one HMM per outer state) ----
    print(f"\n=== Inner HMMs (commodity CS ranks, K_inner={Ki_list}) ===")
    inner = np.full(len(idx), -1, dtype=int)
    inner_models = {}
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
        inner_models[s] = m_i
        dwell_i = mean_dwell(lab, Ks)
        print(f"  S{s} (K={Ks}): seed={seed_i}, LL={ll_i:.1f}, n={len(Xs)}, "
              f"means(gld): {[f'{m_i.means_[order[k],0]:+.2f}' for k in range(Ks)]}, "
              f"dwell: {[f'{dwell_i[k]:.0f}' for k in range(Ks)]}")

    # Composite label: concat unique across outer states.
    # Numbering: S0 gets [0 .. Ki_list[0]-1], S1 gets [Ki_list[0] .. Ki_list[0]+Ki_list[1]-1], etc.
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

    # Outer plots (Ko regimes)
    plot_regimes(outer, dates, nq,  Ko, OUT / f"outer_K{Ko}_nq.png")
    plot_regimes(outer, dates, es,  Ko, OUT / f"outer_K{Ko}_es.png")
    plot_regimes(outer, dates, btc, Ko, OUT / f"outer_K{Ko}_btc.png")
    # Composite plots
    plot_regimes(composite, dates, nq,  Kc, OUT / f"composite_nq.png")
    plot_regimes(composite, dates, es,  Kc, OUT / f"composite_es.png")
    plot_regimes(composite, dates, btc, Kc, OUT / f"composite_btc.png")
    print(f"  wrote outer + composite plots to {OUT}/")

    # Per-regime fwd 20d log return
    print("\n=== Per-regime forward 20d log return ===")
    lines = [f"Hierarchical HMM: outer(rates, K={Ko}) × inner(commodities, K={Ki})\n"
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
                if n == 0: continue
                mu = fwd[mask].mean()
                lines.append(f"  S{s},z{z}: fwd20d = {mu:+.4f}  (n={n})")

    out = "\n".join(lines)
    print(out)
    (OUT / "stats.txt").write_text(out)

    # Save timeline
    tl = pd.DataFrame({
        "outer": outer,
        "inner": inner,
        "composite": composite,
    }, index=dates)
    tl.to_csv(OUT / "timeline.csv")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
