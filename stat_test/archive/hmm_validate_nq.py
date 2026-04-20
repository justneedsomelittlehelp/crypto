# STATUS: ARCHIVED (2026-04-20) — one-off NQ overlay for the rate-change HMM.
# Functionality subsumed by hmm_quality.py which validates against NQ=F, ES=F,
# and BTC for any timeline. See HMM_LOG.md §10.
"""Validate CS-rank + 6m-rate-change HMM regimes against NQ futures.

Refits the current-best model (commodity CS ranks + expanding-rank 6m rate
changes, K=3, seed=7 historically best-LL) then overlays regime bands on
NQ=F daily close.

Also computes per-regime forward 20d log return for NQ, SPY, QQQ, BTC.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from hmm_cs_plus_rates import (
    load_long_history, build_commodity_features, build_rate_features,
)
from hmm_regime import plot_regimes, load_btc_daily

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "cs_plus_rate_changes_nq"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
SEEDS = [42, 0, 1, 2, 7]


def fetch_daily(ticker: str, start, end) -> pd.Series:
    cache = DATA / f"{ticker.lower().replace('^','').replace('=','')}_daily.csv"
    if cache.exists():
        s = pd.read_csv(cache, parse_dates=["date"]).set_index("date")["close"]
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s.loc[start:end]
    print(f"  fetching {ticker} from yfinance...")
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    s = df["Close"].rename("close")
    s.to_frame().reset_index().rename(columns={"Date": "date", "index": "date"}).to_csv(cache, index=False)
    return s


def main():
    print("Building features + fitting HMM (rate-mode=change, best of 5 seeds)...")
    df = load_long_history()
    com = build_commodity_features(df)
    rat = build_rate_features(df, "change")
    feats = com.join(rat, how="inner").dropna()
    X = feats.values
    print(f"  HMM input: {feats.shape}, "
          f"{feats.index[0].date()} → {feats.index[-1].date()}")

    best = None
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=s).fit(X)
        ll = m.score(X)
        print(f"  seed {s}: LL = {ll:,.2f}")
        if best is None or ll > best[0]:
            best = (ll, m, s)
    ll, m, seed = best
    labels = m.predict(X)
    print(f"  picked seed {seed}, LL={ll:,.2f}, freq = "
          f"{[int((labels==k).sum()) for k in range(K)]}")

    dates = feats.index
    start, end = dates[0], dates[-1]

    print("\nLoading NQ + equities...")
    nq  = fetch_daily("NQ=F", start, end).resample("D").last().ffill().loc[start:end]
    spy = fetch_daily("SPY",  start, end).resample("D").last().ffill().loc[start:end]
    qqq = fetch_daily("QQQ",  start, end).resample("D").last().ffill().loc[start:end]

    plot_regimes(labels, dates, nq,  K, OUT / "plot_nq.png")
    plot_regimes(labels, dates, spy, K, OUT / "plot_spy.png")
    plot_regimes(labels, dates, qqq, K, OUT / "plot_qqq.png")
    print(f"  wrote {OUT}/plot_nq.png, plot_spy.png, plot_qqq.png")

    print("\nPer-regime forward 20d log return:")
    lines = [f"HMM: commodity CS ranks + 6m rate-change ranks, K={K}, "
             f"best seed={seed}, LL={ll:,.2f}\n"
             "Per-regime forward 20d log return (mean / count)\n"]
    btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", start, end)
    for name, series in [("NQ", nq), ("SPY", spy), ("QQQ", qqq), ("BTC", btc)]:
        aligned = series.reindex(dates, method="ffill")
        fwd = np.log(aligned.shift(-20) / aligned)
        lines.append(f"\n{name}:")
        for k in range(K):
            mask = (labels == k) & fwd.notna()
            n = int(mask.sum())
            mu = fwd[mask].mean()
            lines.append(f"  r{k}: mean fwd20d = {mu:+.4f}  (n={n})")

    out = "\n".join(lines)
    print(out)
    (OUT / "stats.txt").write_text(out)


if __name__ == "__main__":
    main()
