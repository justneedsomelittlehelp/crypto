# STATUS: ARCHIVED (2026-04-20) — SPY/QQQ overlay for early CS-rank model.
# Going forward we validate against futures (NQ=F, ES=F) inside hmm_quality.py.
# See HMM_LOG.md §7.
"""Validate CS-rank seed=1 HMM regimes against SP500 and NASDAQ-100.

Refits the cross-sectional rank HMM (K=3, seed=1) — same model used for BTC
inspection — then overlays regime bands on SPY and QQQ daily closes.

Also computes per-regime mean forward 20d return for each index, to see
whether the oil-surge regime (r2) looks bearish for equities too.

Outputs:
  hmm_regime_plot_spy.png
  hmm_regime_plot_qqq.png
  hmm_regime_equity_stats.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from hmm_regime import (
    load_regime_daily, build_features, cross_sectional_rank_to_normal,
    plot_regimes,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results" / "cs_rank_equities"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
SEED = 1


def fetch_daily(ticker: str, start, end) -> pd.Series:
    cache = DATA / f"{ticker.lower().replace('^','')}_daily.csv"
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
    print("Building features + fitting HMM (seed=1)...")
    raw = load_regime_daily()
    feats_raw = build_features(raw)
    feats_full = cross_sectional_rank_to_normal(feats_raw)
    feats = feats_full.iloc[:, :-1]
    X = feats.values

    m = GaussianHMM(n_components=K, covariance_type="full",
                    n_iter=500, tol=1e-4, random_state=SEED)
    m.fit(X)
    labels = m.predict(X)
    print(f"  LL = {m.score(X):.2f}, regime freq = "
          f"{[int((labels==k).sum()) for k in range(K)]}")

    dates = feats.index
    start, end = dates[0], dates[-1]

    print("\nLoading equity indices...")
    spy = fetch_daily("SPY", start, end).resample("D").last().ffill()
    qqq = fetch_daily("QQQ", start, end).resample("D").last().ffill()
    spy = spy.loc[start:end]
    qqq = qqq.loc[start:end]

    plot_regimes(labels, dates, spy, K, OUT / "plot_spy.png")
    plot_regimes(labels, dates, qqq, K, OUT / "plot_qqq.png")
    print(f"  wrote {OUT}/plot_spy.png, plot_qqq.png")

    # Per-regime forward 20d log return stats
    print("\nPer-regime forward 20d log return stats:")
    lines = ["Per-regime forward 20d log return (mean / count)\n"]
    for name, series in [("SPY", spy), ("QQQ", qqq), ("BTC", None)]:
        if name == "BTC":
            from hmm_regime import load_btc_daily
            btc = load_btc_daily(DATA / "BTC_1h_RELVP.csv", start, end)
            series = btc
        # align to dates, compute forward 20d log return
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
    (OUT / "equity_stats.txt").write_text(out)


if __name__ == "__main__":
    main()
