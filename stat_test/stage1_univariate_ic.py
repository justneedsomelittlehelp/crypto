"""Stage-1 univariate feature screening.

Computes per-feature Spearman IC vs forward returns at multiple horizons,
reports monthly IC mean + t-stat + hit rate. Pandas-only, runs in seconds.

Goal: rank candidate features cheaply before committing compute to ablation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
BTC_CSV = ROOT / "BTC_1h_RELVP.csv"
FUND_CSV = ROOT / "data" / "funding_rate_merged.csv"
FGI_CSV = ROOT / "data" / "fear_greed_index.csv"
REGIME_H_CSV = ROOT / "data" / "regime_hourly.csv"
REGIME_D_CSV = ROOT / "data" / "regime_daily.csv"

HORIZONS_H = [4, 24, 96, 168]  # 4h, 1d, 4d, 1w


def load_data():
    df = pd.read_csv(BTC_CSV, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    try:
        fund = pd.read_csv(FUND_CSV, parse_dates=["date"]).set_index("date").sort_index()
        fund.index = fund.index.tz_localize("UTC") if fund.index.tz is None else fund.index
        df = df.join(fund["funding_rate"].reindex(df.index, method="ffill"))
    except Exception as e:
        print(f"[warn] funding skipped: {e}")

    try:
        fgi = pd.read_csv(FGI_CSV, parse_dates=["date"]).set_index("date").sort_index()
        fgi.index = fgi.index.tz_localize("UTC") if fgi.index.tz is None else fgi.index
        df = df.join(fgi["fgi_value"].reindex(df.index, method="ffill"))
    except Exception as e:
        print(f"[warn] fgi skipped: {e}")

    try:
        rh = pd.read_csv(REGIME_H_CSV, parse_dates=["datetime_utc"]).set_index("datetime_utc").sort_index()
        rh = rh.reindex(df.index, method="ffill")
        for col in ["vix", "dxy", "gld", "uso"]:
            df[col] = rh[col]
    except Exception as e:
        print(f"[warn] regime hourly skipped: {e}")

    try:
        rd = pd.read_csv(REGIME_D_CSV, parse_dates=["date_utc"]).set_index("date_utc").sort_index()
        rd = rd.reindex(df.index, method="ffill")
        for col in ["ffr", "yield_curve"]:
            df[col] = rd[col]
    except Exception as e:
        print(f"[warn] regime daily skipped: {e}")

    return df


def build_features(df):
    f = pd.DataFrame(index=df.index)
    logc = np.log(df["close"])

    # --- Price/return momentum ---
    for w in [1, 4, 24, 72, 168]:
        f[f"ret_{w}h"] = logc.diff(w)

    # --- Realized vol ---
    r1 = logc.diff(1)
    for w in [24, 72, 168]:
        f[f"rvol_{w}h"] = r1.rolling(w).std()

    # --- Volume features ---
    logv = np.log(df["volume_1h"].replace(0, np.nan))
    f["logvol"] = logv
    f["logvol_z_168"] = (logv - logv.rolling(168).mean()) / logv.rolling(168).std()
    f["vol_surge_24"] = df["volume_1h"] / df["volume_1h"].rolling(24).mean()

    # --- Candle geometry ---
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_ratio"] = body / rng
    f["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    f["ret_1h"] = r1

    # --- VP-derived (from 50 bins, bin 25 = current price) ---
    vp_cols = [f"vp_rel_{i:02d}" for i in range(50)]
    vp = df[vp_cols].values  # (T, 50)

    f["vp_cur_weight"] = vp[:, 25]
    f["vp_max_weight"] = vp.max(axis=1)
    # Peak position relative to current price (bin 25)
    peak_idx = vp.argmax(axis=1)
    f["vp_peak_offset"] = peak_idx - 25
    # Skew: weight above vs below current
    f["vp_above_sum"] = vp[:, 26:].sum(axis=1)
    f["vp_below_sum"] = vp[:, :25].sum(axis=1)
    f["vp_skew"] = f["vp_above_sum"] - f["vp_below_sum"]
    # Concentration (HHI-style, higher = tighter node)
    f["vp_hhi"] = (vp ** 2).sum(axis=1)
    # Distance to nearest high-volume node above/below
    # (bin index where cumulative weight above/below hits 40% of that side's total)
    for name, side in [("ceiling_dist", slice(26, 50)), ("floor_dist", slice(0, 25))]:
        side_vals = vp[:, side]
        total = side_vals.sum(axis=1, keepdims=True) + 1e-9
        cum = side_vals.cumsum(axis=1) / total
        if name == "ceiling_dist":
            idx = (cum >= 0.4).argmax(axis=1)  # 0..23
            f[name] = idx + 1  # bins above current
        else:
            # reverse: start from current and walk down
            rev = side_vals[:, ::-1]
            cum_r = rev.cumsum(axis=1) / total
            idx = (cum_r >= 0.4).argmax(axis=1)
            f[name] = idx + 1

    # --- Funding rate ---
    if "funding_rate" in df.columns:
        f["funding"] = df["funding_rate"]
        f["funding_z_168"] = (df["funding_rate"] - df["funding_rate"].rolling(168).mean()) / df["funding_rate"].rolling(168).std()

    # --- FGI ---
    if "fgi_value" in df.columns:
        f["fgi"] = df["fgi_value"]
        f["fgi_delta_7d"] = df["fgi_value"].diff(168)

    # --- Regime: VIX / DXY / GLD / USO (hourly) ---
    if "vix" in df.columns:
        f["vix"] = df["vix"]
        f["vix_chg_24h"] = df["vix"].diff(24)
        f["vix_chg_168h"] = df["vix"].diff(168)
        f["vix_z_720h"] = (df["vix"] - df["vix"].rolling(720).mean()) / df["vix"].rolling(720).std()

    if "dxy" in df.columns:
        logdxy = np.log(df["dxy"])
        f["dxy_ret_24h"] = logdxy.diff(24)
        f["dxy_ret_168h"] = logdxy.diff(168)
        f["dxy_ret_720h"] = logdxy.diff(720)

    if "gld" in df.columns:
        loggld = np.log(df["gld"])
        f["gld_ret_168h"] = loggld.diff(168)
        f["gld_ret_720h"] = loggld.diff(720)

    if "uso" in df.columns:
        loguso = np.log(df["uso"])
        f["uso_ret_168h"] = loguso.diff(168)
        f["uso_ret_720h"] = loguso.diff(720)

    # Cross-asset: BTC-DXY rolling correlation regime
    if "dxy" in df.columns:
        btc_r = r1
        dxy_r = np.log(df["dxy"]).diff(1)
        f["btc_dxy_corr_720h"] = btc_r.rolling(720).corr(dxy_r)

    # --- Regime: FFR / yield curve (daily) ---
    if "ffr" in df.columns:
        f["ffr"] = df["ffr"]
        f["ffr_chg_30d"] = df["ffr"].diff(24 * 30)

    if "yield_curve" in df.columns:
        f["yield_curve"] = df["yield_curve"]
        f["yield_curve_chg_30d"] = df["yield_curve"].diff(24 * 30)

    # --- Time-of-day / day-of-week ---
    h = df.index.hour
    d = df.index.dayofweek
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    f["dow_sin"] = np.sin(2 * np.pi * d / 7)
    f["dow_cos"] = np.cos(2 * np.pi * d / 7)

    return f


def forward_returns(df, horizons):
    logc = np.log(df["close"])
    return pd.DataFrame({f"fwd_{h}h": logc.shift(-h) - logc for h in horizons}, index=df.index)


def monthly_ic(feat, target):
    """Compute monthly Spearman IC, return mean, std, t-stat, hit-rate."""
    df = pd.DataFrame({"f": feat, "t": target}).dropna()
    if len(df) < 500:
        return np.nan, np.nan, np.nan, np.nan, 0
    df["ym"] = df.index.to_period("M")
    ics = []
    for _, grp in df.groupby("ym"):
        if len(grp) < 30 or grp["f"].std() == 0 or grp["t"].std() == 0:
            continue
        rho, _ = spearmanr(grp["f"], grp["t"])
        if not np.isnan(rho):
            ics.append(rho)
    if len(ics) < 6:
        return np.nan, np.nan, np.nan, np.nan, len(ics)
    ics = np.array(ics)
    mean = ics.mean()
    std = ics.std(ddof=1)
    tstat = mean / (std / np.sqrt(len(ics)))
    hit = (np.sign(ics) == np.sign(mean)).mean()
    return mean, std, tstat, hit, len(ics)


def main():
    print("Loading data...")
    df = load_data()
    print(f"Rows: {len(df):,}  range: {df.index.min()} → {df.index.max()}")

    print("Building features...")
    feats = build_features(df)
    fwd = forward_returns(df, HORIZONS_H)

    # Exclude holdout (folds 11-12 per CLAUDE: data up to ~2026-04-08 end).
    # Use data through 2025-10-01 to keep analysis well before holdout.
    cutoff = pd.Timestamp("2025-10-01", tz="UTC")
    feats = feats[feats.index < cutoff]
    fwd = fwd[fwd.index < cutoff]
    print(f"In-sample rows: {len(feats):,}")

    results = []
    for fname in feats.columns:
        row = {"feature": fname}
        for hcol in fwd.columns:
            mean, std, tstat, hit, n = monthly_ic(feats[fname], fwd[hcol])
            row[f"{hcol}_IC"] = mean
            row[f"{hcol}_t"] = tstat
            row[f"{hcol}_hit"] = hit
        results.append(row)

    res = pd.DataFrame(results)

    print("\n" + "=" * 120)
    print("STAGE-1 UNIVARIATE IC (monthly Spearman, in-sample pre-2025-10)")
    print("=" * 120)
    for h in HORIZONS_H:
        col_ic = f"fwd_{h}h_IC"
        col_t = f"fwd_{h}h_t"
        col_hit = f"fwd_{h}h_hit"
        sub = res[["feature", col_ic, col_t, col_hit]].copy()
        sub = sub.dropna()
        sub["abs_t"] = sub[col_t].abs()
        sub = sub.sort_values("abs_t", ascending=False)
        print(f"\n--- Horizon: {h}h ---  (|t|>2 is interesting, |t|>3 strong)")
        print(f"{'feature':<22} {'IC':>8} {'t-stat':>8} {'hit%':>7}")
        for _, r in sub.head(30).iterrows():
            ic_val = r[col_ic]
            t_val = r[col_t]
            hit_val = r[col_hit]
            marker = "  ***" if abs(t_val) > 3 else ("  **" if abs(t_val) > 2 else "")
            print(f"{r['feature']:<22} {ic_val:>+8.4f} {t_val:>+8.2f} {hit_val*100:>6.1f}%{marker}")

    out = ROOT / "stat_test" / "stage1_univariate_ic_results.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved full results → {out}")


if __name__ == "__main__":
    main()
