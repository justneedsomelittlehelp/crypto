"""Post-hoc filter analysis on any v11 walk-forward prediction file.

Experimental playground edition. Beyond the v10-matched filter stack,
this sweeps:
  - inverted asymmetry filters (asym < 1, asym ∈ [0.3, 0.8], etc.) —
    rationale: the range-derived labels are structurally anti-correlated
    with the asymmetry ratio, so the trades the model actually predicts
    well may live on the opposite side of the filter v10 used.
  - sign-flip baseline: if pred==0 outperforms pred==1 the model is
    anti-aligned with truth rather than uncorrelated.
  - long/short symmetric coverage at the best filter for this model.

Loads a prediction .npz saved by eval_v11.py, applies each filter,
enforces 24h trade-cooldown, compounds per-trade returns at 1x / 20%
position size, and reports in-sample vs holdout split.

Usage:
    python3 -m src.models.analyze_v11_filters                # default 11 tag
    python3 -m src.models.analyze_v11_filters --tag 21
    python3 -m src.models.analyze_v11_filters --npz path/to/file.npz
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

HOLDOUT_START = pd.Timestamp("2025-07-01")
# 24h wall-clock pause = 96 15m bars. Implemented via date check, not bar
# count, so it's resolution-independent and survives fold boundaries.
PAUSE = pd.Timedelta(hours=24)

# Position sizing to match the v6-prime audited baseline (1x / 20%).
POSITION_SIZE = 0.20

# Filter sweep
CONF_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MIN_ASYMMETRY = 1.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def greedy_cooldown(dates: np.ndarray, mask: np.ndarray, pause: pd.Timedelta) -> np.ndarray:
    """Keep trades greedily: after accepting a trade at time t, drop any
    subsequent candidate whose date is within `pause` of t. Mirrors the
    '24h pause' used in the v6-prime audit — avoids counting the same
    signal as many trades because consecutive 15m bars are correlated.
    """
    kept = np.zeros(len(mask), dtype=bool)
    idx = np.where(mask)[0]
    last_accept = None
    for i in idx:
        t = dates[i]
        if last_accept is None or (t - last_accept) >= pause:
            kept[i] = True
            last_accept = t
    return kept


def compound_returns(wins: np.ndarray, tp_pct: np.ndarray, sl_pct: np.ndarray,
                     labels: np.ndarray, size: float) -> tuple[np.ndarray, float, float]:
    """Compound per-trade returns at fixed position size (1x leverage).

    Returns:
        equity curve, final equity multiple, max drawdown (fraction).
    """
    n = len(wins)
    equity = np.ones(n + 1)
    for k in range(n):
        if labels[k] == 1.0:
            r = tp_pct[k]
        elif labels[k] == 0.0:
            r = -sl_pct[k]
        else:
            r = 0.0
        equity[k + 1] = equity[k] * (1.0 + size * r)
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / peak
    return equity, float(equity[-1]), float(dd.max())


def annualize(final_mult: float, days: float) -> float:
    if days <= 0 or final_mult <= 0:
        return 0.0
    years = days / 365.25
    return float(final_mult ** (1.0 / years) - 1.0)


def run_filter(tag: str, mask_in: np.ndarray,
               dates, labels, tp_pct, sl_pct, size=POSITION_SIZE):
    """Apply cooldown + compound. Report acc, EV/trade, CAGR, DD."""
    if mask_in.sum() == 0:
        print(f"  {tag}: no trades")
        return None
    keep = greedy_cooldown(dates, mask_in, PAUSE)
    if keep.sum() == 0:
        print(f"  {tag}: no trades after cooldown")
        return None

    k_labels = labels[keep]
    k_tp = tp_pct[keep]
    k_sl = sl_pct[keep]
    k_dates = dates[keep]
    # Only count trades with a resolved label (not NaN).
    valid = ~np.isnan(k_labels)
    k_labels = k_labels[valid]
    k_tp = k_tp[valid]
    k_sl = k_sl[valid]
    k_dates = k_dates[valid]

    n = len(k_labels)
    if n == 0:
        print(f"  {tag}: no resolved trades")
        return None

    wins = (k_labels == 1.0).astype(np.int32)
    win_rate = wins.mean()
    ev = (k_tp[wins == 1].sum() - k_sl[wins == 0].sum()) / n * 100  # %/trade

    equity, final_mult, max_dd = compound_returns(
        wins, k_tp, k_sl, k_labels, size
    )
    days = (k_dates.max() - k_dates.min()).astype("timedelta64[D]").astype(int)
    cagr = annualize(final_mult, days)

    print(
        f"  {tag:<28} n={n:>5}  win={win_rate*100:5.1f}%  "
        f"EV/tr={ev:+6.2f}%  final={final_mult:5.2f}x  "
        f"DD={max_dd*100:5.1f}%  CAGR={cagr*100:+6.1f}%  "
        f"({days}d)"
    )
    return {
        "tag": tag,
        "n_trades": int(n),
        "win_rate": float(win_rate),
        "ev_per_trade_pct": float(ev),
        "final_mult": float(final_mult),
        "max_dd_pct": float(max_dd * 100),
        "cagr_pct": float(cagr * 100),
        "days": int(days),
    }


def resolve_npz(args) -> Path:
    if args.npz is not None:
        return Path(args.npz)
    # Back-compat: if the user saved a pre-parameterized file named
    # `v11_predictions.npz`, pick it up when --tag is not provided.
    candidates = [
        ROOT / "experiments" / f"v11_{args.tag}_predictions.npz",
        ROOT / "experiments" / "v11_predictions.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find a predictions file. Tried:\n  " +
        "\n  ".join(str(p) for p in candidates)
    )


def parse_args():
    ap = argparse.ArgumentParser(description="v11 filter analysis")
    ap.add_argument("--tag", default="11",
                    help="eval_v11 tag, used to find experiments/v11_{tag}_predictions.npz")
    ap.add_argument("--npz", default=None,
                    help="explicit path to a predictions .npz (overrides --tag)")
    return ap.parse_args()


def main():
    args = parse_args()
    npz_path = resolve_npz(args)
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    preds = d["preds"].astype(np.float64)
    labels = d["labels"].astype(np.float64)
    logits = d["logits"].astype(np.float64)
    dates = pd.to_datetime(d["dates"])
    tp_pct = d["tp_pct"].astype(np.float64)
    sl_pct = d["sl_pct"].astype(np.float64)
    n = len(preds)
    print(f"  {n:,} predictions, {dates.min()} → {dates.max()}")

    # Ensure sorted by date
    order = np.argsort(dates)
    preds = preds[order]
    labels = labels[order]
    logits = logits[order]
    dates = dates[order].values  # numpy datetime64
    tp_pct = tp_pct[order]
    sl_pct = sl_pct[order]

    # Split masks
    holdout_start = np.datetime64(HOLDOUT_START)
    in_sample = dates < holdout_start
    holdout = dates >= holdout_start
    print(f"  in-sample: {in_sample.sum():,}  holdout: {holdout.sum():,}")

    conf = sigmoid(logits)
    asym = np.where(sl_pct > 0, tp_pct / sl_pct, 0.0)

    # ── Quick descriptives on asym / label coupling ──
    print("\n── Label / asymmetry coupling (why filters invert) ──")
    valid = ~np.isnan(labels)
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 1e9)]:
        band = valid & (asym >= lo) & (asym < hi)
        if band.sum() == 0:
            continue
        pos_rate = labels[band].mean() * 100
        print(f"  asym ∈ [{lo:.1f}, {hi:.1f}): n={band.sum():>6,}  pos_rate={pos_rate:5.1f}%")

    # Baseline: unfiltered, long-only (pred == 1), no cooldown → apples to apples with eval summary
    print("\n── Unfiltered baseline (no cooldown, no filters) ──")
    long_mask = preds == 1.0
    run_filter("all / raw long",              long_mask,             dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / raw long",        long_mask & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / raw long",         long_mask & holdout,   dates, labels, tp_pct, sl_pct)

    # Sign-flip baseline: does the model outperform its own negation? If
    # pred==0 is better than pred==1, the model is anti-aligned.
    print("\n── Sign-flip baseline (pred == 0 treated as a long signal) ──")
    short_signal = preds == 0.0
    run_filter("all / pred=0",                short_signal,             dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / pred=0",          short_signal & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / pred=0",           short_signal & holdout,   dates, labels, tp_pct, sl_pct)

    # Cooldown only
    print("\n── With 24h cooldown, no conf/asymmetry filter ──")
    run_filter("all / cooldown",              long_mask,             dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / cooldown",        long_mask & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / cooldown",         long_mask & holdout,   dates, labels, tp_pct, sl_pct)

    # Asymmetry gate
    asym_mask = asym >= MIN_ASYMMETRY
    print(f"\n── 24h cooldown + asymmetry ≥ {MIN_ASYMMETRY} ──")
    run_filter("all / asym",                  long_mask & asym_mask,             dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / asym",            long_mask & asym_mask & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / asym",             long_mask & asym_mask & holdout,   dates, labels, tp_pct, sl_pct)

    # Confidence sweep with asymmetry + cooldown
    for T in CONF_THRESHOLDS:
        conf_mask = conf >= T
        combo = long_mask & conf_mask & asym_mask
        print(f"\n── conf ≥ {T:.2f} + asym ≥ {MIN_ASYMMETRY} + 24h cooldown ──")
        run_filter(f"all / conf_{int(T*100)}",       combo,             dates, labels, tp_pct, sl_pct)
        run_filter(f"in-sample / conf_{int(T*100)}", combo & in_sample, dates, labels, tp_pct, sl_pct)
        run_filter(f"holdout / conf_{int(T*100)}",   combo & holdout,   dates, labels, tp_pct, sl_pct)

    # ── Inverted asymmetry filters ──
    # The range-derived label formula makes labels anti-correlated with
    # asym: low asym → fat SL, skinny TP → TP hits more often → label=1.
    # The standard filter (asym ≥ 1) thus selects labels that skew 0.
    # These variants test whether the model scores its OWN signal on the
    # other side of that asymmetry split.
    print("\n── Inverted asymmetry: asym < 1.0 + 24h cooldown ──")
    inv_mask = asym < 1.0
    run_filter("all / asym<1",                long_mask & inv_mask,             dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / asym<1",          long_mask & inv_mask & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / asym<1",           long_mask & inv_mask & holdout,   dates, labels, tp_pct, sl_pct)

    print("\n── Inverted band: 0.3 ≤ asym < 0.8 + 24h cooldown ──")
    band_mask = (asym >= 0.3) & (asym < 0.8)
    run_filter("all / asym_band",             long_mask & band_mask,             dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / asym_band",       long_mask & band_mask & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / asym_band",        long_mask & band_mask & holdout,   dates, labels, tp_pct, sl_pct)

    print("\n── conf ≥ 0.70 + asym < 1.0 + 24h cooldown (inverted combo) ──")
    inv_combo = long_mask & (conf >= 0.70) & inv_mask
    run_filter("all / conf_70+asym<1",        inv_combo,                        dates, labels, tp_pct, sl_pct)
    run_filter("in-sample / conf_70+asym<1",  inv_combo & in_sample,            dates, labels, tp_pct, sl_pct)
    run_filter("holdout  / conf_70+asym<1",   inv_combo & holdout,              dates, labels, tp_pct, sl_pct)

    # v6-prime audited setup for direct comparison:
    print("\n" + "=" * 70)
    print("  DIRECT v10 AUDITED COMPARISON  (conf≥0.70 + asym≥1.0 + 24h pause + 1x/20%)")
    print(f"  v10 honest:  +6.0% CAGR / −18.4% DD / holdout ≈ −5% / 23 trades/yr")
    print("=" * 70)
    T = 0.70
    conf_mask = conf >= T
    combo = long_mask & conf_mask & asym_mask
    run_filter("v11 all",       combo,             dates, labels, tp_pct, sl_pct)
    run_filter("v11 in-sample", combo & in_sample, dates, labels, tp_pct, sl_pct)
    run_filter("v11 holdout",   combo & holdout,   dates, labels, tp_pct, sl_pct)


if __name__ == "__main__":
    main()
