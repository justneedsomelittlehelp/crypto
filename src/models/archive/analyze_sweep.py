"""Analyze TP/SL sweep results: trade frequency, duration, corrected EV, daily profit.

Now includes long+short analysis: when model predicts 1 → long, predicts 0 → short.
Short side flips TP/SL: short wins SL% when price drops, loses TP% when price rises.

Usage:
    python3 -m src.models.analyze_sweep
"""

import sys
import json
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from src.config import EXPERIMENTS_DIR, LABEL_MAX_BARS, LABEL_REGIME_ADAPTIVE, LABEL_REGIME_SMA_BARS
from src.features.pipeline import build_feature_matrix


ALL_CONFIGS = [
    # Sweep 1: fixed 3% TP, vary SL
    {"name": "3/3 (1:1)",     "tp": 0.03,  "sl": 0.03},
    {"name": "3/4 (1:1.33)",  "tp": 0.03,  "sl": 0.04},
    {"name": "3/5 (1:1.67)",  "tp": 0.03,  "sl": 0.05},
    {"name": "3/6 (1:2)",     "tp": 0.03,  "sl": 0.06},
    {"name": "3/7.5 (1:2.5)", "tp": 0.03,  "sl": 0.075},
    {"name": "3/9 (1:3)",     "tp": 0.03,  "sl": 0.09},
    # Sweep 2: fixed 3% SL, vary TP
    {"name": "4/3 (1.33:1)",  "tp": 0.04,  "sl": 0.03},
    {"name": "5/3 (1.67:1)",  "tp": 0.05,  "sl": 0.03},
    {"name": "6/3 (2:1)",     "tp": 0.06,  "sl": 0.03},
    {"name": "7.5/3 (2.5:1)", "tp": 0.075, "sl": 0.03},
    {"name": "9/3 (3:1)",     "tp": 0.09,  "sl": 0.03},
]


def load_sweep_results():
    lookup = {}
    for fname in ["sweep_tpsl_results.json", "sweep_tpsl_wide_results.json"]:
        path = EXPERIMENTS_DIR / fname
        if path.exists():
            with open(path) as f:
                for r in json.load(f):
                    lookup[r["name"]] = r
    return lookup


def derive_npv(accuracy, precision, base_rate):
    """Derive Negative Predictive Value (short-side win rate) from known metrics.

    Given: accuracy, precision, base_rate (P(label=1))
    Solve for: P(pred=1), then NPV = P(label=0 | pred=0).

    From Bayes:
      base_rate = p1 * precision + (1 - p1) * (1 - NPV)
      accuracy  = p1 * precision + (1 - p1) * NPV

    Adding these: base_rate + accuracy = 2 * p1 * precision + 1
    So: p1 = (accuracy + base_rate - 1) / (2 * precision - 1)
    Then: NPV = (accuracy - p1 * precision) / (1 - p1)
    """
    if abs(2 * precision - 1) < 1e-10:
        return 0.5  # model is random

    p1 = (accuracy + base_rate - 1) / (2 * precision - 1)
    p1 = np.clip(p1, 0.01, 0.99)

    npv = (accuracy - p1 * precision) / (1 - p1)
    npv = np.clip(npv, 0.0, 1.0)
    return float(npv), float(p1)


def analyze():
    print("Loading data...", flush=True)
    df = build_feature_matrix()
    close = df["close"].values
    n = len(close)
    print(f"Data: {n} rows ({n / 24:.0f} days)\n", flush=True)

    # Precompute rolling volatility
    vol_window = 30
    log_returns = np.diff(np.log(close))
    rolling_vol = np.full(n, np.nan)
    for i in range(vol_window, len(log_returns)):
        rolling_vol[i + 1] = np.std(log_returns[i - vol_window + 1 : i + 1])

    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    median_vol = np.median(valid_vol)

    # SMA for regime detection
    sma = np.full(n, np.nan)
    if LABEL_REGIME_ADAPTIVE:
        for i in range(LABEL_REGIME_SMA_BARS, n):
            sma[i] = np.mean(close[i - LABEL_REGIME_SMA_BARS : i])

    warmup = max(vol_window + 1, LABEL_REGIME_SMA_BARS if LABEL_REGIME_ADAPTIVE else 0)

    sweep_results = load_sweep_results()

    # ── LONG-ONLY TABLE ──
    print("=" * 100)
    print("  LONG-ONLY (pred=1 → go long, pred=0 → sit out)")
    print("=" * 100)
    print(f"{'Config':<18} {'Base%':>6} {'AvgDur':>7} {'Tr/d':>6} {'Prec':>6} "
          f"{'EV/tr':>8} {'DailyEV':>9} {'AnnEV%':>8}")
    print("-" * 85)

    all_rows = []

    for config in ALL_CONFIGS:
        tp_base, sl_base = config["tp"], config["sl"]
        name = config["name"]

        durations = []
        label_counts = {1: 0, 0: 0, "nan": 0}

        for i in range(warmup, n - 1):
            if np.isnan(rolling_vol[i]):
                continue

            vol_scale = rolling_vol[i] / median_vol
            vol_scale = max(0.5, min(vol_scale, 3.0))

            if LABEL_REGIME_ADAPTIVE and not np.isnan(sma[i]):
                if close[i] >= sma[i]:
                    tp_pct = tp_base * vol_scale
                    sl_pct = sl_base * vol_scale
                else:
                    tp_pct = sl_base * vol_scale
                    sl_pct = tp_base * vol_scale
            else:
                tp_pct = tp_base * vol_scale
                sl_pct = sl_base * vol_scale

            entry = close[i]
            tp_level = entry * (1 + tp_pct)
            sl_level = entry * (1 - sl_pct)
            max_j = min(i + LABEL_MAX_BARS, n)

            hit = False
            for j in range(i + 1, max_j):
                if close[j] >= tp_level:
                    durations.append(j - i)
                    label_counts[1] += 1
                    hit = True
                    break
                if close[j] <= sl_level:
                    durations.append(j - i)
                    label_counts[0] += 1
                    hit = True
                    break

            if not hit:
                label_counts["nan"] += 1

        total_labeled = label_counts[1] + label_counts[0]
        base_rate = label_counts[1] / total_labeled if total_labeled > 0 else 0
        avg_duration = np.mean(durations) if durations else float("nan")
        max_trades_day = 24 / avg_duration if avg_duration > 0 else 0

        sr = sweep_results.get(name)
        if sr:
            precision = sr["precision"]
            accuracy = sr["accuracy"]
        else:
            precision = base_rate
            accuracy = 0.5

        # Derive short-side win rate (NPV) and prediction rate
        npv_result = derive_npv(accuracy, precision, base_rate)
        npv, pred_long_rate = npv_result

        # Long EV
        ev_long = precision * tp_base * 100 - (1 - precision) * sl_base * 100

        # Short EV: short wins SL% when label=0 (price dropped), loses TP% when label=1
        ev_short = npv * sl_base * 100 - (1 - npv) * tp_base * 100

        # Combined EV per trade (weighted by prediction distribution)
        ev_combined = pred_long_rate * ev_long + (1 - pred_long_rate) * ev_short

        # Long-only: trade only when pred=1, constrained by position overlap
        long_trades_day = max_trades_day * pred_long_rate
        daily_ev_long = ev_long * long_trades_day

        # Long+Short: always in a position (long or short)
        combined_trades_day = max_trades_day  # always trading
        daily_ev_combined = ev_combined * combined_trades_day

        annual_long = daily_ev_long * 365
        annual_combined = daily_ev_combined * 365

        row = {
            "name": name,
            "tp": tp_base * 100,
            "sl": sl_base * 100,
            "base_rate": base_rate,
            "avg_duration_hours": avg_duration,
            "max_trades_day": max_trades_day,
            "pred_long_rate": pred_long_rate,
            "precision": precision,
            "npv": npv,
            "ev_long": ev_long,
            "ev_short": ev_short,
            "ev_combined": ev_combined,
            "long_trades_day": long_trades_day,
            "combined_trades_day": combined_trades_day,
            "daily_ev_long": daily_ev_long,
            "daily_ev_combined": daily_ev_combined,
            "annual_long": annual_long,
            "annual_combined": annual_combined,
        }
        all_rows.append(row)

        print(f"{name:<18} {base_rate*100:>5.1f}% {avg_duration:>6.1f}h {long_trades_day:>5.2f} "
              f"{precision:>6.3f} {ev_long:>+7.3f}% {daily_ev_long:>+8.3f}% {annual_long:>+8.1f}%")

    # ── SHORT-SIDE TABLE ──
    print(f"\n{'=' * 100}")
    print("  SHORT-SIDE (pred=0 → go short: win SL% on drop, lose TP% on rise)")
    print("=" * 100)
    print(f"{'Config':<18} {'NPV':>6} {'ShortEV':>9} {'ShortTr/d':>10} {'DailyEV':>9} {'AnnEV%':>8}")
    print("-" * 70)

    for row in all_rows:
        short_trades_day = row["max_trades_day"] * (1 - row["pred_long_rate"])
        daily_ev_short = row["ev_short"] * short_trades_day
        annual_short = daily_ev_short * 365
        print(f"{row['name']:<18} {row['npv']:>5.3f} {row['ev_short']:>+8.3f}% "
              f"{short_trades_day:>9.2f} {daily_ev_short:>+8.3f}% {annual_short:>+8.1f}%")

    # ── COMBINED TABLE ──
    print(f"\n{'=' * 100}")
    print("  COMBINED LONG+SHORT (always in a position)")
    print("=" * 100)
    print(f"{'Config':<18} {'Prec':>6} {'NPV':>6} {'EVlong':>8} {'EVshort':>9} "
          f"{'Tr/d':>6} {'DailyEV':>9} {'AnnEV%':>8}")
    print("-" * 85)

    for row in all_rows:
        print(f"{row['name']:<18} {row['precision']:>5.3f} {row['npv']:>5.3f} "
              f"{row['ev_long']:>+7.3f}% {row['ev_short']:>+8.3f}% "
              f"{row['combined_trades_day']:>5.2f} {row['daily_ev_combined']:>+8.3f}% "
              f"{row['annual_combined']:>+8.1f}%")

    # ── TOP 5 COMBINED ──
    print(f"\n{'=' * 100}")
    print("  TOP 5 BY COMBINED DAILY EV:")
    print("=" * 100)
    top5 = sorted(all_rows, key=lambda r: r["daily_ev_combined"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"  {i}. {r['name']:<18} DailyEV={r['daily_ev_combined']:>+.3f}%  "
              f"(Long: {r['ev_long']:>+.2f}% × {r['pred_long_rate']:.0%} | "
              f"Short: {r['ev_short']:>+.2f}% × {1-r['pred_long_rate']:.0%})  "
              f"Trades/day={r['combined_trades_day']:.2f}  Annual≈{r['annual_combined']:>+.0f}%")

    # ── COMPARISON: LONG-ONLY vs COMBINED ──
    print(f"\n{'=' * 100}")
    print("  LONG-ONLY vs LONG+SHORT COMPARISON:")
    print("=" * 100)
    print(f"{'Config':<18} {'Long DailyEV':>13} {'Comb DailyEV':>13} {'Improvement':>12}")
    print("-" * 60)
    for row in all_rows:
        diff = row["daily_ev_combined"] - row["daily_ev_long"]
        print(f"{row['name']:<18} {row['daily_ev_long']:>+12.3f}% {row['daily_ev_combined']:>+12.3f}% "
              f"{diff:>+11.3f}%")

    # Save
    out = EXPERIMENTS_DIR / "sweep_analysis_combined.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    analyze()
