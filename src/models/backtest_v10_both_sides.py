"""Short-overlay backtest for v10 predictions.

No retraining. Reads v10_predictions.npz + the close series from
BTC_1h_RELVP.csv (any VP lookback works — close prices are identical),
re-runs first-hit with mirror short thresholds, and reports:

  - Long-only EV (sanity check — should match eval_v10_results.json)
  - Short-only EV (unconditional + threshold sweep)
  - Regime-aware both-sides (long in bull, short in bear)
  - Per-scope: full walk-forward vs holdout-only (post 2025-07-01)

Mirror rule (the user's framing):
  short_tp_level = entry × (1 - long_sl_pct)   # profit on drop
  short_sl_level = entry × (1 + long_tp_pct)   # stop on rise
  i.e., the long's SL distance becomes the short's target, and the
  long's TP distance becomes the short's stop.

Usage:
    python3 -m src.models.backtest_v10_both_sides

This is a diagnostic, not a production backtest — it does not model
capital, leverage, fees, pauses, or slippage. The outputs are per-trade
EV and trade counts, which is the right abstraction to judge whether
the direction is worth a full backtest-engine pass.
"""

import json
import sys
import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_DIR, PROJECT_ROOT, LABEL_MAX_BARS

PREDICTIONS_PATH = EXPERIMENTS_DIR / "v10_predictions.npz"
CSV_PATH = PROJECT_ROOT / "BTC_1h_RELVP.csv"
RESULTS_PATH = EXPERIMENTS_DIR / "backtest_v10_both_sides.json"

HOLDOUT_START = pd.Timestamp("2025-07-01", tz="UTC")
MAX_BARS = LABEL_MAX_BARS  # 336


def compute_short_labels(close_full, pred_row_idx, long_tp, long_sl):
    """Mirror-short first-hit simulation.

    Returns: (labels, tp_pct_for_short, sl_pct_for_short)
      - labels[i] in {0, 1, nan}: 1 if short TP hit first, 0 if short SL first
      - short wins pay long_sl[i] %; short losses cost long_tp[i] %
    """
    n = len(pred_row_idx)
    labels = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        idx = pred_row_idx[i]
        tp = long_tp[i]  # long TP pct → becomes short SL
        sl = long_sl[i]  # long SL pct → becomes short TP
        if np.isnan(tp) or np.isnan(sl):
            continue

        entry = close_full[idx]
        short_tp_level = entry * (1 - sl)
        short_sl_level = entry * (1 + tp)

        end = min(idx + MAX_BARS + 1, len(close_full))
        future = close_full[idx + 1:end]
        if len(future) == 0:
            continue

        tp_hits = future <= short_tp_level
        sl_hits = future >= short_sl_level
        tp_first = np.argmax(tp_hits) if tp_hits.any() else len(future)
        sl_first = np.argmax(sl_hits) if sl_hits.any() else len(future)

        if tp_first >= len(future) and sl_first >= len(future):
            continue  # neither hit within 14d → NaN
        labels[i] = 1.0 if tp_first <= sl_first else 0.0

    return labels


def ev_stats(selected_mask, labels, win_pct, loss_pct):
    """Per-trade EV on selected samples. NaN labels excluded."""
    valid = selected_mask & ~np.isnan(labels)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "ev_per_trade": 0.0}
    wins = valid & (labels == 1.0)
    losses = valid & (labels == 0.0)
    total_win_pct = float(win_pct[wins].sum()) if wins.any() else 0.0
    total_loss_pct = float(loss_pct[losses].sum()) if losses.any() else 0.0
    ev = (total_win_pct - total_loss_pct) / n
    return {
        "n": n,
        "win_rate": float(wins.sum() / n),
        "avg_win_pct": float(win_pct[wins].mean()) if wins.any() else 0.0,
        "avg_loss_pct": float(loss_pct[losses].mean()) if losses.any() else 0.0,
        "ev_per_trade": float(ev),
    }


def print_row(label, stats):
    print(
        f"  {label:<32} n={stats['n']:>5}  "
        f"win%={stats['win_rate']*100:5.1f}  "
        f"avg_win={stats.get('avg_win_pct',0)*100:+5.2f}%  "
        f"avg_loss={stats.get('avg_loss_pct',0)*100:+5.2f}%  "
        f"EV={stats['ev_per_trade']*100:+6.2f}%"
    )


def main():
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        sys.exit(1)
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    print("Loading v10 predictions cache...")
    data = np.load(PREDICTIONS_PATH, allow_pickle=True)
    dates = pd.to_datetime(data["dates"], utc=True)
    logits = data["logits"]
    probs = data["probs"]
    long_labels = data["labels"].astype(np.float32)
    long_tp = data["tp_pct"].astype(np.float64)
    long_sl = data["sl_pct"].astype(np.float64)
    regimes = data["regimes"]  # 1=bull, 0=bear, NaN=unknown
    print(f"  {len(dates)} predictions")

    print("Loading close series from CSV...")
    df = pd.read_csv(CSV_PATH, parse_dates=["date"], usecols=["date", "close"])
    df_dates = pd.to_datetime(df["date"], utc=True).values
    df_close = df["close"].values.astype(np.float64)

    # Map each cached prediction to its row in the full close series
    date_to_idx = {np.datetime64(d, "s"): i for i, d in enumerate(df_dates)}
    pred_row_idx = np.array([
        date_to_idx.get(np.datetime64(pd.Timestamp(d), "s"), -1)
        for d in dates
    ], dtype=np.int64)
    if (pred_row_idx < 0).any():
        n_miss = int((pred_row_idx < 0).sum())
        print(f"  WARNING: {n_miss} predictions could not be mapped to CSV rows")

    print("Simulating mirror-short first-hit labels...")
    short_labels = compute_short_labels(df_close, pred_row_idx, long_tp, long_sl)
    n_short_valid = int((~np.isnan(short_labels)).sum())
    n_short_wins = int((short_labels == 1.0).sum())
    print(f"  Short labels: {n_short_valid} valid, {n_short_wins} wins "
          f"({n_short_wins / max(n_short_valid, 1) * 100:.1f}%)")

    # Per-short payouts: mirror rule
    short_win_pct = long_sl.copy()   # short wins → gains long_sl %
    short_loss_pct = long_tp.copy()  # short loses → costs long_tp %

    # Masks
    holdout_mask = dates >= HOLDOUT_START
    full_mask = np.ones(len(dates), dtype=bool)
    bull_mask = regimes == 1.0
    bear_mask = regimes == 0.0

    scopes = [("full", full_mask), ("holdout", holdout_mask)]
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {}

    # ── 1. Long-only sanity check (should match eval_v10_results.json) ──
    print(f"\n{'=' * 70}\n  LONG ONLY (threshold sweep)\n{'=' * 70}")
    for scope_name, scope_mask in scopes:
        print(f"\n  [{scope_name}]")
        for thr in thresholds:
            mask = (logits > thr) & scope_mask
            s = ev_stats(mask, long_labels, long_tp, long_sl)
            print_row(f"logit>{thr:+.2f}", s)
            results[f"long_only_thr={thr}_{scope_name}"] = s

    # ── 2. Short-only ──
    print(f"\n{'=' * 70}\n  SHORT ONLY (threshold sweep — mirror labels)\n{'=' * 70}")
    for scope_name, scope_mask in scopes:
        print(f"\n  [{scope_name}]")
        for thr in thresholds:
            mask = (logits < -thr) & scope_mask
            s = ev_stats(mask, short_labels, short_win_pct, short_loss_pct)
            print_row(f"logit<{-thr:+.2f}", s)
            results[f"short_only_thr={thr}_{scope_name}"] = s

    # ── 3. Unconditional short (no model signal, just short everything) ──
    print(f"\n{'=' * 70}\n  UNCONDITIONAL SHORT (sanity baseline)\n{'=' * 70}")
    for scope_name, scope_mask in scopes:
        s = ev_stats(scope_mask, short_labels, short_win_pct, short_loss_pct)
        print_row(f"[{scope_name}] short all", s)
        results[f"short_unconditional_{scope_name}"] = s

    # ── 4. Regime-aware both sides: long in bull, short in bear ──
    print(f"\n{'=' * 70}\n  REGIME-AWARE BOTH-SIDES (long in bull, short in bear)\n{'=' * 70}")
    for scope_name, scope_mask in scopes:
        print(f"\n  [{scope_name}]")
        for thr in thresholds:
            long_sel = (logits > thr) & bull_mask & scope_mask
            short_sel = (logits < -thr) & bear_mask & scope_mask
            long_s = ev_stats(long_sel, long_labels, long_tp, long_sl)
            short_s = ev_stats(short_sel, short_labels, short_win_pct, short_loss_pct)
            combined_n = long_s["n"] + short_s["n"]
            if combined_n > 0:
                combined_ev = (
                    long_s["ev_per_trade"] * long_s["n"]
                    + short_s["ev_per_trade"] * short_s["n"]
                ) / combined_n
            else:
                combined_ev = 0.0
            row = {
                "thr": thr,
                "bull_long": long_s,
                "bear_short": short_s,
                "combined_n": combined_n,
                "combined_ev": combined_ev,
            }
            print(f"  thr=|{thr:.2f}|  "
                  f"bull_long n={long_s['n']:>4} EV={long_s['ev_per_trade']*100:+6.2f}%  |  "
                  f"bear_short n={short_s['n']:>4} EV={short_s['ev_per_trade']*100:+6.2f}%  |  "
                  f"combined n={combined_n:>4} EV={combined_ev*100:+6.2f}%")
            results[f"regime_aware_thr={thr}_{scope_name}"] = row

    # ── 5. All-bar regime-agnostic both sides (logit > thr → long, logit < -thr → short) ──
    print(f"\n{'=' * 70}\n  REGIME-AGNOSTIC BOTH-SIDES (pure logit signal)\n{'=' * 70}")
    for scope_name, scope_mask in scopes:
        print(f"\n  [{scope_name}]")
        for thr in thresholds:
            long_sel = (logits > thr) & scope_mask
            short_sel = (logits < -thr) & scope_mask
            long_s = ev_stats(long_sel, long_labels, long_tp, long_sl)
            short_s = ev_stats(short_sel, short_labels, short_win_pct, short_loss_pct)
            combined_n = long_s["n"] + short_s["n"]
            if combined_n > 0:
                combined_ev = (
                    long_s["ev_per_trade"] * long_s["n"]
                    + short_s["ev_per_trade"] * short_s["n"]
                ) / combined_n
            else:
                combined_ev = 0.0
            print(f"  thr=|{thr:.2f}|  "
                  f"long n={long_s['n']:>4} EV={long_s['ev_per_trade']*100:+6.2f}%  |  "
                  f"short n={short_s['n']:>4} EV={short_s['ev_per_trade']*100:+6.2f}%  |  "
                  f"combined n={combined_n:>4} EV={combined_ev*100:+6.2f}%")
            results[f"agnostic_both_thr={thr}_{scope_name}"] = {
                "thr": thr,
                "long": long_s,
                "short": short_s,
                "combined_n": combined_n,
                "combined_ev": combined_ev,
            }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {RESULTS_PATH}")

    print(f"\n{'=' * 70}")
    print("HONEST READ: The holdout column is what matters. If regime-aware")
    print("or agnostic both-sides has positive holdout EV under any reasonable")
    print("threshold, there's signal worth building a real backtest around.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
