"""Backtest driver for v9 wall-aware predictions (binary classification).

Stage 2 backtest: loads predictions from `experiments/v9_predictions.npz`
and runs the backtest engine in binary mode with the same honest config
we locked after the audit:

  - conf_70 + min_asymmetry=1.0 (tp/sl guard) + 1x/20% sizing + 24h pause

Also sweeps two adjacent confidence thresholds (0.65, 0.75) reported once,
no iteration on evaluation period.

The comparison target is the v6-prime backtest with identical config.
If v9 shows a material improvement on the holdout scope, the structure
context token earned its keep.

Usage:
    # After `python -m src.models.eval_v9` on Colab:
    python -m src.models.run_backtest_v9
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from src.config import EXPERIMENTS_DIR
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import compute_metrics


PREDICTIONS_PATH = EXPERIMENTS_DIR / "v9_predictions.npz"
RESULTS_PATH = EXPERIMENTS_DIR / "backtest_results_v9.json"


# ═══════════════════════════════════════════════════════════════════
# Filter and sizing variants — locked before evaluation
# ═══════════════════════════════════════════════════════════════════
FILTER_VARIANTS = {
    # Three confidence thresholds matching the honest audited sweep.
    # tp/sl guard (min_asymmetry=1.0) is load-bearing per the audit.
    "conf_65_guard": {"min_confidence": 0.65, "min_asymmetry": 1.0, "allow_pyramiding": True},
    "conf_70_guard": {"min_confidence": 0.70, "min_asymmetry": 1.0, "allow_pyramiding": True},
    "conf_75_guard": {"min_confidence": 0.75, "min_asymmetry": 1.0, "allow_pyramiding": True},
}

SIZING_VARIANTS = {
    # Same honest config as the audited v6-prime best:
    # 1x leverage, 20% sizing, 24h post-SL pause. No more, no less.
    "sane_1x_pause24": {
        "sizing_mode": "fixed_pct",
        "position_size_pct": 0.20,
        "reserve_pct": 0.0,
        "leverage": 1.0,
        "post_sl_pause_bars": 24,
    },
}

HOLDOUT_START = pd.Timestamp("2025-07-01")


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        print("Run 'python -m src.models.eval_v9' first to generate predictions.")
        sys.exit(1)

    data = np.load(PREDICTIONS_PATH, allow_pickle=True)
    print(f"Loaded predictions from {PREDICTIONS_PATH}")
    print(f"  Total samples: {len(data['preds'])}")
    print(f"  Date range: {data['dates'][0]} → {data['dates'][-1]}")
    print(f"  Close range: ${float(data['close'].min()):,.0f} → ${float(data['close'].max()):,.0f}")
    return data


def slice_data(data, start_ts):
    """Return a dict view of data with rows on or after start_ts."""
    dates = pd.to_datetime(data["dates"]).values
    mask = dates >= np.datetime64(start_ts)
    return {
        "dates": data["dates"][mask],
        "close": data["close"][mask],
        "probs": data["probs"][mask],
        "tp_pct": data["tp_pct"][mask],
        "sl_pct": data["sl_pct"][mask],
    }


def run_one_backtest(data, filter_name, sizing_name):
    filter_cfg = FILTER_VARIANTS[filter_name]
    sizing_cfg = SIZING_VARIANTS[sizing_name]

    config = BacktestConfig(
        starting_capital=5000.0,
        reserve_pct=sizing_cfg.get("reserve_pct", 0.30),
        max_hold_bars=14 * 24,
        direction="long",
        # Filter (binary mode — v9 outputs sigmoid probs like v6-prime)
        prediction_mode="binary",
        min_confidence=filter_cfg["min_confidence"],
        min_asymmetry=filter_cfg["min_asymmetry"],
        allow_pyramiding=filter_cfg["allow_pyramiding"],
        # Sizing
        sizing_mode=sizing_cfg["sizing_mode"],
        position_size_pct=sizing_cfg.get("position_size_pct", 0.20),
        leverage=sizing_cfg.get("leverage", 1.0),
        # Risk controls
        circuit_breaker_dd=sizing_cfg.get("circuit_breaker_dd", 0.0),
        circuit_breaker_pause_bars=sizing_cfg.get("circuit_breaker_pause_bars", 0),
        max_consec_losses=sizing_cfg.get("max_consec_losses", 0),
        killswitch_pause_bars=sizing_cfg.get("killswitch_pause_bars", 0),
        post_sl_pause_bars=sizing_cfg.get("post_sl_pause_bars", 0),
    )

    dates = pd.to_datetime(data["dates"]).values

    portfolio, summary = run_backtest(
        dates=dates,
        close_prices=data["close"],
        tp_pct=data["tp_pct"],
        sl_pct=data["sl_pct"],
        config=config,
        probs=data["probs"],
    )

    metrics = compute_metrics(portfolio, config.starting_capital)
    metrics.update({
        "filter": filter_name,
        "sizing": sizing_name,
        "engine_summary": summary,
    })
    return metrics


def main():
    print("=" * 70)
    print("  v9 WALL-AWARE BACKTEST (Stage 2)")
    print("=" * 70)

    data = load_predictions()

    holdout_data = slice_data(data, HOLDOUT_START)
    print(f"\nHoldout slice: {len(holdout_data['dates'])} bars "
          f"({holdout_data['dates'][0]} → {holdout_data['dates'][-1]})")

    all_results = []
    print(f"\nRunning {len(FILTER_VARIANTS)} filters × {len(SIZING_VARIANTS)} sizings = "
          f"{len(FILTER_VARIANTS) * len(SIZING_VARIANTS)} backtests...\n")

    for filter_name in FILTER_VARIANTS:
        for sizing_name in SIZING_VARIANTS:
            label = f"{filter_name} + {sizing_name}"
            for scope_name, scope_data in [("full", data), ("holdout", holdout_data)]:
                print(f"  Running: {label} [{scope_name}]...", end=" ", flush=True)
                try:
                    metrics = run_one_backtest(scope_data, filter_name, sizing_name)
                    metrics["scope"] = scope_name
                    all_results.append(metrics)
                    print(f"final=${metrics['final_equity']:,.0f}  "
                          f"({metrics['total_return_pct']:+.1f}%)  "
                          f"DD={metrics['max_drawdown_pct']:.1f}%  "
                          f"Sharpe={metrics['sharpe_annualized']:.2f}  "
                          f"trades={metrics['n_trades']}")
                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # ─── Comparison table ───
    print(f"\n{'=' * 100}")
    print(f"  COMPARISON TABLE — Stage 2 (v9 wall-aware, binary labels)")
    print(f"{'=' * 100}")
    print(f"  {'Scope':<8} {'Filter':<18} {'Sizing':<18} {'Final $':>10} {'Return':>9} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'Trades':>7} {'Win%':>7}")
    print(f"  {'-' * 8} {'-' * 18} {'-' * 18} {'-' * 10} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 7}")
    sorted_results = sorted(all_results, key=lambda r: (r.get("scope", ""), -r["total_return_pct"]))
    for r in sorted_results:
        print(
            f"  {r.get('scope','full'):<8} {r['filter']:<18} {r['sizing']:<18} "
            f"${r['final_equity']:>9,.0f} "
            f"{r['total_return_pct']:>+8.1f}% "
            f"{r['annualized_return_pct']:>+7.1f}% "
            f"{r['max_drawdown_pct']:>+7.1f}% "
            f"{r['sharpe_annualized']:>+7.2f} "
            f"{r['n_trades']:>7,} "
            f"{r['win_rate']*100:>6.1f}%"
        )

    # Highlight best by holdout return
    holdout_results = [r for r in all_results if r.get("scope") == "holdout"]
    if holdout_results:
        best = max(holdout_results, key=lambda r: r["total_return_pct"])
        print(f"\n{'=' * 70}")
        print(f"  BEST HOLDOUT: {best['filter']} + {best['sizing']}")
        print(f"{'=' * 70}")
        print(f"  Period: {best['first_trade_date']} → {best['last_trade_date']} ({best['span_years']} years)")
        print(f"  Final equity:      ${best['final_equity']:,.0f}")
        print(f"  Total return:      {best['total_return_pct']:+.2f}%")
        print(f"  CAGR:              {best['annualized_return_pct']:+.2f}%")
        print(f"  Max drawdown:      {best['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe (ann):      {best['sharpe_annualized']:.2f}")
        print(f"  Trades:            {best['n_trades']:,} ({best['trades_per_year']:.1f}/year)")
        print(f"  Win rate:          {best['win_rate']*100:.1f}%")
        print(f"  Avg win:           ${best['avg_win_dollars']:,.2f} ({best['avg_win_pct']:+.2f}%)")
        print(f"  Avg loss:          ${best['avg_loss_dollars']:,.2f} ({best['avg_loss_pct']:+.2f}%)")
        print(f"  Avg hold:          {best['avg_hold_days']:.1f} days")
        print(f"  Total fees:        ${best['total_fees']:,.2f}")
        print(f"  Exit reasons:      {best['exit_reasons']}")

    output = {
        "config": {
            "starting_capital": 5000.0,
            "reserve_pct": 0.0,
            "max_hold_days": 14,
            "fee_taker": 0.0026,
            "fee_maker": 0.0016,
            "slippage_per_side": 0.0005,
            "filters": FILTER_VARIANTS,
            "sizings": SIZING_VARIANTS,
            "holdout_start": str(HOLDOUT_START),
        },
        "results": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
