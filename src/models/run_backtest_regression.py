"""Regression-mode backtest driver for v9 predictions.

Loads predictions from `experiments/v9_regression_predictions.npz` (produced
by `eval_v9_regression.py`), runs the backtest engine in regression mode
with predicted returns instead of sigmoid probabilities.

Stage 1 protocol — locked configuration, no in-sample tuning:
  - Three predicted-return thresholds reported once: 1.5% / 2.0% / 2.5%
  - tp/sl guard (`min_asymmetry = 1.0`) — legitimate pre-entry filter
  - 1x leverage, 20% sizing, 24h post-SL pause — same honest config as
    the audited best
  - Full + holdout scope reported for each

Usage:
    # First, run eval_v9_regression.py on Colab to generate predictions
    python -m src.models.eval_v9_regression

    # Then run this backtest:
    python -m src.models.run_backtest_regression
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


PREDICTIONS_PATH = EXPERIMENTS_DIR / "v9_regression_predictions.npz"
RESULTS_PATH = EXPERIMENTS_DIR / "backtest_results_v9_regression.json"


# ═══════════════════════════════════════════════════════════════════
# Filter and sizing variants
# ═══════════════════════════════════════════════════════════════════
FILTER_VARIANTS = {
    # Three adjacent predicted-return thresholds. Reported once, no
    # iteration. Sweep is over a fixed set chosen before evaluation.
    "reg_015_guard": {"min_predicted_return": 0.015, "min_asymmetry": 1.0, "allow_pyramiding": True},
    "reg_020_guard": {"min_predicted_return": 0.020, "min_asymmetry": 1.0, "allow_pyramiding": True},
    "reg_025_guard": {"min_predicted_return": 0.025, "min_asymmetry": 1.0, "allow_pyramiding": True},
}

SIZING_VARIANTS = {
    # Same honest profile from the audit:
    # 1x leverage, 20% sizing, 24h post-SL pause.
    "sane_1x_pause24": {
        "sizing_mode": "fixed_pct",
        "position_size_pct": 0.20,
        "reserve_pct": 0.0,
        "leverage": 1.0,
        "post_sl_pause_bars": 24,
    },
}

# Out-of-sample holdout window — never used to tune any of the above.
HOLDOUT_START = pd.Timestamp("2025-07-01")


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        print("Run 'python -m src.models.eval_v9_regression' first to generate predictions.")
        sys.exit(1)

    data = np.load(PREDICTIONS_PATH, allow_pickle=True)
    print(f"Loaded predictions from {PREDICTIONS_PATH}")
    print(f"  Total samples: {len(data['predicted_return'])}")
    print(f"  Date range: {data['dates'][0]} → {data['dates'][-1]}")
    print(f"  Close range: ${float(data['close'].min()):,.0f} → ${float(data['close'].max()):,.0f}")
    pred = data["predicted_return"]
    print(f"  Predicted return — mean={pred.mean()*100:+.3f}% std={pred.std()*100:.3f}% "
          f"min={pred.min()*100:+.2f}% max={pred.max()*100:+.2f}%")
    return data


def slice_data(data, start_ts):
    """Return a dict view of data with rows on or after start_ts."""
    dates = pd.to_datetime(data["dates"]).values
    mask = dates >= np.datetime64(start_ts)
    return {
        "dates": data["dates"][mask],
        "close": data["close"][mask],
        "predicted_return": data["predicted_return"][mask],
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
        # Filter (regression mode)
        prediction_mode="regression",
        min_predicted_return=filter_cfg["min_predicted_return"],
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
        predicted_returns=data["predicted_return"],
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
    print("  v9-REGRESSION BACKTEST (Stage 1)")
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
    print(f"  COMPARISON TABLE — Stage 1 (regression labels, v6-prime arch)")
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

    if sorted_results:
        # Best by holdout return — that's the honest verdict
        holdout_results = [r for r in sorted_results if r.get("scope") == "holdout"]
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
            print(f"  Skipped (capital): {best['skipped_no_capital']}")

    # Save
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
