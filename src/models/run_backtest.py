"""Run backtest on cached v6-prime predictions.

Loads predictions from experiments/v6_prime_predictions.npz, then runs the
backtest engine with multiple sizing strategies × filters in parallel.

Each combination is a separate backtest run. Results are saved to
experiments/backtest_results.json with full metrics per combination.

Usage:
    # First, run eval_v6_prime.py to generate the predictions cache
    python -m src.models.eval_v6_prime

    # Then run the backtest:
    python -m src.models.run_backtest
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


PREDICTIONS_PATH = EXPERIMENTS_DIR / "v6_prime_predictions.npz"
RESULTS_PATH = EXPERIMENTS_DIR / "backtest_results.json"


# ═══════════════════════════════════════════════════════════════════
# Filter and sizing variants to test
# ═══════════════════════════════════════════════════════════════════
FILTER_VARIANTS = {
    # Locked to the proven winner: combined_60_20
    "combined_60_20": {
        "min_confidence": 0.60,
        "min_asymmetry": 2.0,
        "allow_pyramiding": True,
    },
}

SIZING_VARIANTS = {
    # All variants use 100% sizing × 3x leverage (deployable winner from Eval 15).
    # Test post-SL pause: pause for N bars after each SL hit, letting market settle.
    "baseline":         {"sizing_mode": "fixed_pct", "position_size_pct": 1.00, "reserve_pct": 0.0, "leverage": 3.0},
    "post_sl_12h":      {"sizing_mode": "fixed_pct", "position_size_pct": 1.00, "reserve_pct": 0.0, "leverage": 3.0,
                         "post_sl_pause_bars": 12},
    "post_sl_24h":      {"sizing_mode": "fixed_pct", "position_size_pct": 1.00, "reserve_pct": 0.0, "leverage": 3.0,
                         "post_sl_pause_bars": 24},
    "post_sl_48h":      {"sizing_mode": "fixed_pct", "position_size_pct": 1.00, "reserve_pct": 0.0, "leverage": 3.0,
                         "post_sl_pause_bars": 48},
    "post_sl_72h":      {"sizing_mode": "fixed_pct", "position_size_pct": 1.00, "reserve_pct": 0.0, "leverage": 3.0,
                         "post_sl_pause_bars": 72},
    "post_sl_24h_kill4L": {"sizing_mode": "fixed_pct", "position_size_pct": 1.00, "reserve_pct": 0.0, "leverage": 3.0,
                         "post_sl_pause_bars": 24, "max_consec_losses": 4, "killswitch_pause_bars": 168},
}


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: {PREDICTIONS_PATH} not found.")
        print("Run 'python -m src.models.eval_v6_prime' first to generate predictions.")
        sys.exit(1)

    data = np.load(PREDICTIONS_PATH, allow_pickle=True)
    print(f"Loaded predictions from {PREDICTIONS_PATH}")
    print(f"  Total samples: {len(data['preds'])}")
    print(f"  Date range: {data['dates'][0]} → {data['dates'][-1]}")
    print(f"  Close range: ${float(data['close'].min()):,.0f} → ${float(data['close'].max()):,.0f}")
    return data


def run_one_backtest(data, filter_name, sizing_name):
    """Run a single backtest with a specific filter × sizing combination."""
    filter_cfg = FILTER_VARIANTS[filter_name]
    sizing_cfg = SIZING_VARIANTS[sizing_name]

    # Build BacktestConfig
    config = BacktestConfig(
        starting_capital=5000.0,
        reserve_pct=sizing_cfg.get("reserve_pct", 0.30),
        max_hold_bars=14 * 24,
        direction="long",
        # Filter
        min_confidence=filter_cfg["min_confidence"],
        min_asymmetry=filter_cfg["min_asymmetry"],
        allow_pyramiding=filter_cfg["allow_pyramiding"],
        # Sizing
        sizing_mode=sizing_cfg["sizing_mode"],
        position_size_pct=sizing_cfg.get("position_size_pct", 0.20),
        # Leverage
        leverage=sizing_cfg.get("leverage", 1.0),
        # Circuit breakers
        circuit_breaker_dd=sizing_cfg.get("circuit_breaker_dd", 0.0),
        circuit_breaker_pause_bars=sizing_cfg.get("circuit_breaker_pause_bars", 0),
        max_consec_losses=sizing_cfg.get("max_consec_losses", 0),
        killswitch_pause_bars=sizing_cfg.get("killswitch_pause_bars", 0),
        post_sl_pause_bars=sizing_cfg.get("post_sl_pause_bars", 0),
    )

    # Parse dates from cached array
    dates = pd.to_datetime(data["dates"]).values

    portfolio, summary = run_backtest(
        dates=dates,
        close_prices=data["close"],
        probs=data["probs"],
        tp_pct=data["tp_pct"],
        sl_pct=data["sl_pct"],
        config=config,
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
    print("  v6-prime BACKTEST")
    print("=" * 70)

    data = load_predictions()

    # Run all combinations
    all_results = []
    print(f"\nRunning {len(FILTER_VARIANTS)} filters × {len(SIZING_VARIANTS)} sizings = "
          f"{len(FILTER_VARIANTS) * len(SIZING_VARIANTS)} backtests...\n")

    for filter_name in FILTER_VARIANTS:
        for sizing_name in SIZING_VARIANTS:
            label = f"{filter_name} + {sizing_name}"
            print(f"  Running: {label}...", end=" ", flush=True)
            try:
                metrics = run_one_backtest(data, filter_name, sizing_name)
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
    print(f"  COMPARISON TABLE — sorted by total return")
    print(f"{'=' * 100}")
    print(f"  {'Filter':<18} {'Sizing':<14} {'Final $':>10} {'Return':>9} {'CAGR':>8} {'DD':>8} {'Sharpe':>8} {'Trades':>7} {'Win%':>7}")
    print(f"  {'-' * 18} {'-' * 14} {'-' * 10} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 7}")
    sorted_results = sorted(all_results, key=lambda r: r["total_return_pct"], reverse=True)
    for r in sorted_results:
        print(
            f"  {r['filter']:<18} {r['sizing']:<14} "
            f"${r['final_equity']:>9,.0f} "
            f"{r['total_return_pct']:>+8.1f}% "
            f"{r['annualized_return_pct']:>+7.1f}% "
            f"{r['max_drawdown_pct']:>+7.1f}% "
            f"{r['sharpe_annualized']:>+7.2f} "
            f"{r['n_trades']:>7,} "
            f"{r['win_rate']*100:>6.1f}%"
        )

    # Best result detail
    if sorted_results:
        best = sorted_results[0]
        print(f"\n{'=' * 70}")
        print(f"  BEST: {best['filter']} + {best['sizing']}")
        print(f"{'=' * 70}")
        print(f"  Period: {best['first_trade_date']} → {best['last_trade_date']} ({best['span_years']} years)")
        print(f"  Starting capital:  ${best['starting_capital']:,.0f}")
        print(f"  Final equity:      ${best['final_equity']:,.0f}")
        print(f"  Total return:      {best['total_return_pct']:+.2f}%")
        print(f"  CAGR:              {best['annualized_return_pct']:+.2f}%")
        print(f"  Max drawdown:      {best['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe (ann):      {best['sharpe_annualized']:.2f}")
        print(f"  Sharpe (per trade): {best['sharpe_per_trade']:.3f}")
        print(f"  Trades:            {best['n_trades']:,} ({best['trades_per_year']:.1f}/year)")
        print(f"  Win rate:          {best['win_rate']*100:.1f}%")
        print(f"  Avg win:           ${best['avg_win_dollars']:,.2f} ({best['avg_win_pct']:+.2f}%)")
        print(f"  Avg loss:          ${best['avg_loss_dollars']:,.2f} ({best['avg_loss_pct']:+.2f}%)")
        print(f"  Avg hold:          {best['avg_hold_days']:.1f} days")
        print(f"  Total fees paid:   ${best['total_fees']:,.2f}")
        print(f"  Max consec losses: {best['max_consec_losses']}")
        print(f"  Exit reasons:      {best['exit_reasons']}")
        print(f"  Skipped (capital): {best['skipped_no_capital']}")
        print(f"  Skipped (pyramid): {best['skipped_pyramid']}")

    # Save
    output = {
        "config": {
            "starting_capital": 5000.0,
            "reserve_pct": 0.30,
            "max_hold_days": 14,
            "fee_taker": 0.0026,
            "fee_maker": 0.0016,
            "slippage_per_side": 0.0005,
            "filters": FILTER_VARIANTS,
            "sizings": SIZING_VARIANTS,
        },
        "results": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
