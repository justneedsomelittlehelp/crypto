"""Run backtest on cached v10 predictions.

Thin wrapper around run_backtest that redirects the predictions cache to
`v10_predictions.npz` and saves results to `backtest_results_v10.json`.

Prereq: download `experiments/v10_predictions.npz` from Colab first.

Usage:
    python3 -m src.models.run_backtest_v10
"""

import src.models.run_backtest as rb
from src.config import EXPERIMENTS_DIR

rb.PREDICTIONS_PATH = EXPERIMENTS_DIR / "v10_predictions.npz"
rb.RESULTS_PATH = EXPERIMENTS_DIR / "backtest_results_v10.json"

if __name__ == "__main__":
    print("=" * 70)
    print("  v10 BACKTEST (90d temporal x 30d VP)")
    print("=" * 70)
    rb.main()
