"""Run the real backtest engine on v11 triple-barrier predictions.

Purpose: the decisive-experiment CAGR numbers in LABEL_REDESIGN.md came
from `analyze_v11_filters.py`, which is a label-accurate per-trade
compound calculation with NO fees, NO slippage, NO execution latency.
v6-prime's "+6.0% CAGR honest" was produced by `src/backtest/engine.py`
which includes all of those frictions. Comparing them side-by-side
was misleading.

This script runs the actual engine on the v11-full-tb and v11-nopv-tb
predictions using the same config shape v6-prime used, so the numbers
are apples-to-apples comparable.

Four things to know about the adaptation:

1. The engine expects `probs` (sigmoid of logits). The v11 npz stores
   logits, so we convert here.

2. Bar counts in BacktestConfig are resolution-aware. v6-prime ran at
   1h, v11 runs at 15m. We scale:
     max_hold_bars:        14 * 24 (1h)  →  14 * 96 (15m) = 1344
     post_sl_pause_bars:   24       (1h) →  24 * 4   (15m) = 96
   Same wall-clock durations, different bar counts.

3. Triple-barrier tp/sl are symmetric (tp_pct ≡ sl_pct ≡ barrier).
   That means the engine's `min_asymmetry` filter is a no-op at any
   threshold ≤ 1.0, and excludes everything at > 1.0. We set it to
   0.0 (keep everything) for tb runs.

4. We run both `full` (entire walk-forward period) and `holdout`
   (>= 2025-07-01) scopes, and both `tb_full` (with VP) and `tb_nopv`
   (candle only) prediction files. 4 scope × filter combinations.

Usage:
    python3 -m src.models.run_backtest_v11_tb
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

from src.config import EXPERIMENTS_DIR
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import compute_metrics


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
BARS_PER_HOUR = 4      # 15m resolution
BARS_PER_DAY = 96

TAGS = ["11_tb_full", "11_tb_nopv"]
HOLDOUT_START = pd.Timestamp("2025-07-01")
RESULTS_PATH = EXPERIMENTS_DIR / "backtest_results_v11_tb.json"


def _variant(
    conf: float,
    pyramid: bool,
    size_pct: float = 0.20,
    reserve: float = 0.30,
    leverage: float = 1.0,
    pause_h: int = 24,
) -> dict:
    return {
        "min_confidence": conf,
        "min_asymmetry": 0.0,                   # symmetric tb barriers
        "position_size_pct": size_pct,
        "reserve_pct": reserve,
        "leverage": leverage,
        "post_sl_pause_bars": pause_h * BARS_PER_HOUR,
        "allow_pyramiding": pyramid,
    }


# Pyramid-off variants (the original four) + pyramid-on at the same
# filters, for direct apples-to-apples on whether the engine is being
# starved of trades by the serialization rule.
VARIANTS = {
    "conf70_1x20_pause24":     _variant(0.70, pyramid=False),
    "conf75_1x20_pause24":     _variant(0.75, pyramid=False),
    "conf80_1x20_pause24":     _variant(0.80, pyramid=False),
    "conf60_1x20_nopause":     _variant(0.60, pyramid=False, pause_h=0),

    "conf70_1x20_pause24_pyr": _variant(0.70, pyramid=True),
    "conf75_1x20_pause24_pyr": _variant(0.75, pyramid=True),
    "conf80_1x20_pause24_pyr": _variant(0.80, pyramid=True),
    "conf60_1x20_nopause_pyr": _variant(0.60, pyramid=True, pause_h=0),
}

SCOPES = ["full", "holdout"]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_tb_predictions(tag: str) -> dict:
    path = EXPERIMENTS_DIR / f"v11_{tag}_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    d = np.load(path, allow_pickle=True)
    probs = sigmoid(d["logits"].astype(np.float64))
    data = {
        "dates": pd.to_datetime(d["dates"]).values,
        "close": d["close"].astype(np.float64),
        "probs": probs,
        "tp_pct": d["tp_pct"].astype(np.float64),
        "sl_pct": d["sl_pct"].astype(np.float64),
    }
    # Labels are NaN for timeout samples under triple-barrier; the engine
    # doesn't care about labels (it walks the close path), but we drop
    # rows with NaN tp/sl to avoid any ambiguity.
    valid = np.isfinite(data["tp_pct"]) & np.isfinite(data["sl_pct"])
    for k in data:
        data[k] = data[k][valid]
    print(
        f"  {path.name}: {len(data['dates']):,} rows  "
        f"{data['dates'][0]} → {data['dates'][-1]}  "
        f"probs p50={np.median(probs):.3f}"
    )
    return data


def slice_by_scope(data: dict, scope: str) -> dict:
    if scope == "full":
        return data
    start = np.datetime64(HOLDOUT_START)
    mask = data["dates"] >= start
    return {k: v[mask] for k, v in data.items()}


def build_config(cfg: dict) -> BacktestConfig:
    return BacktestConfig(
        starting_capital=5000.0,
        reserve_pct=cfg["reserve_pct"],
        position_size_pct=cfg["position_size_pct"],
        sizing_mode="fixed_pct",
        max_hold_bars=14 * BARS_PER_DAY,   # 14 days at 15m = 1344
        direction="long",
        min_confidence=cfg["min_confidence"],
        min_asymmetry=cfg["min_asymmetry"],
        allow_pyramiding=cfg["allow_pyramiding"],
        leverage=cfg["leverage"],
        post_sl_pause_bars=cfg["post_sl_pause_bars"],
    )


def run_one(data: dict, cfg: BacktestConfig) -> dict:
    portfolio, engine_summary = run_backtest(
        dates=data["dates"],
        close_prices=data["close"],
        probs=data["probs"],
        tp_pct=data["tp_pct"],
        sl_pct=data["sl_pct"],
        config=cfg,
    )
    metrics = compute_metrics(portfolio, cfg.starting_capital)
    metrics["engine_summary"] = engine_summary
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print("  v11 TRIPLE-BARRIER BACKTEST — real engine (fees, slippage, latency)")
    print("=" * 78)

    # Preload both prediction files once
    print("\nLoading predictions...")
    preds = {tag: load_tb_predictions(tag) for tag in TAGS}

    # Run all (tag × variant × scope) combinations
    all_results = []
    print("\nRunning backtests...")
    for tag in TAGS:
        for variant_name, variant_cfg in VARIANTS.items():
            for scope in SCOPES:
                data = slice_by_scope(preds[tag], scope)
                if len(data["dates"]) == 0:
                    continue
                cfg = build_config(variant_cfg)
                try:
                    m = run_one(data, cfg)
                except Exception as e:
                    print(f"  [{tag} / {variant_name} / {scope}] FAILED: {e}")
                    continue
                m.update({
                    "tag": tag,
                    "variant": variant_name,
                    "scope": scope,
                })
                all_results.append(m)
                print(
                    f"  {tag:<12} {variant_name:<22} {scope:<8} "
                    f"final=${m['final_equity']:>9,.0f}  "
                    f"ret={m['total_return_pct']:>+7.1f}%  "
                    f"CAGR={m['annualized_return_pct']:>+7.1f}%  "
                    f"DD={m['max_drawdown_pct']:>+6.1f}%  "
                    f"Sh={m['sharpe_annualized']:>+5.2f}  "
                    f"n={m['n_trades']:>4}  "
                    f"win={m['win_rate']*100:>5.1f}%"
                )

    # ─── Comparison table: full vs nopv at each (variant, scope) ───
    print("\n" + "=" * 102)
    print("  COMPARISON — v11-full (with VP) vs v11-nopv (candle only) — same engine, same config")
    print("=" * 102)
    print(f"  {'Variant':<22} {'Scope':<8} "
          f"{'full CAGR':>11} {'nopv CAGR':>11} {'Δ CAGR':>10}  "
          f"{'full DD':>9} {'nopv DD':>9}  "
          f"{'full trades':>12} {'nopv trades':>12}")
    print("  " + "-" * 100)

    # Index by (variant, scope) for easy lookup
    by_key = {(r["tag"], r["variant"], r["scope"]): r for r in all_results}
    for variant_name in VARIANTS:
        for scope in SCOPES:
            f = by_key.get(("11_tb_full", variant_name, scope))
            n = by_key.get(("11_tb_nopv", variant_name, scope))
            if f is None or n is None:
                continue
            dC = f["annualized_return_pct"] - n["annualized_return_pct"]
            print(
                f"  {variant_name:<22} {scope:<8} "
                f"{f['annualized_return_pct']:>+10.1f}% "
                f"{n['annualized_return_pct']:>+10.1f}% "
                f"{dC:>+9.1f}  "
                f"{f['max_drawdown_pct']:>+8.1f}% "
                f"{n['max_drawdown_pct']:>+8.1f}% "
                f"{f['n_trades']:>12} "
                f"{n['n_trades']:>12}"
            )

    # Save
    output = {
        "metadata": {
            "source_npz_files": [f"v11_{tag}_predictions.npz" for tag in TAGS],
            "engine": "src.backtest.engine.run_backtest",
            "resolution_bars_per_day": BARS_PER_DAY,
            "holdout_start": HOLDOUT_START.isoformat(),
            "variants": VARIANTS,
            "notes": (
                "Real backtest engine on v11 triple-barrier predictions. "
                "Includes Kraken fees (taker 0.26%, maker 0.16%), slippage "
                "(0.05%/side), execution latency, 5k starting capital, "
                "reserve + pyramid rules. Compare to analyze_v11_filters "
                "CAGR numbers — those omit all frictions and will be higher."
            ),
        },
        "results": [
            {k: v for k, v in r.items() if k != "engine_summary"}
            for r in all_results
        ],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
