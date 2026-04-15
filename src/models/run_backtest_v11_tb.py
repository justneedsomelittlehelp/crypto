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

2. Bar counts in BacktestConfig are resolution-aware. Underlying data
   is 15m regardless of the model's prediction cadence (15m, 1h, 4h).
   The engine walks the full 15m close path, so bar-based windows are
   always in 15m:
     max_hold_bars:        14 * 96 (15m) = 1344
     post_sl_pause_bars:   24 * 4  (15m) = 96
   Prediction cadence only changes how many anchors are fed in.

3. Triple-barrier tp/sl are symmetric (tp_pct ≡ sl_pct ≡ barrier).
   That means the engine's `min_asymmetry` filter is a no-op at any
   threshold ≤ 1.0, and excludes everything at > 1.0. We set it to
   0.0 (keep everything) for tb runs.

4. We run both `full` (entire walk-forward period) and `holdout`
   (>= 2025-07-01) scopes for every tag passed in.

Usage:
    # Default (15m cadence, compat with existing runs):
    python3 -m src.models.run_backtest_v11_tb

    # 1h cadence ablation pair:
    python3 -m src.models.run_backtest_v11_tb \\
        --tags 11_tb_full_c60m,11_tb_nopv_c60m \\
        --out backtest_results_v11_tb_c60m.json

    # Both cadences in one run, to get the paired comparison table:
    python3 -m src.models.run_backtest_v11_tb \\
        --tags 11_tb_full,11_tb_nopv,11_tb_full_c60m,11_tb_nopv_c60m \\
        --out backtest_results_v11_tb_cadence_ablation.json
"""

import argparse
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
BARS_PER_HOUR = 4      # 15m resolution of underlying price path
BARS_PER_DAY = 96

DEFAULT_TAGS = ["11_tb_full", "11_tb_nopv"]
DEFAULT_OUT = "backtest_results_v11_tb.json"
HOLDOUT_START = pd.Timestamp("2025-07-01")


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
    # Prediction cadence is stored in npz for runs produced after the
    # cadence knob was added. Old files default to 15m (stride=1).
    stride_bars = int(d["stride_bars"]) if "stride_bars" in d.files else 1
    cadence_minutes = (
        int(d["cadence_minutes"]) if "cadence_minutes" in d.files
        else stride_bars * 15
    )
    data = {
        "dates": pd.to_datetime(d["dates"]).values,
        "close": d["close"].astype(np.float64),
        "probs": probs,
        "tp_pct": d["tp_pct"].astype(np.float64),
        "sl_pct": d["sl_pct"].astype(np.float64),
        "stride_bars": stride_bars,
        "cadence_minutes": cadence_minutes,
    }
    # Labels are NaN for timeout samples under triple-barrier; the engine
    # doesn't care about labels (it walks the close path), but we drop
    # rows with NaN tp/sl to avoid any ambiguity.
    valid = np.isfinite(data["tp_pct"]) & np.isfinite(data["sl_pct"])
    array_keys = ["dates", "close", "probs", "tp_pct", "sl_pct"]
    for k in array_keys:
        data[k] = data[k][valid]
    print(
        f"  {path.name}: {len(data['dates']):,} rows  "
        f"cadence={cadence_minutes}m  "
        f"{data['dates'][0]} → {data['dates'][-1]}  "
        f"probs p50={np.median(probs):.3f}"
    )
    return data


ARRAY_KEYS = ("dates", "close", "probs", "tp_pct", "sl_pct")


def slice_by_scope(data: dict, scope: str) -> dict:
    if scope == "full":
        return dict(data)
    start = np.datetime64(HOLDOUT_START)
    mask = data["dates"] >= start
    out = {k: (data[k][mask] if k in ARRAY_KEYS else data[k]) for k in data}
    return out


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


def _pair_tag(tag: str) -> tuple[str, str]:
    """Split a tag like '11_tb_full_c60m' into (base='full', cadence_suffix='_c60m').

    Base is 'full' if the tag ends with 'full[...cadence]' and 'nopv' if it
    ends with 'nopv[...cadence]'. Cadence suffix is '' for legacy 15m.
    """
    core = tag
    cadence_suffix = ""
    # Accept a trailing _c{N}m cadence marker
    if "_c" in core and core.rsplit("_c", 1)[1].endswith("m"):
        core, cad = core.rsplit("_c", 1)
        cadence_suffix = f"_c{cad}"
    if core.endswith("_full"):
        return ("full", cadence_suffix)
    if core.endswith("_nopv"):
        return ("nopv", cadence_suffix)
    return (tag, cadence_suffix)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="v11-tb real-engine backtest")
    p.add_argument("--tags", type=str,
                   default=",".join(DEFAULT_TAGS),
                   help="comma-separated tag list. Each tag resolves to "
                        "experiments/v11_{tag}_predictions.npz. For the "
                        "cadence ablation pass "
                        "'11_tb_full,11_tb_nopv,11_tb_full_c60m,11_tb_nopv_c60m'.")
    p.add_argument("--out", type=str, default=DEFAULT_OUT,
                   help="output json filename (under experiments/). "
                        f"default: {DEFAULT_OUT}")
    return p.parse_args()


def main():
    args = parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    results_path = EXPERIMENTS_DIR / args.out

    print("=" * 78)
    print("  v11 TRIPLE-BARRIER BACKTEST — real engine (fees, slippage, latency)")
    print("=" * 78)
    print(f"  Tags: {', '.join(tags)}")
    print(f"  Out : {results_path.name}")

    # Preload prediction files once
    print("\nLoading predictions...")
    preds = {tag: load_tb_predictions(tag) for tag in tags}

    # Run all (tag × variant × scope) combinations
    all_results = []
    print("\nRunning backtests...")
    for tag in tags:
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
                    "cadence_minutes": preds[tag]["cadence_minutes"],
                })
                # Guard: when a variant produces 0 trades compute_metrics
                # returns a sparse dict without annualized_return_pct. Fill
                # a neutral row so downstream comparisons don't crash.
                m.setdefault("final_equity", cfg.starting_capital)
                m.setdefault("total_return_pct", 0.0)
                m.setdefault("annualized_return_pct", 0.0)
                m.setdefault("max_drawdown_pct", 0.0)
                m.setdefault("sharpe_annualized", 0.0)
                m.setdefault("n_trades", 0)
                m.setdefault("win_rate", 0.0)
                all_results.append(m)
                note = "  [NO TRADES]" if m["n_trades"] == 0 else ""
                print(
                    f"  {tag:<22} {variant_name:<22} {scope:<8} "
                    f"final=${m['final_equity']:>9,.0f}  "
                    f"ret={m['total_return_pct']:>+7.1f}%  "
                    f"CAGR={m['annualized_return_pct']:>+7.1f}%  "
                    f"DD={m['max_drawdown_pct']:>+6.1f}%  "
                    f"Sh={m['sharpe_annualized']:>+5.2f}  "
                    f"n={m['n_trades']:>4}  "
                    f"win={m['win_rate']*100:>5.1f}%{note}"
                )

    by_key = {(r["tag"], r["variant"], r["scope"]): r for r in all_results}

    # ─── Per-cadence comparison: full vs nopv at each (variant, scope) ───
    # Group tags by their cadence suffix, emit one comparison block per cadence.
    pair_tags = {tag: _pair_tag(tag) for tag in tags}
    cadence_suffixes = sorted({cs for _, cs in pair_tags.values()})

    for cad_suffix in cadence_suffixes:
        label = "15m (legacy)" if cad_suffix == "" else f"{cad_suffix[2:]} cadence"
        full_tag = next(
            (t for t, (b, c) in pair_tags.items() if b == "full" and c == cad_suffix),
            None,
        )
        nopv_tag = next(
            (t for t, (b, c) in pair_tags.items() if b == "nopv" and c == cad_suffix),
            None,
        )
        if full_tag is None or nopv_tag is None:
            continue

        print("\n" + "=" * 102)
        print(f"  COMPARISON @ {label} — v11-full (VP) vs v11-nopv (candle only)")
        print("=" * 102)
        print(f"  {'Variant':<22} {'Scope':<8} "
              f"{'full CAGR':>11} {'nopv CAGR':>11} {'Δ CAGR':>10}  "
              f"{'full DD':>9} {'nopv DD':>9}  "
              f"{'full trades':>12} {'nopv trades':>12}")
        print("  " + "-" * 100)
        for variant_name in VARIANTS:
            for scope in SCOPES:
                f = by_key.get((full_tag, variant_name, scope))
                n = by_key.get((nopv_tag, variant_name, scope))
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

    # ─── Cadence ablation: same base (full or nopv) across cadences ───
    if len(cadence_suffixes) > 1:
        for base in ("full", "nopv"):
            base_tags = [
                (cs, t) for t, (b, cs) in pair_tags.items() if b == base
            ]
            if len(base_tags) < 2:
                continue
            base_tags.sort(key=lambda x: 0 if x[0] == "" else int(x[0][2:-1]))

            print("\n" + "=" * 102)
            print(f"  CADENCE ABLATION — v11-{base}-tb across prediction cadences")
            print("=" * 102)
            header = f"  {'Variant':<22} {'Scope':<8}"
            for cs, _ in base_tags:
                lbl = "15m" if cs == "" else cs[2:]
                header += f" {lbl+' CAGR':>12} {lbl+' DD':>10} {lbl+' n':>7}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for variant_name in VARIANTS:
                for scope in SCOPES:
                    row = f"  {variant_name:<22} {scope:<8}"
                    ok = True
                    for _, tag in base_tags:
                        r = by_key.get((tag, variant_name, scope))
                        if r is None:
                            ok = False
                            break
                        row += (
                            f" {r['annualized_return_pct']:>+11.1f}%"
                            f" {r['max_drawdown_pct']:>+9.1f}%"
                            f" {r['n_trades']:>7}"
                        )
                    if ok:
                        print(row)

    # Save
    output = {
        "metadata": {
            "source_npz_files": [f"v11_{tag}_predictions.npz" for tag in tags],
            "cadence_minutes_by_tag": {
                tag: preds[tag]["cadence_minutes"] for tag in tags
            },
            "engine": "src.backtest.engine.run_backtest",
            "resolution_bars_per_day": BARS_PER_DAY,
            "holdout_start": HOLDOUT_START.isoformat(),
            "variants": VARIANTS,
            "notes": (
                "Real backtest engine on v11 triple-barrier predictions. "
                "Includes Kraken fees (taker 0.26%, maker 0.16%), slippage "
                "(0.05%/side), execution latency, 5k starting capital, "
                "reserve + pyramid rules. Compare to analyze_v11_filters "
                "CAGR numbers — those omit all frictions and will be higher. "
                "Cadence is read from each npz file; multi-cadence runs "
                "produce both per-cadence comparisons and a cadence ablation."
            ),
        },
        "results": [
            {k: v for k, v in r.items() if k != "engine_summary"}
            for r in all_results
        ],
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
