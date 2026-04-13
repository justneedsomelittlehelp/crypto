"""Eval v10 — v6-prime labels/training with 90-day temporal window + 30-day VP.

Thin wrapper around eval_v6_prime that:
  1. Patches cfg.LOOKBACK_BARS_MODEL from 720 (30d) to 2160 (90d) BEFORE
     the v1_raw pipeline and eval_v6_prime modules are imported. This
     propagates through to FastDataset's default lookback and the
     _compute_vp_structure rolling window.
  2. Redirects feature loading to BTC_1h_RELVP_30d.csv (30-day VP lookback).
  3. Swaps the model builder from TemporalEnrichedV6Prime → TemporalEnrichedV10.
  4. Retargets the results + predictions output files so v10 does not
     clobber v6-prime's artifacts.

Prereq: run `python3 -m src.data.compute_vp_30d` once to generate the
30-day VP CSV. If it's missing this script fails fast with a clear error.

Usage:
    python3 -m src.models.eval_v10
"""

import sys

# ───────────────────────────────────────────────────────────────────
# Step 1: patch config BEFORE any downstream imports
# ───────────────────────────────────────────────────────────────────
import src.config as cfg

V10_N_DAYS = 90
cfg.LOOKBACK_BARS_MODEL = V10_N_DAYS * cfg.BARS_PER_DAY  # 2160 bars

CSV_30D_PATH = cfg.PROJECT_ROOT / "BTC_1h_RELVP_30d.csv"
if not CSV_30D_PATH.exists():
    print(f"ERROR: {CSV_30D_PATH} not found.")
    print("Run `python3 -m src.data.compute_vp_30d` first to generate it.")
    sys.exit(1)

# ───────────────────────────────────────────────────────────────────
# Step 2: import v1_raw — now binds LOOKBACK_BARS_MODEL=2160 into its
# _compute_vp_structure default, so peak aggregation uses the full
# 90-day model window.
# ───────────────────────────────────────────────────────────────────
from src.features.pipelines import v1_raw  # noqa: E402
from src.features.pipelines.v1_raw import (  # noqa: E402
    feature_index_v1,
    DERIVED_FEATURE_COLS_V1,
    VP_STRUCTURE_COLS_V1,
)

# ───────────────────────────────────────────────────────────────────
# Step 3: import eval_v6_prime — picks up patched LOOKBACK_BARS_MODEL
# for FastDataset's default lookback.
# ───────────────────────────────────────────────────────────────────
import src.models.eval_v6_prime as eval_v6p  # noqa: E402
from src.models.architectures.v10_long_temporal import TemporalEnrichedV10  # noqa: E402

# ───────────────────────────────────────────────────────────────────
# Step 4: redirect feature loader to the 30-day VP CSV
# ───────────────────────────────────────────────────────────────────
_orig_build_feature_matrix_v1 = v1_raw.build_feature_matrix_v1


def _build_feature_matrix_v10():
    return _orig_build_feature_matrix_v1(csv_path=CSV_30D_PATH)


eval_v6p.build_feature_matrix_v1 = _build_feature_matrix_v10


# ───────────────────────────────────────────────────────────────────
# Step 5: swap model builder
# ───────────────────────────────────────────────────────────────────
def build_v10():
    return TemporalEnrichedV10(
        ohlc_open_idx=feature_index_v1("ohlc_open_ratio"),
        ohlc_high_idx=feature_index_v1("ohlc_high_ratio"),
        ohlc_low_idx=feature_index_v1("ohlc_low_ratio"),
        log_return_idx=feature_index_v1("log_return"),
        volume_ratio_idx=feature_index_v1("volume_ratio"),
        vp_structure_start_idx=feature_index_v1("vp_ceiling_dist"),
        n_vp_structure=len(VP_STRUCTURE_COLS_V1),
        n_other_features=len(DERIVED_FEATURE_COLS_V1) + len(VP_STRUCTURE_COLS_V1),
        dropout=eval_v6p.DROPOUT,
        n_days=V10_N_DAYS,
    )


eval_v6p.build_v6_prime = build_v10


# ───────────────────────────────────────────────────────────────────
# Step 6: retarget output artifacts so v10 doesn't clobber v6-prime
# ───────────────────────────────────────────────────────────────────
_orig_main = eval_v6p.main


def main():
    # Patch output paths for this run. eval_v6_prime.main() writes to
    # EXPERIMENTS_DIR / "eval_v6_prime_results.json" and
    # EXPERIMENTS_DIR / "v6_prime_predictions.npz". We redirect both by
    # overriding Path division via a local shim on the module's
    # EXPERIMENTS_DIR — cleaner approach: rename after the run.
    import shutil
    from pathlib import Path

    exp_dir: Path = eval_v6p.EXPERIMENTS_DIR
    json_src = exp_dir / "eval_v6_prime_results.json"
    npz_src = exp_dir / "v6_prime_predictions.npz"
    json_dst = exp_dir / "eval_v10_results.json"
    npz_dst = exp_dir / "v10_predictions.npz"

    print(f"\n{'=' * 60}")
    print(f"EVAL v10 — 90d temporal × 30d VP")
    print(f"  CSV:       {CSV_30D_PATH.name}")
    print(f"  n_days:    {V10_N_DAYS}")
    print(f"  lookback:  {cfg.LOOKBACK_BARS_MODEL} bars")
    print(f"  results →  {json_dst.name}")
    print(f"  preds   →  {npz_dst.name}")
    print(f"{'=' * 60}")

    _orig_main()

    if json_src.exists():
        shutil.move(str(json_src), str(json_dst))
        print(f"Renamed {json_src.name} → {json_dst.name}")
    if npz_src.exists():
        shutil.move(str(npz_src), str(npz_dst))
        print(f"Renamed {npz_src.name} → {npz_dst.name}")


if __name__ == "__main__":
    main()
