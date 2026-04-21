# STATUS: LIVE — canonical K-selection for the frozen global-outer HMM.
# Recommends K_outer=3 (by CV-LL), K_inner=[5,5,5] (by CV-LL across all three
# outer states). Results: results/k_selection_hier_global/. See HMM_LOG.md.
# Supersedes hmm_k_selection_hierarchical.py.
"""K-selection for hmm_hierarchical_global.py, per layer.

Outer layer (global macro panel, 3y rolling rank→inv-normal):
  features: dxy, hy_oas, net_fed_liq_yoy, global_m2_yoy
  Sweeps K ∈ {2..5}, reports: LL, BIC, BIC_neff, AIC, CV-LL, BTC-VR.

Inner layer (commodity CS ranks, unchanged):
  For each outer state, sweeps K ∈ {2..5}.

Same protocol/caveats as hmm_k_selection_hierarchical.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_hierarchical_global import build_outer_global, fit_best
from hmm_k_selection_hierarchical import sweep, build_btc_fwd
from hmm_regime import load_long_history, build_commodity_features

HERE = Path(__file__).parent
DATA = HERE / "data"
PROJECT_ROOT = HERE.parent
DATA_PROJECT = PROJECT_ROOT / "data"
OUT = HERE / "results" / "k_selection_hier_global"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-outer-chosen", type=int, default=None,
                   help="If set, run inner sweep at this K_outer instead of argmax-CV-LL")
    p.add_argument("--outer-csv", type=str,
                   default=str(DATA_PROJECT / "regime_global_outer.csv"),
                   help="Path to regime_global_outer.csv")
    args = p.parse_args()

    outer_path = Path(args.outer_csv)
    if not outer_path.exists():
        raise FileNotFoundError(
            f"{outer_path} not found. Run:\n"
            f"  python3 -m src.data.fetch_regime --global-outer --fred-key YOUR_KEY"
        )

    # ---- Build outer (global macro, 3y rank→inv-normal) ----
    rat = build_outer_global(outer_path)

    # ---- Build inner (commodity CS ranks) ----
    df = load_long_history()
    com = build_commodity_features(df)
    idx = rat.index.intersection(com.index)
    rat = rat.loc[idx]
    com = com.loc[idx]
    Xr, Xc = rat.values, com.values

    print(f"Panel: {len(idx)} days, {idx[0].date()} → {idx[-1].date()}")
    print(f"  outer feats: {list(rat.columns)}  (d={Xr.shape[1]})")
    print(f"  inner feats: {list(com.columns)}  (d={Xc.shape[1]})")

    y_btc = build_btc_fwd(idx)

    # Outer sweep
    outer_df = sweep(Xr, y_btc, label="OUTER: global macro (dxy, credit_spread, net_fed_liq_yoy, global_m2_yoy)")
    outer_df.to_csv(OUT / "outer_sweep.csv", index=False)

    # Pick K_outer
    if args.k_outer_chosen is not None:
        K_outer = args.k_outer_chosen
        print(f"\n[User-specified K_outer = {K_outer}]")
    else:
        K_outer = int(outer_df.loc[outer_df['CV_ll_test'].idxmax(), 'K'])
        print(f"\n[Auto-picked K_outer = {K_outer} by CV-LL]")

    # Refit outer at K_outer, get labels
    _, m_outer, _ = fit_best(Xr, K_outer)
    outer_labels = m_outer.predict(Xr)

    # Inner sweeps — one per outer state
    inner_dfs = {}
    for s in range(K_outer):
        mask = outer_labels == s
        Xs = Xc[mask]
        ys = y_btc[mask]
        if len(Xs) < 200:
            print(f"\n[S{s}: only {len(Xs)} rows — skip]")
            continue
        df_s = sweep(Xs, ys, label=f"INNER | outer state S{s} (n={len(Xs)})")
        df_s.insert(0, "outer_state", s)
        df_s.to_csv(OUT / f"inner_sweep_S{s}.csv", index=False)
        inner_dfs[s] = df_s

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Outer recommendation:")
    k1 = int(outer_df.loc[outer_df['CV_ll_test'].idxmax(), 'K'])
    k2 = int(outer_df.loc[outer_df['BIC_neff'].idxmin(), 'K'])
    print(f"  K_outer = {k1} (CV-LL) / {k2} (BIC_neff)")
    print(f"Inner recommendations:")
    for s, df_s in inner_dfs.items():
        k1 = int(df_s.loc[df_s['CV_ll_test'].idxmax(), 'K'])
        k2 = int(df_s.loc[df_s['BIC_neff'].idxmin(), 'K'])
        vr_row = df_s.loc[df_s['BTC_fwd20_VR'].idxmax()] if df_s['BTC_fwd20_VR'].notna().any() else None
        k3 = int(vr_row['K']) if vr_row is not None else None
        print(f"  S{s}: K_inner = {k1} (CV-LL) / {k2} (BIC_neff) / {k3} (BTC-VR)")

    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
