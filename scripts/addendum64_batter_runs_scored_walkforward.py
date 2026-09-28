"""
Addendum 64: walk-forward CV-tuned model for batter_runs_scored on the
CORRECTED pool (both fan-out bugs from Addendum 62/63 fixed). Same
discipline as every other walk-forward this session: CV-AUC-only
hyperparameter selection per fold (never ROI), expanding-window folds,
held-out predictions pooled across folds, official gate touched exactly
once at the very end.

Reuses PARAM_GRID_WIDE/FIXED_PARAMS/cv_select unmodified from
addendum49_h2h_walkforward.py (the same fixed, pre-established grid used
for h2h/pitcher_strikeouts) -- not a new grid tuned for this market.

Given Addendum 62's EDA already found ~zero correlation between every
non-market_prob feature and the actual outcome (at n=156,788, the largest
sample tested this session), this is not expected to flip the verdict --
but it's the correct way to get the most decisive, best-possible number
on this corrected pool, using the full walk-forward pooled sample rather
than one static split.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum62_batter_runs_scored_eda import build_pool  # noqa: E402
from addendum63_batter_runs_scored_corrected_baseline import FEATURES  # noqa: E402

N_FOLDS = 5


def main():
    con = duckdb.connect(DB, read_only=True)

    print("rebuilding batter_runs_scored pool (both fan-out bugs fixed)...")
    pool = build_pool(con)
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"  final pool n={len(pool)}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds) ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        X_train, y_train = train_df[FEATURES], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[FEATURES])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()


if __name__ == "__main__":
    main()
