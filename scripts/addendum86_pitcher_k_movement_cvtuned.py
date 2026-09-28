"""
Addendum 86: CV-tuned re-test of the pitcher_strikeouts line-movement
signal (currently MONITORING: n=156, ROI=+2.24%, CI=[-10.37,14.86]).
Every prior line-movement script (56/56b/60/61/65/66/69) deliberately
used an UNTUNED logistic regression, specifically to keep that first
test simple and pre-registered. This is the first genuine hyperparameter
search on this signal -- same CV-AUC-only selection discipline (never
ROI) used for every XGBoost model this session, same PARAM_GRID_WIDE/
FIXED_PARAMS imported unmodified from Addendum 49, applied to the exact
same pool/features (market_prob, movement, n_snapshots) Addendum 61 used.

Same walk-forward folds, same single edge>0.0 cut, gate touched exactly
once on the pooled held-out set. Reported honestly regardless of outcome.
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
from addendum61_pitcher_k_movement_walkforward import CACHE_DB, build_pool, FEATURES  # noqa: E402

N_FOLDS = 5


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    pool = build_pool(con, con2)
    print(f"pool: n={len(pool)}  date range {pool.game_date.min()} -> {pool.game_date.max()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), CV-AUC-tuned XGBoost ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 50 or len(test_df) < 10:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
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
    if len(sub) == 0:
        print("\n  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
