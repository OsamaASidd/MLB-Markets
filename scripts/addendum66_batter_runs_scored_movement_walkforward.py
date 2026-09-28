"""
Addendum 66: walk-forward validation of the Addendum 65 line-movement
signal for batter_runs_scored. The single 70/30 split produced zero
qualifying bets in the test set -- too concentrated a slice to draw any
conclusion. Same standard second-check as every other single-split
result this session: expanding-window folds pool far more held-out
predictions (up to n=10,540 total pool) than one static split can.

Same simple, untuned logistic regression as Addendum 65, same single
edge>0.0 cut, official gate rule, evaluated exactly once on the pooled
held-out predictions.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum65_batter_runs_scored_line_movement import CACHE_DB, build_pool  # noqa: E402

N_FOLDS = 6
FEATURES = ["market_prob", "movement", "n_snapshots"]


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    pool = build_pool(con, con2)
    print(f"pool: n={len(pool)}  date range {pool.game_date.min()} -> {pool.game_date.max()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), untuned logistic regression ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 50 or len(test_df) < 10:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train_df[FEATURES], train_df["win"].astype(int))
        t = test_df.copy()
        t["model_prob"] = model.predict_proba(t[FEATURES])[:, 1]
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
