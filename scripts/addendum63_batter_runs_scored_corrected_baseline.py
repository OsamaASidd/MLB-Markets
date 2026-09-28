"""
Addendum 63: corrected reproduction of the batter_runs_scored baseline.

Addendum 62's EDA discovered that the ORIGINAL Addendum 30/55c pipeline
(run_runs_scored in addendum55c_novig_batter_markets.py) has TWO real
join fan-out bugs it never dealt with:
  1. build_team_game_log()'s `games` has 3,858 rows but only 3,845 unique
     game_pk (pre-existing bug, documented in Addendum 37 for a DIFFERENT
     pipeline -- never fixed here).
  2. client_games has 8,678 unique event_id but only 7,029 unique game_pk
     (multiple event_id rows per real game -- NEWLY found here). This is
     the bigger one: it inflates the final pool by ~27%.

Net effect: the published baseline (n=52,981 ROI=-2.54% CI=[-4.05%,
-1.03%] FAIL) was computed on a fanned-out dataset with real duplicate
rows. This script re-fits the EXACT SAME model (same features, same
XGBClassifier hyperparameters, same 75/25 chronological split -- nothing
new tuned or searched) on the corrected, deduped pool, to get an honest
number to replace the erroneous published one.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from addendum62_batter_runs_scored_eda import build_pool  # noqa: E402

FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
            "runs_factor", "hr_factor", "k_factor", "hits_factor"]


def main():
    con = duckdb.connect(DB, read_only=True)

    print("rebuilding batter_runs_scored pool with both fan-out bugs fixed...")
    pool = build_pool(con)
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)

    split = int(len(pool) * 0.75)
    train, test = pool.iloc[:split].copy(), pool.iloc[split:].copy()
    print(f"\n  TRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    X_train, y_train = train[FEATURES], train["win"].astype(int)
    X_test, y_test = test[FEATURES], test["win"].astype(int)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
        reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred >= 0.5)
    print(f"  AUC TEST={auc:.4f}  accuracy TEST={acc:.4f}  (same untuned hyperparameters as Addendum 30)")

    test = test.copy()
    test["model_prob"] = test_pred
    test["edge"] = test["model_prob"].values - test["market_prob"].values
    sub = test[(test.odds < 0) & (test.edge > 0.0)]
    if len(sub) == 0:
        print("\n  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\n=== CORRECTED result (fan-out bugs fixed, same model/features/split as Addendum 30) ===")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    print(f"\n  published (buggy) baseline was: n=52,981  ROI=-2.54%  CI=[-4.05%,-1.03%]  FAIL")

    con.close()


if __name__ == "__main__":
    main()
