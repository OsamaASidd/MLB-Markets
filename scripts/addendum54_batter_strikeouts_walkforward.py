"""
Addendum 54: genuine walk-forward validation of Addendum 53's
batter_strikeouts "enriched" result (n=1,211, ROI=+1.24%, CI=[-3.08,5.55],
mechanical PASS).

WHY: Addendum 53's enrichment added ZERO real predictive power (marginal
CV-AUC contribution -0.0002 versus the no-enrichment baseline) -- the
FAIL->PASS flip came from a different subset of bets crossing the edge
threshold, not a better model. That is the exact fingerprint of the
near-misses that evaporated under honest holdout testing earlier this
session (batter_total_bases, batter_runs_scored's original tuned-XGBoost
result). Applying the same walk-forward method that separated the real
finding (runs_scored, confirmed) from the fake ones (total_bases,
confirmed fake) here too, rather than trusting a single split.

Reuses build_pool/build_games_v3/attach_features from
addendum53_batter_strikeouts_enrichment.py UNMODIFIED (import). Same
CV-AUC-only selection, ROI touched exactly once on the full pooled
held-out set at the end.
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum53_batter_strikeouts_enrichment import build_pool, build_games_v3, attach_features, ENRICHED_FEATURES  # noqa: E402
from addendum37_historical_enrichment import build_box_ids  # noqa: E402

N_FOLDS = 6
PARAM_GRID_WIDE = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "n_estimators": [100, 200, 300, 500],
    "min_child_weight": [5, 10, 20, 30, 50],
}
FIXED_PARAMS = dict(
    subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
    eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
)


def cv_select(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID_WIDE, scoring="roc_auc", cv=cv, n_jobs=joblib.cpu_count())
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_, search.best_params_


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building batter_strikeouts pool (Addendum 53, unmodified import)...")
    pool = build_pool(con)
    games_v3 = build_games_v3(con, con2)
    box_ids = build_box_ids(con)
    n_before = len(pool)
    pool = attach_features(pool, games_v3, box_ids, con2)
    print(f"row-count sanity check: before={n_before}  after={len(pool)}  {'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    print(f"total pool n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds, ENRICHED_FEATURES) ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 500 or len(test_df) < 50:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        X_train, y_train = train_df[ENRICHED_FEATURES], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[ENRICHED_FEATURES])[:, 1]
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

    print("\nFor reference:")
    print("  Addendum 53 single-split (enriched): n=1211  ROI=1.24%  CI=[-3.08,5.55]  PASS (marginal AUC contribution ~0, suspect)")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
