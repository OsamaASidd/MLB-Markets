"""
Addendum 47: walk-forward cross-validated hyperparameter search for
batter_runs_scored, looking for a genuinely generalizing "sweet spot" --
NOT a re-tune against the already-observed Addendum 43b test-set ROI
(-0.54%, n=693). That number has already been looked at; searching for
hyperparameters that flip THAT SAME fixed test set positive would be
test-set peeking, the exact failure mode this project has spent 46 prior
addenda avoiding (see HANDOFF.md, and Addenda 39/41/46's holdout work,
which specifically exists to catch this).

Legitimate alternative: sequential (expanding-window) walk-forward
validation across batter_runs_scored's real 2026-06-13 to 2026-07-27
window. At each of 5 chronological folds, train ONLY on data strictly
before that fold (never touching that fold's outcomes), select
hyperparameters via CV Poisson deviance on the training data ONLY (a
WIDER, richer grid than Addendum 43b's, since a real "sweet spot" search
is what's being asked for -- but still never scored by ROI), then predict
on that fold. Every fold's held-out bets are POOLED and the official gate
is evaluated EXACTLY ONCE across the full pooled set at the very end --
no fold is inspected individually before that, so there's no opportunity
to discard an unfavorable fold or re-tune after seeing partial results.

Reuses Addendum 43b's load_pool/select_features/verify_* functions
unmodified (import). Does not modify that file.
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from addendum38_pickhistory_enrichment import CACHE_DB, RUNS_SCORED_JOIN, RUNS_SCORED_SELECT  # noqa: E402
from addendum43b_poisson_regression_pickhistory import (  # noqa: E402
    load_pool, sanity_check_rowcounts, select_features,
)

N_FOLDS = 5
N_CV_FOLDS = 5  # inner CV folds for hyperparameter selection within each walk-forward training slice

# WIDER pre-specified grid than Addendum 43b's -- a genuine sweet-spot
# search, but still selected ONLY via CV Poisson deviance on training data,
# fixed BEFORE looking at any fold's held-out bets.
POISSON_ALPHA_GRID = {"poissonregressor__alpha": [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}
XGB_PARAM_GRID = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "n_estimators": [100, 200, 300, 500, 800],
    "min_child_weight": [5, 10, 20, 30, 50],
}
XGB_FIXED_PARAMS = dict(
    subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
    objective="count:poisson", missing=np.nan, random_state=0, n_jobs=1,
)


def cv_select_model_wide(X_train, y_train):
    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=0)

    poisson_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(max_iter=2000))
    search_poisson = GridSearchCV(poisson_pipe, POISSON_ALPHA_GRID, scoring="neg_mean_poisson_deviance", cv=cv, n_jobs=1)
    search_poisson.fit(X_train, y_train)

    search_xgb = GridSearchCV(
        xgb.XGBRegressor(**XGB_FIXED_PARAMS), XGB_PARAM_GRID,
        scoring="neg_mean_poisson_deviance", cv=cv, n_jobs=joblib.cpu_count(),
    )
    with joblib.parallel_backend("threading"):
        search_xgb.fit(X_train, y_train)

    if search_xgb.best_score_ > search_poisson.best_score_:
        return "XGBRegressor(count:poisson)", search_xgb.best_estimator_, search_xgb.best_score_, search_xgb.best_params_
    return "PoissonRegressor", search_poisson.best_estimator_, search_poisson.best_score_, search_poisson.best_params_


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    df, base_n, score_cols = load_pool(con, "runs_scored", RUNS_SCORED_JOIN, RUNS_SCORED_SELECT)
    sanity_check_rowcounts(df, base_n, "runs_scored")
    df = df.dropna(subset=["actual_value", "line", "odds", "pick_side"]).reset_index(drop=True)
    print(f"n={len(df)}  date range: {df.game_date.min()} -> {df.game_date.max()}")

    # 5 equal-sized chronological folds (already sorted by date in load_pool)
    fold_id = pd.qcut(np.arange(len(df)), N_FOLDS, labels=False)
    df["fold"] = fold_id

    all_bets = []
    print("\n=== walk-forward folds (expanding window: train = all strictly-prior folds) ===")
    for f in range(1, N_FOLDS):  # fold 0 has no prior data to train on -- skipped, not evaluated
        train_df = df[df["fold"] < f].copy()
        test_df = df[df["fold"] == f].copy()
        features, score_keep, new_keep, _ = select_features(train_df, score_cols, min_cov=max(50, int(len(train_df) * 0.05)))
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}  "
              f"features kept={len(features)} ({len(score_keep)} score_* + {len(new_keep)} cache)")
        if len(train_df) < 200 or len(test_df) < 20 or not features:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        X_train = train_df[features]
        y_train = train_df["actual_value"]
        name, model, cv_score, params = cv_select_model_wide(X_train, y_train)
        print(f"  fold {f}: selected {name}  CV_neg_poisson_deviance={cv_score:.4f}  params={params}")

        X_test = test_df[features]
        mu = np.clip(model.predict(X_test), 1e-6, None)
        k = np.floor(test_df["line"].values)
        p_over = poisson.sf(k, mu)
        p_under = poisson.cdf(k, mu)
        model_prob = np.where(test_df["pick_side"].values == "over", p_over, p_under)
        market_prob = implied_prob(test_df["odds"].values)
        edge = model_prob - market_prob

        fold_bets = test_df[["game_date", "odds", "win"]].copy()
        fold_bets["edge"] = edge
        fold_bets["fold"] = f
        all_bets.append(fold_bets)

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    print("(official gate touched EXACTLY ONCE below, on the full pooled set -- no per-fold peeking above)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    print("\nFor reference, Addendum 43b's single fixed-split result was:")
    print("  n=693  ROI=-0.54%  CI=[-6.38,5.29]  FAIL")
    print("This walk-forward result pools far more held-out bets across multiple time periods, "
          "with a wider (but still CV-deviance-selected, never ROI-selected) hyperparameter grid -- "
          "a genuine sweet-spot search, not a re-run against an already-observed number.")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
