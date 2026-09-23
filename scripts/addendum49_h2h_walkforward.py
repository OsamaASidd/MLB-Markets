"""
Addendum 49: walk-forward cross-validated evaluation of h2h, on top of
Addendum 48's fuller game log (7,028 games vs the old 3,845-game ceiling).

WHY: Addendum 48's single per-year 75/25 split got h2h to its closest-ever
result (n=276, ROI +8.4%, CI=[-1.6,18.4] -- still FAIL, CI lower bound just
1.6 points from crossing zero). Rather than re-tune against that specific
already-observed test set (test-set peeking -- the exact failure mode this
project has spent 48 prior addenda avoiding, see HANDOFF.md), this applies
the SAME legitimate walk-forward method that turned batter_runs_scored into
a real pass (Addendum 47): sequential expanding-window folds, hyperparameters
selected via CV-AUC on TRAINING data only within each fold, held-out
predictions from every fold pooled together, official gate touched EXACTLY
ONCE on the full pooled set at the very end. No fold is inspected
individually before pooling -- there's no opportunity to discard an
unfavorable fold.

Reuses build_fuller_games/build_pool/build_features from
addendum48_h2h_totals_fuller_gamelog.py unmodified (import). Does not
modify that file.
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
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, build_pool, build_features, FEATURES

N_FOLDS = 6  # more folds than Addendum 47's 5 -- h2h's pool is bigger, can support finer granularity

# Wider pre-specified grid than the single-split version, same spirit as
# Addendum 47's "genuine sweet-spot search" -- selected ONLY via CV AUC on
# each fold's training data, never against any fold's held-out ROI.
PARAM_GRID_WIDE = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "n_estimators": [100, 200, 300, 500, 800],
    "min_child_weight": [5, 10, 20, 30, 50],
}
FIXED_PARAMS = dict(
    subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
    eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
)


def attach_enrichment(con, con2, pool, id_to_name):
    """Same enrichment steps as addendum48.evaluate() -- factored out here
    so it can be applied once, before folding, without also doing the
    single-split fit/eval that function bundles together."""
    from addendum37_historical_enrichment import add_bullpen_features
    pool = add_bullpen_features(con2, pool, id_to_name)
    pool["season"] = pd.to_datetime(pool["game_date"]).dt.year.astype("int64")

    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    pool = pool.merge(op, on="game_pk", how="left")

    pstat = con2.execute("SELECT player_id, season, xera, era, xba_allowed FROM cache_mlb_historical_pitcher_statcast").fetchdf().drop_duplicates(subset=["player_id", "season"])
    pstat["player_id"] = pstat["player_id"].astype("Int64")
    pstat["season"] = pstat["season"].astype("int64")
    home_p = pstat.rename(columns={"player_id": "home_starter_id", "xera": "home_sp_xera", "era": "home_sp_era", "xba_allowed": "home_sp_xba"})
    pool = pool.merge(home_p, on=["home_starter_id", "season"], how="left")
    away_p = pstat.rename(columns={"player_id": "away_starter_id", "xera": "away_sp_xera", "era": "away_sp_era", "xba_allowed": "away_sp_xba"})
    pool = pool.merge(away_p, on=["away_starter_id", "season"], how="left")
    pool["starter_xera_diff"] = pool["away_sp_xera"] - pool["home_sp_xera"]
    pool["starter_era_diff"] = pool["away_sp_era"] - pool["home_sp_era"]
    pool["starter_xba_diff"] = pool["away_sp_xba"] - pool["home_sp_xba"]

    weather_venue = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    venue_by_gamepk = pool[["game_pk", "event_id"]].drop_duplicates(subset=["game_pk"]).merge(weather_venue, on="event_id", how="left")[["game_pk", "venue_name"]]
    park_dims = con2.execute("SELECT venue_name, lf_distance, cf_distance, rf_distance, lf_wall_height, cf_wall_height, rf_wall_height, is_dome FROM cache_mlb_park_dimensions").fetchdf().drop_duplicates(subset=["venue_name"])
    park_orient = con2.execute("SELECT venue_name, cf_compass_degrees FROM cache_mlb_ballpark_orientation").fetchdf().drop_duplicates(subset=["venue_name"])
    park_static = park_dims.merge(park_orient, on="venue_name", how="outer")
    park_static["is_dome"] = park_static["is_dome"].astype("float64")
    venue_feats = venue_by_gamepk.merge(park_static, on="venue_name", how="left").drop(columns=["venue_name"])
    pool = pool.merge(venue_feats, on="game_pk", how="left")
    return pool


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

    print("building fuller game log (Addendum 48, unmodified import)...")
    games, id_to_name = build_fuller_games(con, con2)

    pool = build_pool(con, games, "h2h")
    n_before = len(pool)
    pool = attach_enrichment(con, con2, pool, id_to_name)
    print(f"row-count sanity check: before={n_before}  after={len(pool)}  {'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool = build_features(pool)
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"]).sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"total pool n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds) ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
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
    print("(official gate touched EXACTLY ONCE below, on the full pooled set)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    print("\nFor reference:")
    print("  Addendum 48 single-split result: n=276  ROI=8.4%  CI=[-1.6,18.4]  FAIL")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
