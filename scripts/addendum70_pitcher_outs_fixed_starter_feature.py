"""
Addendum 70: pitcher_outs, re-run with the leaked starter-quality feature
fixed. add_starter_game_features() (addendum37_historical_enrichment.py)
uses cache_mlb_historical_pitcher_statcast keyed only by (player_id,
season) -- confirmed (Addendum 68) to be each pitcher's FINAL, whole-
season ERA/xERA/xBA, applied identically to every start that season,
including starts before the season ended. Genuine lookahead leakage.
Addendum 51 (pitcher_outs full treatment) used this exact feature group
(STARTER_GAME_V2) -- this script reruns it with Addendum 68's
build_starter_rolling_era/attach_fixed_starter_feature (point-in-time,
shift(1)-before-rolling) substituted in, everything else unchanged
(same pool, same bullpen/park enrichment, same walk-forward discipline,
same PARAM_GRID_WIDE).
"""
import os
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
from xgboost_individual_markets import DB, norm_name, implied_prob, profit, stat, gate  # noqa: E402
from test_backfilled_markets import build_market_dataset  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features, add_park_features, TEAM_NAME_ALIAS  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum49_h2h_walkforward import PARAM_GRID_WIDE, FIXED_PARAMS  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402

N_FOLDS = 6
BULLPEN_V2 = ["bullpen_era_diff", "bullpen_k9_diff"]
PARK_V2 = ["lf_distance", "cf_distance", "rf_distance",
           "lf_wall_height", "cf_wall_height", "rf_wall_height",
           "cf_compass_degrees", "is_dome"]
STARTER_FIXED = ["sp_roll_era_diff", "sp_roll_ip_diff"]
BASE_FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                  "runs_factor", "hr_factor", "k_factor", "hits_factor"]
FEATURES_FIXED = BASE_FEATURES + BULLPEN_V2 + STARTER_FIXED + PARK_V2


def cv_select(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID_WIDE,
                           scoring="roc_auc", cv=cv, n_jobs=os.cpu_count())
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_, search.best_params_


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building real backfilled-odds pitcher_outs pool matched to real box-score outcomes...")
    box = con.execute("""
        SELECT game_pk, player_name, outs AS pitcher_outs_stat, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    po_cache = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"
    pool = build_market_dataset(po_cache, "pitcher_outs", "pitcher_outs_stat", box, ["Pitcher"], starter_only=True)
    print(f"  pool (both over+under sides): n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")

    games_v3, _ = build_fuller_games(con, con2)
    n0 = len(games_v3)

    cg = con.execute("SELECT event_id, game_pk, home_team, away_team FROM client_games").fetchdf()
    cg["game_pk"] = pd.to_numeric(cg["game_pk"], errors="coerce")
    g2 = games_v3[["game_pk", "home_team_id", "away_team_id"]].merge(
        cg[["game_pk", "home_team", "away_team"]], on="game_pk", how="inner")
    id_to_name = {}
    for tid, name in g2.groupby("home_team_id")["home_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    for tid, name in g2.groupby("away_team_id")["away_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    id_to_name = {tid: TEAM_NAME_ALIAS.get(name, name) for tid, name in id_to_name.items()}

    print("\napplying bullpen + park enrichment (Addendum 37, unmodified)...")
    games_full = add_bullpen_features(con2, games_v3, id_to_name)
    assert len(games_full) == n0
    games_full = add_park_features(con, con2, games_full)
    assert len(games_full) == n0

    print("applying CORRECTED point-in-time starter rolling ERA (Addendum 68)...")
    roll = build_starter_rolling_era(con)
    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    games_full = games_full.merge(op, on="game_pk", how="left")
    assert len(games_full) == n0

    roll_h = roll.rename(columns={"player_id": "home_starter_id", "sp_roll_era": "home_sp_roll_era", "sp_roll_ip": "home_sp_roll_ip"})
    roll_h["home_starter_id"] = roll_h["home_starter_id"].astype("Int64")
    games_full = games_full.merge(roll_h[["home_starter_id", "game_pk", "home_sp_roll_era", "home_sp_roll_ip"]],
                                   on=["home_starter_id", "game_pk"], how="left")
    roll_a = roll.rename(columns={"player_id": "away_starter_id", "sp_roll_era": "away_sp_roll_era", "sp_roll_ip": "away_sp_roll_ip"})
    roll_a["away_starter_id"] = roll_a["away_starter_id"].astype("Int64")
    games_full = games_full.merge(roll_a[["away_starter_id", "game_pk", "away_sp_roll_era", "away_sp_roll_ip"]],
                                   on=["away_starter_id", "game_pk"], how="left")
    assert len(games_full) == n0, "starter rolling ERA merge fan-out -- STOP"
    games_full["sp_roll_era_diff"] = games_full["away_sp_roll_era"] - games_full["home_sp_roll_era"]
    games_full["sp_roll_ip_diff"] = games_full["home_sp_roll_ip"] - games_full["away_sp_roll_ip"]

    merge_cols = (["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                    "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
                  + BULLPEN_V2 + STARTER_FIXED + PARK_V2)
    games_full_dedup = games_full.drop_duplicates(subset=["game_pk"])[merge_cols]
    n_before = len(pool)
    pool_full = pool.merge(games_full_dedup, on="game_pk", how="inner")
    print(f"\n  final feature merge: before={n_before:,}  after={len(pool_full):,}  "
          f"{'OK' if len(pool_full) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool_full) == n_before

    pool_full["game_date"] = pd.to_datetime(pool_full["game_date"])
    pool_full["side_code"] = pool_full["side"].astype("category").cat.codes
    pool_full["elo_diff"] = (pool_full["home_elo"] + 24) - pool_full["away_elo"]
    pool_full["l10_diff"] = pool_full["home_l10"] - pool_full["away_l10"]
    pool_full["fatigue_diff"] = pool_full["home_fatigue"] - pool_full["away_fatigue"]
    pool_full["market_prob"] = implied_prob(pool_full["odds"])
    pool_full = pool_full.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"  pool ready for walk-forward: n={len(pool_full):,}  "
          f"date range {pool_full.game_date.min().date()} -> {pool_full.game_date.max().date()}")

    fold_id = pd.qcut(np.arange(len(pool_full)), N_FOLDS, labels=False)
    pool_full["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds) ===")
    for f in range(1, N_FOLDS):
        train_df = pool_full[pool_full["fold"] < f]
        test_df = pool_full[pool_full["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df):,}  test n={len(test_df):,}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        X_train, y_train = train_df[FEATURES_FIXED], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[FEATURES_FIXED])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled):,} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    print("\n  For reference, Addendum 51 (leaked starter feature): n=? ROI=? -- see original addendum log")
    print("  Addendum 30 (baseline, no enrichment): n=3,334 ROI=-1.21% CI=[-7.2,4.78] FAIL")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
