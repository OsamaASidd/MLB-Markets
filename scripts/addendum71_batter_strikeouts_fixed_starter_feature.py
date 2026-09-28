"""
Addendum 71: batter_strikeouts, re-run with the leaked starter-quality
feature fixed. add_opp_pitcher_features() (addendum37_historical_enrichment.py)
uses the SAME cache_mlb_historical_pitcher_statcast season-aggregate table
(confirmed leaky in Addendum 68/70) for the opposing starter's quality.
This script substitutes Addendum 68's build_starter_rolling_era
(point-in-time, shift(1)-before-rolling) for the batter's SPECIFIC
opposing starter, everything else unchanged (same pool, same bullpen/
park enrichment, same CV-AUC-tuned XGBoost, same 75/25 chronological
split Addendum 53 used).

Note: Addendum 53 already independently found and fixed the OTHER known
fan-out bug (client_games multi-event_id-per-game_pk) for this market by
using build_fuller_games -- only the starter-quality leak is new here.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, norm_name, implied_prob  # noqa: E402
from test_backfilled_markets import build_market_dataset  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features, add_park_features, build_box_ids  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402
from addendum53_batter_strikeouts_enrichment import BULLPEN_V2, PARK_V2, build_pool  # noqa: E402

BASE_FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                  "runs_factor", "hr_factor", "k_factor", "hits_factor"]
FIXED_STARTER = ["opp_sp_roll_era", "opp_sp_roll_ip"]
ENRICHED_FEATURES_FIXED = BASE_FEATURES + BULLPEN_V2 + FIXED_STARTER + PARK_V2

GAME_LEVEL_COLS = ["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                    "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"] + BULLPEN_V2 + PARK_V2

BASELINE_ADDENDUM30 = {"n": 41489, "roi": -0.87, "lo": -3.1, "hi": 1.35, "verdict": "FAIL"}


def build_games_v3_fixed(con, con2):
    games, id_to_name = build_fuller_games(con, con2)
    n0 = len(games)
    games = add_bullpen_features(con2, games, id_to_name)
    assert len(games) == n0
    games = add_park_features(con, con2, games)
    assert len(games) == n0
    games["game_date"] = pd.to_datetime(games["game_date"])
    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    games = games.merge(op, on="game_pk", how="left")
    assert len(games) == n0
    games["season"] = games["game_date"].dt.year.astype("int64")
    return games


def split_chronological(pool):
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.75)
    return pool.iloc[:cut].copy(), pool.iloc[cut:].copy()


def tune_and_evaluate(label, train, test, features):
    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True)
    import joblib
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)
    test_pred = search.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred >= 0.5)
    print(f"\n  [{label}]  CV AUC(train)={search.best_score_:.4f}  test AUC={auc:.4f}  test acc={acc:.4f}  params={search.best_params_}")

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  edge>0.0: n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return s, verdict


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building batter_strikeouts pool (Addendum 53 pipeline, unmodified)...")
    pool = build_pool(con)
    games_v3 = build_games_v3_fixed(con, con2)
    box_ids = build_box_ids(con)

    n0 = len(pool)
    pool = pool.merge(games_v3[GAME_LEVEL_COLS], on="game_pk", how="inner")
    pool["name_norm"] = pool["tiebreak"]
    n1 = len(pool)
    pool = pool.merge(box_ids[["game_pk", "name_norm", "team_id"]], on=["game_pk", "name_norm"], how="left")
    ctx = games_v3.drop_duplicates(subset=["game_pk"])[
        ["game_pk", "home_team_id", "away_team_id", "home_starter_id", "away_starter_id"]]
    pool = pool.merge(ctx, on="game_pk", how="left")
    pool["opp_starter_id"] = pd.Series(np.where(pool["team_id"] == pool["home_team_id"],
                                       pool["away_starter_id"], pool["home_starter_id"]), index=pool.index).astype("Int64")
    print(f"  row-count check (game-level + opp-starter-id merges): before={n0} after={len(pool)} "
          f"{'OK' if len(pool) == n0 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n0

    print("attaching CORRECTED point-in-time opposing-starter rolling ERA (Addendum 68)...")
    roll = build_starter_rolling_era(con)
    roll_o = roll.rename(columns={"player_id": "opp_starter_id", "sp_roll_era": "opp_sp_roll_era", "sp_roll_ip": "opp_sp_roll_ip"})
    roll_o["opp_starter_id"] = roll_o["opp_starter_id"].astype("Int64")
    n2 = len(pool)
    pool = pool.merge(roll_o[["opp_starter_id", "game_pk", "opp_sp_roll_era", "opp_sp_roll_ip"]],
                       on=["opp_starter_id", "game_pk"], how="left")
    print(f"  row-count check (rolling ERA merge): before={n2} after={len(pool)} "
          f"{'OK' if len(pool) == n2 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n2

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"])

    train, test = split_chronological(pool)
    print(f"\n  TRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    result_a, verdict_a = tune_and_evaluate("baseline (no enrichment)", train, test, BASE_FEATURES)
    result_b, verdict_b = tune_and_evaluate("enriched, FIXED starter feature", train, test, ENRICHED_FEATURES_FIXED)

    print(f"\n  published baseline (Addendum 30): {BASELINE_ADDENDUM30}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
