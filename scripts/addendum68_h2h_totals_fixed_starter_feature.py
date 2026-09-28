"""
Addendum 68: replace the leaked starter-quality feature (season-aggregate
xera/era/xba from cache_mlb_historical_pitcher_statcast, keyed only by
(player_id, season) -- confirmed via DESCRIBE + row check to be one row
per pitcher per season, i.e. that pitcher's FINAL, whole-season stat line
applied identically to every start he made that year, including starts
that happened before the season ended) with a genuinely point-in-time
rolling ERA built directly from real boxscore pitcher lines (same
shift(1)-before-rolling discipline as Addendum 58's pitcher_strikeouts
dedicated model).

Starter identity comes from `opposing_pitcher` (home_starter_id/
away_starter_id), joined directly to boxscore.player_id -- no name
normalization needed, avoids that whole class of matching error.

Re-runs the EDA correlation check on the corrected feature first (sanity
check that the fix actually changed the number), then a full CV-tuned
walk-forward test for h2h and totals.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, build_pool, build_features  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402

N_FOLDS = 5

FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
            "runs_factor", "hr_factor", "k_factor", "hits_factor",
            "bullpen_era_diff", "bullpen_k9_diff",
            "sp_roll_era_diff", "sp_roll_ip_diff",
            "lf_distance", "cf_distance", "rf_distance",
            "lf_wall_height", "cf_wall_height", "rf_wall_height", "cf_compass_degrees", "is_dome"]


def build_starter_rolling_era(con):
    """Point-in-time rolling ERA per starting pitcher: shift(1) BEFORE any
    rolling/expanding aggregation, same discipline as Addendum 58."""
    box = con.execute("""
        SELECT player_id, game_pk, game_date, innings_pitched, pitcher_earned_runs
        FROM boxscore WHERE position_type = 'Pitcher' AND is_starter = True
    """).fetchdf()
    box["game_date"] = pd.to_datetime(box["game_date"])
    box["ip_float"] = pd.to_numeric(box["innings_pitched"], errors="coerce")
    box = box.dropna(subset=["ip_float", "pitcher_earned_runs", "game_date"])
    box = box.sort_values(["player_id", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    g = box.groupby("player_id")
    box["roll_er_sum"] = g["pitcher_earned_runs"].transform(lambda s: s.shift(1).rolling(8, min_periods=2).sum())
    box["roll_ip_sum"] = g["ip_float"].transform(lambda s: s.shift(1).rolling(8, min_periods=2).sum())
    box["sp_roll_era"] = 9 * box["roll_er_sum"] / box["roll_ip_sum"]
    box["sp_roll_ip"] = box["roll_ip_sum"] / 8
    return box[["player_id", "game_pk", "sp_roll_era", "sp_roll_ip"]].drop_duplicates(subset=["player_id", "game_pk"])


def attach_fixed_starter_feature(con, pool):
    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    n0 = len(pool)
    pool = pool.merge(op, on="game_pk", how="left")
    assert len(pool) == n0, "opposing_pitcher merge fan-out -- STOP"

    roll = build_starter_rolling_era(con)
    roll_h = roll.rename(columns={"player_id": "home_starter_id", "sp_roll_era": "home_sp_roll_era", "sp_roll_ip": "home_sp_roll_ip"})
    roll_h["home_starter_id"] = roll_h["home_starter_id"].astype("Int64")
    pool = pool.merge(roll_h[["home_starter_id", "game_pk", "home_sp_roll_era", "home_sp_roll_ip"]],
                       on=["home_starter_id", "game_pk"], how="left")
    roll_a = roll.rename(columns={"player_id": "away_starter_id", "sp_roll_era": "away_sp_roll_era", "sp_roll_ip": "away_sp_roll_ip"})
    roll_a["away_starter_id"] = roll_a["away_starter_id"].astype("Int64")
    pool = pool.merge(roll_a[["away_starter_id", "game_pk", "away_sp_roll_era", "away_sp_roll_ip"]],
                       on=["away_starter_id", "game_pk"], how="left")
    assert len(pool) == n0, "starter rolling ERA merge fan-out -- STOP"

    pool["sp_roll_era_diff"] = pool["away_sp_roll_era"] - pool["home_sp_roll_era"]
    pool["sp_roll_ip_diff"] = pool["home_sp_roll_ip"] - pool["away_sp_roll_ip"]
    return pool


def build_enriched_pool(con, con2, games, id_to_name, market_key):
    pool = build_pool(con, games, market_key)
    n_before = len(pool)
    pool = add_bullpen_features(con2, pool, id_to_name)

    weather_venue = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    venue_by_gamepk = pool[["game_pk", "event_id"]].drop_duplicates(subset=["game_pk"]).merge(
        weather_venue, on="event_id", how="left")[["game_pk", "venue_name"]]
    park_dims = con2.execute("""
        SELECT venue_name, lf_distance, cf_distance, rf_distance,
               lf_wall_height, cf_wall_height, rf_wall_height, is_dome
        FROM cache_mlb_park_dimensions
    """).fetchdf().drop_duplicates(subset=["venue_name"])
    park_orient = con2.execute("SELECT venue_name, cf_compass_degrees FROM cache_mlb_ballpark_orientation").fetchdf().drop_duplicates(subset=["venue_name"])
    park_static = park_dims.merge(park_orient, on="venue_name", how="outer")
    park_static["is_dome"] = park_static["is_dome"].astype("float64")
    venue_feats = venue_by_gamepk.merge(park_static, on="venue_name", how="left").drop(columns=["venue_name"])
    pool = pool.merge(venue_feats, on="game_pk", how="left")

    pool = attach_fixed_starter_feature(con, pool)

    print(f"  row-count sanity check: before enrichment n={n_before}  after n={len(pool)}  "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool = build_features(pool)
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool.dropna(subset=["win", "odds"]).reset_index(drop=True)


def print_correlation_check(pool, market_key):
    print(f"\n  === {market_key}: per-feature correlation with win, AFTER fixing starter leakage ===")
    rows = []
    for f in ["market_prob", "elo_diff", "bullpen_era_diff", "l10_diff",
              "sp_roll_era_diff", "sp_roll_ip_diff"]:
        if f not in pool.columns:
            continue
        sub = pool.dropna(subset=[f, "win"])
        if len(sub) < 30:
            continue
        r, p = stats.pointbiserialr(sub["win"].astype(int), sub[f])
        rows.append({"feature": f, "n": len(sub), "corr_with_win": r, "p_value": p})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def run_walkforward(market_key, pool):
    print(f"\n{'=' * 78}\n{market_key} walk-forward (corrected starter feature)\n{'=' * 78}")
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
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
    print(f"\n=== {market_key} pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\n{market_key} edge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")


def main(market_key):
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log (Addendum 48, unmodified import)...")
    games, id_to_name = build_fuller_games(con, con2)

    print(f"\nbuilding {market_key} pool with corrected point-in-time starter feature...")
    pool = build_enriched_pool(con, con2, games, id_to_name, market_key)
    print(f"  final n={len(pool)}")
    print_correlation_check(pool, market_key)
    run_walkforward(market_key, pool)

    con.close()
    con2.close()


if __name__ == "__main__":
    market = sys.argv[1] if len(sys.argv) > 1 else "h2h"
    main(market)
