"""
Addendum 74: market-offset ridge-logistic GLM for pitcher_outs, same
architecture as Addendum 72/73. Reuses Addendum 70's pool construction
(real backfilled odds + corrected point-in-time starter feature)
unmodified up to the point of model fitting -- de-vig computed from the
best_over_odds/best_under_odds columns that survive build_market_dataset's
merge (both odds recorded per (game,player) before one side is picked).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, norm_name, implied_prob  # noqa: E402
from addendum55c_novig_batter_markets import build_market_dataset_both_sides  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features, add_park_features, TEAM_NAME_ALIAS  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402
from glm_market_offset_walkforward import run_walkforward  # noqa: E402

BULLPEN_V2 = ["bullpen_era_diff", "bullpen_k9_diff"]
PARK_V2 = ["lf_distance", "cf_distance", "rf_distance",
           "lf_wall_height", "cf_wall_height", "rf_wall_height",
           "cf_compass_degrees", "is_dome"]
STARTER_FIXED = ["sp_roll_era_diff", "sp_roll_ip_diff"]
BASE_FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
GLM_FEATURES = BASE_FEATURES + BULLPEN_V2 + STARTER_FIXED + PARK_V2


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building real backfilled-odds pitcher_outs pool (both sides kept for de-vig)...")
    box = con.execute("""
        SELECT game_pk, player_name, outs AS pitcher_outs_stat, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    po_cache = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"
    pool = build_market_dataset_both_sides(po_cache, "pitcher_outs", "pitcher_outs_stat", box, ["Pitcher"], starter_only=True)
    print(f"  pool (both over+under sides): n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")
    assert {"best_over_odds", "best_under_odds"}.issubset(pool.columns), "both-sides odds missing -- STOP"

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

    games_full = add_bullpen_features(con2, games_v3, id_to_name)
    assert len(games_full) == n0
    games_full = add_park_features(con, con2, games_full)
    assert len(games_full) == n0

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
    assert len(games_full) == n0
    games_full["sp_roll_era_diff"] = games_full["away_sp_roll_era"] - games_full["home_sp_roll_era"]
    games_full["sp_roll_ip_diff"] = games_full["home_sp_roll_ip"] - games_full["away_sp_roll_ip"]

    merge_cols = (["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                    "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
                  + BULLPEN_V2 + STARTER_FIXED + PARK_V2)
    games_full_dedup = games_full.drop_duplicates(subset=["game_pk"])[merge_cols]
    n_before = len(pool)
    pool_full = pool.merge(games_full_dedup, on="game_pk", how="inner")
    print(f"  final feature merge: before={n_before:,}  after={len(pool_full):,}  "
          f"{'OK' if len(pool_full) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool_full) == n_before

    pool_full["game_date"] = pd.to_datetime(pool_full["game_date"])
    pool_full["elo_diff"] = (pool_full["home_elo"] + 24) - pool_full["away_elo"]
    pool_full["l10_diff"] = pool_full["home_l10"] - pool_full["away_l10"]
    pool_full["fatigue_diff"] = pool_full["home_fatigue"] - pool_full["away_fatigue"]

    over_p = implied_prob(pool_full["best_over_odds"].values)
    under_p = implied_prob(pool_full["best_under_odds"].values)
    total = over_p + under_p
    fair_over = over_p / total
    fair_under = under_p / total
    pool_full["market_prob"] = np.where(pool_full["side"].values == "over", fair_over, fair_under)
    pool_full["hold"] = total - 1.0
    print(f"  de-vig applied: mean overround={round(100 * (total.mean() - 1), 2)}%")

    pool_full = pool_full.dropna(subset=["win", "odds", "market_prob"])
    print(f"  pool ready: n={len(pool_full):,}  date range {pool_full.game_date.min().date()} -> {pool_full.game_date.max().date()}")

    run_walkforward(pool_full, GLM_FEATURES, "pitcher_outs", hold_col="hold")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
