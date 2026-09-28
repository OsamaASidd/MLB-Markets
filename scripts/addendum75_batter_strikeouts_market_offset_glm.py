"""
Addendum 75: market-offset ridge-logistic GLM for batter_strikeouts.
Reuses Addendum 55c's build_market_dataset_both_sides (keeps both real
odds sides for de-vig) + Addendum 71's corrected point-in-time opposing-
starter rolling ERA, everything else from the established Addendum 53
pipeline (fuller game log, bullpen/park enrichment, box_ids for
opposing-starter resolution).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, norm_name, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features, add_park_features, build_box_ids  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum55c_novig_batter_markets import build_market_dataset_both_sides  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402
from glm_market_offset_walkforward import run_walkforward  # noqa: E402

BULLPEN_V2 = ["bullpen_era_diff", "bullpen_k9_diff"]
PARK_V2 = ["lf_distance", "cf_distance", "rf_distance",
           "lf_wall_height", "cf_wall_height", "rf_wall_height",
           "cf_compass_degrees", "is_dome"]
BASE_FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
GLM_FEATURES = BASE_FEATURES + BULLPEN_V2 + ["opp_sp_roll_era", "opp_sp_roll_ip"] + PARK_V2


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building batter_strikeouts pool, both sides kept for de-vig...")
    box_bk = con.execute("""
        SELECT game_pk, player_name, batter_strikeouts, is_starter, position_type FROM boxscore
    """).fetchdf()
    box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
    cache_path = ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl"
    pool = build_market_dataset_both_sides(cache_path, "batter_strikeouts", "batter_strikeouts", box_bk,
                                            ["Catcher", "Hitter", "Infielder", "Outfielder"], starter_only=False)
    print(f"  pool: n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")

    games_v3, id_to_name = build_fuller_games(con, con2)
    n0 = len(games_v3)
    games_v3 = add_bullpen_features(con2, games_v3, id_to_name)
    assert len(games_v3) == n0
    games_v3 = add_park_features(con, con2, games_v3)
    assert len(games_v3) == n0
    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    games_v3 = games_v3.merge(op, on="game_pk", how="left")
    assert len(games_v3) == n0
    games_v3["game_date"] = pd.to_datetime(games_v3["game_date"])
    box_ids = build_box_ids(con)

    GAME_LEVEL_COLS = ["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                        "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"] + BULLPEN_V2 + PARK_V2
    n0p = len(pool)
    pool = pool.merge(games_v3[GAME_LEVEL_COLS], on="game_pk", how="inner")
    pool["name_norm"] = pool["tiebreak"]
    n1 = len(pool)
    pool = pool.merge(box_ids[["game_pk", "name_norm", "team_id"]], on=["game_pk", "name_norm"], how="left")
    ctx = games_v3.drop_duplicates(subset=["game_pk"])[
        ["game_pk", "home_team_id", "away_team_id", "home_starter_id", "away_starter_id"]]
    pool = pool.merge(ctx, on="game_pk", how="left")
    pool["opp_starter_id"] = pd.Series(np.where(pool["team_id"] == pool["home_team_id"],
                                        pool["away_starter_id"], pool["home_starter_id"]), index=pool.index).astype("Int64")
    print(f"  row-count check: before={n0p} after={len(pool)} {'OK' if len(pool) == n0p else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n0p

    roll = build_starter_rolling_era(con)
    roll_o = roll.rename(columns={"player_id": "opp_starter_id", "sp_roll_era": "opp_sp_roll_era", "sp_roll_ip": "opp_sp_roll_ip"})
    roll_o["opp_starter_id"] = roll_o["opp_starter_id"].astype("Int64")
    n2 = len(pool)
    pool = pool.merge(roll_o[["opp_starter_id", "game_pk", "opp_sp_roll_era", "opp_sp_roll_ip"]],
                       on=["opp_starter_id", "game_pk"], how="left")
    print(f"  row-count check (rolling ERA merge): before={n2} after={len(pool)} "
          f"{'OK' if len(pool) == n2 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n2

    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["game_date"] = pd.to_datetime(pool["game_date"])

    over_p = implied_prob(pool["best_over_odds"].values)
    under_p = implied_prob(pool["best_under_odds"].values)
    total = over_p + under_p
    fair_over = over_p / total
    fair_under = under_p / total
    pool["market_prob"] = np.where(pool["side"].values == "over", fair_over, fair_under)
    pool["hold"] = total - 1.0
    print(f"  de-vig applied: mean overround={round(100 * (total.mean() - 1), 2)}%")

    pool = pool.dropna(subset=["win", "odds", "market_prob"])
    print(f"  pool ready: n={len(pool):,}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    run_walkforward(pool, GLM_FEATURES, "batter_strikeouts", n_folds=6, hold_col="hold")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
