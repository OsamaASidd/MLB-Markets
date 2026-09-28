"""
Addendum 62: EDA-first pass on batter_runs_scored, BEFORE building any new
model -- same discipline just established for pitcher_strikeouts
(Addendum 59). Reuses the exact, unmodified pool-construction from
Addendum 55c's run_runs_scored() (Addendum 30's real-odds pipeline:
build_team_game_log + Elo/L10/bullpen fatigue + ballpark_factors, merged
onto the real backfilled batter_runs_scored odds cache, matched to real
boxscore runs_scored outcomes). n=52,981, already a large, definitive
sample (n>=500 -> ROI>0-only gate rule applies, no CI hurdle) -- so if
EDA finds real correlation here, there's already enough data to prove it
decisively; no "wait for more data" caveat like pitcher_strikeouts had.

Published baseline (Addendum 30, RAW market_prob): n=52,981 ROI=-2.54%
CI=[-4.05%,-1.03%] FAIL. Addendum 55c's de-vig correction on the same
pipeline also did not flip it (else it would already be in the 4 passing
markets). This script asks: is there ANY feature here with real signal
that a differently-built model could exploit, or is this market also
already efficiently priced against every feature we have (like
pitcher_strikeouts turned out to be)?
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob, norm_name, build_team_game_log, build_elo_l10, build_bullpen_fatigue  # noqa: E402
from addendum55c_novig_batter_markets import build_market_dataset_both_sides  # noqa: E402

pd.set_option("display.width", 140)


def build_pool(con):
    games = build_team_game_log(con)
    n_games_raw = len(games)
    games = games.drop_duplicates(subset=["game_pk"])
    print(f"  build_team_game_log() dedup: {n_games_raw} raw rows -> {len(games)} unique game_pk "
          f"(pre-existing duplicate-game_pk bug, documented in Addendum 37 -- NOT fixed at the source in "
          f"the original Addendum 30/55c batter_runs_scored pipeline, which merges onto this table directly)")
    elo_l10 = build_elo_l10(games)
    fatigue = build_bullpen_fatigue(con)
    games = games.merge(elo_l10, on="game_pk")
    games["home_fatigue"] = games.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    games["away_fatigue"] = games.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)

    parks = con.execute("SELECT * FROM ballpark_factors").fetchdf()
    for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
        parks[c] = pd.to_numeric(parks[c], errors="coerce")
    park_by_event = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    park_by_event = park_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    event_to_gamepk = con.execute("SELECT event_id, game_pk FROM client_games").fetchdf()
    event_to_gamepk["game_pk"] = pd.to_numeric(event_to_gamepk["game_pk"], errors="coerce")
    park_by_gamepk = park_by_event.merge(event_to_gamepk, on="event_id", how="inner")[
        ["game_pk", "runs_factor", "hr_factor", "k_factor", "hits_factor"]]
    n_park_raw = len(park_by_gamepk)
    park_by_gamepk = park_by_gamepk.drop_duplicates(subset=["game_pk"])
    print(f"  park_by_gamepk dedup: {n_park_raw} raw rows -> {len(park_by_gamepk)} unique game_pk "
          f"(client_games has multiple event_id rows per real game_pk -- NEWLY FOUND fan-out source, "
          f"also NOT fixed at the source in the original Addendum 30/55c pipeline)")
    games = games.merge(park_by_gamepk, on="game_pk", how="left")

    box = con.execute("""
        SELECT game_pk, player_name, runs_scored, batting_order_slot, plate_appearances,
               is_starter, position_type FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    market_key, stat_col = "batter_runs_scored", "runs_scored"
    cache_path = ROOT / "data_raw" / "runs_scored_odds_cache.jsonl"
    position_filter, starter_only = ["Catcher", "Hitter", "Infielder", "Outfielder"], False

    pool = build_market_dataset_both_sides(cache_path, market_key, stat_col, box, position_filter, starter_only)
    print(f"  pool (pre game-log join): n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")

    n0 = len(pool)
    pool = pool.merge(games[["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                              "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]],
                       on="game_pk", how="inner")
    print(f"  row-count check (game-log merge, inner join -- decrease OK, increase is NOT): "
          f"before={n0} after={len(pool)} {'OK' if len(pool) <= n0 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) <= n0

    pool["name_norm"] = pool["tiebreak"]
    n1 = len(pool)
    pool = pool.merge(box[["game_pk", "name_norm", "batting_order_slot", "plate_appearances"]].drop_duplicates(subset=["game_pk", "name_norm"]),
                       on=["game_pk", "name_norm"], how="left")
    print(f"  row-count check (batting-order merge, left join): before={n1} after={len(pool)} "
          f"{'OK' if len(pool) == n1 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n1

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool.dropna(subset=["win", "odds"]).reset_index(drop=True)


def main():
    con = duckdb.connect(DB, read_only=True)

    print("building batter_runs_scored pool (Addendum 30/55c pipeline, unmodified)...")
    pool = build_pool(con)
    print(f"  final n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    print("\n=== 1. MARKET CALIBRATION (decile of market_prob vs actual win rate) ===")
    pool["prob_decile"] = pd.qcut(pool["market_prob"], 10, labels=False, duplicates="drop")
    calib = pool.groupby("prob_decile").agg(
        n=("win", "size"), mean_market_prob=("market_prob", "mean"), actual_win_rate=("win", "mean")
    )
    calib["gap"] = calib["actual_win_rate"] - calib["mean_market_prob"]
    print(calib.round(4).to_string())
    corr_calib = np.corrcoef(calib["mean_market_prob"], calib["actual_win_rate"])[0, 1]
    print(f"  decile-level correlation(market_prob, actual_win_rate) = {corr_calib:.4f}")

    print("\n=== 2. PER-FEATURE POINT-BISERIAL CORRELATION WITH WIN ===")
    FEATURES = ["market_prob", "side_code", "elo_diff", "l10_diff", "fatigue_diff",
                "runs_factor", "hr_factor", "k_factor", "hits_factor",
                "batting_order_slot", "plate_appearances"]
    rows = []
    for f in FEATURES:
        sub = pool.dropna(subset=[f, "win"])
        if sub[f].nunique() < 2:
            continue
        r, p = stats.pointbiserialr(sub["win"].astype(int), sub[f])
        rows.append({"feature": f, "n": len(sub), "corr_with_win": r, "p_value": p})
    corr_df = pd.DataFrame(rows).sort_values("corr_with_win", key=lambda s: s.abs(), ascending=False)
    print(corr_df.round(4).to_string(index=False))

    print("\n=== 3. SEGMENT BREAKDOWNS ===")
    print("\n-- by side (over vs under) --")
    print(pool.groupby("side").agg(n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n-- by batting_order_slot (lineup position -- more PAs = more scoring chances) --")
    pool["slot_bucket"] = pd.cut(pool["batting_order_slot"], [0, 2, 4, 6, 9], labels=["1-2", "3-4", "5-6", "7-9"])
    print(pool.groupby("slot_bucket", observed=True).agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n-- by market_prob extremity (favorite vs coinflip vs dog, this side) --")
    pool["fav_bucket"] = pd.cut(pool["market_prob"], [0, 0.40, 0.47, 0.53, 0.60, 1.0],
                                 labels=["dog<40%", "40-47%", "47-53% (coinflip)", "53-60%", "fav>60%"])
    print(pool.groupby("fav_bucket", observed=True).agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n=== 4. RESIDUAL / MISPRICING CHECK (gap by segment) ===")
    for seg_col in ["slot_bucket", "fav_bucket"]:
        seg = pool.groupby(seg_col, observed=True).apply(
            lambda g: pd.Series({"n": len(g), "gap": g["win"].mean() - g["market_prob"].mean()}),
            include_groups=False)
        print(f"\n  gap by {seg_col}:")
        print(seg.round(4).to_string())

    con.close()


if __name__ == "__main__":
    main()
