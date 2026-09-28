"""
Addendum 67: EDA-first pass on h2h, before building any new model. Same
discipline as pitcher_strikeouts (Addendum 59) and batter_runs_scored
(Addendum 62). Reuses Addendum 48's fuller game log + full enrichment
pipeline (bullpen quality, starter quality, park dims) unmodified,
stopping before model fit to inspect calibration and per-feature
correlation with the actual outcome first.

h2h is structurally different from every player-prop market tested so
far: it's a team win/loss market, not a discrete counting stat, so the
mechanism for any inefficiency (public bias toward popular teams,
recency overreaction, etc) would be different in kind from a
pre-computed player projection. Worth checking properly rather than
assuming the same "efficiently priced" conclusion carries over.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, build_pool, build_features, FEATURES  # noqa: E402

pd.set_option("display.width", 140)


def build_enriched_pool(con, con2, games, id_to_name, market_key):
    pool = build_pool(con, games, market_key)
    n_before = len(pool)
    pool = add_bullpen_features(con2, pool, id_to_name)
    pool["season"] = pd.to_datetime(pool["game_date"]).dt.year.astype("int64")

    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    pool = pool.merge(op, on="game_pk", how="left")

    pstat = con2.execute("""
        SELECT player_id, season, xera, era, xba_allowed FROM cache_mlb_historical_pitcher_statcast
    """).fetchdf().drop_duplicates(subset=["player_id", "season"])
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

    print(f"  row-count sanity check: before enrichment n={n_before}  after n={len(pool)}  "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool = build_features(pool)
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool.dropna(subset=["win", "odds"]).reset_index(drop=True)


def run_eda(market_key, con, con2, games, id_to_name):
    print(f"\n{'=' * 78}\n{market_key} EDA\n{'=' * 78}")
    pool = build_enriched_pool(con, con2, games, id_to_name, market_key)
    print(f"  final n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    print("\n  === MARKET CALIBRATION (decile of market_prob vs actual win rate) ===")
    pool["prob_decile"] = pd.qcut(pool["market_prob"], 10, labels=False, duplicates="drop")
    calib = pool.groupby("prob_decile").agg(
        n=("win", "size"), mean_market_prob=("market_prob", "mean"), actual_win_rate=("win", "mean")
    )
    calib["gap"] = calib["actual_win_rate"] - calib["mean_market_prob"]
    print(calib.round(4).to_string())
    corr_calib = np.corrcoef(calib["mean_market_prob"], calib["actual_win_rate"])[0, 1]
    print(f"  decile-level correlation(market_prob, actual_win_rate) = {corr_calib:.4f}")

    print("\n  === PER-FEATURE POINT-BISERIAL CORRELATION WITH WIN ===")
    rows = []
    for f in FEATURES:
        if f not in pool.columns:
            continue
        sub = pool.dropna(subset=[f, "win"])
        if len(sub) < 30 or sub[f].nunique() < 2:
            continue
        r, p = stats.pointbiserialr(sub["win"].astype(int), sub[f])
        rows.append({"feature": f, "n": len(sub), "corr_with_win": r, "p_value": p})
    corr_df = pd.DataFrame(rows).sort_values("corr_with_win", key=lambda s: s.abs(), ascending=False)
    print(corr_df.round(4).to_string(index=False))
    return pool


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log (Addendum 48, unmodified import)...")
    games, id_to_name = build_fuller_games(con, con2)

    run_eda("h2h", con, con2, games, id_to_name)
    run_eda("totals", con, con2, games, id_to_name)

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
