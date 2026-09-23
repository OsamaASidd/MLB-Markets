"""
Addendum 37: historical enrichment with real player-level + point-in-time
team-level signal, on the 5 FAIL/near-miss markets that use the real
2023-2025 warehouse odds (batter_total_bases, batter_home_runs,
pitcher_strikeouts, h2h, totals). `spreads` already passes and `hits`/
`rbis` already pass -- none of the three are in scope here.

New data source: db/cache_features.duckdb (Supabase cache_* tables pulled
read-only via scripts/pull_cache_features.py, already run). Two genuinely
historical (2023-2025) tables used here for the first time in this project:

  - cache_mlb_historical_bullpen: team_name, snapshot_date (569 distinct
    dates, 2023-05-04 to 2025-10-31), rolling_14d_era, rolling_14d_whip,
    rolling_14d_k_per_9, rolling_14d_bb_per_9. Point-in-time rolling
    snapshot (NOT one row per game) -- joined per team+game_date via
    pandas merge_asof(direction="backward"), i.e. the most recent
    snapshot_date <= game_date, never a future snapshot (no lookahead).
    VERIFIED before writing any join code: rolling_14d_whip and
    rolling_14d_bb_per_9 are 100% NULL in every one of 13,141 rows
    (COUNT(col) vs COUNT(*) checked directly) -- not a join bug, the
    source cache table itself never populated these two columns. Dropped
    from FEATURES_V2 as dead columns; only rolling_14d_era and
    rolling_14d_k_per_9 carry real signal.
  - cache_mlb_historical_pitcher_statcast: player_id, season (int year),
    xera, era, babip_allowed, xba_allowed, barrel_rate_allowed,
    hard_hit_pct_allowed, fly_ball_rate. Joined by player_id + season
    (season = year of game_date). Same verification: babip_allowed,
    barrel_rate_allowed, hard_hit_pct_allowed, fly_ball_rate, hr_per_9,
    k_per_9, bb_per_9, ip are 100% NULL across all 1,087 rows. Only xera,
    era, and xba_allowed are populated -- those 3 are the only pitcher
    statcast columns actually used below.
  - cache_mlb_park_dimensions / cache_mlb_ballpark_orientation: static,
    joined by venue_name (same venue_name already used for the existing
    ballpark_factors/weather join in build_games_and_box()).

Starting-pitcher IDs per game: opposing_pitcher table (event_id,
home_starter_id, away_starter_id, game_pk already resolved -- game_pk is
BIGINT here, no event_id->game_pk hop needed for this table specifically).
Team name for the bullpen join: client_games.home_team/away_team (real
team names) cross-walked to the team_id space already used throughout this
project (build_team_game_log's home_team_id/away_team_id) via a join on
game_pk. One naming alias needed and applied: client_games sometimes has
"Athletics" where the cache table always says "Oakland Athletics" for the
2023-2025 window -- same franchise, aliased so the bullpen join doesn't
silently miss it.

New features are added via LEFT joins (never inner), matching the exact
convention build_games_and_box() already uses for ballpark_factors -- a
game/player without a matching historical snapshot gets NaN, which XGBoost
(missing=np.nan, same as every other addendum) handles natively, rather
than the join silently shrinking the sample. This means n(train/test) for
each market matches its pre-enrichment baseline exactly; per-feature
coverage (% non-null) is printed per market instead, so any genuine
coverage gap is visible rather than hidden inside a smaller n.

Method is the exact tuning discipline already established in Addendum 34
(scripts/tuned_xgboost_4_markets.py), extended with the new columns:
  - ONE model family: XGBoost (project baseline throughout).
  - ONE pre-specified hyperparameter grid: PARAM_GRID/FIXED_PARAMS
    imported as-is from tuned_xgboost_4_markets.py, not widened.
  - Selection metric: CV-averaged ROC AUC on the TRAIN split only
    (5-fold StratifiedKFold via GridSearchCV). ROI/gate is never touched
    during selection.
  - Refit once, score once on the untouched test split, single
    pre-specified edge>0.0 cut, official gate rule (gate() in
    xgboost_individual_markets.py: n>=500 -> ROI>0; else CI lower
    bound>0).
  - Same per-market train/test split each market already used
    (per_year for all 5 of these markets, per multi_model_comparison.py /
    tuned_xgboost_4_markets.py).
  - Every market reported, pass or fail. No threshold search folded into
    the official verdict.

Reuses build_games_and_box / build_warehouse_pool / build_game_pool /
build_features / split_per_year / split_chronological from
multi_model_comparison.py, and PARAM_GRID / FIXED_PARAMS from
tuned_xgboost_4_markets.py, UNMODIFIED (imported, not edited). FEATURES_V2
is this script's own extended list -- the imported FEATURES list from
multi_model_comparison.py is never mutated.
"""
import os
import pathlib
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, gate, stat, profit, norm_name  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    FEATURES, build_games_and_box, build_warehouse_pool, build_game_pool, build_features,
    split_per_year, split_chronological,
)
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")

TEAM_NAME_ALIAS = {"Athletics": "Oakland Athletics"}

# New engineered feature groups (Addendum 37). Only columns verified to
# carry real (non-100%-null) signal in the pulled cache tables are here --
# see module docstring for what was checked and dropped.
BULLPEN_V2 = ["bullpen_era_diff", "bullpen_k9_diff"]
STARTER_GAME_V2 = ["starter_xera_diff", "starter_era_diff", "starter_xba_diff"]
OPP_PITCHER_V2 = ["opp_sp_xera", "opp_sp_era", "opp_sp_xba"]
PARK_V2 = ["lf_distance", "cf_distance", "rf_distance",
           "lf_wall_height", "cf_wall_height", "rf_wall_height",
           "cf_compass_degrees", "is_dome"]

FEATURES_V2 = FEATURES + BULLPEN_V2 + STARTER_GAME_V2 + OPP_PITCHER_V2 + PARK_V2

# Known baseline numbers for direct before/after comparison, cited from
# already-committed output (not re-derived here):
#   batter_total_bases / h2h / totals: Addendum 32/33 untuned XGBoost row,
#     reports/multi_model_comparison_output.txt (same pool/feature/split
#     code this script reuses).
#   batter_home_runs / pitcher_strikeouts: Addendum 34 tuned XGBoost,
#     reports/tuned_xgboost_4_markets_output.txt (same tuning discipline
#     this script reuses, pre-enrichment).
BASELINES = {
    "batter_total_bases": {"desc": "Addendum 32 XGBoost (untuned, pre-enrichment)",
                            "n": 7132, "roi": -0.17, "lo": -2.04, "hi": 1.71, "verdict": "FAIL"},
    "batter_home_runs": {"desc": "Addendum 34 tuned XGBoost (pre-enrichment)",
                          "n": 2386, "roi": 0.68, "lo": -1.07, "hi": 2.42,
                          "verdict": "PASS (n>=500+ROI>0 arm; CI crosses 0 -- flagged near-miss, not confirmed)"},
    "pitcher_strikeouts": {"desc": "Addendum 34 tuned XGBoost (pre-enrichment)",
                            "n": 672, "roi": -12.15, "lo": -18.86, "hi": -5.45, "verdict": "FAIL"},
    "h2h": {"desc": "Addendum 32 XGBoost (untuned, pre-enrichment)",
            "n": 526, "roi": -34.16, "lo": -41.37, "hi": -26.96, "verdict": "FAIL"},
    "totals": {"desc": "Addendum 32 XGBoost (untuned, pre-enrichment)",
               "n": 434, "roi": 3.69, "lo": -5.27, "hi": 12.64, "verdict": "FAIL"},
}


def build_team_name_map(con, games):
    """team_id -> real team name (matching cache_mlb_historical_bullpen's
    team_name), cross-walked via client_games.home_team/away_team joined
    to game_pk. Looked at client_games/warehouse tables before hardcoding
    a 30-team id->name map, per the task's join-key notes -- client_games
    already carries real team names, so used that instead."""
    cg = con.execute("SELECT event_id, game_pk, home_team, away_team FROM client_games").fetchdf()
    cg["game_pk"] = pd.to_numeric(cg["game_pk"], errors="coerce")
    cg = cg.dropna(subset=["game_pk"])
    cg["game_pk"] = cg["game_pk"].astype("int64")

    g2 = games[["game_pk", "home_team_id", "away_team_id"]].merge(
        cg[["game_pk", "home_team", "away_team"]], on="game_pk", how="inner")

    id_to_name = {}
    for tid, name in g2.groupby("home_team_id")["home_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    for tid, name in g2.groupby("away_team_id")["away_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    return {tid: TEAM_NAME_ALIAS.get(name, name) for tid, name in id_to_name.items()}


def add_bullpen_features(con2, games, id_to_name):
    """Point-in-time bullpen quality: for each team+game_date, join to the
    cache_mlb_historical_bullpen snapshot with the closest snapshot_date <=
    game_date (merge_asof, direction='backward') -- never a future
    snapshot. rolling_14d_whip / rolling_14d_bb_per_9 excluded: verified
    100% NULL in the source table (see module docstring)."""
    games = games.copy()
    games["home_team_name"] = games["home_team_id"].map(id_to_name)
    games["away_team_name"] = games["away_team_id"].map(id_to_name)

    bullpen = con2.execute("""
        SELECT team_name, snapshot_date, rolling_14d_era, rolling_14d_k_per_9
        FROM cache_mlb_historical_bullpen
    """).fetchdf()
    bullpen["snapshot_date"] = pd.to_datetime(bullpen["snapshot_date"]).astype("datetime64[ns]")
    bullpen = bullpen.sort_values("snapshot_date").reset_index(drop=True)

    def asof(team_col):
        df = pd.DataFrame({
            "_idx": games.index,
            "team_name": games[team_col].values,
            "game_date": pd.to_datetime(games["game_date"]).astype("datetime64[ns]").values,
        }).sort_values("game_date")
        merged = pd.merge_asof(df, bullpen, left_on="game_date", right_on="snapshot_date",
                                by="team_name", direction="backward")
        return merged.sort_values("_idx").set_index("_idx")[["rolling_14d_era", "rolling_14d_k_per_9"]]

    h = asof("home_team_name")
    a = asof("away_team_name")
    games["home_bp_era"] = h["rolling_14d_era"].values
    games["home_bp_k9"] = h["rolling_14d_k_per_9"].values
    games["away_bp_era"] = a["rolling_14d_era"].values
    games["away_bp_k9"] = a["rolling_14d_k_per_9"].values
    # Lower ERA is better for the pitching side: away_era - home_era > 0
    # means the home bullpen has the lower (better) ERA, same "higher diff
    # favors home" sign convention as elo_diff/l10_diff already use.
    games["bullpen_era_diff"] = games["away_bp_era"] - games["home_bp_era"]
    games["bullpen_k9_diff"] = games["home_bp_k9"] - games["away_bp_k9"]
    return games


def add_starter_game_features(con, con2, games):
    """Both starters' statcast quality, diffed at the game level -- used
    for h2h/totals/pitcher_strikeouts (and also carried into the batter
    markets as game-context, alongside the batter-specific opposing-starter
    features added separately in add_opp_pitcher_features)."""
    games = games.copy()
    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")

    games = games.merge(op, on="game_pk", how="left")
    games["season"] = pd.to_datetime(games["game_date"]).dt.year.astype("int64")

    pstat = con2.execute("""
        SELECT player_id, season, xera, era, xba_allowed FROM cache_mlb_historical_pitcher_statcast
    """).fetchdf().drop_duplicates(subset=["player_id", "season"])
    pstat["player_id"] = pstat["player_id"].astype("Int64")
    pstat["season"] = pstat["season"].astype("int64")

    home_p = pstat.rename(columns={"player_id": "home_starter_id", "xera": "home_sp_xera",
                                    "era": "home_sp_era", "xba_allowed": "home_sp_xba"})
    games = games.merge(home_p, on=["home_starter_id", "season"], how="left")
    away_p = pstat.rename(columns={"player_id": "away_starter_id", "xera": "away_sp_xera",
                                    "era": "away_sp_era", "xba_allowed": "away_sp_xba"})
    games = games.merge(away_p, on=["away_starter_id", "season"], how="left")

    # Lower allowed xERA/ERA/xBA is better pitching: away - home > 0 means
    # the home starter is better, same sign convention as bullpen_era_diff.
    games["starter_xera_diff"] = games["away_sp_xera"] - games["home_sp_xera"]
    games["starter_era_diff"] = games["away_sp_era"] - games["home_sp_era"]
    games["starter_xba_diff"] = games["away_sp_xba"] - games["home_sp_xba"]
    return games


def add_park_features(con, con2, games):
    """Static park dimension/orientation features, joined by venue_name --
    same venue_name already used for the existing ballpark_factors/weather
    join in build_games_and_box()."""
    games = games.copy()
    weather_venue = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    event_to_gamepk = con.execute("SELECT event_id, game_pk FROM client_games").fetchdf()
    event_to_gamepk["game_pk"] = pd.to_numeric(event_to_gamepk["game_pk"], errors="coerce")
    event_to_gamepk = event_to_gamepk.dropna(subset=["game_pk"])
    event_to_gamepk["game_pk"] = event_to_gamepk["game_pk"].astype("int64")
    venue_by_gamepk = weather_venue.merge(event_to_gamepk, on="event_id", how="inner")[["game_pk", "venue_name"]]
    venue_by_gamepk = venue_by_gamepk.drop_duplicates(subset=["game_pk"])

    park_dims = con2.execute("""
        SELECT venue_name, lf_distance, cf_distance, rf_distance,
               lf_wall_height, cf_wall_height, rf_wall_height, is_dome
        FROM cache_mlb_park_dimensions
    """).fetchdf().drop_duplicates(subset=["venue_name"])
    park_orient = con2.execute("""
        SELECT venue_name, cf_compass_degrees FROM cache_mlb_ballpark_orientation
    """).fetchdf().drop_duplicates(subset=["venue_name"])
    park_static = park_dims.merge(park_orient, on="venue_name", how="outer")
    park_static["is_dome"] = park_static["is_dome"].astype("float64")

    venue_feats = venue_by_gamepk.merge(park_static, on="venue_name", how="left").drop(columns=["venue_name"])
    games = games.merge(venue_feats, on="game_pk", how="left")
    return games


def build_games_v2(con, con2, games):
    # PRE-EXISTING bug in the established (unmodified) build_team_game_log()
    # pipeline, discovered while building this addendum, NOT introduced by
    # any code here: `games` has 4,547 rows but only 3,845 unique game_pk
    # (verified root cause: SELECT DISTINCT game_pk, game_date FROM
    # boxscore returns >1 game_date for 13 game_pk values, some >2 months
    # apart -- looks like a game_pk collision/data bug in the source data,
    # not a timezone artifact). Merging pool data onto games ONCE (as every
    # prior addendum's build_features() call does) mostly doesn't show
    # visible inflation for batter-prop pools (verified: batter_total_bases
    # pool size unaffected), but DOES inflate game-level pools built via
    # build_game_pool (verified: h2h pool 5,907 -> 11,393 rows, ~1.93x,
    # reproduced with ZERO changes from this file -- 100% inherited, not
    # something this addendum caused). Duplicated rows don't move the ROI
    # point estimate much but DO inflate n, which narrows the CI and risks
    # a real FAIL crossing the CI-based gate threshold spuriously. Dedupe
    # here so this addendum's own numbers are computed on a clean base --
    # this makes Addendum 37 NOT byte-for-byte reproducing the exact
    # (buggy) methodology of Addenda 32/34/36, which inherited this same
    # issue unknowingly; flagged for the project separately as something
    # those addenda's h2h/totals numbers should be re-checked against.
    games = games.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    id_to_name = build_team_name_map(con, games)
    games = add_bullpen_features(con2, games, id_to_name)
    games = add_starter_game_features(con, con2, games)
    games = add_park_features(con, con2, games)
    return games


def build_box_ids(con):
    """player_id + team_id per (game_pk, normalized player name) -- needed
    to resolve, for a given batter, which team they're on and therefore
    which starter (home or away) is the OPPOSING pitcher they actually
    face. Not in build_games_and_box()'s box selection, so pulled
    separately here rather than modifying that function."""
    box_ids = con.execute("SELECT DISTINCT player_id, game_pk, player_name, team_id FROM boxscore").fetchdf()
    box_ids["name_norm"] = box_ids["player_name"].map(norm_name)
    box_ids = box_ids.dropna(subset=["name_norm"])
    return box_ids.drop_duplicates(subset=["game_pk", "name_norm"])


def add_opp_pitcher_features(pool, games_v2, box_ids, con2):
    """Batter-prop-specific: the OPPOSING starting pitcher's statcast
    quality for that exact game (not both starters averaged/diffed --
    the specific pitcher the batter actually faces)."""
    pool = pool.merge(box_ids[["game_pk", "name_norm", "team_id"]], on=["game_pk", "name_norm"], how="left")
    # Same duplicate-game_pk fan-out risk as build_features_v2 -- dedupe
    # before merging (see comment there for the full explanation).
    ctx = games_v2.drop_duplicates(subset=["game_pk"])[
        ["game_pk", "home_team_id", "away_team_id", "home_starter_id", "away_starter_id", "season"]]
    pool = pool.merge(ctx, on="game_pk", how="left")
    pool["opp_starter_id"] = np.where(pool["team_id"] == pool["home_team_id"],
                                       pool["away_starter_id"], pool["home_starter_id"])
    pool["opp_starter_id"] = pool["opp_starter_id"].astype("Int64")

    pstat = con2.execute("""
        SELECT player_id, season, xera, era, xba_allowed FROM cache_mlb_historical_pitcher_statcast
    """).fetchdf().drop_duplicates(subset=["player_id", "season"])
    pstat = pstat.rename(columns={"player_id": "opp_starter_id", "xera": "opp_sp_xera",
                                   "era": "opp_sp_era", "xba_allowed": "opp_sp_xba"})
    pstat["opp_starter_id"] = pstat["opp_starter_id"].astype("Int64")
    pool = pool.merge(pstat, on=["opp_starter_id", "season"], how="left")
    return pool


def build_features_v2(pool, games_v2, kind, box_ids, con2):
    """Same pattern as multi_model_comparison.build_features(), extended:
    calls the original (unmodified) build_features() for the established
    FEATURES columns, then LEFT-merges in the new Addendum 37 game-level
    columns, then (batter markets only) the opposing-starter row-level
    columns."""
    pool = build_features(pool, games_v2)
    extra_cols = ["game_pk"] + BULLPEN_V2 + STARTER_GAME_V2 + PARK_V2
    # games_v2 inherits duplicate game_pk rows from the pre-existing,
    # unmodified build_team_game_log() pipeline (verified: 3,845 unique of
    # 4,547 total rows -- a pre-existing characteristic every prior addendum
    # only ever merged onto ONCE, so it stayed harmless). This is the SECOND
    # merge onto games_v2 in this function (build_features() above already
    # did the first) -- merging twice on a key with duplicates compounds the
    # fan-out multiplicatively (verified: caused a ~200x row-count explosion
    # and spurious 0.95-0.997 CV AUCs on the first run of this script before
    # this fix). Dedupe by game_pk immediately before this second merge so
    # it stays a clean many-to-one join, same as every prior addendum's
    # single merge onto this table.
    games_v2_dedup = games_v2.drop_duplicates(subset=["game_pk"])[extra_cols]
    pool = pool.merge(games_v2_dedup, on="game_pk", how="left")
    if kind == "batter":
        pool = add_opp_pitcher_features(pool, games_v2, box_ids, con2)
    else:
        for c in OPP_PITCHER_V2:
            pool[c] = np.nan
    return pool


def tune_and_evaluate(market_name, pool, games_v2, split_mode, kind, box_ids, con2):
    pool = build_features_v2(pool, games_v2, kind, box_ids, con2)
    if len(pool) < 500:
        print(f"  {market_name}: n={len(pool)} INSUFFICIENT DATA (<500)")
        return None
    train, test = (split_per_year(pool) if split_mode == "per_year" else split_chronological(pool))
    if len(train) < 50 or len(test) < 50:
        print(f"  {market_name}: INSUFFICIENT DATA after split")
        return None

    X_train, y_train = train[FEATURES_V2], train["win"].astype(int)
    X_test, y_test = test[FEATURES_V2], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same fix already noted/used in
    # tuned_xgboost_4_markets.py).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    coverage_cols = BULLPEN_V2 + STARTER_GAME_V2 + (OPP_PITCHER_V2 if kind == "batter" else []) + PARK_V2
    coverage = {c: round(100 * pool[c].notna().mean(), 1) for c in coverage_cols}

    print(f"  {market_name}")
    print(f"    n(train/test)={len(train)}/{len(test)}")
    print(f"    new-feature coverage (% non-null, full pool): {coverage}")
    print(f"    best CV AUC={search.best_score_:.3f}  params={search.best_params_}")
    print(f"    edge>0.0 bets: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"market": market_name, "n_train": len(train), "n_test": len(test),
            "cv_auc": search.best_score_, "best_params": search.best_params_,
            "coverage": coverage, **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building real team game log, Elo, L10, bullpen fatigue, park factors (existing pipeline)...")
    games, box = build_games_and_box(con)

    print("extending with Addendum 37 historical features: point-in-time bullpen quality, "
          "opposing/both-starters statcast quality, park dimensions/orientation...")
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    print(f"\nFEATURES_V2 ({len(FEATURES_V2)} total, extends multi_model_comparison.FEATURES "
          f"[{len(FEATURES)}] without mutating it):")
    print(f"  {FEATURES_V2}")
    print("  NOTE: rolling_14d_whip / rolling_14d_bb_per_9 (cache_mlb_historical_bullpen) and "
          "babip_allowed / barrel_rate_allowed / hard_hit_pct_allowed / fly_ball_rate "
          "(cache_mlb_historical_pitcher_statcast) are verified 100% NULL in every row of the "
          "pulled cache tables (COUNT(col) vs COUNT(*) checked directly before writing any join "
          "code) -- excluded as dead columns, not a join bug on this side.")

    all_results = []
    print("\n=== Addendum 37: tuned XGBoost + historical enrichment, single pre-specified grid, "
          "CV-AUC selection only, single pre-specified edge>0.0 cut ===")

    for market_key, stat_col, kind in [
        ("batter_total_bases", "total_bases", "batter"),
        ("batter_home_runs", "home_runs", "batter"),
        ("pitcher_strikeouts", "strikeouts", "pitcher"),
    ]:
        pool = build_warehouse_pool(con, box, market_key, stat_col, kind)
        r = tune_and_evaluate(market_key, pool, games_v2, "per_year", kind, box_ids, con2)
        all_results.append(r)

    for market_key in ["h2h", "totals"]:
        pool = build_game_pool(con, games_v2, market_key)
        r = tune_and_evaluate(market_key, pool, games_v2, "per_year", "game", box_ids, con2)
        all_results.append(r)

    con.close()
    con2.close()

    print("\n=== SUMMARY: BEFORE (cited baseline) vs AFTER (Addendum 37 enrichment), all 5 markets ===")
    for market_key in ["batter_total_bases", "batter_home_runs", "pitcher_strikeouts", "h2h", "totals"]:
        b = BASELINES[market_key]
        print(f"\n{market_key}")
        print(f"  BEFORE [{b['desc']}]: n={b['n']}  ROI={b['roi']}  CI=[{b['lo']},{b['hi']}]  {b['verdict']}")
        r = next((x for x in all_results if x and x["market"] == market_key), None)
        if r is None:
            print("  AFTER  [Addendum 37]: could not be evaluated (insufficient data after enrichment -- see above)")
        else:
            print(f"  AFTER  [Addendum 37]: n(train/test)={r['n_train']}/{r['n_test']}  "
                  f"CV_AUC={r['cv_auc']:.3f}  params={r['best_params']}")
            print(f"                        test: n={r['n']}  ROI={r['roi']}  CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    passes = [r for r in all_results if r and r["verdict"] == "PASS"]
    print(f"\n{len(passes)} of {len([r for r in all_results if r])} evaluated markets pass the "
          f"official gate rule after enrichment. Reported as-is, pass or fail, single "
          f"pre-specified grid and cut -- same discipline as every prior addendum.")


if __name__ == "__main__":
    main()
