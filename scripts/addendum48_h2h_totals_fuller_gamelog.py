"""
Addendum 48: rebuild the h2h/totals game log directly from client_games +
cache_mlb_historical_outcomes, bypassing the lineups-dependent
build_team_game_log() pipeline every prior addendum (27 onward) has used
for these two markets.

WHY: client_closing_odds already carries real odds for 7,088 games with a
resolvable game_pk. But build_team_game_log() (xgboost_individual_markets.py)
only ever recovers 3,845 usable games -- it derives home/away team identity
by chaining lineups -> client_games -> boxscore (player-level roster
matching for BOTH sides), and that chain silently drops ~45% of otherwise
perfectly good games. Verified directly this session:
  lineups: 7,518 distinct events
  -> client_games join: 6,225 game_pks
  -> + boxscore player join: 5,699 game_pks
  -> both-sides-present + team_runs inner joins: 3,845 (final, current ceiling)

client_games already has home_team/away_team BY NAME directly (7,440 rows,
covering the same odds), and cache_mlb_historical_outcomes (pulled this
session from the client's live system) has home_score/away_score for
8,324 of those events -- verified directly, no lineups or boxscore
player-matching involved at all. This is a completely independent,
MORE COMPLETE path to exactly the same information (who played, who won,
final score) -- roughly 2.2x the usable games versus the current pipeline.

Excludes 3 non-team rows ("American League"/"National League" -- All-Star
game entries, not real MLB games between two franchises) and applies the
one known team-name alias (Athletics -> Oakland Athletics, same as
addendum37_historical_enrichment.TEAM_NAME_ALIAS) before mapping team names
to team_id.

Bullpen fatigue (build_bullpen_fatigue) and Addendum 37's cache-based
enrichment (bullpen quality, starter quality, park dimensions) are reused
unmodified -- none of them actually depend on the lineups table either;
they were just being STARVED by the smaller game log inherited from it.

Same disciplined methodology as every other addendum: ONE model family
(XGBoost, PARAM_GRID/FIXED_PARAMS reused unmodified from
tuned_xgboost_4_markets.py), CV-AUC selection on TRAIN only, ROI/gate
touched exactly once at the single pre-specified edge>0.0 cut, official
gate rule (gate() in xgboost_individual_markets.py).
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
from xgboost_individual_markets import (  # noqa: E402
    DB, build_elo_l10, build_bullpen_fatigue, pick_main_line,
    gate, stat, profit, implied_prob,
)
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, TEAM_NAME_ALIAS, add_bullpen_features, add_starter_game_features, add_park_features,
)
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

MIN_GRADED = 500
EXCLUDE_TEAMS = {"American League", "National League"}


def build_fuller_games(con, con2):
    """games_v3: client_games x cache_mlb_historical_outcomes, no lineups
    or boxscore player-matching dependency at all."""
    raw = con.execute("""
        SELECT g.event_id, g.game_pk, o.home_team, o.away_team,
               o.home_score, o.away_score, TRY_CAST(g.game_date AS DATE) AS game_date
        FROM client_games g
        JOIN cache.cache_mlb_historical_outcomes o ON g.event_id = o.event_id
        WHERE o.game_completed = true
          AND o.home_score IS NOT NULL AND o.away_score IS NOT NULL
          AND g.game_pk IS NOT NULL AND g.game_pk != ''
    """).fetchdf()
    n_raw = len(raw)
    raw = raw[~raw.home_team.isin(EXCLUDE_TEAMS) & ~raw.away_team.isin(EXCLUDE_TEAMS)]
    raw["home_team"] = raw["home_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["away_team"] = raw["away_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["game_pk"] = pd.to_numeric(raw["game_pk"], errors="coerce")
    raw = raw.dropna(subset=["game_pk", "game_date"]).copy()
    raw["game_pk"] = raw["game_pk"].astype("int64")
    raw = raw.drop_duplicates(subset=["game_pk"]).reset_index(drop=True)
    print(f"  games_v3 build: {n_raw} raw event x outcome rows -> {len(raw)} after excluding "
          f"exhibitions, applying name alias, dropping null game_pk/date, and deduping game_pk")

    # team name -> team_id: derive from boxscore's own team_id + player_metadata-free approach --
    # use the EXISTING (smaller) lineups-based game log purely as a name<->id lookup table (MLB
    # has a fixed ~30-team universe; a team's id doesn't change based on which games we can see).
    from multi_model_comparison import build_games_and_box  # noqa: E402 (local import, avoids
    # pulling its heavier dependency chain in at module load time for scripts that don't need it)
    old_games, _ = build_games_and_box(con)
    cg = con.execute("SELECT event_id, game_pk, home_team, away_team FROM client_games").fetchdf()
    cg["game_pk"] = pd.to_numeric(cg["game_pk"], errors="coerce")
    g2 = old_games[["game_pk", "home_team_id", "away_team_id"]].merge(
        cg[["game_pk", "home_team", "away_team"]], on="game_pk", how="inner")
    id_to_name = {}
    for tid, name in g2.groupby("home_team_id")["home_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    for tid, name in g2.groupby("away_team_id")["away_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    name_to_id = {TEAM_NAME_ALIAS.get(name, name): tid for tid, name in id_to_name.items()}
    print(f"  team name -> id lookup built from {len(name_to_id)} names (expect ~30 real MLB teams)")

    raw["home_team_id"] = raw["home_team"].map(name_to_id)
    raw["away_team_id"] = raw["away_team"].map(name_to_id)
    unmapped = raw[raw.home_team_id.isna() | raw.away_team_id.isna()]
    if len(unmapped):
        print(f"  WARNING: {len(unmapped)} games have an unmapped team name, dropped: "
              f"{sorted(set(unmapped.home_team.tolist() + unmapped.away_team.tolist()))}")
    raw = raw.dropna(subset=["home_team_id", "away_team_id"]).copy()
    raw["home_team_id"] = raw["home_team_id"].astype("int64")
    raw["away_team_id"] = raw["away_team_id"].astype("int64")
    raw["home_runs_"] = raw["home_score"].astype(float)
    raw["away_runs_"] = raw["away_score"].astype(float)

    games = raw.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    print(f"  games_v3 final: n={len(games)}  unique game_pk={games.game_pk.nunique()}  "
          f"date range {games.game_date.min()} -> {games.game_date.max()}")
    assert len(games) == games.game_pk.nunique(), "duplicate game_pk in games_v3 -- STOP"

    elo_l10 = build_elo_l10(games)
    games = games.merge(elo_l10, on="game_pk", how="inner")
    assert len(games) == games.game_pk.nunique(), "elo/l10 merge changed row count -- STOP"

    fatigue = build_bullpen_fatigue(con)
    games["home_fatigue"] = games.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    games["away_fatigue"] = games.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)

    parks = con.execute("SELECT * FROM ballpark_factors").fetchdf()
    for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
        parks[c] = pd.to_numeric(parks[c], errors="coerce")
    park_by_event = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    park_by_event = park_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    venue_by_gamepk = games[["event_id", "game_pk"]].merge(park_by_event, on="event_id", how="left")[
        ["game_pk", "runs_factor", "hr_factor", "k_factor", "hits_factor"]].drop_duplicates(subset=["game_pk"])
    games = games.merge(venue_by_gamepk, on="game_pk", how="left")
    assert len(games) == games.game_pk.nunique(), "park factor merge changed row count -- STOP"

    return games, id_to_name


def build_pool(con, games, market_key):
    if market_key == "h2h":
        odds = con.execute("""
            SELECT co.game_pk, co.best_over_odds AS odds
            FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                  JOIN client_games g ON co.event_id = g.event_id) co
            WHERE co.market_key = 'h2h__away' AND co.game_pk IS NOT NULL AND co.game_pk != ''
        """).fetchdf()
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
        odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
        odds = odds.dropna(subset=["game_pk"])
        odds["game_pk"] = odds["game_pk"].astype("int64")
        m = odds.merge(games, on="game_pk", how="inner")
        m["win"] = m["away_runs_"] > m["home_runs_"]
        m["side"] = "away"
        m["tiebreak"] = ""
        return m.dropna(subset=["odds"])
    else:  # totals
        odds = con.execute("""
            SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds
            FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                  JOIN client_games g ON co.event_id = g.event_id) co
            WHERE co.market_key = 'totals' AND co.game_pk IS NOT NULL AND co.game_pk != ''
        """).fetchdf()
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
        odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
        odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
        odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
        odds = odds.dropna(subset=["game_pk"])
        odds["game_pk"] = odds["game_pk"].astype("int64")
        odds = pick_main_line(odds, ["game_pk"], over_odds_col="best_over_odds")
        m = odds.merge(games, on="game_pk", how="inner")
        m["total_runs"] = m["home_runs_"] + m["away_runs_"]
        under = m.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = ""
        over = m.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = ""
        return pd.concat([under, over]).dropna(subset=["odds"])


def build_features(pool):
    pool = pool.copy()
    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    return pool


def split_per_year(pool):
    train_parts, test_parts = [], []
    for _, grp in pool.groupby(pool.game_date.dt.year if hasattr(pool.game_date, "dt") else pd.to_datetime(pool.game_date).dt.year):
        grp = grp.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort")
        cut = int(len(grp) * 0.75)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
            "runs_factor", "hr_factor", "k_factor", "hits_factor",
            "bullpen_era_diff", "bullpen_k9_diff", "starter_xera_diff", "starter_era_diff",
            "starter_xba_diff", "lf_distance", "cf_distance", "rf_distance",
            "lf_wall_height", "cf_wall_height", "rf_wall_height", "cf_compass_degrees", "is_dome"]


def evaluate(market_key, con, con2, games, id_to_name):
    print(f"\n{'=' * 78}\n{market_key}\n{'=' * 78}")
    pool = build_pool(con, games, market_key)
    n_before = len(pool)
    pool = add_bullpen_features(con2, pool.rename(columns={}), id_to_name) if False else pool
    # add_bullpen_features expects a `games`-shaped frame w/ home_team_id/away_team_id/game_date;
    # pool already carries those (merged in via build_pool's join onto `games`), so call it on pool directly.
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
    assert len(pool) == n_before, "enrichment merge changed row count -- aborting"

    pool = build_features(pool)
    pool["game_date"] = pd.to_datetime(pool["game_date"])

    if len(pool) < MIN_GRADED:
        print(f"  n={len(pool)} INSUFFICIENT DATA")
        return None
    train, test = split_per_year(pool)
    if len(train) < 50 or len(test) < 50:
        print("  INSUFFICIENT DATA after split")
        return None

    # Do NOT dropna on FEATURES -- XGBoost is configured with missing=np.nan
    # specifically to handle partial feature missingness natively (same as
    # every other addendum's XGBoost usage in this project). Only drop rows
    # missing the label/odds themselves, which really can't be used at all.
    train = train.dropna(subset=["win", "odds"])
    test = test.dropna(subset=["win", "odds"])
    print(f"  n(train/test)={len(train)}/{len(test)}")

    X_train, y_train = train[FEATURES], train["win"].astype(int)
    X_test = test[FEATURES]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=6, refit=True)
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)
    test_pred = search.best_estimator_.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  CV AUC={search.best_score_:.4f}  params={search.best_params_}")
    print(f"  edge>0.0 test: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"market": market_key, "n_train": len(train), "n_test": len(test), **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log from client_games x cache_mlb_historical_outcomes...")
    games, id_to_name = build_fuller_games(con, con2)

    results = []
    for mk in ["h2h", "totals"]:
        results.append(evaluate(mk, con, con2, games, id_to_name))

    print("\n" + "=" * 78)
    print("SUMMARY -- Addendum 48 (fuller game log) vs best prior result")
    print("=" * 78)
    print("  h2h    BEFORE (Addendum 43c regression, best prior): n=242  ROI=6.9  CI=[-3.98,17.78]  FAIL")
    print("  totals BEFORE (Addendum 32 baseline, best prior): n=434  ROI=3.69  CI=[-5.27,12.64]  FAIL")
    for r in results:
        if r:
            print(f"  {r['market']} AFTER: n(train/test)={r['n_train']}/{r['n_test']}  "
                  f"test n={r['n']}  ROI={r['roi']}  CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
