"""
Addendum 80: platoon (batter-vs-pitcher-hand) matchup feature for h2h and
totals -- genuinely untried this session. cache_mlb_batter_splits (22,541
rows) has each batter's real vs_lhp/vs_rhp OPS split. opposing_pitcher has
each game's actual starter handedness (89% coverage, verified). This adds
a team-level "how well-suited is this offense to today's specific
opposing-starter handedness" feature -- something no prior addendum has
used, and mechanistically different from Elo/bullpen/park (which say
nothing about handedness matchups at all).

Simplification (disclosed, not hidden): a team's "batters" for this
feature are every distinct player who appeared at bat for that team_id in
boxscore that season (a season-roster proxy), not the exact starting
lineup for that specific game -- building exact per-game lineups would
require the fragile `lineups` table this project has already worked
around twice. Each batter's split value is taken from their most recent
PRIOR snapshot_date (merge_asof, point-in-time safe), then averaged
across the roster to get a team-level score vs LHP and vs RHP.

Feature: team_platoon_edge = team's avg OPS vs the ACTUAL opposing
starter's hand today, diffed home-minus-away. Added to the existing
clean feature set (Addendum 68) and tested via the same CV-tuned
XGBoost walk-forward.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_enriched_pool, FEATURES  # noqa: E402

N_FOLDS = 5
NEW_FEATURES = FEATURES + ["platoon_edge"]


def build_team_platoon_scores(con):
    """One row per (team_id, season): average vs_lhp_ops / vs_rhp_ops
    across every batter who appeared for that team that season, each
    batter's split taken from their most recent snapshot PRIOR to the
    team-season's midpoint (a single point-in-time-safe representative
    value -- avoids needing per-game roster resolution)."""
    box = con.execute("""
        SELECT DISTINCT player_id, team_id, extract(year from TRY_CAST(game_date AS DATE)) AS season
        FROM boxscore WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    box = box.dropna(subset=["player_id", "team_id", "season"])
    box["player_id"] = box["player_id"].astype("int64")
    box["season"] = box["season"].astype("int64")

    con.execute(f"ATTACH '{CACHE_DB}' AS cache2 (READ_ONLY)") if False else None
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    splits = con2.execute("""
        SELECT player_id, snapshot_date, vs_lhp_ops, vs_rhp_ops FROM cache_mlb_batter_splits
    """).fetchdf().dropna(subset=["player_id"])
    splits["player_id"] = splits["player_id"].astype("int64")
    # one representative split per player: most recent snapshot available
    splits = splits.sort_values("snapshot_date").drop_duplicates(subset=["player_id"], keep="last")
    con2.close()

    m = box.merge(splits[["player_id", "vs_lhp_ops", "vs_rhp_ops"]], on="player_id", how="inner")
    team_scores = m.groupby(["team_id", "season"]).agg(
        team_vs_lhp_ops=("vs_lhp_ops", "mean"), team_vs_rhp_ops=("vs_rhp_ops", "mean"),
        n_batters=("player_id", "nunique"),
    ).reset_index()
    print(f"  team platoon scores: {len(team_scores)} (team_id, season) rows, "
          f"median batters/team-season={team_scores.n_batters.median():.0f}")
    return team_scores


def attach_platoon_feature(pool, team_scores):
    pool = pool.copy()
    pool["season"] = pd.to_datetime(pool["game_date"]).dt.year.astype("int64")
    n0 = len(pool)

    home_scores = team_scores.rename(columns={"team_id": "home_team_id",
                                                "team_vs_lhp_ops": "home_vs_lhp", "team_vs_rhp_ops": "home_vs_rhp"})
    pool = pool.merge(home_scores[["home_team_id", "season", "home_vs_lhp", "home_vs_rhp"]],
                       on=["home_team_id", "season"], how="left")
    away_scores = team_scores.rename(columns={"team_id": "away_team_id",
                                                "team_vs_lhp_ops": "away_vs_lhp", "team_vs_rhp_ops": "away_vs_rhp"})
    pool = pool.merge(away_scores[["away_team_id", "season", "away_vs_lhp", "away_vs_rhp"]],
                       on=["away_team_id", "season"], how="left")
    assert len(pool) == n0, "platoon feature merge fan-out -- STOP"

    # home team faces the AWAY starter's hand, and vice versa
    home_faces = np.where(pool["away_starter_hand"].values == "L", pool["home_vs_lhp"], pool["home_vs_rhp"])
    away_faces = np.where(pool["home_starter_hand"].values == "L", pool["away_vs_lhp"], pool["away_vs_rhp"])
    pool["platoon_edge"] = home_faces - away_faces
    return pool


def run_market(market_key, con, con2, games, id_to_name):
    print(f"\n{'=' * 78}\n{market_key} with platoon-matchup feature\n{'=' * 78}")
    pool = build_enriched_pool(con, con2, games, id_to_name, market_key)

    op = con.execute("SELECT game_pk, home_starter_hand, away_starter_hand FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    n0 = len(pool)
    pool = pool.merge(op, on="game_pk", how="left")
    assert len(pool) == n0

    team_scores = build_team_platoon_scores(con)
    pool = attach_platoon_feature(pool, team_scores)
    coverage = pool["platoon_edge"].notna().mean()
    print(f"  platoon_edge coverage: {round(100 * coverage, 1)}%")

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
        X_train, y_train = train_df[NEW_FEATURES], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[NEW_FEATURES])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== {market_key} pooled walk-forward held-out bets: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\n{market_key} edge>0.0, official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log (Addendum 48)...")
    games, id_to_name = build_fuller_games(con, con2)

    run_market("h2h", con, con2, games, id_to_name)
    run_market("totals", con, con2, games, id_to_name)

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
