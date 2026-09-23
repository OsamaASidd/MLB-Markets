"""
Addendum 51: pitcher_outs, full treatment -- combining all three enrichment
improvements proven elsewhere in this project onto the real backfilled-odds
pitcher_outs dataset (Addendum 30: real odds from The Odds API, matched to
real box-score outcomes, the closest-ever FAIL in this project at -1.21%
ROI, CI [-7.2%, 4.78%], n=3,334 test).

pitcher_outs has been re-tested many times (Addenda 27, 30, 34, 38, 43b) and
always FAILs, but it has NEVER been enriched with the richer signal already
proven valuable on other markets (Addendum 37: bullpen quality, starting-
pitcher quality, park dimensions/orientation) and has NEVER been evaluated
via walk-forward cross-validation (only a single 75/25 split). This script
combines both, following the exact pattern that turned batter_runs_scored
and h2h into this project's closest results (Addenda 47/48/49):

  1. Real backfilled odds pool (test_backfilled_markets.parse_cache /
     build_market_dataset, reused UNMODIFIED) -- 5,451 real (event,player)
     rows matched to real box-score pitcher_outs outcomes, over+under both
     sides = 10,896 raw bet-rows.
  2. Game log: empirically checked whether Addendum 48's fuller game log
     (build_fuller_games, client_games x cache_mlb_historical_outcomes, no
     lineups/boxscore-roster dependency, 7,028 usable games vs the old
     build_team_game_log()'s 3,845) helps HERE too. Verified directly
     (see "GAME LOG CHECK" section below, printed at runtime): the old,
     lineups-based game log already covers all 3,122 unique game_pk this
     pool needs -- no additional games are recovered. BUT the old table
     itself carries 3,858 rows for 3,845 unique game_pk (a small residual
     duplicate-game_pk artifact, the same join-fan-out bug class flagged in
     Addenda 37/40/48) which inflates the pool by ~60 spurious duplicate
     rows on a plain merge. The fuller, de-duplicated games_v3 table avoids
     this for free while losing nothing, so it's used here -- not because
     it recovers more rows (it doesn't, for this specific pool), but
     because it's a strictly cleaner join target. Reported honestly either
     way.
  3. Addendum 37's three enrichment feature groups (add_bullpen_features,
     add_starter_game_features, add_park_features -- reused UNMODIFIED,
     imported, not duplicated). pitcher_outs has no "batter vs opposing
     pitcher" structure (it doesn't ask about a specific matchup batter);
     per the task brief it's treated like h2h/totals: both starters'
     quality diffed at the game level (STARTER_GAME_V2), not a single
     opposing-pitcher row-level feature.
  4. The SAME walk-forward pattern as Addenda 47/49: expanding-window
     folds, hyperparameters selected via CV-AUC (GridSearchCV, PARAM_GRID_
     WIDE, imported unmodified from addendum49_h2h_walkforward.py) on
     TRAINING data only within each fold, held-out predictions from every
     fold pooled, official gate touched EXACTLY ONCE on the full pooled
     set at the very end.

Mandatory per the task brief: pool row counts printed and asserted at every
merge step below -- the exact join-fan-out bug class found in Addenda
37/40/48 has recurred multiple times this session.
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

from xgboost_individual_markets import (  # noqa: E402
    DB, build_team_game_log, norm_name, implied_prob, profit, stat, gate,
)
from test_backfilled_markets import build_market_dataset  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, add_bullpen_features, add_starter_game_features, add_park_features,
)
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum49_h2h_walkforward import PARAM_GRID_WIDE, FIXED_PARAMS  # noqa: E402

N_FOLDS = 6  # same as Addendum 49 (h2h) -- pool is comparable order of magnitude

BULLPEN_V2 = ["bullpen_era_diff", "bullpen_k9_diff"]
STARTER_GAME_V2 = ["starter_xera_diff", "starter_era_diff", "starter_xba_diff"]
PARK_V2 = ["lf_distance", "cf_distance", "rf_distance",
           "lf_wall_height", "cf_wall_height", "rf_wall_height",
           "cf_compass_degrees", "is_dome"]
BASE_FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                  "runs_factor", "hr_factor", "k_factor", "hits_factor"]
FEATURES_V2 = BASE_FEATURES + BULLPEN_V2 + STARTER_GAME_V2 + PARK_V2

BASELINE_ADDENDUM30 = {"n": 3334, "roi": -1.21, "lo": -7.2, "hi": 4.78, "verdict": "FAIL"}


def build_pool(con, box):
    po_cache = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"
    pool = build_market_dataset(po_cache, "pitcher_outs", "pitcher_outs_stat", box,
                                 ["Pitcher"], starter_only=True)
    print(f"  pool (both over+under sides): n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")
    return pool


def check_game_log_choice(con, con2, pool):
    """Empirically verify whether Addendum 48's fuller game log recovers
    more usable rows for pitcher_outs than the old lineups-based one, and
    whether the old one has the known duplicate-game_pk fan-out issue."""
    print("\n=== GAME LOG CHECK (Addendum 48 fuller log vs old lineups-based log) ===")
    old_games = build_team_game_log(con)
    print(f"  old build_team_game_log(): rows={len(old_games):,}  unique game_pk={old_games.game_pk.nunique():,}"
          f"  {'(DUPLICATE game_pk ROWS PRESENT)' if len(old_games) != old_games.game_pk.nunique() else '(clean)'}")
    merged_old = pool.merge(old_games[["game_pk"]], on="game_pk", how="inner")
    print(f"  pool JOINED to OLD games: n={len(merged_old):,} (pool was {len(pool):,}) "
          f"unique game_pk={merged_old.game_pk.nunique():,}  "
          f"{'*** FAN-OUT vs raw pool ***' if len(merged_old) != len(pool) else 'no fan-out'}")

    games_v3, _ = build_fuller_games(con, con2)
    print(f"  Addendum 48 build_fuller_games(): rows={len(games_v3):,}  unique game_pk={games_v3.game_pk.nunique():,}")
    merged_new = pool.merge(games_v3[["game_pk"]], on="game_pk", how="inner")
    print(f"  pool JOINED to NEW games_v3: n={len(merged_new):,} (pool was {len(pool):,}) "
          f"unique game_pk={merged_new.game_pk.nunique():,}  "
          f"{'*** FAN-OUT vs raw pool ***' if len(merged_new) != len(pool) else 'no fan-out'}")

    recovered = merged_new.game_pk.nunique() - merged_old.game_pk.nunique()
    print(f"  VERDICT: fuller game log recovers {recovered} additional unique game_pk for this pool "
          f"(old log's {old_games.game_pk.nunique():,} unique games already covers "
          f"{merged_old.game_pk.nunique():,} of the {pool.game_pk.nunique():,} this pool needs). "
          f"Using games_v3 anyway: it is de-duplicated by construction (avoids the "
          f"{len(merged_old) - len(pool)}-row fan-out seen on the old table above), a strictly "
          f"cleaner join target even though it adds zero net rows here.")
    return games_v3


def build_games_full(con, con2, games_v3):
    n0 = len(games_v3)
    assert games_v3.game_pk.nunique() == n0, "games_v3 should already be unique per game_pk -- STOP"

    id_to_name = {}
    cg = con.execute("SELECT event_id, game_pk, home_team, away_team FROM client_games").fetchdf()
    cg["game_pk"] = pd.to_numeric(cg["game_pk"], errors="coerce")
    from addendum37_historical_enrichment import TEAM_NAME_ALIAS  # noqa: E402
    g2 = games_v3[["game_pk", "home_team_id", "away_team_id"]].merge(
        cg[["game_pk", "home_team", "away_team"]], on="game_pk", how="inner")
    for tid, name in g2.groupby("home_team_id")["home_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    for tid, name in g2.groupby("away_team_id")["away_team"].agg(lambda s: s.mode().iat[0]).items():
        id_to_name.setdefault(tid, name)
    id_to_name = {tid: TEAM_NAME_ALIAS.get(name, name) for tid, name in id_to_name.items()}

    print("\n=== Addendum 37 enrichment (bullpen quality, starter quality, park dims), reused unmodified ===")
    games_full = add_bullpen_features(con2, games_v3, id_to_name)
    assert len(games_full) == n0, "bullpen-feature merge changed games row count -- STOP"
    print(f"  after add_bullpen_features: rows={len(games_full):,} (unchanged, OK)")

    games_full = add_starter_game_features(con, con2, games_full)
    assert len(games_full) == n0, "starter-feature merge changed games row count -- STOP"
    print(f"  after add_starter_game_features: rows={len(games_full):,} (unchanged, OK)")

    games_full = add_park_features(con, con2, games_full)
    assert len(games_full) == n0, "park-feature merge changed games row count -- STOP"
    print(f"  after add_park_features: rows={len(games_full):,} (unchanged, OK)")

    return games_full


def cv_select(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID_WIDE,
                           scoring="roc_auc", cv=cv, n_jobs=os.cpu_count())
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same fix used throughout this
    # project's addenda).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here
    return search.best_estimator_, search.best_score_, search.best_params_


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=== Addendum 51: pitcher_outs, full treatment (enrichment + walk-forward CV) ===")
    print("\nbuilding real backfilled-odds pool matched to real box-score outcomes...")
    box = con.execute("""
        SELECT game_pk, player_name, outs AS pitcher_outs_stat, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    pool = build_pool(con, box)

    games_v3 = check_game_log_choice(con, con2, pool)
    games_full = build_games_full(con, con2, games_v3)

    merge_cols = (["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                    "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
                  + BULLPEN_V2 + STARTER_GAME_V2 + PARK_V2)
    games_full_dedup = games_full.drop_duplicates(subset=["game_pk"])[merge_cols]
    n_before = len(pool)
    pool_full = pool.merge(games_full_dedup, on="game_pk", how="inner")
    print(f"\n=== final feature merge (pool x enriched games_v3) ===")
    print(f"  pool before merge: n={n_before:,}   after merge: n={len(pool_full):,}  "
          f"{'OK -- no fan-out' if len(pool_full) == n_before else '*** ROW COUNT MISMATCH, STOP ***'}")
    assert len(pool_full) == n_before, "final feature merge changed row count -- STOP, investigate fan-out/drop"

    coverage_cols = BULLPEN_V2 + STARTER_GAME_V2 + PARK_V2
    coverage = {c: round(100 * pool_full[c].notna().mean(), 1) for c in coverage_cols}
    print(f"\n  new-feature coverage (% non-null, full pool n={len(pool_full):,}):")
    for c, pct in coverage.items():
        print(f"    {c:<20} {pct}%")

    pool_full["game_date"] = pd.to_datetime(pool_full["game_date"])
    pool_full["side_code"] = pool_full["side"].astype("category").cat.codes
    pool_full["elo_diff"] = (pool_full["home_elo"] + 24) - pool_full["away_elo"]
    pool_full["l10_diff"] = pool_full["home_l10"] - pool_full["away_l10"]
    pool_full["fatigue_diff"] = pool_full["home_fatigue"] - pool_full["away_fatigue"]
    pool_full["market_prob"] = implied_prob(pool_full["odds"])

    pool_full = pool_full.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"\n  pool ready for walk-forward: n={len(pool_full):,}  "
          f"date range {pool_full.game_date.min().date()} -> {pool_full.game_date.max().date()}")

    fold_id = pd.qcut(np.arange(len(pool_full)), N_FOLDS, labels=False)
    pool_full["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds, CV-AUC hyperparameter "
          f"selection on TRAIN only, PARAM_GRID_WIDE from Addendum 49) ===")
    for f in range(1, N_FOLDS):  # fold 0 has no prior data to train on -- skipped, not evaluated
        train_df = pool_full[pool_full["fold"] < f]
        test_df = pool_full[pool_full["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df):,}  test n={len(test_df):,}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        X_train, y_train = train_df[FEATURES_V2], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")

        test_pred = model.predict_proba(test_df[FEATURES_V2])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled):,} ===")
    print("(official gate touched EXACTLY ONCE below, on the full pooled set -- no per-fold peeking above)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    print("\n=== HONEST COMPARISON: Addendum 51 (full treatment) vs Addendum 30 (baseline) ===")
    b = BASELINE_ADDENDUM30
    print(f"  Addendum 30 (single 75/25 split, basic features only):"
          f"  n={b['n']}  ROI={b['roi']}%  CI=[{b['lo']},{b['hi']}]  {b['verdict']}")
    print(f"  Addendum 51 (walk-forward, enriched features):"
          f"  n={s['n']}  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
