"""
Addendum 53: Addendum-37-style historical enrichment (opposing-pitcher
quality, real bullpen quality, park dimensions/orientation) applied to
batter_strikeouts's real backfilled-odds dataset -- the largest sample of
any market in this project (Addendum 30: n=60,862 matched rows, TEST
n=41,489, FAIL but genuinely flat: ROI=-0.87%, CI=[-3.1%,1.35%]).

batter_strikeouts has NEVER been enriched with:
  - opposing-pitcher quality: the SPECIFIC starter the batter faced that
    game (not both starters averaged), season-level xERA/ERA/xBA from
    cache_mlb_historical_pitcher_statcast, matched by player_id+season.
    Resolved via boxscore team_id (build_box_ids) -> which side the batter
    is on -> opposing_pitcher.{home,away}_starter_id for the other side.
  - real bullpen quality: rolling 14-day ERA/K9 from
    cache_mlb_historical_bullpen, point-in-time ASOF-joined by
    team_name+game_date (never a future snapshot).
  - park dimensions/orientation: fence distances, wall heights, CF compass
    degrees from cache_mlb_park_dimensions / cache_mlb_ballpark_orientation.

All three groups reused UNMODIFIED (imported, not copy-pasted) from
addendum37_historical_enrichment.py: add_bullpen_features,
add_starter_game_features, add_park_features, add_opp_pitcher_features,
build_box_ids. The odds-cache parsing / box-score matching is reused
UNMODIFIED from test_backfilled_markets.py: build_market_dataset (which
itself calls parse_cache).

Game-log choice (checked empirically before deciding, not assumed): the
market's real bottleneck is name-matching odds-cache players to boxscore
rows (3,113 unique game_pk after that match), not the lineups-dependent
build_team_game_log() pipeline (3,845 unique game_pk) that the h2h/totals
fan-out bug (Addendum 48) was about. Verified directly: joining the
batter_strikeouts pool to build_team_game_log()'s output keeps 121,637 of
121,695 pre-join rows; joining to Addendum 48's fuller game log
(build_fuller_games, 7,028 unique game_pk, no lineups dependency) keeps all
121,695. A real but small (+58 row, +0.05%) recovery -- unlike h2h/totals,
this market was never meaningfully constrained by the lineups pipeline.
Used the fuller game log anyway (strictly more complete, free, already
built and imported) as the single base for every model below.

SEPARATE, real finding surfaced while doing the mandatory row-count sanity
checks (not assumed, reproduced exactly): Addendum 30's own published
TRAIN=124,467/TEST=41,489 for batter_strikeouts is itself inflated by a
previously-undocumented fan-out bug in test_backfilled_markets.py's game-log
builder -- distinct from the lineups-dependent build_team_game_log() dup
(3,845 vs 4,547 rows) already documented in Addendum 37. client_games has
1,216 distinct game_pk values that map to TWO different event_id values
each (verified directly via SQL -- looks like doubleheader/duplicate-
ingestion rows, not investigated further here since it's out of this
addendum's scope). park_by_event.merge(event_to_gamepk, on="event_id")
therefore produces MULTIPLE park-factor rows per game_pk for those 1,216
games (7,237 rows for 5,975 unique game_pk), and the final
`games.merge(park_by_gamepk, on="game_pk", how="left")` duplicates every
row of `games` for any affected game_pk (3,845 unique game_pk -> 4,547
rows). When the batter_strikeouts pool (121,695 rows spanning 3,113 unique
game_pk, 486 of which are among the 1,216 affected) inner-joins onto that
bloated `games` frame, the pool balloons from 121,695 to 165,956 rows --
reproduced exactly against Addendum 30's cited TRAIN+TEST (124,467+41,489
= 165,956). Addendum 30's TEST n=41,489 is therefore ~36% inflated by
duplicate rows, not 41,489 independent betting opportunities. This
addendum's own pipeline (build_fuller_games, Addendum 48) is NOT affected
-- it dedupes `games` to unique game_pk before any park-factor merge and
asserts the row count is unchanged after every one (see build_games_v3()
below). This is why this addendum's clean TEST n is smaller than Addendum
30's cited TEST n -- an honest, verified difference in data quality, not a
smaller real sample. Flagged for the project separately (same treatment
Addendum 37 gave the build_team_game_log dup); not fixed retroactively in
pitcher_outs/batter_runs_scored here, out of this addendum's scope.

Method (per the task's explicit instruction to reuse the CV-AUC-only tuning
discipline from Addendum 37/48, not Addendum 30's fixed-hyperparameter
run):
  - ONE model family: XGBoost.
  - ONE pre-specified hyperparameter grid: PARAM_GRID/FIXED_PARAMS imported
    unmodified from tuned_xgboost_4_markets.py.
  - Selection metric: CV-averaged ROC AUC on the TRAIN split only (5-fold
    StratifiedKFold via GridSearchCV, threading backend -- loky/process
    backend fails on this sandboxed Windows Python, same fix already used
    in Addendum 37/48). ROI/gate is never touched during selection.
  - Same single chronological 75/25 split Addendum 30 used for this market
    (sort by game_date/game_pk/side/tiebreak, cut at 75%) -- computed ONCE,
    reused for both models below so the enrichment's marginal effect is
    isolated on identical rows.
  - Two models, both pre-registered, both reported regardless of outcome:
      Model A: baseline features only (Elo/L10/fatigue/park-factor -- same
        9 features Addendum 30 used), tuned. Isolates the marginal effect
        of CV-AUC tuning alone vs Addendum 30's untuned fixed-params run.
      Model B: Model A's features + the three Addendum 37 enrichment
        groups. This addendum's actual finding.
  - Refit once per model, score once on the untouched test split, single
    pre-specified edge>0.0 cut, official gate rule (gate() in
    xgboost_individual_markets.py: n>=500 -> ROI>0; else CI lower
    bound>0). No threshold search folded into either verdict.

Mandatory per the task: row counts printed before/after every merge, with
explicit fan-out assertions -- the exact bug class Addenda 37/40/48 found
(a game_pk with duplicate rows on one side of a merge silently multiplying
the pool and narrowing the CI enough to flip a FAIL to a spurious PASS).
"""
import os
import pathlib
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, gate, stat, profit, norm_name, implied_prob  # noqa: E402
from test_backfilled_markets import build_market_dataset  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, add_bullpen_features, add_starter_game_features, add_park_features,
    add_opp_pitcher_features, build_box_ids,
    BULLPEN_V2, STARTER_GAME_V2, OPP_PITCHER_V2, PARK_V2,
)
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

BASE_FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                  "runs_factor", "hr_factor", "k_factor", "hits_factor"]
ENRICHED_FEATURES = BASE_FEATURES + BULLPEN_V2 + STARTER_GAME_V2 + OPP_PITCHER_V2 + PARK_V2

GAME_LEVEL_COLS = (["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                     "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
                    + BULLPEN_V2 + STARTER_GAME_V2 + PARK_V2)

# Addendum 30's published result, cited (not re-derived): untuned fixed
# XGBoost hyperparameters, no enrichment, build_team_game_log() game log.
BASELINE_ADDENDUM30 = {"n": 41489, "roi": -0.87, "lo": -3.1, "hi": 1.35, "verdict": "FAIL"}


def build_pool(con):
    box_bk = con.execute("""
        SELECT game_pk, player_name, batter_strikeouts, is_starter, position_type FROM boxscore
    """).fetchdf()
    box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
    bk_cache = ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl"
    pool = build_market_dataset(bk_cache, "batter_strikeouts", "batter_strikeouts", box_bk,
                                 ["Catcher", "Hitter", "Infielder", "Outfielder"], starter_only=False)
    print(f"  pool (pre game-log join): n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")
    return pool


def build_games_v3(con, con2):
    print("building fuller game log (client_games x cache_mlb_historical_outcomes, Addendum 48 -- "
          "empirically checked above the lineups-dependent build_team_game_log() pipeline)...")
    games, id_to_name = build_fuller_games(con, con2)
    n0 = len(games)
    print(f"  games_v3 base: n={n0:,}  unique game_pk={games.game_pk.nunique():,}")

    games = add_bullpen_features(con2, games, id_to_name)
    assert len(games) == n0, "*** bullpen merge changed row count -- FAN-OUT, STOP ***"
    print(f"  + real bullpen quality (Addendum 37, ASOF by team+date): n={len(games):,}  OK (row count unchanged)")

    games = add_starter_game_features(con, con2, games)
    assert len(games) == n0, "*** starter merge changed row count -- FAN-OUT, STOP ***"
    print(f"  + both-starters statcast quality (Addendum 37): n={len(games):,}  OK (row count unchanged)")

    games = add_park_features(con, con2, games)
    assert len(games) == n0, "*** park merge changed row count -- FAN-OUT, STOP ***"
    print(f"  + park dimensions/orientation (Addendum 37): n={len(games):,}  OK (row count unchanged)")

    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def attach_features(pool, games_v3, box_ids, con2):
    n0 = len(pool)
    pool = pool.merge(games_v3[GAME_LEVEL_COLS], on="game_pk", how="inner")
    print(f"  pool + game-level features (elo/l10/fatigue/park-factor + bullpen/starter/park V2), "
          f"INNER on game_pk (same convention Addendum 30 used): n={n0:,} -> {len(pool):,}")

    pool["name_norm"] = pool["tiebreak"]  # build_market_dataset keeps name_norm only as `tiebreak`
    n1 = len(pool)
    pool = add_opp_pitcher_features(pool, games_v3, box_ids, con2)
    assert len(pool) == n1, "*** opposing-pitcher merge changed row count -- FAN-OUT, STOP ***"
    print(f"  pool + opposing-starting-pitcher quality (Addendum 37, batter-specific, LEFT join): "
          f"n={n1:,} -> {len(pool):,}  OK (row count unchanged)")

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool


def split_chronological(pool):
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.75)
    return pool.iloc[:cut].copy(), pool.iloc[cut:].copy()


def tune_and_evaluate(label, train, test, features):
    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same fix already used in
    # tuned_xgboost_4_markets.py / addendum37 / addendum48).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    test_acc = accuracy_score(y_test, test_pred >= 0.5)

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    print(f"\n  [{label}]")
    print(f"    features ({len(features)}): {features}")
    print(f"    best CV AUC (train only, 5-fold)={search.best_score_:.4f}  params={search.best_params_}")
    print(f"    test AUC={test_auc:.4f}  test acc={test_acc:.4f}  (reported only, not used for selection)")
    print(f"    edge>0.0 bets (single pre-specified cut): n={s['n']}  ROI={s['roi']}  "
          f"CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"label": label, "cv_auc": search.best_score_, "best_params": search.best_params_,
            "test_auc": round(test_auc, 4), "test_acc": round(test_acc, 4), **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=== Addendum 53: batter_strikeouts real-backfilled odds + Addendum-37-style "
          "historical enrichment (opposing-pitcher quality, bullpen quality, park dimensions) ===\n")

    print("--- building batter_strikeouts pool from real backfilled odds (unmodified reuse of "
          "test_backfilled_markets.build_market_dataset / parse_cache) ---")
    pool = build_pool(con)

    print("\n--- building enriched game log ---")
    games_v3 = build_games_v3(con, con2)
    box_ids = build_box_ids(con)

    print(f"\ncoverage check: pool touches {pool.game_pk.nunique():,} unique games; "
          f"games_v3 covers {games_v3.game_pk.nunique():,} unique games")
    print("NOTE: Addendum 30's cited TRAIN=124,467/TEST=41,489 for this market is inflated ~36% by a "
          "real, separately-verified park-factor join fan-out in test_backfilled_markets.py's original "
          "game-log builder (1,216 client_games game_pk values map to 2 different event_id values each; "
          "reproduces exactly to 165,956 rows). games_v3 above (build_fuller_games) is NOT affected -- "
          "deduped to unique game_pk with an explicit assert before any park merge. See module docstring.")

    print("\n--- attaching features to pool ---")
    pool = attach_features(pool, games_v3, box_ids, con2)

    coverage_cols = BULLPEN_V2 + STARTER_GAME_V2 + OPP_PITCHER_V2 + PARK_V2
    coverage = {c: round(100 * pool[c].notna().mean(), 1) for c in coverage_cols}
    print(f"\nnew-feature coverage (% non-null, full pool n={len(pool):,}):")
    for c in coverage_cols:
        print(f"    {c:<22} {coverage[c]}%")
    opp_pitcher_cov = round(100 * pool[OPP_PITCHER_V2].notna().all(axis=1).mean(), 1)
    print(f"  opposing-pitcher-quality row coverage (all 3 opp_sp_* cols present): {opp_pitcher_cov}%")

    train, test = split_chronological(pool)
    print(f"\nTRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    if len(pool) < 500 or len(train) < 50 or len(test) < 50:
        print("INSUFFICIENT DATA -- stopping")
        con.close(); con2.close()
        return

    print("\n=== Model A: tuned XGBoost, BASELINE features only (same 9 features Addendum 30 used -- "
          "isolates the marginal effect of CV-AUC tuning alone vs Addendum 30's untuned fixed-params run) ===")
    result_a = tune_and_evaluate("baseline features (no enrichment), tuned", train, test, BASE_FEATURES)

    print("\n=== Model B: tuned XGBoost, ENRICHED features (+ real bullpen quality, both-starters "
          "statcast, opposing-starter statcast, park dimensions/orientation -- this addendum's finding) ===")
    result_b = tune_and_evaluate("enriched features (Addendum 37 groups)", train, test, ENRICHED_FEATURES)

    con.close()
    con2.close()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    b = BASELINE_ADDENDUM30
    print(f"Addendum 30 (cited, untuned fixed hyperparams, no enrichment, build_team_game_log): "
          f"n={b['n']:,}  ROI={b['roi']}%  CI=[{b['lo']}%,{b['hi']}%]  {b['verdict']}")
    print(f"Model A   (this run, CV-AUC-tuned, NO enrichment, fuller game log): "
          f"n(train/test)={len(train):,}/{len(test):,}  CV_AUC={result_a['cv_auc']:.4f}  "
          f"test n={result_a['n']}  ROI={result_a['roi']}%  CI=[{result_a['lo']}%,{result_a['hi']}%]  {result_a['verdict']}")
    print(f"Model B   (this run, CV-AUC-tuned, WITH Addendum 37 enrichment, fuller game log): "
          f"n(train/test)={len(train):,}/{len(test):,}  CV_AUC={result_b['cv_auc']:.4f}  "
          f"test n={result_b['n']}  ROI={result_b['roi']}%  CI=[{result_b['lo']}%,{result_b['hi']}%]  {result_b['verdict']}")
    print(f"\nMarginal CV-AUC contribution of enrichment on top of tuning (Model B - Model A): "
          f"{result_b['cv_auc'] - result_a['cv_auc']:+.4f}")
    print(f"\nVerdict: {result_b['verdict']} on the official gate rule (single pre-specified edge>0.0 cut, "
          f"n>=500 -> ROI>0 else CI lower bound>0). Reported as-is, pass or fail.")


if __name__ == "__main__":
    main()
