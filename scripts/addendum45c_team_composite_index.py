"""
Addendum 45c: team-level "current form" composite -- aggregating individual
batter/pitcher recent performance (not just team win-loss record) up to a
team-level differential feature, tested on the two remaining severely
sample-constrained markets, h2h and totals.

WHY THIS IS DIFFERENT FROM WHAT'S ALREADY THERE: elo_diff/l10_diff (already
in FEATURES_V2, via multi_model_comparison.py) capture whether a team has
been WINNING games. They say nothing about whether the team's individual
PLAYERS are hitting/pitching well right now -- a team can be riding a lucky
win streak with cold bats, or slumping in the standings despite hot
individual form. This addendum builds a composite directly from real
boxscore batting/pitching production (not runs/wins) to target that gap.

POINT-IN-TIME CONSTRUCTION (the critical ground rule for this addendum):
  1. From `boxscore` (spans 2020-07-23 to 2026-07-27, real data), build one
     row per (team_id, game_pk, game_date): team-wide SUM(hits)+SUM(total_bases)
     and SUM(at_bats) across every batter who played for that team that game
     (position_type in Catcher/Hitter/Infielder/Outfielder), and team-wide
     SUM(strikeouts) and SUM(outs) across every pitcher who threw for that
     team that game (position_type='Pitcher'). `outs` is boxscore's own
     already-computed outs-recorded column (verified consistent with
     innings_pitched on a manual spot check -- e.g. innings_pitched=1.3 ->
     outs=4 -- so no re-parsing of the IP string needed here).
  2. For each team INDEPENDENTLY, sort that team's own games chronologically
     (game_date, then game_pk as a tiebreak, mergesort for determinism), then
     take a rolling SUM over the trailing 10 games, having first shift(1)'d
     the per-game totals by one game. The shift is what makes this strictly
     point-in-time: the value attached to game i uses only games
     i-10 .. i-1 of THAT team's own schedule, never game i itself and never
     a future game. min_periods=3 (need at least 3 prior team-games of
     history) avoids a single-game's noise driving the composite to a
     spurious extreme early in a team's tracked history; below that, the
     composite is NaN and XGBoost (missing=np.nan, the project's standard
     handling throughout) treats it as missing rather than fabricating a
     number.
  3. Two rates, each from the trailing sums: batting_form_rate =
     (trailing hits+total_bases) / (trailing at_bats) -- a standard combined
     contact+power rate; pitching_form_rate = (trailing strikeouts) /
     (trailing outs recorded) -- a standard dominance rate for the pitching
     staff actually used recently. Both are built ONLY from the team's own
     past games, so both are point-in-time safe by construction, independent
     of which games later end up in the odds-based h2h/totals pool (the
     rolling history is built across boxscore's FULL span, so a team's
     October 2023 composite already reflects real 2023 games earlier that
     season, not a cold start at the odds-warehouse boundary).
  4. COMBINATION METHOD (decided here, once, before touching any ROI number
     -- not grid-searched): the two rates live on very different scales
     (~0.55-0.70 for batting_form_rate, ~0.15-0.30 for pitching_form_rate),
     so a raw average would silently be almost all batting. Each rate is
     z-scored using ITS OWN population mean/std computed across the full
     panel of trailing-window values (a fixed rescaling constant, not a
     per-split or ROI-fit parameter -- this project already accepts static,
     dataset-wide normalization constants elsewhere, e.g. ballpark_factors
     are season-level aggregates, not point-in-time either). The composite
     is then the simple, equal-weighted average of the two z-scores: no a
     priori reason to weight recent hitting over recent pitching (or vice
     versa) for "how good is this team playing right now" -- equal weight is
     the most defensible default absent such a reason, and it is fixed
     before any test-set number is ever computed.
  5. home_team_form_diff = home team's composite - away team's composite,
     same "higher = better for home" sign convention as elo_diff/l10_diff.

Reuses UNMODIFIED, via import: build_games_and_box, build_game_pool,
split_per_year (multi_model_comparison.py); build_games_v2, build_box_ids,
build_features_v2, FEATURES_V2 (addendum37_historical_enrichment.py) --
build_games_v2 is also where the real duplicate-game_pk bug (Addendum 37)
was fixed, so re-verified again below, not assumed; GAME_FEATURES_V2, the
prior baseline BASELINES dict (addendum43c_regression_game_markets.py);
gate/stat/profit/DB (xgboost_individual_markets.py); PARAM_GRID/FIXED_PARAMS
(tuned_xgboost_4_markets.py). This script's own new code is only the team
composite itself and the one new feature it produces.

DISCIPLINE (same as every prior addendum): the composite's construction
(window=10, min_periods=3, equal-weight z-score average) is fixed BEFORE
looking at any CV or test number -- not grid-searched. Model/feature
selection uses CV-averaged ROC AUC on the TRAIN split ONLY (5-fold
StratifiedKFold via GridSearchCV, PARAM_GRID/FIXED_PARAMS unmodified). The
CV AUC of FEATURES_V2 (baseline, no composite) vs FEATURES_V2 + composite is
compared on TRAIN ONLY to isolate the composite's marginal contribution --
that comparison never touches the test set. The test set is then touched
EXACTLY ONCE per market, using the with-composite model (the hypothesis
under test here), at the single pre-specified edge>0.0 / odds<0 cut, scored
by the project's official gate() rule. A FAIL is reported as FAIL.
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

from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_game_pool, split_per_year,
)
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, build_games_v2, build_box_ids, build_features_v2, FEATURES_V2,
)
from addendum43c_regression_game_markets import GAME_FEATURES_V2, BASELINES  # noqa: E402
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

ROLL_WINDOW = 10
ROLL_MIN_PERIODS = 3
COMPOSITE_FEATURE = "home_team_form_diff"
FEATURES_WITH_COMPOSITE = GAME_FEATURES_V2 + [COMPOSITE_FEATURE]


def build_team_game_panel(con):
    """One row per (team_id, game_pk, game_date): real team-wide batting and
    pitching box totals for that single game, built directly from `boxscore`
    across its FULL span (2020-2026) -- independent of the odds-based game
    pool, so trailing windows have real history before the odds-warehouse
    era (2023) starts, not a cold start."""
    dates = con.execute("SELECT DISTINCT game_pk, game_date FROM boxscore").fetchdf()
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    # A handful of game_pk values have >1 distinct game_date logged (the
    # same real data quirk documented in Addendum 37) -- take the earliest
    # deterministically so this script has one date per game_pk, same
    # dedup discipline as build_games_v2's drop_duplicates fix.
    dates = dates.sort_values("game_date", kind="mergesort").drop_duplicates(subset=["game_pk"], keep="first")

    bat = con.execute("""
        SELECT game_pk, team_id, SUM(hits) AS b_hits, SUM(total_bases) AS b_tb, SUM(at_bats) AS b_ab
        FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
        GROUP BY 1, 2
    """).fetchdf()
    pit = con.execute("""
        SELECT game_pk, team_id, SUM(strikeouts) AS p_k, SUM(outs) AS p_outs
        FROM boxscore
        WHERE position_type = 'Pitcher'
        GROUP BY 1, 2
    """).fetchdf()
    panel = bat.merge(pit, on=["game_pk", "team_id"], how="outer")
    panel = panel.merge(dates, on="game_pk", how="inner")
    return panel


def add_rolling_form(panel):
    """Per-team, strictly-prior-games rolling sums -> two rates. shift(1)
    before rolling is what excludes the current game (no same-game or
    future data); window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS fixed in
    advance, not tuned against any downstream number."""
    out = []
    for _, grp in panel.groupby("team_id", sort=False):
        grp = grp.sort_values(["game_date", "game_pk"], kind="mergesort").copy()
        prior_hits_tb = (grp["b_hits"] + grp["b_tb"]).shift(1)
        prior_ab = grp["b_ab"].shift(1)
        prior_k = grp["p_k"].shift(1)
        prior_outs = grp["p_outs"].shift(1)
        grp["roll_hits_tb"] = prior_hits_tb.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).sum()
        grp["roll_ab"] = prior_ab.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).sum()
        grp["roll_k"] = prior_k.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).sum()
        grp["roll_outs"] = prior_outs.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).sum()
        out.append(grp)
    panel = pd.concat(out).reset_index(drop=True)

    panel["batting_form_rate"] = np.where(panel["roll_ab"] > 0, panel["roll_hits_tb"] / panel["roll_ab"], np.nan)
    panel["pitching_form_rate"] = np.where(panel["roll_outs"] > 0, panel["roll_k"] / panel["roll_outs"], np.nan)
    return panel


def add_composite(panel):
    """Equal-weight average of two z-scored rates -- see module docstring
    ("COMBINATION METHOD") for why: different scales, no a priori reason to
    weight one over the other, fixed here before any ROI is computed."""
    bmean, bstd = panel["batting_form_rate"].mean(), panel["batting_form_rate"].std()
    pmean, pstd = panel["pitching_form_rate"].mean(), panel["pitching_form_rate"].std()
    print(f"  batting_form_rate: mean={bmean:.4f} std={bstd:.4f} (n non-null={panel['batting_form_rate'].notna().sum()})")
    print(f"  pitching_form_rate: mean={pmean:.4f} std={pstd:.4f} (n non-null={panel['pitching_form_rate'].notna().sum()})")
    panel["z_bat"] = (panel["batting_form_rate"] - bmean) / bstd
    panel["z_pit"] = (panel["pitching_form_rate"] - pmean) / pstd
    panel["team_form"] = 0.5 * panel["z_bat"] + 0.5 * panel["z_pit"]
    return panel


def add_team_form_diff(games_v2, panel):
    """LEFT-merges the point-in-time team composite onto games_v2 for both
    home and away teams, keyed on (game_pk, team_id) -- panel has exactly
    one row per (team_id, game_pk), so this is a clean many-to-one join,
    verified below by an explicit row-count assert (the exact bug class
    Addendum 37 found: never assume a join stays 1:1)."""
    games = games_v2.copy()
    n0 = len(games)

    home = panel[["team_id", "game_pk", "team_form"]].rename(
        columns={"team_id": "home_team_id", "team_form": "home_team_form"})
    away = panel[["team_id", "game_pk", "team_form"]].rename(
        columns={"team_id": "away_team_id", "team_form": "away_team_form"})
    games = games.merge(home, on=["game_pk", "home_team_id"], how="left")
    games = games.merge(away, on=["game_pk", "away_team_id"], how="left")

    n1 = len(games)
    assert n1 == n0, f"team-composite merge changed games_v2 row count ({n0} -> {n1}) -- investigate"
    print(f"  team-composite merge onto games_v2: n={n0} -> n={n1} (OK, no fan-out)")

    games[COMPOSITE_FEATURE] = games["home_team_form"] - games["away_team_form"]
    return games


def cv_auc(features, train):
    """CV-averaged ROC AUC on the TRAIN split only, via GridSearchCV over
    the project's own pre-specified PARAM_GRID/FIXED_PARAMS
    (tuned_xgboost_4_markets.py, unmodified) -- never touches ROI/test."""
    X_train, y_train = train[features], train["win"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead, same fix already used throughout
    # this project (Addendum 34/37).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)
    return search


def evaluate_market(market_name, pool):
    train, test = split_per_year(pool)
    print(f"\n  [{market_name}] n(train/test)={len(train)}/{len(test)}")

    print(f"  [{market_name}] CV AUC (TRAIN only, marginal-contribution check, ROI untouched):")
    search_base = cv_auc(GAME_FEATURES_V2, train)
    print(f"    WITHOUT composite (FEATURES_V2, {len(GAME_FEATURES_V2)} features): "
          f"best CV AUC={search_base.best_score_:.4f}  params={search_base.best_params_}")
    search_comp = cv_auc(FEATURES_WITH_COMPOSITE, train)
    print(f"    WITH    composite (FEATURES_V2 + {COMPOSITE_FEATURE}, {len(FEATURES_WITH_COMPOSITE)} features): "
          f"best CV AUC={search_comp.best_score_:.4f}  params={search_comp.best_params_}")
    delta = search_comp.best_score_ - search_base.best_score_
    print(f"    marginal contribution of {COMPOSITE_FEATURE}: {delta:+.4f} CV AUC")

    # Test set touched EXACTLY ONCE, using the with-composite model -- this
    # is the hypothesis under test in this addendum.
    best_model = search_comp.best_estimator_
    X_test = test[FEATURES_WITH_COMPOSITE]
    test_pred = best_model.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    coverage = round(100 * pool[COMPOSITE_FEATURE].notna().mean(), 1)
    print(f"  [{market_name}] {COMPOSITE_FEATURE} coverage (% non-null, full pool): {coverage}%")
    print(f"  [{market_name}] TEST (touched once, with-composite model): "
          f"edge>0.0 bets n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"market": market_name, "n_train": len(train), "n_test": len(test),
            "cv_auc_base": search_base.best_score_, "cv_auc_composite": search_comp.best_score_,
            "cv_auc_delta": delta, "coverage_pct": coverage, **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building real team game log, Elo, L10, bullpen fatigue, park factors (existing pipeline)...")
    games, box = build_games_and_box(con)
    n_games_raw, n_games_raw_unique = len(games), games["game_pk"].nunique()
    print(f"MANDATORY SANITY CHECK -- games (build_games_and_box): n={n_games_raw}  unique game_pk={n_games_raw_unique}")

    print("extending with Addendum 37 historical features (imported unmodified: build_games_v2) -- "
          "this is also where the real duplicate-game_pk bug is fixed via drop_duplicates...")
    games_v2 = build_games_v2(con, con2, games)
    n_v2, n_v2_unique = len(games_v2), games_v2["game_pk"].nunique()
    print(f"MANDATORY SANITY CHECK -- games_v2 (build_games_v2, post-dedup): n={n_v2}  unique game_pk={n_v2_unique}")
    assert n_v2 == n_v2_unique, "games_v2 has duplicate game_pk rows -- the exact Addendum 37 bug class, NOT fixed"
    print("  CONFIRMED: games_v2 has exactly one row per game_pk, no duplicates.")

    box_ids = build_box_ids(con)

    print("\nbuilding Addendum 45c team-level 'current form' composite from real boxscore "
          "batting/pitching production (point-in-time, trailing 10 team-games)...")
    panel = build_team_game_panel(con)
    print(f"  team-game panel: n={len(panel)} rows (team_id x game_pk), "
          f"{panel['team_id'].nunique()} teams, {panel['game_pk'].nunique()} unique games")
    panel = add_rolling_form(panel)
    panel = add_composite(panel)

    games_v3 = add_team_form_diff(games_v2, panel)
    n_v3, n_v3_unique = len(games_v3), games_v3["game_pk"].nunique()
    assert n_v3 == n_v3_unique, "games_v3 (post team-composite merge) has duplicate game_pk rows -- investigate"
    print(f"MANDATORY SANITY CHECK -- games_v3 (post team-composite merge): n={n_v3}  unique game_pk={n_v3_unique}")
    print("  CONFIRMED: games_v3 still has exactly one row per game_pk, no duplicates.")

    print(f"\nFEATURES_V2 baseline ({len(GAME_FEATURES_V2)} features, unmodified import): {GAME_FEATURES_V2}")
    print(f"FEATURES_V2 + composite ({len(FEATURES_WITH_COMPOSITE)} features): "
          f"adds {COMPOSITE_FEATURE!r} on top, nothing else")

    print("\n=== Addendum 45c: team current-form composite as a new feature, h2h and totals ===")
    results = []
    for market_key in ["h2h", "totals"]:
        print(f"\n--- {market_key} ---")
        pool_raw = build_game_pool(con, games_v3, market_key)
        n_before, n_games_before = len(pool_raw), pool_raw["game_pk"].nunique()
        print(f"  [{market_key}] pool rows before feature merge: n={n_before}  (unique game_pk={n_games_before})")

        pool = build_features_v2(pool_raw, games_v3, "game", box_ids, con2)
        n_after1 = len(pool)
        print(f"  [{market_key}] pool rows after build_features_v2 (unmodified import): n={n_after1}  "
              f"{'OK, no fan-out' if n_after1 == n_before else 'MISMATCH -- investigate'}")
        assert n_after1 == n_before, f"{market_key}: build_features_v2 changed row count"

        games_v3_dedup = games_v3.drop_duplicates(subset=["game_pk"])[["game_pk", COMPOSITE_FEATURE]]
        pool = pool.merge(games_v3_dedup, on="game_pk", how="left")
        n_after2 = len(pool)
        print(f"  [{market_key}] pool rows after merging {COMPOSITE_FEATURE}: n={n_after2}  "
              f"{'OK, no fan-out' if n_after2 == n_before else 'MISMATCH -- investigate'}")
        assert n_after2 == n_before, f"{market_key}: composite merge changed row count"

        r = evaluate_market(market_key, pool)
        results.append(r)

    con.close()
    con2.close()

    print("\n=== SUMMARY: Addendum 45c (team current-form composite) vs best prior result, both markets ===")
    for r in results:
        b = BASELINES[r["market"]]
        print(f"\n{r['market']}")
        print(f"  BEFORE [{b['desc']}]: n={b['n']}  ROI={b['roi']}  CI=[{b['lo']},{b['hi']}]  {b['verdict']}")
        print(f"  CV AUC (TRAIN only): without composite={r['cv_auc_base']:.4f}  "
              f"with composite={r['cv_auc_composite']:.4f}  delta={r['cv_auc_delta']:+.4f}")
        print(f"  AFTER  [Addendum 45c, with-composite model]: n(train/test)={r['n_train']}/{r['n_test']}  "
              f"composite coverage={r['coverage_pct']}%")
        print(f"                        test: n={r['n']}  ROI={r['roi']}  CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    passes = [r for r in results if r["verdict"] == "PASS"]
    print(f"\n{len(passes)} of {len(results)} markets pass the official gate rule with the team "
          f"current-form composite added. Reported as-is -- both markets are already documented as "
          f"severely sample-size-constrained after edge-filtering (test n in the 180-550 range); a "
          f"FAIL here is a valid, expected outcome given that constraint, not a shortfall in this "
          f"analysis, same discipline as every prior addendum on these two markets.")


if __name__ == "__main__":
    main()
