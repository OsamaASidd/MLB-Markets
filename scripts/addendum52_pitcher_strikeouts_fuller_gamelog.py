"""
Addendum 52: rebuild pitcher_strikeouts's pool on Addendum 48's fuller game
log (client_games x cache_mlb_historical_outcomes, no lineups/boxscore
player-matching dependency), keeping everything else (features, model,
split, evaluation) identical to the established Addendum 37 pattern.

WHY: pitcher_strikeouts has real 2023-2025 warehouse odds
(client_closing_odds market_key='pitcher_strikeouts') joined to real
boxscore strikeout counts -- that part of the pool never depended on
lineups. But every prior addendum (27, 32, 34, 36, 37, 40, 43a, 45b) then
attached team-level context features (Elo, L10, bullpen fatigue, park
factors) via an INNER merge onto build_team_game_log()'s output -- the
SAME lineups-dependent, artificially-shrunk game log that was capping
h2h/totals until Addendum 48 rebuilt it from client_games +
cache_mlb_historical_outcomes directly (7,028 usable games instead of the
old ~3,845-game ceiling). That inner merge is exactly where
pitcher_strikeouts's pool was being starved by the same bug, without
anyone ever swapping in the fuller log for this specific market. This
addendum does that swap and nothing else.

Best prior result (Addendum 43a, Poisson regression): n=823, ROI=-5.28%,
CI=[-11.38,0.82], FAIL -- close on the CI upper bound but still FAIL.

Method, unchanged from Addendum 37/48/49:
  - Pool: multi_model_comparison.build_warehouse_pool (real odds x real
    boxscore strikeouts), reused unmodified -- it already doesn't depend on
    any team game log (odds come from client_closing_odds/client_games,
    the batter/pitcher side comes from the raw boxscore table). The
    dependency on the old game log only entered downstream, when team-level
    context features were attached.
  - Team-level context (Elo, L10, fatigue, park factors): attached via
    Addendum 48's build_fuller_games() output instead of
    xgboost_individual_markets.build_team_game_log() -- the one substantive
    change this addendum makes.
  - Enrichment (bullpen quality, starter quality, park dimensions):
    Addendum 49's attach_enrichment(), reused unmodified -- it already
    operates directly on a pool carrying home_team_id/away_team_id/
    game_date/event_id/game_pk, which is exactly the shape produced here.
    OPP_PITCHER_V2 (opposing-starter-specific columns) stays NaN, same as
    Addendum 37's kind='pitcher' branch (pitcher_strikeouts uses the
    game-level starter diff features, not a batter's specific opposing
    starter).
  - Model/tuning: XGBoost, PARAM_GRID/FIXED_PARAMS from
    tuned_xgboost_4_markets.py, CV-AUC (5-fold StratifiedKFold via
    GridSearchCV) selection on TRAIN only -- ROI/gate never touched until
    the single final test-split evaluation.
  - Split: per_year 75/25 (multi_model_comparison.split_per_year), same as
    every prior pitcher_strikeouts addendum.
  - Single pre-specified edge>0.0 cut, official gate() rule
    (xgboost_individual_markets.py, verified against the client's own
    betgenius/harness/lib/metrics.ts).

Ground rules: pool row counts printed before/after every merge; the fuller
game log's one-row-per-game_pk property is explicitly re-verified here
(not assumed from Addendum 48's own internal assert), and the OLD game
log's known duplicate-game_pk characteristic (Addendum 37: 4,547 rows,
3,845 unique) is deduped before use so the old-vs-new pool-size comparison
below is apples-to-apples.
"""
import os
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
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_warehouse_pool, split_per_year,
)
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, BULLPEN_V2, STARTER_GAME_V2, OPP_PITCHER_V2, PARK_V2, FEATURES_V2,
)
from addendum49_h2h_walkforward import attach_enrichment  # noqa: E402
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402
from addendum49_h2h_walkforward import PARAM_GRID_WIDE, cv_select  # noqa: E402

MARKET_KEY = "pitcher_strikeouts"
STAT_COL = "strikeouts"
KIND = "pitcher"

CONTEXT_COLS = ["game_pk", "game_date", "home_team_id", "away_team_id",
                "home_elo", "away_elo", "home_l10", "away_l10",
                "home_fatigue", "away_fatigue",
                "runs_factor", "hr_factor", "k_factor", "hits_factor"]

BASELINE = {"desc": "Addendum 43a Poisson regression (best prior, pre-fuller-gamelog)",
            "n": 823, "roi": -5.28, "lo": -11.38, "hi": 0.82, "verdict": "FAIL"}


def build_derived_features(pool):
    pool = pool.copy()
    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    return pool


def single_split_eval(pool):
    """Same tuning/eval discipline as Addendum 37/48: CV-AUC selection on
    TRAIN only, single edge>0.0 cut, official gate rule, evaluated once."""
    if len(pool) < 500:
        print(f"  n={len(pool)} INSUFFICIENT DATA (<500)")
        return None
    train, test = split_per_year(pool)
    if len(train) < 50 or len(test) < 50:
        print("  INSUFFICIENT DATA after split")
        return None

    X_train, y_train = train[FEATURES_V2], train["win"].astype(int)
    X_test = test[FEATURES_V2]
    print(f"  n(train/test)={len(train)}/{len(test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID,
                           scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True)
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    test_pred = search.best_estimator_.predict_proba(X_test)[:, 1]
    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  best CV AUC={search.best_score_:.4f}  params={search.best_params_}")
    print(f"  edge>0.0 test: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"n_train": len(train), "n_test": len(test), "cv_auc": search.best_score_,
            "best_params": search.best_params_, **s, "verdict": verdict}


def walk_forward_eval(pool, n_folds=5):
    """Same expanding-window walk-forward pattern as Addendum 47/49: folds
    built on the full chronologically-sorted pool, hyperparameters selected
    via CV-AUC on each fold's TRAINING data only, held-out predictions from
    every fold pooled, official gate touched EXACTLY ONCE on the pooled
    set at the very end. No fold is inspected individually before pooling."""
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), n_folds, labels=False)
    pool = pool.copy()
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n  === walk-forward folds (expanding window, {n_folds} folds) ===")
    for f in range(1, n_folds):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n    fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"    fold {f}: INSUFFICIENT DATA, skipped")
            continue

        X_train, y_train = train_df[FEATURES_V2], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"    fold {f}: CV_AUC={cv_score:.4f}  params={params}")

        test_pred = model.predict_proba(test_df[FEATURES_V2])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    if not all_bets:
        print("  walk-forward: no fold produced usable predictions")
        return None
    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n  pooled walk-forward held-out bets across folds 1-{n_folds - 1}: n={len(pooled)}")
    print("  (official gate touched EXACTLY ONCE below, on the full pooled set)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  edge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {**s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=" * 78)
    print(f"Addendum 52: {MARKET_KEY} rebuilt on Addendum 48's fuller game log")
    print("=" * 78)

    print("\nbuilding OLD (lineups-dependent) team game log + box, for the row-count baseline...")
    old_games, box = build_games_and_box(con)
    n_old_raw = len(old_games)
    old_games = old_games.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)
    print(f"  old game log: {n_old_raw} rows -> {len(old_games)} after deduping game_pk "
          f"(Addendum 37's documented pre-existing dupe-game_pk bug, deduped for a fair comparison)")
    assert len(old_games) == old_games.game_pk.nunique(), "old game log still has duplicate game_pk after dedup -- STOP"

    print("\nbuilding FULLER game log (Addendum 48, unmodified import)...")
    fuller_games, id_to_name = build_fuller_games(con, con2)
    print(f"  fuller game log: n={len(fuller_games)} rows, unique game_pk={fuller_games.game_pk.nunique()}")
    assert len(fuller_games) == fuller_games.game_pk.nunique(), \
        "fuller game log has duplicate game_pk -- STOP (re-verifying Addendum 48's own claim, not assuming it)"

    print(f"\nbuilding {MARKET_KEY} pool (real closing odds x real boxscore strikeouts, "
          f"independent of any team game log)...")
    pool_raw = build_warehouse_pool(con, box, MARKET_KEY, STAT_COL, KIND)
    print(f"  pool_raw (odds x boxscore, pre-game-log-merge): n={len(pool_raw)}")

    pool_old = pool_raw.merge(old_games[CONTEXT_COLS], on="game_pk", how="inner")
    pool_new = pool_raw.merge(fuller_games[CONTEXT_COLS], on="game_pk", how="inner")
    print(f"\n  ROW-COUNT RECOVERY for {MARKET_KEY}'s pool:")
    print(f"    OLD game log join:    n={len(pool_old)}  (unique game_pk={pool_old.game_pk.nunique()})")
    print(f"    FULLER game log join: n={len(pool_new)}  (unique game_pk={pool_new.game_pk.nunique()})  "
          f"({len(pool_new) / len(pool_old):.2f}x)" if len(pool_old) else "")

    pool = build_derived_features(pool_new)
    n_before_enrich = len(pool)
    pool = attach_enrichment(con, con2, pool, id_to_name)
    print(f"\n  row-count sanity check (Addendum 37/49 enrichment merges): before={n_before_enrich}  "
          f"after={len(pool)}  {'OK' if len(pool) == n_before_enrich else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before_enrich, "enrichment merge changed row count -- aborting"

    # kind='pitcher': no batter-specific opposing-starter row, matching
    # Addendum 37's build_features_v2() else-branch exactly.
    for c in OPP_PITCHER_V2:
        pool[c] = np.nan

    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"])
    print(f"  pool after dropping rows missing win/odds: n={len(pool)}")

    print(f"\nFEATURES_V2 ({len(FEATURES_V2)} total): {FEATURES_V2}")

    print(f"\n{'=' * 78}\n{MARKET_KEY} -- single per-year 75/25 split (Addendum 37 pattern)\n{'=' * 78}")
    single_result = single_split_eval(pool)

    print(f"\n{'=' * 78}\nSUMMARY -- Addendum 52 vs best prior result\n{'=' * 78}")
    print(f"  BEFORE [{BASELINE['desc']}]: n={BASELINE['n']}  ROI={BASELINE['roi']}%  "
          f"CI=[{BASELINE['lo']},{BASELINE['hi']}]  {BASELINE['verdict']}")
    if single_result:
        print(f"  AFTER  [Addendum 52, fuller game log]: n(train/test)={single_result['n_train']}/"
              f"{single_result['n_test']}  CV_AUC={single_result['cv_auc']:.4f}")
        print(f"                        test: n={single_result['n']}  ROI={single_result['roi']}  "
              f"CI=[{single_result['lo']},{single_result['hi']}]  {single_result['verdict']}")
    else:
        print("  AFTER  [Addendum 52, fuller game log]: could not be evaluated (insufficient data)")

    # Judgment call, per task instructions: only run walk-forward validation
    # if the single-split result looks close (CI upper bound near/above 0,
    # or an outright but narrow PASS/near-PASS) -- same trigger condition
    # that led to Addendum 49 being run on top of Addendum 48's h2h result.
    run_walkforward = False
    if single_result and single_result["hi"] is not None:
        if single_result["verdict"] == "PASS" or single_result["hi"] > -5.0:
            run_walkforward = True

    if run_walkforward:
        print(f"\n{'=' * 78}\n{MARKET_KEY} -- walk-forward validation (Addendum 47/49 pattern), "
              f"since the single-split result looks close enough to be worth it\n{'=' * 78}")
        wf_result = walk_forward_eval(pool, n_folds=5)
        print(f"\n{'=' * 78}\nFINAL SUMMARY\n{'=' * 78}")
        print(f"  single-split : n={single_result['n']}  ROI={single_result['roi']}  "
              f"CI=[{single_result['lo']},{single_result['hi']}]  {single_result['verdict']}")
        if wf_result:
            print(f"  walk-forward : n={wf_result['n']}  ROI={wf_result['roi']}  "
                  f"CI=[{wf_result['lo']},{wf_result['hi']}]  {wf_result['verdict']}")
    else:
        print(f"\nSingle-split result not close enough to the gate to justify walk-forward "
              f"validation (see judgment-call note in this script) -- reporting the single-split "
              f"result as final, per the task's own instruction that a FAIL here is valid and "
              f"honestly-reportable.")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
