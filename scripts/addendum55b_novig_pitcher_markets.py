"""
Addendum 55b: recompute pitcher_outs (Addendum 51's best-known result) and
pitcher_strikeouts (Addendum 52's best-known results, both single-split and
walk-forward) with a DE-VIGGED market_prob instead of the raw single-side
implied probability every prior addendum in this project has used.

BACKGROUND: every edge calculation in this project's history has computed
edge = model_prob - implied_prob(odds), where implied_prob() (in
xgboost_individual_markets.py) is the raw, single-side implied probability --
it never removes the bookmaker's overround. Real `totals` odds in this
warehouse were measured this session at ~2.9% average overround. This is a
standard, well-established methodological correction (de-vig the market
probability before computing edge), not a new feature or a new model. Per
the task brief: no retraining/re-tuning/re-selection of either model --
reuse the EXACT SAME established pipeline for each market, the ONLY change
is substituting a de-vigged market_prob for the raw one before edge is
computed. Each market's official gate is touched EXACTLY ONCE.

THE NO-VIG FORMULA (two-sided prop, both real odds available for the same
player/game/line):
    fair_prob_side = implied_prob(this_side_odds)
                     / (implied_prob(over_odds) + implied_prob(under_odds))

BOTH-SIDES-ODDS AVAILABILITY -- investigated empirically, not assumed:
  - pitcher_outs (test_backfilled_markets.build_market_dataset): parse_cache()
    DOES parse both best_over_dec/best_under_dec per (event,player) row from
    the raw jsonl cache. But build_market_dataset()'s final column selection
    (["market_key","game_pk","commence_time","odds","win","side","tiebreak"])
    explicitly drops the counterpart side's odds after splitting each row
    into an "over" bet-row and an "under" bet-row. Recovered here by calling
    parse_cache() a second time (same unmodified import, not a different
    raw-jsonl re-parse) and rejoining onto the addendum51 pool by
    (game_pk, tiebreak==name_norm) -- verified below to be a lossless,
    non-fan-out join.
  - pitcher_strikeouts (multi_model_comparison.build_warehouse_pool): this
    function's final pd.concat([under, over]) never narrows columns, so
    best_over_odds/best_under_odds from the original client_closing_odds
    query survive on every row it returns already. No re-parsing or rejoin
    needed here -- verified below directly against pool.columns.

Reused UNMODIFIED (imported, not copied): build_pool, check_game_log_choice,
build_games_full, cv_select, BASE_FEATURES/BULLPEN_V2/STARTER_GAME_V2/
PARK_V2/FEATURES_V2, N_FOLDS, BASELINE_ADDENDUM30 (addendum51);
build_derived_features, CONTEXT_COLS, BASELINE (addendum52);
build_games_and_box, build_warehouse_pool, split_per_year
(multi_model_comparison); build_fuller_games (addendum48); attach_enrichment,
PARAM_GRID_WIDE, cv_select (addendum49); PARAM_GRID, FIXED_PARAMS
(tuned_xgboost_4_markets); add_bullpen_features/add_starter_game_features/
add_park_features, CACHE_DB, FEATURES_V2/BULLPEN_V2/STARTER_GAME_V2/
OPP_PITCHER_V2/PARK_V2 (addendum37); DB, norm_name, implied_prob, profit,
stat, gate (xgboost_individual_markets); parse_cache, decimal_to_american
(test_backfilled_markets).

The walk-forward loops themselves are re-inlined here (not called as a
black-box function) ONLY so a per-bet "what if we'd used raw market_prob
instead" column can be tracked for the mandatory sign-flip sanity check --
every model-fitting/CV-selection/feature/split/gate-evaluation call inside
each loop is the identical imported function used unmodified by the
original addendum. The official PASS/FAIL number comes from the de-vigged
edge column and is touched exactly once, same as the original.
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
    DB, norm_name, implied_prob, profit, stat, gate,
)
from test_backfilled_markets import parse_cache, decimal_to_american  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum49_h2h_walkforward import PARAM_GRID_WIDE, FIXED_PARAMS, attach_enrichment  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_warehouse_pool, split_per_year,
)
from addendum37_historical_enrichment import (  # noqa: E402
    BULLPEN_V2, STARTER_GAME_V2, OPP_PITCHER_V2, PARK_V2, FEATURES_V2 as FEATURES_V2_52,
    add_bullpen_features, add_starter_game_features, add_park_features,
)
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS as FIXED_PARAMS_52  # noqa: E402

import addendum51_pitcher_outs_full_treatment as A51  # noqa: E402
import addendum52_pitcher_strikeouts_fuller_gamelog as A52  # noqa: E402

RAW_BASELINES = {
    "pitcher_outs": {"n": 6, "roi": 13.78, "lo": -56.74, "hi": 84.31, "verdict": "FAIL",
                      "desc": "Addendum 51 (raw market_prob, walk-forward+enrichment)"},
    "pitcher_strikeouts_single": {"n": 304, "roi": -7.84, "verdict": "FAIL",
                                    "desc": "Addendum 52 single-split (raw market_prob)"},
    "pitcher_strikeouts_walkforward": {"n": 310, "roi": 4.74, "verdict": "FAIL",
                                        "desc": "Addendum 52 walk-forward (raw market_prob)"},
}


def cv_select_51(X_train, y_train):
    """Byte-identical to A51.cv_select -- reused unmodified; wrapped only so
    this module doesn't need `import addendum51...cv_select` shadowed below."""
    return A51.cv_select(X_train, y_train)


# ============================================================================
# PITCHER_OUTS (Addendum 51 pipeline, de-vigged market_prob)
# ============================================================================

def run_pitcher_outs(con, con2, box):
    print("\n" + "=" * 78)
    print("PITCHER_OUTS -- Addendum 51 pipeline, de-vigged market_prob")
    print("=" * 78)

    pool = A51.build_pool(con, box)
    games_v3 = A51.check_game_log_choice(con, con2, pool)
    games_full = A51.build_games_full(con, con2, games_v3)

    merge_cols = (["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                    "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]
                  + A51.BULLPEN_V2 + A51.STARTER_GAME_V2 + A51.PARK_V2)
    games_full_dedup = games_full.drop_duplicates(subset=["game_pk"])[merge_cols]
    n_before = len(pool)
    pool_full = pool.merge(games_full_dedup, on="game_pk", how="inner")
    print(f"\n  feature merge (pool x enriched games_v3): before={n_before:,}  after={len(pool_full):,}  "
          f"{'OK -- no fan-out' if len(pool_full) == n_before else '*** ROW COUNT MISMATCH, STOP ***'}")
    assert len(pool_full) == n_before, "feature merge changed row count -- STOP"

    pool_full["game_date"] = pd.to_datetime(pool_full["game_date"])
    pool_full["side_code"] = pool_full["side"].astype("category").cat.codes
    pool_full["elo_diff"] = (pool_full["home_elo"] + 24) - pool_full["away_elo"]
    pool_full["l10_diff"] = pool_full["home_l10"] - pool_full["away_l10"]
    pool_full["fatigue_diff"] = pool_full["home_fatigue"] - pool_full["away_fatigue"]
    pool_full["market_prob_raw"] = implied_prob(pool_full["odds"])

    # ---- recover both-sides odds: re-call parse_cache (unmodified import) ----
    po_cache = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"
    both = parse_cache(po_cache, "pitcher_outs")
    both["game_pk"] = pd.to_numeric(both["game_pk"], errors="coerce")
    both["name_norm"] = both["player_name"].map(norm_name)
    both["both_over_odds"] = both["best_over_dec"].map(decimal_to_american)
    both["both_under_odds"] = both["best_under_dec"].map(decimal_to_american)
    both = both.dropna(subset=["game_pk"])
    n_raw_both = len(both)
    dupe_ct = int(both.duplicated(subset=["game_pk", "name_norm"]).sum())
    print(f"\n  both-sides-odds recovery: parse_cache('{po_cache.name}') re-parsed -> {n_raw_both:,} "
          f"(event,player) rows; {dupe_ct} duplicate (game_pk,name_norm) rows "
          f"{'found, deduped keep-first' if dupe_ct else '(none, clean)'}")
    both = both.sort_values(["game_pk", "name_norm"]).drop_duplicates(subset=["game_pk", "name_norm"], keep="first")
    both = both[["game_pk", "name_norm", "both_over_odds", "both_under_odds"]]

    n_before_join = len(pool_full)
    pool_full = pool_full.merge(both, left_on=["game_pk", "tiebreak"], right_on=["game_pk", "name_norm"], how="left")
    print(f"  join pool_full x both-sides-odds on (game_pk, tiebreak=name_norm): "
          f"before={n_before_join:,}  after={len(pool_full):,}  "
          f"{'OK -- no fan-out' if len(pool_full) == n_before_join else '*** ROW COUNT MISMATCH, STOP ***'}")
    assert len(pool_full) == n_before_join, "both-sides join changed row count -- STOP"

    missing = pool_full["both_over_odds"].isna() | pool_full["both_under_odds"].isna()
    print(f"  rows missing one/both sides after join: {missing.sum():,} of {len(pool_full):,} "
          f"({100 * missing.mean():.1f}%) -- de-vig is impossible for these (no counterpart price "
          f"was ever quoted), dropped from this de-vigged analysis only")
    pool_full = pool_full[~missing].copy()

    over_prob = pd.Series(implied_prob(pool_full["both_over_odds"]), index=pool_full.index)
    under_prob = pd.Series(implied_prob(pool_full["both_under_odds"]), index=pool_full.index)
    vig_sum = over_prob + under_prob
    pool_full["market_prob"] = implied_prob(pool_full["odds"]) / vig_sum
    vig_pct = (vig_sum - 1.0) * 100
    print(f"\n  *** MEASURED VIG, pitcher_outs pool (both-sides rows, n={len(pool_full):,}) ***")
    print(f"      mean={vig_pct.mean():.2f}%  median={vig_pct.median():.2f}%  "
          f"min={vig_pct.min():.2f}%  max={vig_pct.max():.2f}%  std={vig_pct.std():.2f}%")

    pool_full = pool_full.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"\n  pool ready for walk-forward: n={len(pool_full):,}  "
          f"date range {pool_full.game_date.min().date()} -> {pool_full.game_date.max().date()}")

    fold_id = pd.qcut(np.arange(len(pool_full)), A51.N_FOLDS, labels=False)
    pool_full["fold"] = fold_id

    all_bets = []
    print(f"\n  === walk-forward folds (expanding window, {A51.N_FOLDS} folds, CV-AUC selection "
          f"on TRAIN only, PARAM_GRID_WIDE from Addendum 49, DE-VIGGED market_prob) ===")
    for f in range(1, A51.N_FOLDS):
        train_df = pool_full[pool_full["fold"] < f]
        test_df = pool_full[pool_full["fold"] == f]
        print(f"\n    fold {f}: train n={len(train_df):,}  test n={len(test_df):,}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"    fold {f}: INSUFFICIENT DATA, skipped")
            continue

        X_train, y_train = train_df[A51.FEATURES_V2], train_df["win"].astype(int)
        model, cv_score, params = cv_select_51(X_train, y_train)
        print(f"    fold {f}: CV_AUC={cv_score:.4f}  params={params}")

        test_pred = model.predict_proba(test_df[A51.FEATURES_V2])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        t["edge_raw"] = t["model_prob"] - t["market_prob_raw"]
        all_bets.append(t[["game_date", "odds", "win", "edge", "edge_raw"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n  pooled walk-forward held-out bets across folds 1-{A51.N_FOLDS - 1}: n={len(pooled):,}")
    print("  (official gate touched EXACTLY ONCE below, on the de-vigged edge column)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\n  DE-VIGGED edge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"    n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    flips = int(((pooled.edge > 0.0) != (pooled.edge_raw > 0.0)).sum())
    print(f"\n  sign-flip sanity check (same model_prob predictions, raw vs de-vigged market_prob "
          f"used only in the edge subtraction): {flips:,} of {len(pooled):,} pooled bets' edge sign changed")

    b = RAW_BASELINES["pitcher_outs"]
    print(f"\n  HONEST COMPARISON:")
    print(f"    {b['desc']}: n={b['n']}  ROI={b['roi']}%  CI=[{b['lo']},{b['hi']}]  {b['verdict']}")
    print(f"    Addendum 55b (SAME pipeline, DE-VIGGED market_prob): "
          f"n={s['n']}  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"market": "pitcher_outs", "n": s["n"], "roi": s["roi"], "lo": s["lo"], "hi": s["hi"],
            "verdict": verdict, "flips": flips, "pooled_n": len(pooled),
            "vig_mean_pct": round(vig_pct.mean(), 2)}


# ============================================================================
# PITCHER_STRIKEOUTS (Addendum 52 pipeline, de-vigged market_prob)
# ============================================================================

def eval_edge(pool, edge_col):
    sub = pool[(pool.odds < 0) & (pool[edge_col] > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    return s, verdict


def single_split_eval_devig(pool):
    """Same discipline/config as A52.single_split_eval, reused via the exact
    same imports (PARAM_GRID/FIXED_PARAMS/CV setup) -- inlined only to also
    track the raw-market_prob edge for the sign-flip diagnostic. The official
    gate call at the bottom uses the de-vigged `edge` column exactly once,
    identical in every other respect to calling A52.single_split_eval(pool)
    directly on a pool whose market_prob column has been overwritten."""
    if len(pool) < 500:
        print(f"  n={len(pool)} INSUFFICIENT DATA (<500)")
        return None
    train, test = split_per_year(pool)
    if len(train) < 50 or len(test) < 50:
        print("  INSUFFICIENT DATA after split")
        return None

    X_train, y_train = train[FEATURES_V2_52], train["win"].astype(int)
    X_test = test[FEATURES_V2_52]
    print(f"  n(train/test)={len(train)}/{len(test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS_52), PARAM_GRID,
                           scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True)
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    test_pred = search.best_estimator_.predict_proba(X_test)[:, 1]
    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    t["edge_raw"] = t["model_prob"] - t["market_prob_raw"]

    s, verdict = eval_edge(t, "edge")
    print(f"  best CV AUC={search.best_score_:.4f}  params={search.best_params_}")
    print(f"  DE-VIGGED edge>0.0 test: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")
    flips = int(((t.edge > 0.0) != (t.edge_raw > 0.0)).sum())
    print(f"  sign-flip check (single-split): {flips:,} of {len(t):,} test bets' edge sign changed "
          f"(raw vs de-vigged, same fitted model)")
    return {"n_train": len(train), "n_test": len(test), "cv_auc": search.best_score_,
            "best_params": search.best_params_, **s, "verdict": verdict, "flips": flips}


def walk_forward_eval_devig(pool, n_folds=5):
    """Same expanding-window pattern/config as A52.walk_forward_eval,
    inlined only to track the raw-market_prob edge for the sign-flip
    diagnostic alongside the de-vigged one. cv_select is A49's imported,
    unmodified function -- identical CV-AUC selection either way."""
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), n_folds, labels=False)
    pool = pool.copy()
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n  === walk-forward folds (expanding window, {n_folds} folds, DE-VIGGED market_prob) ===")
    for f in range(1, n_folds):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n    fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"    fold {f}: INSUFFICIENT DATA, skipped")
            continue

        X_train, y_train = train_df[FEATURES_V2_52], train_df["win"].astype(int)
        model, cv_score, params = A52.cv_select(X_train, y_train)
        print(f"    fold {f}: CV_AUC={cv_score:.4f}  params={params}")

        test_pred = model.predict_proba(test_df[FEATURES_V2_52])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        t["edge_raw"] = t["model_prob"] - t["market_prob_raw"]
        all_bets.append(t[["game_date", "odds", "win", "edge", "edge_raw"]])

    if not all_bets:
        print("  walk-forward: no fold produced usable predictions")
        return None
    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n  pooled walk-forward held-out bets across folds 1-{n_folds - 1}: n={len(pooled)}")
    print("  (official gate touched EXACTLY ONCE below, on the de-vigged edge column)")

    s, verdict = eval_edge(pooled, "edge")
    print(f"  DE-VIGGED edge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    flips = int(((pooled.edge > 0.0) != (pooled.edge_raw > 0.0)).sum())
    print(f"  sign-flip check (walk-forward): {flips:,} of {len(pooled):,} pooled bets' edge sign "
          f"changed (raw vs de-vigged)")
    return {**s, "verdict": verdict, "flips": flips, "pooled_n": len(pooled)}


def run_pitcher_strikeouts(con, con2):
    print("\n" + "=" * 78)
    print("PITCHER_STRIKEOUTS -- Addendum 52 pipeline, de-vigged market_prob")
    print("=" * 78)

    print("\nbuilding OLD (lineups-dependent) team game log + box (row-count baseline, unmodified reuse)...")
    old_games, box = build_games_and_box(con)
    old_games = old_games.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    print("building FULLER game log (Addendum 48, unmodified import)...")
    fuller_games, id_to_name = build_fuller_games(con, con2)
    assert len(fuller_games) == fuller_games.game_pk.nunique(), "fuller game log has duplicate game_pk -- STOP"

    print(f"\nbuilding {A52.MARKET_KEY} pool (real closing odds x real boxscore strikeouts, "
          f"multi_model_comparison.build_warehouse_pool, unmodified)...")
    pool_raw = build_warehouse_pool(con, box, A52.MARKET_KEY, A52.STAT_COL, A52.KIND)
    print(f"  pool_raw (odds x boxscore, pre-game-log-merge): n={len(pool_raw)}")

    pool_old = pool_raw.merge(old_games[A52.CONTEXT_COLS], on="game_pk", how="inner")
    pool_new = pool_raw.merge(fuller_games[A52.CONTEXT_COLS], on="game_pk", how="inner")
    print(f"  ROW-COUNT RECOVERY: OLD game log join n={len(pool_old)}  FULLER game log join n={len(pool_new)}")

    # ---- both-sides-odds check: verify they already ride along on pool_new ----
    has_both_cols = {"best_over_odds", "best_under_odds"} <= set(pool_new.columns)
    print(f"\n  both-sides-odds check: best_over_odds/best_under_odds present on "
          f"build_warehouse_pool's own output columns = {has_both_cols} (no re-parse/rejoin needed "
          f"for this market -- multi_model_comparison.build_warehouse_pool's final concat never "
          f"drops the counterpart side's odds column)")
    assert has_both_cols, "expected best_over_odds/best_under_odds on pool_new -- STOP, investigate"

    pool = A52.build_derived_features(pool_new)   # sets market_prob = implied_prob(odds) [raw], for now
    pool["market_prob_raw"] = pool["market_prob"]
    n_before_enrich = len(pool)
    pool = attach_enrichment(con, con2, pool, id_to_name)
    print(f"\n  row-count sanity check (enrichment merges): before={n_before_enrich}  after={len(pool)}  "
          f"{'OK' if len(pool) == n_before_enrich else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before_enrich, "enrichment merge changed row count -- aborting"

    for c in OPP_PITCHER_V2:
        pool[c] = np.nan

    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"])
    print(f"  pool after dropping rows missing win/odds: n={len(pool)}")

    has_both = pool["best_over_odds"].notna() & pool["best_under_odds"].notna()
    print(f"  rows with both sides' odds present: {has_both.sum():,} of {len(pool):,} "
          f"({100 * has_both.mean():.1f}%)")
    n_before_devig_filter = len(pool)
    pool = pool[has_both].copy()
    print(f"  pool restricted to both-sides rows for the de-vig analysis: {n_before_devig_filter:,} -> "
          f"{len(pool):,}")

    over_prob = pd.Series(implied_prob(pool["best_over_odds"]), index=pool.index)
    under_prob = pd.Series(implied_prob(pool["best_under_odds"]), index=pool.index)
    vig_sum = over_prob + under_prob
    pool["market_prob"] = implied_prob(pool["odds"]) / vig_sum   # OVERWRITE -- de-vigged, feeds model + edge
    vig_pct = (vig_sum - 1.0) * 100
    print(f"\n  *** MEASURED VIG, pitcher_strikeouts pool (both-sides rows, n={len(pool):,}) ***")
    print(f"      mean={vig_pct.mean():.2f}%  median={vig_pct.median():.2f}%  "
          f"min={vig_pct.min():.2f}%  max={vig_pct.max():.2f}%  std={vig_pct.std():.2f}%")

    print(f"\nFEATURES_V2 ({len(FEATURES_V2_52)} total): {FEATURES_V2_52}")

    print(f"\n{'=' * 78}\n{A52.MARKET_KEY} -- single per-year 75/25 split, DE-VIGGED market_prob\n{'=' * 78}")
    single_result = single_split_eval_devig(pool)

    print(f"\n{'=' * 78}\n{A52.MARKET_KEY} -- walk-forward validation, DE-VIGGED market_prob\n{'=' * 78}")
    wf_result = walk_forward_eval_devig(pool, n_folds=5)

    print(f"\n{'=' * 78}\nHONEST COMPARISON -- Addendum 55b vs Addendum 52 (raw market_prob)\n{'=' * 78}")
    b1 = RAW_BASELINES["pitcher_strikeouts_single"]
    b2 = RAW_BASELINES["pitcher_strikeouts_walkforward"]
    print(f"  {b1['desc']}: n={b1['n']}  ROI={b1['roi']}%  {b1['verdict']}")
    if single_result:
        print(f"  Addendum 55b single-split (de-vigged): n={single_result['n']}  ROI={single_result['roi']}%  "
              f"CI=[{single_result['lo']},{single_result['hi']}]  {single_result['verdict']}")
    print(f"  {b2['desc']}: n={b2['n']}  ROI={b2['roi']}%  {b2['verdict']}")
    if wf_result:
        print(f"  Addendum 55b walk-forward (de-vigged): n={wf_result['n']}  ROI={wf_result['roi']}%  "
              f"CI=[{wf_result['lo']},{wf_result['hi']}]  {wf_result['verdict']}")

    return {"market": "pitcher_strikeouts", "single": single_result, "walkforward": wf_result,
            "vig_mean_pct": round(vig_pct.mean(), 2)}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=" * 78)
    print("Addendum 55b: de-vigged market_prob, pitcher_outs + pitcher_strikeouts")
    print("Ground rule: no retraining/re-tuning/re-selection -- reuse each market's exact")
    print("established pipeline (Addenda 51/52) unmodified except market_prob. Gate touched")
    print("EXACTLY ONCE per market. A FAIL here is a valid, honestly-reportable outcome.")
    print("=" * 78)

    box = con.execute("""
        SELECT game_pk, player_name, outs AS pitcher_outs_stat, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    result_po = run_pitcher_outs(con, con2, box)
    result_ks = run_pitcher_strikeouts(con, con2)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY -- Addendum 55b")
    print("=" * 78)
    print(f"  pitcher_outs        : vig(mean)={result_po['vig_mean_pct']}%  "
          f"n={result_po['n']}  ROI={result_po['roi']}%  CI=[{result_po['lo']},{result_po['hi']}]  "
          f"{result_po['verdict']}  (edge-sign flips: {result_po['flips']} of {result_po['pooled_n']})")
    if result_ks["single"]:
        print(f"  pitcher_strikeouts (single-split)     : vig(mean)={result_ks['vig_mean_pct']}%  "
              f"n={result_ks['single']['n']}  ROI={result_ks['single']['roi']}%  "
              f"CI=[{result_ks['single']['lo']},{result_ks['single']['hi']}]  "
              f"{result_ks['single']['verdict']}  (edge-sign flips: {result_ks['single']['flips']})")
    if result_ks["walkforward"]:
        print(f"  pitcher_strikeouts (walk-forward)     : vig(mean)={result_ks['vig_mean_pct']}%  "
              f"n={result_ks['walkforward']['n']}  ROI={result_ks['walkforward']['roi']}%  "
              f"CI=[{result_ks['walkforward']['lo']},{result_ks['walkforward']['hi']}]  "
              f"{result_ks['walkforward']['verdict']}  (edge-sign flips: {result_ks['walkforward']['flips']})")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
