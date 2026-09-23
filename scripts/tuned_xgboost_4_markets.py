"""
Addendum 34: disciplined hyperparameter tuning on the 4 markets where the
Addendum 33 multi-model sweep produced an isolated single-model pass
(batter_home_runs, pitcher_strikeouts, pitcher_outs, batter_strikeouts).

Pre-registered BEFORE running, to keep this from becoming the exact
threshold/model search HANDOFF.md's guardrails warn against:

  - ONE model family across all 4 markets, chosen once and not per-market:
    XGBoost, the project's already-established primary/baseline model
    throughout every prior addendum. Picking a different model per market
    based on which one happened to pass in Addendum 33 would itself be a
    form of cherry-picking -- avoided here on purpose.
  - ONE pre-specified hyperparameter grid (below), identical for all 4
    markets. Not widened or re-run if the first pass doesn't look good.
  - Tuning selection metric is CV-averaged ROC AUC on the TRAINING split
    ONLY (5-fold StratifiedKFold), via GridSearchCV. ROI/gate/edge is never
    computed on training data and never used to pick hyperparameters --
    only AUC, a metric decoupled from the pass/fail decision.
  - The tuned model is refit once on the full training split and scored
    ONCE on the untouched held-out test split, at the same single
    pre-specified edge>0.0 cut and same official gate rule used throughout
    this project. No re-splitting, no re-tuning based on the test result.
  - Every market's result is reported below, pass or fail.

Reuses the same pool/feature-building code already published in
multi_model_comparison.py (Addendum 32/33) -- does not modify it.
"""
import pathlib
import sys

import os

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, norm_name  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    FEATURES, build_games_and_box, build_warehouse_pool, build_features,
    split_per_year, split_chronological,
)
from test_backfilled_markets import build_market_dataset  # noqa: E402

# Pre-specified grid -- modest on purpose (this is a disciplined tuning
# pass, not a new search-for-a-winner). Fixed params below match the
# project's established XGBoost baseline; only these 4 are swept.
PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.02, 0.03, 0.05],
    "n_estimators": [200, 300, 500],
    "min_child_weight": [10, 20, 30],
}
FIXED_PARAMS = dict(
    subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
    eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
)


def tune_and_evaluate(market_name, pool, games, split_mode):
    pool = build_features(pool, games)
    if len(pool) < 500:
        print(f"  {market_name}: n={len(pool)} INSUFFICIENT DATA (<500)")
        return
    train, test = (split_per_year(pool) if split_mode == "per_year" else split_chronological(pool))
    if len(train) < 50 or len(test) < 50:
        print(f"  {market_name}: INSUFFICIENT DATA after split")
        return

    X_train, y_train = train[FEATURES], train["win"].astype(int)
    X_test, y_test = test[FEATURES], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python (_posixsubprocess missing) -- threading backend instead, which
    # still parallelizes real work since XGBoost's fit releases the GIL.
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # same single pre-specified cut as Addenda 32/33
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    print(f"  {market_name}")
    print(f"    best CV AUC={search.best_score_:.3f}  params={search.best_params_}")
    print(f"    test: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")


def main():
    con = duckdb.connect(DB, read_only=True)
    print("building real team game log, Elo, L10, bullpen fatigue, park factors...")
    games, box = build_games_and_box(con)

    print("\n=== Addendum 34: tuned XGBoost, single pre-specified grid, CV-AUC selection only ===")

    for market_key, stat_col, kind in [
        ("batter_home_runs", "home_runs", "batter"),
        ("pitcher_strikeouts", "strikeouts", "pitcher"),
    ]:
        pool = build_warehouse_pool(con, box, market_key, stat_col, kind)
        tune_and_evaluate(market_key, pool, games, "per_year")

    backfill_specs = [
        ("pitcher_outs", ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl", "pitcher_outs_stat",
         ["Pitcher"], True, "outs AS pitcher_outs_stat"),
        ("batter_strikeouts", ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl", "batter_strikeouts",
         ["Catcher", "Hitter", "Infielder", "Outfielder"], False, "batter_strikeouts"),
    ]
    for market_name, cache_path, sc, position_filter, starter_only, select_col in backfill_specs:
        if not cache_path.exists():
            print(f"  {market_name}: cache file missing ({cache_path.name}), skipping")
            continue
        box_bk = con.execute(f"""
            SELECT game_pk, player_name, {select_col}, is_starter, position_type FROM boxscore
        """).fetchdf()
        box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
        pool = build_market_dataset(cache_path, market_name, sc, box_bk, position_filter, starter_only)
        tune_and_evaluate(market_name, pool, games, "chronological")

    con.close()


if __name__ == "__main__":
    main()
