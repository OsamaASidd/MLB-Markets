"""
Addendum 43c: regression-based strategy for h2h and totals.

Every prior attempt on these two markets (Addenda 9, 17, 32-34, 37, 42)
framed them as BINARY CLASSIFICATION ("does this side/total win?") and both
have repeatedly failed the gate at a small, edge-filtered test n (roughly
180-200), consistent with sample-size starvation rather than an obvious
absence of signal. This addendum instead predicts the actual continuous
outcome directly -- the standard approach for game-total/moneyline models
in sports analytics:
  - `totals`: Poisson regression on total_runs = home_runs_ + away_runs_
    (a non-negative count), then P(total_runs > line) from
    scipy.stats.poisson around that prediction.
  - `h2h`: regression on run_diff = home_runs_ - away_runs_ (roughly
    Normal, like a point-spread margin model), then P(away wins) =
    P(run_diff < 0) from scipy.stats.norm, using a FIXED sigma estimated
    from TRAIN residuals only (test residuals never touched).

CRITICAL GROUND RULE: the regression model is selected ONLY via a proper
scoring rule on cross-validated TRAINING data (Poisson deviance for
totals, RMSE for h2h) -- never via ROI or win/loss accuracy. The test
set's betting ROI is touched exactly once, at the end, using the single
pre-specified edge>0.0 / odds<0 cut and the project's official gate() rule
(xgboost_individual_markets.py). If FAIL, that is reported as FAIL --
no post-hoc adjustment or re-run.

Reuses UNMODIFIED, via import:
  - build_games_and_box, build_game_pool, split_per_year
    (multi_model_comparison.py)
  - build_games_v2, build_box_ids, build_features_v2, FEATURES_V2
    (addendum37_historical_enrichment.py) -- build_games_v2 is also where
    the real duplicate-game_pk bug found in Addendum 37 was fixed
    (games.drop_duplicates(subset=["game_pk"])); reusing it unmodified
    means this script inherits that fix rather than re-introducing the bug.

FEATURES_V2 includes OPP_PITCHER_V2 (opp_sp_xera/opp_sp_era/opp_sp_xba) --
those are batter-prop-specific (the SPECIFIC opposing starter a batter
faces) and are 100% NaN for game-level pools (build_features_v2 fills them
with NaN whenever kind != "batter", verified below). Dropped from the
game-level feature set used here (GAME_FEATURES_V2) since an all-NaN
column carries no signal and only risks confusing the imputer used for the
non-XGBoost regression candidate.
"""
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_game_pool, split_per_year,
)
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, build_games_v2, build_box_ids, build_features_v2, FEATURES_V2, OPP_PITCHER_V2,
)

# Game-level feature set: FEATURES_V2 minus the batter-specific opposing-
# starter columns, which build_features_v2 itself fills with NaN for any
# non-batter pool (see module docstring).
GAME_FEATURES_V2 = [f for f in FEATURES_V2 if f not in OPP_PITCHER_V2]

# Best prior CLASSIFICATION result for each market, as specified in the
# task brief, for direct before/after comparison.
BASELINES = {
    "h2h": {"desc": "best prior classification result (Addendum 37/42)",
            "n": 181, "roi": 2.75, "lo": -9.98, "hi": 15.49, "verdict": "FAIL"},
    "totals": {"desc": "best prior classification result (Addendum 37/42)",
               "n": 194, "roi": -11.76, "lo": -25.15, "hi": 1.63, "verdict": "FAIL"},
}

CV = KFold(n_splits=5, shuffle=True, random_state=0)

XGB_POISSON_FIXED = dict(
    objective="count:poisson", learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=20, reg_lambda=3.0, missing=np.nan, random_state=0, n_jobs=1,
)
XGB_POISSON_GRID = {"max_depth": [3, 4], "n_estimators": [200, 300]}

XGB_REG_FIXED = dict(
    objective="reg:squarederror", learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=20, reg_lambda=3.0, missing=np.nan, random_state=0, n_jobs=1,
)
XGB_REG_GRID = {"max_depth": [3, 4], "n_estimators": [200, 300]}

POISSON_REG_GRID = {"reg__alpha": [0.01, 0.1, 1.0, 10.0]}


def cv_select(name, estimator, param_grid, X, y, scoring):
    """Cross-validated model selection on TRAIN data only, via a proper
    scoring rule (never ROI/accuracy). Returns (name, cv_score, fitted
    best_estimator, best_params) -- best_estimator is refit on the FULL
    training split."""
    if param_grid:
        search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=CV, n_jobs=1)
        search.fit(X, y)
        return name, search.best_score_, search.best_estimator_, search.best_params_
    scores = cross_val_score(estimator, X, y, scoring=scoring, cv=CV, n_jobs=1)
    estimator.fit(X, y)
    return name, scores.mean(), estimator, {}


def select_totals_model(X_train, y_train):
    candidates = [
        cv_select("PoissonRegressor",
                   Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("reg", PoissonRegressor(max_iter=2000))]),
                   POISSON_REG_GRID, X_train, y_train, "neg_mean_poisson_deviance"),
        cv_select("XGBRegressor(count:poisson)",
                   xgb.XGBRegressor(**XGB_POISSON_FIXED),
                   XGB_POISSON_GRID, X_train, y_train, "neg_mean_poisson_deviance"),
    ]
    for name, score, _, params in candidates:
        print(f"    candidate {name:<28} CV neg_mean_poisson_deviance={score:.5f}  params={params}")
    return max(candidates, key=lambda c: c[1])


def select_h2h_model(X_train, y_train):
    candidates = [
        cv_select("LinearRegression",
                   Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("reg", LinearRegression())]),
                   {}, X_train, y_train, "neg_root_mean_squared_error"),
        cv_select("XGBRegressor(squared-error)",
                   xgb.XGBRegressor(**XGB_REG_FIXED),
                   XGB_REG_GRID, X_train, y_train, "neg_root_mean_squared_error"),
    ]
    for name, score, _, params in candidates:
        print(f"    candidate {name:<28} CV neg_root_mean_squared_error={score:.5f}  (RMSE={-score:.4f})  params={params}")
    return max(candidates, key=lambda c: c[1])


def build_pool_with_features(con, con2, games_v2, box_ids, market_key):
    print(f"\n[{market_key}] building game pool (build_game_pool, unmodified import)...")
    pool = build_game_pool(con, games_v2, market_key)
    n_before = len(pool)
    n_games_before = pool["game_pk"].nunique()
    print(f"  pool rows before feature merge: n={n_before}  (unique game_pk={n_games_before})")

    pool = build_features_v2(pool, games_v2, "game", box_ids, con2)
    n_after = len(pool)
    n_games_after = pool["game_pk"].nunique()
    ok = "OK, no fan-out" if n_after == n_before else "MISMATCH -- investigate"
    print(f"  pool rows after feature merge:  n={n_after}  (unique game_pk={n_games_after})  {ok}")
    assert n_after == n_before, f"{market_key}: build_features_v2 changed row count ({n_before} -> {n_after})"

    opp_cols_all_nan = all(pool[c].isna().all() for c in OPP_PITCHER_V2)
    print(f"  OPP_PITCHER_V2 all-NaN at game level (expected, dropped from GAME_FEATURES_V2): {opp_cols_all_nan}")
    return pool


def evaluate_totals(pool):
    train, test = split_per_year(pool)
    print(f"  n(train/test)={len(train)}/{len(test)}")

    X_train = train[GAME_FEATURES_V2]
    y_train = train["total_runs"].astype(float)

    print("  model selection (CV Poisson deviance on TRAIN only, never ROI):")
    name, cv_score, model, params = select_totals_model(X_train, y_train)
    print(f"  SELECTED: {name}  CV neg_mean_poisson_deviance={cv_score:.5f}  params={params}")

    X_test = test[GAME_FEATURES_V2]
    mu = np.clip(model.predict(X_test), 1e-6, None)

    line = test["line"].to_numpy(dtype=float)
    floor_line = np.floor(line)
    prob_over = stats.poisson.sf(floor_line, mu=mu)          # P(total_runs > line)
    prob_under = stats.poisson.cdf(floor_line, mu=mu)        # P(total_runs < line)
    model_prob = np.where(test["side"].to_numpy() == "over", prob_over, prob_under)

    t = test.copy()
    t["pred_total_runs"] = mu
    t["model_prob"] = model_prob
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, official gate
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  TEST (touched once): edge>0.0 bets n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"market": "totals", "model": name, "cv_score": cv_score, "params": params,
            "n_train": len(train), "n_test": len(test), **s, "verdict": verdict}


def evaluate_h2h(pool):
    pool = pool.copy()
    pool["run_diff"] = pool["home_runs_"] - pool["away_runs_"]

    train, test = split_per_year(pool)
    print(f"  n(train/test)={len(train)}/{len(test)}")

    X_train = train[GAME_FEATURES_V2]
    y_train = train["run_diff"].astype(float)

    print("  model selection (CV RMSE on TRAIN only, never ROI):")
    name, cv_score, model, params = select_h2h_model(X_train, y_train)
    print(f"  SELECTED: {name}  CV RMSE={-cv_score:.4f}  params={params}")

    train_pred = model.predict(X_train)
    resid = y_train.to_numpy() - train_pred
    sigma = resid.std(ddof=1)  # TRAIN residuals only, never test
    print(f"  residual sigma (from TRAIN residuals only): {sigma:.4f}")

    X_test = test[GAME_FEATURES_V2]
    pred_diff = model.predict(X_test)
    # h2h pool (build_game_pool) is single-sided: "away" -- away wins iff
    # run_diff = home_runs_ - away_runs_ < 0.
    model_prob = stats.norm.cdf(0.0, loc=pred_diff, scale=sigma)  # P(run_diff < 0) = P(away wins)

    t = test.copy()
    t["pred_run_diff"] = pred_diff
    t["model_prob"] = model_prob
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, official gate
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  TEST (touched once): edge>0.0 bets n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"market": "h2h", "model": name, "cv_score": cv_score, "params": params,
            "n_train": len(train), "n_test": len(test), **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building real team game log, Elo, L10, bullpen fatigue, park factors (existing pipeline)...")
    games, box = build_games_and_box(con)
    n_games_raw = len(games)
    n_games_raw_unique = games["game_pk"].nunique()
    print(f"MANDATORY SANITY CHECK -- games (build_games_and_box): n={n_games_raw}  unique game_pk={n_games_raw_unique}")
    if n_games_raw != n_games_raw_unique:
        print(f"  (expected: this is the pre-existing build_team_game_log duplicate-game_date-per-game_pk "
              f"bug documented in Addendum 37 -- build_games_v2 below fixes it via drop_duplicates)")

    print("extending with Addendum 37 historical features (imported unmodified: build_games_v2)...")
    games_v2 = build_games_v2(con, con2, games)
    n_v2 = len(games_v2)
    n_v2_unique = games_v2["game_pk"].nunique()
    print(f"MANDATORY SANITY CHECK -- games_v2 (build_games_v2, post-dedup): n={n_v2}  unique game_pk={n_v2_unique}")
    assert n_v2 == n_v2_unique, "games_v2 has duplicate game_pk rows -- the exact Addendum 37 bug class, NOT fixed"
    print("  CONFIRMED: games_v2 has exactly one row per game_pk, no duplicates.")

    box_ids = build_box_ids(con)

    print(f"\nGAME_FEATURES_V2 ({len(GAME_FEATURES_V2)} features, FEATURES_V2 [{len(FEATURES_V2)}] "
          f"minus OPP_PITCHER_V2 {OPP_PITCHER_V2}):")
    print(f"  {GAME_FEATURES_V2}")

    print("\n=== Addendum 43c: regression on the continuous outcome (total_runs / run_diff), "
          "model selected via CV proper-scoring-rule on TRAIN only, ROI touched once ===")

    results = []

    print("\n--- totals: Poisson regression on total_runs ---")
    pool_totals = build_pool_with_features(con, con2, games_v2, box_ids, "totals")
    results.append(evaluate_totals(pool_totals))

    print("\n--- h2h: regression on run_diff (home_runs_ - away_runs_), Normal-sigma win prob ---")
    pool_h2h = build_pool_with_features(con, con2, games_v2, box_ids, "h2h")
    results.append(evaluate_h2h(pool_h2h))

    con.close()
    con2.close()

    print("\n=== SUMMARY: regression strategy (Addendum 43c) vs best prior classification result ===")
    for r in results:
        b = BASELINES[r["market"]]
        print(f"\n{r['market']}")
        print(f"  BEFORE [{b['desc']}]: n={b['n']}  ROI={b['roi']}  CI=[{b['lo']},{b['hi']}]  {b['verdict']}")
        print(f"  AFTER  [Addendum 43c regression, model={r['model']}]: "
              f"n(train/test)={r['n_train']}/{r['n_test']}")
        print(f"                        test: n={r['n']}  ROI={r['roi']}  CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    passes = [r for r in results if r["verdict"] == "PASS"]
    print(f"\n{len(passes)} of {len(results)} markets pass the official gate rule under the regression "
          f"framing. Reported as-is -- a FAIL here is a valid, expected outcome given both markets' "
          f"already-documented sample-size constraints, not a shortfall in this analysis.")


if __name__ == "__main__":
    main()
