"""
Addendum 43a: Poisson regression on the ACTUAL count outcome, not
classification on win/loss -- pitcher_strikeouts and batter_total_bases.

Every prior attempt on these two markets (Addenda 32-37/40:
reports/MILESTONE_1_GATE_REPORT.md, reports/addendum40_pitcher_k_lineup_output.txt)
framed the problem as binary classification: "does the over/under bet win?"
This script tries something genuinely different: predict the expected count
directly (a number -- expected strikeouts / expected total bases), then
derive a betting probability from that prediction via the Poisson
distribution, the standard statistical model for count data like this.

CRITICAL GROUND RULE (see HANDOFF.md's guardrails against p-hacking /
threshold-shopping, repeated here on purpose): the regression model and its
hyperparameters are selected ONLY via a proper scoring rule (CV-averaged
Poisson deviance, with RMSE reported alongside for context) computed on the
TRAIN split. ROI/gate/edge is NEVER computed on training data and never
used to pick a model or hyperparameters. The chosen model is refit once on
the full train split and scored ONCE on the untouched test split, at the
single pre-specified edge>0.0 cut and the official gate rule
(gate() in xgboost_individual_markets.py: n>=500 -> ROI>0; else CI lower
bound>0). If the result is FAIL, it is reported FAIL -- no re-tuning, no
threshold search, no re-running hoping for a different number.

Data / pipeline, reused UNMODIFIED from already-published, already-verified
code (per the task's explicit instruction -- nothing here re-derives or
edits any of these):
  - build_games_and_box, build_warehouse_pool, split_per_year
    (multi_model_comparison.py) -- build_warehouse_pool already returns,
    per bet-observation row, the REAL count outcome (the stat_col column,
    e.g. "strikeouts"/"total_bases") alongside the binarized `win` column;
    it was never dropped, just unused by every prior classification-only
    script. That real count is exactly what this script regresses on.
  - build_games_v2, build_box_ids, build_features_v2, FEATURES_V2,
    CACHE_DB, BASELINES (addendum37_historical_enrichment.py) -- the
    corrected, deduplicated, historically-enriched feature pipeline for
    these exact two markets.

Pool structure note: build_warehouse_pool returns TWO rows per real
player-game bet-observation (one "over" row, one "under" row -- same real
game, same real count outcome, different side/odds/market_prob), the same
structure every classification addendum in this project already uses and
reports n(train/test) against. Kept as-is here for direct comparability:
the Poisson model is fit on this same duplicated-row train/test split,
using FEATURES_V2 (which includes side_code/market_prob) exactly as
specified. Each row still gets its OWN predicted mean (a real, if unusual,
consequence of literally reusing FEATURES_V2 including the side-varying
columns) and its own row-appropriate probability:
  - "over" row:  predicted_prob = P(X > line)   = 1 - poisson.cdf(line, mu)
  - "under" row: predicted_prob = P(X < line)   = poisson.cdf(ceil(line)-1, mu)
This handles the half-integer (X.5, no push possible) vs whole-integer
(push possible) line convention correctly: P(X>line) and P(X<line) are only
complementary (sum to 1) when line is a whole integer would otherwise
double count the push -- computed here as two separate strict
inequalities, matching this project's own `win` definition elsewhere
(under[stat] < line, over[stat] > line -- a push counts as a loss for
both sides, never a win for either).

MANDATORY sanity check (same bug class caught in Addendum 37 and again in
Addendum 40 -- a join-key duplication silently inflating n): pool row
counts are printed before and after the build_features_v2 merge for every
market; a >20% unexpected change would be flagged loudly rather than
trusted silently.
"""
import os
import pathlib
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, gate, stat, profit, norm_name  # noqa: E402
from multi_model_comparison import build_games_and_box, build_warehouse_pool, split_per_year  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    build_games_v2, build_box_ids, build_features_v2, FEATURES_V2, CACHE_DB, BASELINES,
)

# Pre-specified, modest grids -- chosen once, before looking at any test-set
# ROI. Selection metric is CV Poisson deviance on the TRAIN split only.
POISSON_ALPHA_GRID = {"poissonregressor__alpha": [0.01, 0.1, 1.0, 10.0]}
XGB_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.02, 0.05],
    "n_estimators": [200, 400],
    "min_child_weight": [10, 20],
}
XGB_FIXED_PARAMS = dict(
    objective="count:poisson", subsample=0.8, colsample_bytree=0.7,
    reg_lambda=3.0, missing=np.nan, random_state=0, n_jobs=1,
)
N_JOBS = os.cpu_count() or 4

MARKETS = [
    ("pitcher_strikeouts", "strikeouts", "pitcher"),
    ("batter_total_bases", "total_bases", "batter"),
]


def under_threshold(line):
    """Integer k such that P(X < line) = P(X <= k): k = floor(line) when
    line is a half-integer (X.5, the common prop-line convention -- no push
    possible), k = line - 1 when line is a whole integer (so the push at
    X==line is correctly excluded from the under side, matching this
    project's own strict-inequality `win` definition elsewhere)."""
    return np.ceil(np.asarray(line, dtype=float)) - 1.0


def poisson_probs(line, mu):
    """(prob_over, prob_under) = (P(X > line), P(X < line)) under a
    Poisson(mu) model of the count. NOT forced to sum to 1 on whole-integer
    lines (the push probability P(X==line) is real and belongs to neither
    side) -- only complementary on the X.5 lines this project's props
    mostly use, exactly as intended."""
    line = np.asarray(line, dtype=float)
    mu = np.asarray(mu, dtype=float)
    prob_over = 1.0 - poisson.cdf(line, mu)
    prob_under = poisson.cdf(under_threshold(line), mu)
    return prob_over, prob_under


def fit_poisson_regressor(X_train, y_train, cv):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("poissonregressor", PoissonRegressor(max_iter=3000)),
    ])
    search = GridSearchCV(pipe, POISSON_ALPHA_GRID, cv=cv,
                           scoring="neg_mean_poisson_deviance", n_jobs=N_JOBS)
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same fix already used/noted in
    # tuned_xgboost_4_markets.py / addendum37_historical_enrichment.py).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV Poisson deviance only -- no ROI/gate touched here
    return search


def fit_xgb_poisson(X_train, y_train, cv):
    search = GridSearchCV(
        xgb.XGBRegressor(**XGB_FIXED_PARAMS), XGB_PARAM_GRID, cv=cv,
        scoring="neg_mean_poisson_deviance", n_jobs=N_JOBS,
    )
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV Poisson deviance only -- no ROI/gate touched here
    return search


def cv_rmse(estimator, X_train, y_train, cv):
    with joblib.parallel_backend("threading"):
        scores = cross_val_score(estimator, X_train, y_train, cv=cv,
                                  scoring="neg_mean_squared_error", n_jobs=N_JOBS)
    return float(np.sqrt(-scores.mean()))


def run_market(market_key, stat_col, kind, con, box, games_v2, box_ids, con2):
    print(f"\n=== {market_key} ===")
    pool0 = build_warehouse_pool(con, box, market_key, stat_col, kind)
    n0 = len(pool0)
    print(f"  [row-count check] raw warehouse pool (over+under rows, real count outcome "
          f"'{stat_col}' already present, not re-derived): n={n0}")

    pool1 = build_features_v2(pool0, games_v2, kind, box_ids, con2)
    n1 = len(pool1)
    pct = 100.0 * (n1 - n0) / n0 if n0 else float("nan")
    print(f"  [row-count check] after build_features_v2 (Addendum 37, unmodified, LEFT joins only): "
          f"n={n1} ({pct:+.1f}% vs raw pool)")
    if abs(pct) > 20:
        print("  *** WARNING: row count changed by more than 20% vs the raw pool -- "
              "this is exactly the join-key-duplication bug class caught in Addenda 37/40. "
              "STOP and investigate before trusting any number below. ***")
    else:
        print(f"  row-count sanity check: OK ({pct:+.1f}% change from left-joins that should not "
              f"fan out -- consistent with Addendum 37/40's own verification of this same merge).")

    n_missing_stat = pool1[stat_col].isna().sum()
    n_missing_line = pool1["line"].isna().sum()
    if n_missing_stat or n_missing_line:
        print(f"  NOTE: {n_missing_stat} rows missing real count outcome, {n_missing_line} rows "
              f"missing line -- dropped before modeling (build_warehouse_pool already drops "
              f"missing stat_col; any line NaNs here would be a new finding).")
        pool1 = pool1.dropna(subset=[stat_col, "line"])

    train, test = split_per_year(pool1)
    print(f"  n(train/test)={len(train)}/{len(test)}  (same per-year 75/25 split every prior "
          f"addendum uses for this market)")

    X_train, y_train = train[FEATURES_V2], train[stat_col].astype(float)
    X_test = test[FEATURES_V2]

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # ---- Model / hyperparameter selection: CV Poisson deviance, TRAIN split
    # only. ROI/gate/edge is not computed anywhere in this block. ----
    pr_search = fit_poisson_regressor(X_train, y_train, cv)
    xg_search = fit_xgb_poisson(X_train, y_train, cv)

    pr_dev = -pr_search.best_score_
    xg_dev = -xg_search.best_score_
    pr_rmse = cv_rmse(pr_search.best_estimator_, X_train, y_train, cv)
    xg_best_est = xgb.XGBRegressor(**{**XGB_FIXED_PARAMS, **xg_search.best_params_})
    xg_rmse = cv_rmse(xg_best_est, X_train, y_train, cv)

    print(f"  CV (TRAIN only) PoissonRegressor:            mean_poisson_deviance={pr_dev:.4f}  "
          f"RMSE={pr_rmse:.4f}  best_params={pr_search.best_params_}")
    print(f"  CV (TRAIN only) XGBRegressor(count:poisson):  mean_poisson_deviance={xg_dev:.4f}  "
          f"RMSE={xg_rmse:.4f}  best_params={xg_search.best_params_}")

    if xg_dev < pr_dev:
        chosen_name, chosen_model = "XGBRegressor(count:poisson)", xg_best_est
    else:
        chosen_name, chosen_model = "PoissonRegressor", pr_search.best_estimator_
    print(f"  >>> primary model, chosen by CV Poisson deviance on TRAIN only (lower is better): "
          f"{chosen_name}")

    chosen_model.fit(X_train, y_train)
    mu_test = chosen_model.predict(X_test)
    n_nonpositive = int((mu_test <= 0).sum())
    if n_nonpositive:
        print(f"  NOTE: {n_nonpositive} test predictions were <=0 (should not happen under a "
              f"log-link Poisson model) -- clipped to a small positive epsilon before poisson.cdf.")
    mu_test = np.clip(mu_test, 1e-6, None)

    # ---- Single, final, one-shot test-set evaluation. Not touched above. ----
    t = test.copy()
    t["predicted_mean"] = mu_test
    prob_over, prob_under = poisson_probs(t["line"].values, t["predicted_mean"].values)
    t["model_prob"] = np.where(t["side"].values == "over", prob_over, prob_under)
    t["edge"] = t["model_prob"] - t["market_prob"]

    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  TEST SET, edge>0.0 (odds<0) bets, single pre-specified cut, official gate rule:")
    print(f"    n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  win_rate={s['wr']}  -> {verdict}")

    baseline = BASELINES.get(market_key)
    if baseline:
        print(f"  prior classification-based result [{baseline['desc']}]: n={baseline['n']} "
              f"ROI={baseline['roi']} CI=[{baseline['lo']},{baseline['hi']}]  {baseline['verdict']}")

    return {
        "market": market_key, "n_train": len(train), "n_test": len(test),
        "chosen_model": chosen_name, "pr_dev": round(pr_dev, 4), "xg_dev": round(xg_dev, 4),
        "pr_rmse": round(pr_rmse, 4), "xg_rmse": round(xg_rmse, 4),
        "pr_best_params": pr_search.best_params_, "xg_best_params": xg_search.best_params_,
        **s, "verdict": verdict,
    }


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building real team game log, Elo, L10, bullpen fatigue, park factors (existing pipeline, "
          "unmodified import from multi_model_comparison.py)...")
    games, box = build_games_and_box(con)
    print("extending with Addendum 37 historical features (unmodified import from "
          "addendum37_historical_enrichment.py): point-in-time bullpen quality, both-starters + "
          "opposing-starter statcast quality, park dimensions/orientation...")
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    print(f"\nFEATURES_V2 ({len(FEATURES_V2)} total): {FEATURES_V2}")

    print("\n" + "=" * 100)
    print("Addendum 43a: Poisson REGRESSION on the actual count outcome (strikeouts / total_bases),")
    print("not classification on win/loss. predicted_prob_over/under derived from the fitted mean")
    print("via scipy.stats.poisson.cdf, compared to the market's own implied probability for the")
    print("same edge>0.0 cut and official gate rule used everywhere else in this project. Model and")
    print("hyperparameters selected ONLY by CV Poisson deviance (RMSE reported alongside for context)")
    print("on the TRAIN split -- ROI/gate is never touched until the single, final, one-shot")
    print("test-set evaluation below. A FAIL is reported as a FAIL, no re-tuning after the fact.")
    print("=" * 100)

    results = []
    for market_key, stat_col, kind in MARKETS:
        r = run_market(market_key, stat_col, kind, con, box, games_v2, box_ids, con2)
        results.append(r)

    con.close()
    con2.close()

    print("\n=== SUMMARY: Addendum 43a Poisson regression, both markets ===")
    for r in results:
        print(f"\n{r['market']}")
        print(f"  chosen model: {r['chosen_model']}")
        print(f"  CV (TRAIN only): PoissonRegressor deviance={r['pr_dev']} RMSE={r['pr_rmse']}  |  "
              f"XGBRegressor(count:poisson) deviance={r['xg_dev']} RMSE={r['xg_rmse']}")
        print(f"  n(train/test)={r['n_train']}/{r['n_test']}")
        print(f"  test: edge>0.0 n={r['n']} ROI={r['roi']} CI=[{r['lo']},{r['hi']}]  {r['verdict']}")
        baseline = BASELINES.get(r["market"])
        if baseline:
            print(f"  vs prior classification result [{baseline['desc']}]: n={baseline['n']} "
                  f"ROI={baseline['roi']} CI=[{baseline['lo']},{baseline['hi']}]  {baseline['verdict']}")

    passes = [r for r in results if r["verdict"] == "PASS"]
    print(f"\n{len(passes)} of {len(results)} markets pass the official gate rule under the Poisson-"
          f"regression reframing. Reported as-is, pass or fail -- single pre-specified model-"
          f"selection procedure (CV Poisson deviance, train split only), single pre-specified "
          f"edge>0.0 cut, no threshold search, no re-running after seeing the test-set result.")


if __name__ == "__main__":
    main()
