"""
Addendum 43b: Poisson REGRESSION on the actual count stat, for the two
pick_history-only markets (pitcher_outs / batter_runs_scored), instead of the
BINARY CLASSIFICATION ("does the over/under bet win?") every prior attempt
(Addenda 27, 34, 38, 41) has used.

WHY: pick_history carries the raw outcome count directly in `actual_value`
(confirmed empirically below -- 100% populated on every graded row for both
markets, and >99.9% consistent with the recorded `hit`/`pick_side`/`line`
relationship) -- there is no need to reconstruct it from boxscore. That means
the natural model here is not "predict win probability of THIS specific bet"
but "predict the expected count (outs recorded / runs scored), then derive a
betting probability from the Poisson distribution" -- the standard
statistical model for count data, and the one that's directly explainable to
a client: "we predict this pitcher gets X outs, the market line is Y, here's
the Poisson-implied probability the over/under hits."

REAL DATA WINDOW -- VERIFIED DIRECTLY, NOT ASSUMED (see print output below):
Both markets' real graded (hit IS NOT NULL) pick_history window is actually
**2026-06-13 to 2026-07-27** for BOTH pitcher_outs (n=798) and runs_scored
(n=6,859) -- not 2026-05-17 as an earlier docstring in
xgboost_individual_markets.py states for a *different* purpose (that date is
when hits/total_bases/rbis/home_runs -- a different set of markets -- start;
pitcher_outs/runs_scored picks only start 2026-06-07 and only start grading
2026-06-13). Confirmed by direct query against db/mlb_markets.duckdb with the
exact same filter run_pick_history_market() uses. This matches Addendum 41's
own verified runs_scored window exactly, and turns out to be identical for
pitcher_outs too -- a real finding, not an assumption carried over.

GROUND RULE (followed exactly): model/hyperparameter selection uses ONLY a
proper scoring rule (mean Poisson deviance, primary; RMSE, secondary) via
K-fold CV on the TRAINING split. The test set's betting ROI is touched
EXACTLY ONCE, at the very end, at a single pre-specified edge>0.0 cut, the
official gate rule (gate() in xgboost_individual_markets.py). No threshold
search, no re-running if the result displeases.

FEATURE SET: reuses Addendum 38's point-in-time cache-table enrichment
(pitcher_outs: last3/inn1/bullpen/pen-rest/high-leverage/manager-hook;
runs_scored: batter_splits/team_batting_stats/team_oaa) joined via the exact
same ASOF-join SQL (imported from addendum38_pickhistory_enrichment.py, not
re-derived), on top of pick_history's own score_* columns, same coverage
floor (>=200 rows or >=5% of pool, whichever larger) used throughout this
project's pick_history addenda.

TWO CANDIDATE MODEL FAMILIES, chosen via CV Poisson deviance only:
  A. sklearn PoissonRegressor (median-impute + standardize + Poisson GLM,
     small alpha grid) -- the simplest, most directly explainable regression.
  B. xgboost.XGBRegressor(objective="count:poisson") -- same PARAM_GRID
     already pre-registered and reused (unmodified) from
     tuned_xgboost_4_markets.py for every other XGBoost tuning pass in this
     project, just with a regression/Poisson objective substituted for the
     classification-specific eval_metric.
Both are scored with identical KFold(5, shuffle=True, random_state=0) folds
on the training split only, scoring="neg_mean_poisson_deviance". Whichever
has the better (less negative) CV score is refit once and carried to test.

BETTING PROBABILITY FROM THE POISSON MODEL: every line in both markets is an
X.5 half-integer (verified below -- 100% of graded rows, both markets), so
there is never a push to handle. For k = floor(line):
  P(over wins)  = P(count > k)  = poisson.sf(k, mu)   = 1 - poisson.cdf(k, mu)
  P(under wins) = P(count < k+1)= poisson.cdf(k, mu)
using the model's predicted mean mu for that row, matched to whichever side
(`pick_side`) the real pick was actually placed on.

MANDATORY SANITY CHECK (Addendum 37/40 bug class -- join row-count
inflation): row counts printed before/after every merge below. All cache
joins are ASOF LEFT JOIN (or a single-row-per-team static LEFT JOIN) keyed to
match at most one row per pick -- if the post-join count differs from the
pre-join count, that's a fan-out bug and this script says so explicitly
rather than silently trusting the result.
"""
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
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, MIN_GRADED, gate, stat, profit, implied_prob  # noqa: E402
from addendum38_pickhistory_enrichment import (  # noqa: E402
    CACHE_DB, PITCHER_OUTS_JOIN, PITCHER_OUTS_SELECT,
    RUNS_SCORED_JOIN, RUNS_SCORED_SELECT,
)
from tuned_xgboost_4_markets import PARAM_GRID  # noqa: E402  (reused unmodified)

MARKET_SPECS = {
    "pitcher_outs": dict(prop_type="pitcher_outs", join_sql=PITCHER_OUTS_JOIN, select_sql=PITCHER_OUTS_SELECT),
    "runs_scored": dict(prop_type="runs_scored", join_sql=RUNS_SCORED_JOIN, select_sql=RUNS_SCORED_SELECT),
}

# XGBRegressor fixed params: same subsample/colsample/reg_lambda already
# pre-registered for every other XGBoost pass in this project
# (tuned_xgboost_4_markets.PARAM_GRID's sibling FIXED_PARAMS) -- only the
# classification-specific eval_metric is swapped for a Poisson objective,
# since this is a regression on the count, not a classifier.
XGB_FIXED_PARAMS = dict(
    subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
    objective="count:poisson", missing=np.nan, random_state=0, n_jobs=1,
)
# PoissonRegressor's only real tuning knob: L2 regularization strength.
# Small, pre-specified grid -- not searched against ROI.
POISSON_ALPHA_GRID = {"poissonregressor__alpha": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]}

N_CV_FOLDS = 5


def verify_real_window(con, prop_type):
    """Print the ACTUAL graded date range for this market, queried directly
    -- per task instructions, never assumed from another addendum's docstring."""
    row = con.execute(f"""
        SELECT count(*), min(TRY_CAST(game_date AS DATE)), max(TRY_CAST(game_date AS DATE))
        FROM pick_history
        WHERE prop_type = '{prop_type}'
          AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchone()
    print(f"  {prop_type}: n={row[0]}  real graded window: {row[1]} -> {row[2]}")
    return row[1], row[2]


def verify_line_convention(con, prop_type):
    frac = con.execute(f"""
        SELECT TRY_CAST(line AS DOUBLE) % 1.0 AS frac, count(*) n
        FROM pick_history
        WHERE prop_type = '{prop_type}' AND hit IS NOT NULL
        GROUP BY 1
    """).fetchdf()
    print(f"  {prop_type} line fractional-part distribution:\n{frac.to_string(index=False)}")
    all_half = (len(frac) == 1) and abs(frac["frac"].iloc[0] - 0.5) < 1e-9
    print(f"  -> all lines are X.5 (no push possible): {all_half}")
    return all_half


def load_pool(con, prop_type, join_sql, select_sql):
    """Base pick_history rows/window/win/odds logic identical to
    run_pick_history_market() -- PLUS the actual raw count (`actual_value`),
    the line, and pick_side, needed for regression + Poisson-CDF betting
    probability -- PLUS player_id/team for the Addendum 38 cache joins."""
    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    score_select = ", ".join(f'TRY_CAST(ph."{c}" AS DOUBLE) AS "{c}"' for c in score_cols)

    base_n = con.execute(f"""
        SELECT count(*) FROM pick_history ph
        WHERE ph.prop_type = '{prop_type}'
          AND lower(coalesce(ph.is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(ph.voided,'false')) NOT IN ('true','t','1')
          AND ph.hit IS NOT NULL AND TRY_CAST(ph.odds AS INTEGER) IS NOT NULL
    """).fetchone()[0]

    query = f"""
        WITH base AS (
            SELECT ph.id, TRY_CAST(ph.game_date AS DATE) AS game_date,
                   TRY_CAST(ph.odds AS INTEGER) AS odds,
                   TRY_CAST(ph.line AS DOUBLE) AS line,
                   ph.pick_side,
                   TRY_CAST(ph.actual_value AS DOUBLE) AS actual_value,
                   lower(ph.hit) IN ('true','t','1') AS win,
                   TRY_CAST(ph.player_id AS BIGINT) AS player_id, ph.team,
                   {score_select}
            FROM pick_history ph
            WHERE ph.prop_type = '{prop_type}'
              AND lower(coalesce(ph.is_synthetic,'false')) NOT IN ('true','t','1')
              AND lower(coalesce(ph.voided,'false')) NOT IN ('true','t','1')
              AND ph.hit IS NOT NULL AND TRY_CAST(ph.odds AS INTEGER) IS NOT NULL
        )
        SELECT base.*, {select_sql}
        FROM base
        {join_sql}
    """
    df = con.execute(query).fetchdf()
    df = df.dropna(subset=["game_date"]).sort_values(
        ["game_date", "id"], kind="mergesort"
    ).reset_index(drop=True)
    return df, base_n, score_cols


def sanity_check_rowcounts(df, base_n, market_key):
    print(f"\n  [{market_key}] row-count sanity check (Addendum 37/40 bug class):")
    print(f"    pre-join base pool (pick_history filter only): n={base_n}")
    print(f"    post-join pool (all cache ASOF/static joins applied): n={len(df)}")
    if len(df) != base_n:
        print(f"    *** WARNING: post-join n ({len(df)}) != pre-join n ({base_n}) -- "
              f"a join is fanning out to multiple rows per pick. Investigate before "
              f"trusting anything downstream. ***")
        return False
    print("    OK: exact match -- every cache join is a true at-most-one-row-per-pick "
          "ASOF/static join, no fan-out.")
    return True


def check_actual_value_consistency(df, market_key):
    """actual_value/line/pick_side should reconstruct `win` (`hit`) almost
    exactly -- a second, independent sanity check that we're reading the raw
    count column correctly."""
    d = df.dropna(subset=["actual_value", "line", "pick_side"])
    recon = np.where(d["pick_side"] == "over", d["actual_value"] > d["line"], d["actual_value"] < d["line"])
    match = (recon == d["win"].values)
    print(f"  [{market_key}] actual_value/line/pick_side reconstructs recorded `hit` in "
          f"{match.mean() * 100:.2f}% of {len(d)} graded rows "
          f"({(~match).sum()} mismatches -- recorded `hit` is still what's used for realized "
          f"ROI below, never overridden by a recomputed value).")


def select_features(df, score_cols, min_cov):
    new_cols = [c for c in df.columns if c.startswith("ck_")]
    score_keep = [c for c in score_cols if df[c].notna().sum() >= min_cov]
    new_keep = [c for c in new_cols if df[c].notna().sum() >= min_cov]
    return score_keep + new_keep, score_keep, new_keep, new_cols


def cv_select_model(X_train, y_train):
    """Model/hyperparameter selection via CV Poisson deviance (primary) and
    CV RMSE (secondary, reported only) on the TRAINING split -- ROI/gate
    never touched here. Identical KFold splits reused for both candidate
    families for a fair, apples-to-apples comparison."""
    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=0)

    poisson_pipe = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        PoissonRegressor(max_iter=2000),
    )
    search_poisson = GridSearchCV(
        poisson_pipe, POISSON_ALPHA_GRID, scoring="neg_mean_poisson_deviance",
        cv=cv, n_jobs=1, refit=True,
    )
    search_poisson.fit(X_train, y_train)

    search_xgb = GridSearchCV(
        xgb.XGBRegressor(**XGB_FIXED_PARAMS), PARAM_GRID,
        scoring="neg_mean_poisson_deviance", cv=cv, n_jobs=joblib.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (XGBoost's fit releases the GIL,
    # so this still parallelizes real work). Same pattern as Addendum 34/38.
    with joblib.parallel_backend("threading"):
        search_xgb.fit(X_train, y_train)

    print(f"    PoissonRegressor : best CV neg-mean-Poisson-deviance = {search_poisson.best_score_:.4f}  "
          f"(alpha={search_poisson.best_params_['poissonregressor__alpha']})")
    print(f"    XGBRegressor     : best CV neg-mean-Poisson-deviance = {search_xgb.best_score_:.4f}  "
          f"(params={search_xgb.best_params_})")

    if search_xgb.best_score_ > search_poisson.best_score_:
        chosen_name, chosen = "XGBRegressor(count:poisson)", search_xgb.best_estimator_
        chosen_cv_score = search_xgb.best_score_
    else:
        chosen_name, chosen = "PoissonRegressor", search_poisson.best_estimator_
        chosen_cv_score = search_poisson.best_score_
    print(f"    -> selected via CV Poisson deviance only: {chosen_name} "
          f"(CV neg-mean-Poisson-deviance={chosen_cv_score:.4f})")
    return chosen_name, chosen


def cv_rmse(estimator_ctor, X_train, y_train):
    """Secondary CV metric (RMSE), reported for context, never used to pick
    between models -- refits `estimator_ctor()` fresh on each fold."""
    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=0)
    errs = []
    for tr, te in cv.split(X_train):
        m = estimator_ctor()
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = m.predict(X_train.iloc[te])
        errs.append(mean_squared_error(y_train.iloc[te], pred))
    return float(np.sqrt(np.mean(errs)))


def evaluate_market(con, market_key, prop_type, join_sql, select_sql):
    print(f"\n{'=' * 78}\n{market_key}\n{'=' * 78}")
    verify_real_window(con, prop_type)
    verify_line_convention(con, prop_type)

    df, base_n, score_cols = load_pool(con, prop_type, join_sql, select_sql)
    ok = sanity_check_rowcounts(df, base_n, market_key)
    check_actual_value_consistency(df, market_key)

    n_total = len(df)
    if n_total < MIN_GRADED:
        print(f"  n={n_total} < {MIN_GRADED} -- INSUFFICIENT DATA, not evaluated")
        return {"market": market_key, "n": n_total, "verdict": "insufficient data"}

    min_cov = max(200, n_total * 0.05)
    features, score_keep, new_keep, new_cols = select_features(df, score_cols, min_cov)
    print(f"\n  features kept (coverage >= {min_cov:.0f} rows): "
          f"{len(score_keep)}/{len(score_cols)} score_* + {len(new_keep)}/{len(new_cols)} cache "
          f"= {len(features)} total")

    pool = df.dropna(subset=["actual_value", "line", "pick_side", "odds", "game_date"]).copy()
    print(f"  after dropping rows missing target/line/side/odds: n={len(pool)} "
          f"(dropped {n_total - len(pool)})")

    pool = pool.sort_values(["game_date", "id"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.70)  # chronological 70/30, same convention as Addendum 41
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"  chronological 70/30 split (Addendum 41 convention): "
          f"train n={len(train)} ({train['game_date'].min()} -> {train['game_date'].max()}), "
          f"test n={len(test)} ({test['game_date'].min()} -> {test['game_date'].max()})")

    if len(train) < 100 or len(test) < 20:
        print("  INSUFFICIENT DATA for a real holdout -- not evaluated")
        return {"market": market_key, "n": n_total, "verdict": "insufficient data"}

    X_train, y_train = train[features], train["actual_value"].astype(float)
    X_test = test[features]

    print("\n  --- model selection: CV Poisson deviance on TRAIN only, ROI untouched ---")
    chosen_name, chosen_model = cv_select_model(X_train, y_train)

    def poisson_ctor():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              PoissonRegressor(max_iter=2000, alpha=0.1))
    def xgb_ctor():
        return xgb.XGBRegressor(**XGB_FIXED_PARAMS, n_estimators=300, max_depth=4,
                                 learning_rate=0.03, min_child_weight=20)
    rmse_poisson = cv_rmse(poisson_ctor, X_train, y_train)
    rmse_xgb = cv_rmse(xgb_ctor, X_train, y_train)
    print(f"    (secondary, reported only) CV RMSE: PoissonRegressor={rmse_poisson:.3f}  "
          f"XGBRegressor={rmse_xgb:.3f}")

    # --- refit chosen model once on full train, predict ONCE on test ---
    mu_test = chosen_model.predict(X_test)
    mu_test = np.clip(mu_test, 1e-6, None)  # Poisson mean must be > 0

    k = np.floor(test["line"].values).astype(int)  # all lines are X.5 -> k=floor is exact, no push
    p_over = poisson.sf(k, mu_test)
    p_under = poisson.cdf(k, mu_test)
    test = test.copy()
    test["mu"] = mu_test
    test["model_prob"] = np.where(test["pick_side"].values == "over", p_over, p_under)
    test["market_prob"] = implied_prob(test["odds"])
    test["edge"] = test["model_prob"] - test["market_prob"]

    test_dev = mean_poisson_deviance(test["actual_value"], mu_test)
    test_rmse = np.sqrt(mean_squared_error(test["actual_value"], mu_test))
    print(f"\n  test-set fit quality (regression, not betting): "
          f"Poisson deviance={test_dev:.4f}  RMSE={test_rmse:.3f}  "
          f"mean actual={test['actual_value'].mean():.2f}  mean predicted={mu_test.mean():.2f}")

    # --- SINGLE pre-specified cut, official gate rule, touched exactly once ---
    sub = test[(test.odds < 0) & (test.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    passed = gate(s)
    underpowered = (s["n"] is not None and s["n"] > 0 and s["n"] < 500
                    and s["lo"] is not None and s["hi"] is not None and (s["hi"] - s["lo"] > 40))
    verdict = "PASS" if passed else ("UNDERPOWERED" if underpowered else "FAIL")

    print(f"\n  edge>0.0 (single pre-specified cut, official gate rule, TOUCHED ONCE): "
          f"n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {
        "market": market_key, "model": chosen_name,
        "n_train": len(train), "n_test": len(test),
        "test_poisson_deviance": round(test_dev, 4), "test_rmse": round(test_rmse, 3),
        **{f"stat_{k2}": v for k2, v in s.items()}, "verdict": verdict,
    }


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=" * 78)
    print("Addendum 43b: Poisson REGRESSION on the actual count stat")
    print("(pitcher_outs -> expected outs recorded, batter_runs_scored -> expected")
    print("runs scored), predicting the count directly and deriving a betting")
    print("probability via the Poisson CDF -- instead of binarized win/loss")
    print("classification (every prior attempt: Addenda 27, 34, 38, 41).")
    print("Model/hyperparameter selection: CV Poisson deviance (+ CV RMSE, reported)")
    print("on TRAIN only. Test ROI touched exactly once, single edge>0.0 cut, official")
    print("gate rule (gate() in xgboost_individual_markets.py).")
    print("=" * 78)

    results = []
    for market_key, spec in MARKET_SPECS.items():
        res = evaluate_market(con, market_key, spec["prop_type"], spec["join_sql"], spec["select_sql"])
        results.append(res)

    print("\n" + "=" * 78)
    print("SUMMARY -- Addendum 43b (Poisson regression) vs best prior classification result")
    print("=" * 78)
    print("  pitcher_outs  prior best (Addendum 27/38, classification): severely "
          "sample-size-constrained FAIL")
    print("  runs_scored   prior best (Addendum 41, classification): FAIL, "
          "ROI=+0.89%, CI=[-6.13%,7.91%]")
    print()
    for r in results:
        if r.get("verdict") in ("insufficient data",):
            print(f"  {r['market']:<15} n={r['n']}  {r['verdict']}")
            continue
        print(f"  {r['market']:<15} model={r['model']}  n(train/test)={r['n_train']:,}/{r['n_test']:,}  "
              f"test Poisson-deviance={r['test_poisson_deviance']}  test RMSE={r['test_rmse']}  "
              f"edge>0.0: n={r['stat_n']} ROI={r['stat_roi']}% CI=[{r['stat_lo']},{r['stat_hi']}]  "
              f"{r['verdict']}")

    con.close()


if __name__ == "__main__":
    main()
