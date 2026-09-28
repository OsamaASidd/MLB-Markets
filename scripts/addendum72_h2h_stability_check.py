"""
Addendum 72: reproducibility/stability check on Addendum 49's h2h walk-forward
result, requested directly after a plain re-run of addendum49_h2h_walkforward.py
(unmodified) produced n=1133 ROI=+0.49% PASS -- which does not match the
number written into reports/milestone2_handover.html for the same addendum
(n=1,114 ROI=-0.37% FAIL). Every parameter in Addendum 49 that looks
seed-controlled is already hardcoded (FIXED_PARAMS random_state=0,
StratifiedKFold random_state=0), so a flip between two "identical" runs
means the result is unstable under something OTHER than a changed seed
(most likely floating-point non-determinism from GridSearchCV's threading
backend + XGBoost's histogram tree method on multiple cores -- the exact
same class of instability this project already documented once, in
Addendum 22/23's pooled Model B).

WHAT THIS SCRIPT DOES: builds the Addendum 48/49 pool and features exactly
once (unmodified imports, no changes to feature engineering or the walk-
forward fold structure), then repeats the per-fold CV-select + fit +
predict sequence across every (seed, fold) combination, each with an
explicit random_state (0..N_RUNS-1) fed into both XGBoost and the inner
StratifiedKFold. Every (seed, fold) pair is an independent unit of work
(a fold's train/test slice doesn't depend on any other fold's model), so
all of them run concurrently on a thread pool -- XGBoost's native fit
releases the GIL, so threads give real parallelism here even without
process-based parallelism (which fails in this sandboxed environment:
joblib's default loky/process backend raises
`ModuleNotFoundError: No module named '_posixsubprocess'`).

Every seed's n/ROI/CI/verdict is reported once pooled -- pass and fail
alike, no selection, no "best of" reporting. This is a diagnostic, not a
new pass/fail finding for h2h itself; h2h's real verdict is "unresolved,
unstable" until/unless this comes back consistent.
"""
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, build_pool, build_features, FEATURES
from addendum49_h2h_walkforward import attach_enrichment, N_FOLDS, PARAM_GRID_WIDE

N_RUNS = 8
# First parallel attempt (outer ThreadPoolExecutor + inner GridSearchCV
# joblib "threading" backend, nested) died silently -- exit code 4, no
# Python traceback captured even with stderr redirected, consistent with a
# native-level crash from nested joblib thread-backend contexts across
# real OS threads, not a catchable Python exception. Fix: no nested
# parallelism. GridSearchCV runs fully serial internally (n_jobs=1); all
# concurrency comes from the OUTER ThreadPoolExecutor across the 40
# independent (seed, fold) tasks instead. XGBoost's native fit still
# releases the GIL, so this still gets real wall-clock parallelism.


def cv_select(X_train, y_train, seed):
    fixed = dict(subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
                 eval_metric="logloss", missing=np.nan, random_state=seed, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(xgb.XGBClassifier(**fixed), PARAM_GRID_WIDE, scoring="roc_auc", cv=cv, n_jobs=1)
    search.fit(X_train, y_train)
    return search.best_estimator_


def fold_task(pool, seed, f):
    try:
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        if len(train_df) < 200 or len(test_df) < 20:
            return seed, None, None
        X_train, y_train = train_df[FEATURES], train_df["win"].astype(int)
        model = cv_select(X_train, y_train, seed)
        test_pred = model.predict_proba(test_df[FEATURES])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        return seed, t[["game_date", "odds", "win", "edge"]], None
    except Exception as e:
        return seed, None, f"{type(e).__name__}: {e}"


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log + features (Addendum 48/49, unmodified imports)...", flush=True)
    games, id_to_name = build_fuller_games(con, con2)
    pool = build_pool(con, games, "h2h")
    n_before = len(pool)
    pool = attach_enrichment(con, con2, pool, id_to_name)
    assert len(pool) == n_before, "*** FAN-OUT, STOP ***"
    pool = build_features(pool)
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort"
    ).reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id
    print(f"pool built once: n={len(pool)}, reused for all {N_RUNS} seeded runs", flush=True)
    con.close()
    con2.close()

    tasks = [(seed, f) for seed in range(N_RUNS) for f in range(1, N_FOLDS)]
    print(f"\n=== Addendum 72: {N_RUNS} seeds x {N_FOLDS - 1} folds = {len(tasks)} tasks, "
          f"running concurrently (thread pool, serial grid search per task) ===", flush=True)

    bets_by_seed = {seed: [] for seed in range(N_RUNS)}
    errors = []
    t_start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(fold_task, pool, seed, f) for seed, f in tasks]
        for fut in as_completed(futures):
            seed, bets, err = fut.result()
            done += 1
            if err is not None:
                errors.append((seed, err))
                print(f"  [{done}/{len(tasks)}] seed={seed} FAILED: {err} ({time.time() - t_start:.0f}s elapsed)", flush=True)
                continue
            if bets is not None:
                bets_by_seed[seed].append(bets)
            print(f"  [{done}/{len(tasks)}] task done ({time.time() - t_start:.0f}s elapsed)", flush=True)

    if errors:
        print(f"\n*** {len(errors)} task(s) raised exceptions -- see above ***")

    results = []
    print(f"\n=== per-seed pooled results (official gate touched EXACTLY ONCE per seed) ===")
    for seed in range(N_RUNS):
        if not bets_by_seed[seed]:
            print(f"  seed={seed}  NO SUCCESSFUL FOLDS -- skipped")
            continue
        pooled = pd.concat(bets_by_seed[seed], ignore_index=True)
        sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        verdict = "PASS" if gate(s) else "FAIL"
        print(f"  seed={seed}  n={s['n']:<5} WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
        results.append({"seed": seed, **s, "verdict": verdict})

    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_fail = N_RUNS - n_pass
    rois = [r["roi"] for r in results if r["roi"] is not None]
    print(f"\n=== SUMMARY ===")
    print(f"  {n_pass}/{N_RUNS} PASS, {n_fail}/{N_RUNS} FAIL")
    print(f"  ROI range across runs: {min(rois):.2f}% to {max(rois):.2f}%  (mean {sum(rois)/len(rois):.2f}%)")
    if 0 < n_pass < N_RUNS:
        print("  VERDICT: UNSTABLE -- verdict flips with the random seed alone. "
              "Not a real, repeatable edge. Do not report either a PASS or FAIL "
              "from any single run of Addendum 49 as h2h's verdict.")
    elif n_pass == N_RUNS:
        print("  VERDICT: every seed passes -- worth a closer look, but still not "
              "the same as an independent holdout confirmation.")
    else:
        print("  VERDICT: every seed fails -- Addendum 49's single PASS-looking run "
              "(if any) was the outlier, not the signal. FAIL stands.")
    print(f"\ntotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
