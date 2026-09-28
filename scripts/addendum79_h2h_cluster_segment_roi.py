"""
Addendum 79: clustering + empirical segment ROI for h2h -- a genuinely
different paradigm from every regression-family model tried so far
(XGBoost, logistic regression, market-offset GLM, Negative Binomial).
Instead of fitting a function that predicts probability from features,
this partitions games into unsupervised similarity groups (KMeans on the
SAME clean point-in-time features used everywhere else -- elo/l10/
fatigue/bullpen/starter-rolling-ERA/park), blind to the outcome, and
checks whether the market-offset GLM's existing edge concentrates
profitably in any specific feature-space region that the pooled result
averages away.

GROUND RULES (stated up front, followed exactly, to avoid this becoming
subset-selection p-hacking):
  - Number of clusters FIXED at 5 BEFORE running anything. Not tuned,
    not chosen after seeing which K produces a nice-looking segment.
  - Clustering is fit on TRAIN data only per walk-forward fold (KMeans on
    standardized features, outcome/edge never used as a clustering
    input), and test rows are assigned to their nearest TRAIN centroid --
    same point-in-time discipline as every other walk-forward this
    session.
  - The betting rule (edge = market-offset GLM's model_prob minus
    de-vigged market_prob, bet when edge>0 and odds<0) is UNCHANGED from
    Addendum 72 -- clustering only partitions the SAME already-computed
    bets after the fact, it does not add a new free parameter to the
    betting decision itself.
  - ALL 5 clusters are reported below, every time, not just the best one.
    With 5 segments tested, a single segment clearing the gate on its own
    has a real, quantifiable higher chance of being noise than the pooled
    result -- flagged explicitly, not glossed over.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum72_h2h_market_offset_glm import build_devigged_pool, FEATURES, ALPHA_GRID, logit, cv_select_alpha  # noqa: E402

N_FOLDS = 5
N_CLUSTERS = 5  # fixed BEFORE running -- see docstring


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    pool = build_devigged_pool(con, con2)
    print(f"\nfinal de-vigged h2h pool: n={len(pool)}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds): GLM edge + blind KMeans segment ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f].copy()
        test_df = pool[pool["fold"] == f].copy()
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(train_df[FEATURES]))
        y_train = train_df["win"].astype(int).values
        offset_train = logit(train_df["market_prob"].values)

        best_alpha, cv_auc = cv_select_alpha(X_train, y_train, offset_train)
        print(f"  fold {f}: CV_AUC={cv_auc:.4f}  best_ridge_alpha={best_alpha}")
        X_train_c = sm.add_constant(X_train, has_constant="add")
        final_model = sm.GLM(y_train, X_train_c, family=sm.families.Binomial(), offset=offset_train).fit_regularized(
            alpha=best_alpha, L1_wt=0.0)

        X_test = scaler.transform(imputer.transform(test_df[FEATURES]))
        X_test_c = sm.add_constant(X_test, has_constant="add")
        offset_test = logit(test_df["market_prob"].values)
        test_pred = final_model.predict(X_test_c, offset=offset_test)

        # Blind clustering: fit on TRAIN features only, outcome/edge never used.
        km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
        km.fit(X_train)
        test_cluster = km.predict(X_test)

        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        t["cluster"] = test_cluster
        all_bets.append(t[["game_date", "odds", "win", "edge", "cluster"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")

    sub_all = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits_all = sub_all.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s_all = stat(profits_all, sub_all["win"])
    verdict_all = "PASS" if gate(s_all) else "FAIL"
    print(f"\nOVERALL pooled result (same as Addendum 72, reproduced): "
          f"n={s_all['n']}  ROI={s_all['roi']}%  CI=[{s_all['lo']},{s_all['hi']}]  {verdict_all}")

    print(f"\n=== PER-CLUSTER breakdown (all {N_CLUSTERS} clusters shown, none cherry-picked) ===")
    print("NOTE: with 5 segments tested, a single segment clearing the gate on its own carries a real "
          "multiple-comparisons risk that the pooled result doesn't -- treat any per-cluster PASS below as "
          "a lead requiring fresh confirmatory data, not a standalone finding.")
    results = []
    for c in range(N_CLUSTERS):
        sub_c = sub_all[sub_all.cluster == c]
        if len(sub_c) < 10:
            print(f"\n  cluster {c}: n={len(sub_c)} -- too few qualifying bets to evaluate")
            continue
        profits_c = sub_c.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s_c = stat(profits_c, sub_c["win"])
        verdict_c = "PASS" if gate(s_c) else "FAIL"
        print(f"\n  cluster {c}: n={s_c['n']}  WR={s_c['wr']}%  ROI={s_c['roi']}%  CI=[{s_c['lo']},{s_c['hi']}]  {verdict_c}")
        results.append({"cluster": c, **s_c, "verdict": verdict_c})

    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n{n_pass} of {len(results)} evaluable clusters individually clear the gate.")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
