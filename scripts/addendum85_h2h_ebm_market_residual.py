"""
Addendum 85: Explainable Boosting Machine (EBM) on the market residual
for h2h -- keeps the market-offset philosophy (Addendum 72) but replaces
the ridge GLM's strictly linear terms with smooth, per-feature learned
shape functions (an additive model, heavily regularized by construction,
unlike a deep XGBoost tree that can overfit joint interactions).

Simplification, disclosed: EBM (interpret library) doesn't support a
literal fixed-weight-1.0 offset like statsmodels GLM does. Instead,
logit(de-vigged market_prob) is included as one of the model's own input
features -- the EBM's additive structure means it still gets its own
independent shape function, and in practice tends to learn something
close to a near-linear, near-unit-weight relationship for a
well-calibrated input like this, but it is NOT constrained to exactly
that the way Addendum 72's offset was. Reported honestly either way.

Same walk-forward discipline: EBM's own internal regularization
(interaction limit, smoothing rounds) is fixed to reasonable defaults
(not tuned), single edge>0.0 cut, gate touched once on the pooled
held-out set.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.impute import SimpleImputer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum72_h2h_market_offset_glm import build_devigged_pool, FEATURES, logit  # noqa: E402

N_FOLDS = 5
EBM_FEATURES = FEATURES + ["market_logit"]


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    pool = build_devigged_pool(con, con2)
    pool["market_logit"] = logit(pool["market_prob"].values)
    print(f"\nfinal de-vigged h2h pool: n={len(pool)}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), EBM market-residual model ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f].copy()
        test_df = pool[pool["fold"] == f].copy()
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train_df[EBM_FEATURES])
        y_train = train_df["win"].astype(int).values

        ebm = ExplainableBoostingClassifier(
            interactions=5, max_bins=64, learning_rate=0.02,
            min_samples_leaf=20, random_state=0, n_jobs=1,
        )
        ebm.fit(X_train, y_train)

        X_test = imputer.transform(test_df[EBM_FEATURES])
        test_pred = ebm.predict_proba(X_test)[:, 1]

        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
