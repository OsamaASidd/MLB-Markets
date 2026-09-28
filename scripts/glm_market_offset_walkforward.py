"""
Shared market-offset ridge-logistic GLM walk-forward runner, factored out
of Addendum 72 so the same architecture can be applied to every
remaining failed market without re-deriving the CV/fold/gate logic each
time. Takes an already-built pool (must have: game_date, game_pk, odds,
win, market_prob [de-vigged where possible], and the given feature
columns) and runs the identical procedure Addendum 72 used for h2h.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from xgboost_individual_markets import gate, stat, profit  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cv_select_alpha(X_train, y_train, offset_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = {a: [] for a in ALPHA_GRID}
    for tr_idx, va_idx in cv.split(X_train):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        otr, ova = offset_train[tr_idx], offset_train[va_idx]
        Xtr_c = sm.add_constant(Xtr, has_constant="add")
        Xva_c = sm.add_constant(Xva, has_constant="add")
        for a in ALPHA_GRID:
            try:
                res = sm.GLM(ytr, Xtr_c, family=sm.families.Binomial(), offset=otr).fit_regularized(alpha=a, L1_wt=0.0)
                pred = res.predict(Xva_c, offset=ova)
                scores[a].append(roc_auc_score(yva, pred))
            except Exception:
                scores[a].append(np.nan)
    mean_scores = {a: np.nanmean(v) for a, v in scores.items()}
    best_alpha = max(mean_scores, key=mean_scores.get)
    return best_alpha, mean_scores[best_alpha]


def run_walkforward(pool, features, market_name, n_folds=5, hold_col=None):
    pool = pool.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), n_folds, labels=False)
    pool = pool.copy()
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== {market_name}: walk-forward (expanding window, {n_folds} folds), market-offset ridge-logistic GLM ===")
    for f in range(1, n_folds):
        train_df = pool[pool["fold"] < f].copy()
        test_df = pool[pool["fold"] == f].copy()
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(train_df[features]))
        y_train = train_df["win"].astype(int).values
        offset_train = logit(train_df["market_prob"].values)

        best_alpha, cv_auc = cv_select_alpha(X_train, y_train, offset_train)
        print(f"  fold {f}: CV_AUC={cv_auc:.4f}  best_ridge_alpha={best_alpha}")

        X_train_c = sm.add_constant(X_train, has_constant="add")
        final_model = sm.GLM(y_train, X_train_c, family=sm.families.Binomial(), offset=offset_train).fit_regularized(
            alpha=best_alpha, L1_wt=0.0)

        X_test = scaler.transform(imputer.transform(test_df[features]))
        X_test_c = sm.add_constant(X_test, has_constant="add")
        offset_test = logit(test_df["market_prob"].values)
        test_pred = final_model.predict(X_test_c, offset=offset_test)

        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        cols = ["game_date", "odds", "win", "edge", "model_prob", "market_prob"]
        if hold_col:
            cols.append(hold_col)
        all_bets.append(t[cols])

    if not all_bets:
        print(f"  {market_name}: INSUFFICIENT DATA across all folds")
        return None
    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== {market_name} pooled walk-forward held-out bets across folds 1-{n_folds - 1}: n={len(pooled)} ===")

    # Diagnostic: does the model beat the de-vigged closing price on proper
    # scoring (log-loss), on the FULL held-out set (not just qualifying
    # bets, to avoid selection bias in this specific comparison)?
    y_all = pooled["win"].astype(int)
    ll_model = log_loss(y_all, pooled["model_prob"].clip(1e-6, 1 - 1e-6))
    ll_market = log_loss(y_all, pooled["market_prob"].clip(1e-6, 1 - 1e-6))
    print(f"\n  DIAGNOSTIC: log-loss on full held-out set (n={len(pooled)}) -- "
          f"model={ll_model:.4f}  market(de-vigged close)={ll_market:.4f}  "
          f"{'model beats close' if ll_model < ll_market else 'market close still better'} "
          f"(delta={ll_market - ll_model:+.4f}, positive = model better)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print(f"  {market_name}: edge>0.0 cut produced ZERO bets -- not evaluated")
        return None
    if hold_col:
        mean_hold = sub[hold_col].mean()
        print(f"  DIAGNOSTIC: mean hold (vig) on the {len(sub)} qualifying bets = {round(100*mean_hold, 2)}%")
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\n{market_name} edge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"market": market_name, **s, "verdict": verdict,
            "log_loss_model": ll_model, "log_loss_market": ll_market}
