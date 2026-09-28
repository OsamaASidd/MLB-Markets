"""
Addendum 82: discrete-time survival/hazard model for pitcher_outs --
genuinely different from Addendum 78's Negative Binomial count regression.
NB predicts the total count in one shot; this instead models the actual
data-generating mechanism: at each out recorded, what's the hazard the
manager pulls this pitcher before the next out?

Person-period expansion: a start with K outs recorded becomes K rows
(step=1..K), label=1 only on the final row (pulled here), 0 otherwise.
Hazard model: logistic regression on (step, step^2, existing point-in-time
features). Survival function P(T > k) = prod_{j=1}^{k} (1 - h_j), used to
derive P(over line) = P(T > floor(line)).

Uses the SAME corrected, point-in-time feature set as Addendum 74/78
(fixed starter-quality rolling ERA, real bullpen/park enrichment) --
only the model family/target formulation changes. No dependency on the
just-audited (and blocked) manager_hook/pen_rest tables -- those would
need their own leakage check before use, which they fail (see chat).
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum78_pitcher_outs_negbinom import build_unique_starts_pool, FEATURES  # noqa: E402

N_FOLDS = 5


def build_person_period(pool):
    """Expand each start into (step=1..outs) rows, label=1 on the final
    (pulled-here) row only."""
    rows = []
    for r in pool.itertuples():
        k = int(r.outs)
        if k < 1:
            continue
        steps = np.arange(1, k + 1)
        df = pd.DataFrame({"step": steps})
        df["label"] = 0
        df.iloc[-1, df.columns.get_loc("label")] = 1
        df["game_pk"] = r.game_pk
        df["game_date"] = r.game_date
        for f in FEATURES:
            df[f] = getattr(r, f)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def fit_hazard(train_pp, features):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(train_pp[features])
    Xs = scaler.fit_transform(X_imp)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xs, train_pp["label"])
    return imputer, scaler, lr


def survival_p_over(imputer, scaler, lr, row, features, line, max_step=27):
    """P(T > floor(line)) = prod_{j=1}^{floor(line)} (1 - hazard_j).
    `features` = the game-level FEATURES only (no 'step' on the source
    row -- that's added below to match the haz_features column order
    the imputer/scaler/model were fit on: FEATURES + ['step'])."""
    k = int(np.floor(line))
    k = min(k, max_step)
    if k <= 0:
        return 0.999
    steps = np.arange(1, k + 1)
    base = pd.DataFrame([row[features].to_dict()] * k)
    base["step"] = steps
    X = scaler.transform(imputer.transform(base[features + ["step"]]))
    hz = lr.predict_proba(X)[:, 1]
    surv = np.prod(1 - hz)
    return float(np.clip(surv, 1e-6, 1 - 1e-6))


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building unique real-starts pool (reusing Addendum 78's corrected features)...")
    pool = build_unique_starts_pool(con, con2)
    haz_features = FEATURES + ["step"]
    print(f"  pool: n={len(pool)}")

    pool = pool.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), discrete-time hazard model ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        train_pp = build_person_period(train_df)
        print(f"  fold {f}: person-period expansion train n={len(train_pp)}")
        imputer, scaler, lr = fit_hazard(train_pp, haz_features)

        for _, row in test_df.iterrows():
            p_over = survival_p_over(imputer, scaler, lr, row, FEATURES, row["line"])
            for side, model_p, odds_col in [("over", p_over, "best_over_odds"), ("under", 1 - p_over, "best_under_odds")]:
                win = (row["outs"] > row["line"]) if side == "over" else (row["outs"] < row["line"])
                market_p = row["fair_over_prob"] if side == "over" else (1 - row["fair_over_prob"])
                edge = model_p - market_p
                all_bets.append({"game_date": row["game_date"], "odds": row[odds_col], "win": win, "edge": edge})

    pooled = pd.DataFrame(all_bets)
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
