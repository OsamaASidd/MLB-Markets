"""
Proper gradient-boosted model (XGBoost) on the real score_* factors already
logged in pick_history, replacing the crude linear z-score reweight from
Addendum 2/8 with something that can actually learn non-linear interactions
and find the best-suited weight on each factor itself (feature_importances_
is the literal answer to "weight the features").

Per market:
  1. Load every score_* column + odds + game_date + hit outcome.
  2. Split by date: first 60% = TRAIN, last 40% = TEST (never shuffled --
     this is a temporal holdout, same discipline as every other addendum).
  3. Fit XGBClassifier on TRAIN only, regularized against overfitting
     (shallow trees, subsampling) given the row-to-feature ratio.
  4. Evaluate raw discrimination (AUC) on TEST -- can the model tell
     winners from losers at all on data it never saw?
  5. Turn it into a bet: edge = model's predicted win probability minus
     the market's own implied probability from the real logged odds. Bet
     when edge clears a threshold, minus-money only (same convention as
     the rest of this report). Compute real ROI on TEST with a 95% CI.
  6. Report feature importances -- the actual learned weights.

This is real modeling work. It will report whatever TEST actually shows,
including if that's negative -- an out-of-sample ROI number from this
script is only meaningful if it comes with its own CI and its own honest
failure mode disclosed the same as every other result in this repo.
"""
import pathlib
import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def profit(win, odds):
    if not win:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_prob(odds):
    return np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100))


def stat(profits, wins):
    n = len(profits)
    if n == 0:
        return {"n": 0, "roi": None, "lo": None, "hi": None, "wr": None}
    m = profits.mean() * 100
    se = profits.std(ddof=1) / np.sqrt(n) * 100 if n > 1 else np.nan
    return {"n": n, "wr": round(wins.mean() * 100, 1), "roi": round(m, 2),
            "lo": round(m - 1.96 * se, 2) if not np.isnan(se) else None,
            "hi": round(m + 1.96 * se, 2) if not np.isnan(se) else None}


def gate(s):
    return s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0


def run_market(con, market):
    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    select_cols = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in score_cols)
    df = con.execute(f"""
        SELECT TRY_CAST(game_date AS DATE) AS game_date,
               TRY_CAST(odds AS INTEGER) AS odds,
               lower(hit) IN ('true','t','1') AS win,
               {select_cols}
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL AND prop_type = '{market}'
          AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchdf()
    df = df.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    # keep only factor columns with reasonable non-null coverage for this market
    keep = [c for c in score_cols if df[c].notna().sum() >= max(200, len(df) * 0.05)]
    print(f"\n{'='*100}\n{market}  (n={len(df):,}, {len(keep)} usable factor columns)\n{'='*100}")
    if len(df) < 400 or len(keep) < 3:
        print("  too few rows or factors -- skipped")
        return None

    split = int(len(df) * 0.6)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    X_train, X_test = train[keep], test[keep]
    y_train, y_test = train["win"].astype(int), test["win"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_lambda=2.0, eval_metric="logloss", missing=np.nan,
        random_state=0,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, train_pred)
    auc_test = roc_auc_score(y_test, test_pred)
    print(f"  AUC  TRAIN={auc_train:.3f}   TEST={auc_test:.3f}   (0.500 = no better than chance)")

    imp = pd.Series(model.feature_importances_, index=keep).sort_values(ascending=False)
    print("  top 10 learned feature weights:")
    for f, v in imp.head(10).items():
        print(f"    {f:<40} {v:.4f}")

    test = test.copy()
    test["model_prob"] = test_pred
    test["market_prob"] = implied_prob(test["odds"])
    test["edge"] = test["model_prob"] - test["market_prob"]

    print("\n  -- betting on model edge, minus-money only, TEST (holdout) --")
    results = []
    for thresh in [0.0, 0.03, 0.05, 0.08, 0.10]:
        sub = test[(test.odds < 0) & (test.edge > thresh)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        passed = gate(s)
        print(f"    edge>{thresh:<5} n={s['n']:<6} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")
        results.append((thresh, s))
    return {"market": market, "auc_train": auc_train, "auc_test": auc_test, "results": results}


def main():
    con = duckdb.connect(DB, read_only=True)
    for market in ["hits", "total_bases", "pitcher_strikeouts", "rbis", "runs_scored"]:
        run_market(con, market)
    con.close()


if __name__ == "__main__":
    main()
