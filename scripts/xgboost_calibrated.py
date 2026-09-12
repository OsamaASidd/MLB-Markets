"""
Addendum 20 found total_bases has real out-of-sample ranking skill (TEST
AUC 0.626) that didn't turn into a profitable edge. One real, unexplored
explanation: the raw model outputs a score that ranks outcomes correctly
but isn't a properly-calibrated probability -- so comparing it directly to
the market's implied probability produces a wrong "edge" even when the
ranking itself is right. This fits calibration on its own carved-out slice
to test that honestly.

Three-way temporal split (fixes the leakage risk of calibrating on the
same data the model was fit on):
  - FIT   (first 50% of dates)  -- train the XGBoost model
  - CAL   (next 20% of dates)   -- fit an isotonic calibration mapping on
                                   the model's raw outputs vs actual outcomes
  - TEST  (last 30% of dates)   -- true final holdout, touched only once,
                                   at the end, for the honest answer

Reports calibration quality (a reliability table: predicted-probability
bucket vs actual win rate) on TEST, then re-runs the same edge-threshold
betting sweep as Addendum 20 using the calibrated probabilities.
"""
import pathlib
import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500
MARKET = "total_bases"


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


def main():
    con = duckdb.connect(DB, read_only=True)
    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    select_cols = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in score_cols)
    df = con.execute(f"""
        SELECT TRY_CAST(game_date AS DATE) AS game_date,
               TRY_CAST(odds AS INTEGER) AS odds,
               lower(hit) IN ('true','t','1') AS win,
               {select_cols}
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL AND prop_type = '{MARKET}'
          AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchdf()
    df = df.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)
    con.close()

    keep = [c for c in score_cols if df[c].notna().sum() >= max(200, len(df) * 0.05)]
    n = len(df)
    fit_end, cal_end = int(n * 0.5), int(n * 0.7)
    fit_df, cal_df, test_df = df.iloc[:fit_end], df.iloc[fit_end:cal_end], df.iloc[cal_end:]
    print(f"{MARKET}: n={n:,}  FIT={len(fit_df):,}  CAL={len(cal_df):,}  TEST={len(test_df):,}")

    X_fit, y_fit = fit_df[keep], fit_df["win"].astype(int)
    X_cal, y_cal = cal_df[keep], cal_df["win"].astype(int)
    X_test, y_test = test_df[keep], test_df["win"].astype(int)

    base = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_lambda=2.0, eval_metric="logloss", missing=np.nan, random_state=0,
    )
    base.fit(X_fit, y_fit)

    raw_test_pred = base.predict_proba(X_test)[:, 1]
    raw_cal_pred = base.predict_proba(X_cal)[:, 1]
    print(f"\nraw model AUC on TEST: {roc_auc_score(y_test, raw_test_pred):.3f}")
    print(f"raw model Brier score on TEST (lower=better calibrated): {brier_score_loss(y_test, raw_test_pred):.4f}")

    calibrator = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrator.fit(X_cal, y_cal)
    cal_test_pred = calibrator.predict_proba(X_test)[:, 1]
    print(f"calibrated model AUC on TEST: {roc_auc_score(y_test, cal_test_pred):.3f}  (ranking is unchanged by calibration, this should match)")
    print(f"calibrated model Brier score on TEST: {brier_score_loss(y_test, cal_test_pred):.4f}")

    print("\n-- reliability check on TEST: predicted-probability bucket vs actual win rate --")
    test_df = test_df.copy()
    test_df["raw_pred"] = raw_test_pred
    test_df["cal_pred"] = cal_test_pred
    for lo, hi in [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        sub = test_df[(test_df.raw_pred >= lo) & (test_df.raw_pred < hi)]
        if len(sub) < 20:
            continue
        print(f"  raw_pred [{lo},{hi})  n={len(sub):<6} mean_raw_pred={sub.raw_pred.mean():.3f}  "
              f"mean_cal_pred={sub.cal_pred.mean():.3f}  actual_WR={sub.win.mean():.3f}")

    test_df["market_prob"] = implied_prob(test_df["odds"])
    test_df["edge_raw"] = test_df["raw_pred"] - test_df["market_prob"]
    test_df["edge_cal"] = test_df["cal_pred"] - test_df["market_prob"]

    for label, edge_col in [("RAW model edge", "edge_raw"), ("CALIBRATED model edge", "edge_cal")]:
        print(f"\n-- betting on {label}, minus-money only, TRUE final TEST holdout --")
        for thresh in [0.0, 0.03, 0.05, 0.08, 0.10]:
            sub = test_df[(test_df.odds < 0) & (test_df[edge_col] > thresh)]
            profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
            s = stat(profits, sub["win"])
            passed = gate(s)
            print(f"    edge>{thresh:<5} n={s['n']:<6} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")


if __name__ == "__main__":
    main()
