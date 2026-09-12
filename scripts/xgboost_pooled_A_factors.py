"""
Model A: one unified XGBoost model, pooled across ALL markets, with market
type itself as a feature (categorical) rather than a separate model per
market. Uses the full ~150-factor production feature set -- which only
exists for the real window it was actually logged in: 2026-05-17 to
2026-07-27 (confirmed by direct query; the 2023-2026 odds warehouse has no
factor columns at all, only raw odds+outcomes -- that's Model B).

75/25 split done chronologically within this window (there's only one
real "year" of factor data, so a per-year split isn't meaningful here --
Model B is where the per-year pooling happens across 2023-2026).

Reports honest accuracy/AUC on the holdout -- NOT targeting a specific
number. For a real win/loss prediction task like this, anything near 90%+
raw accuracy would itself be a red flag for data leakage, not a sign of
success; that's disclosed here rather than chased.
"""
import pathlib
import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500

MARKETS = ["hits", "total_bases", "rbis", "home_runs", "runs_scored",
           "pitcher_strikeouts", "pitcher_outs", "h2h", "spreads", "totals"]


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

    frames = []
    for m in MARKETS:
        df = con.execute(f"""
            SELECT '{m}' AS market, TRY_CAST(game_date AS DATE) AS game_date,
                   TRY_CAST(odds AS INTEGER) AS odds,
                   lower(hit) IN ('true','t','1') AS win,
                   {select_cols}
            FROM pick_history
            WHERE mlb_market_type IS NOT NULL AND prop_type = '{m}'
              AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
              AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
              AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
        """).fetchdf()
        frames.append(df)
    con.close()

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)
    print(f"pooled dataset: n={len(all_df):,} across {len(MARKETS)} markets, "
          f"{all_df.game_date.min()} -> {all_df.game_date.max()}")
    print(all_df.groupby("market").size().sort_values(ascending=False))

    all_df["market_code"] = all_df["market"].astype("category").cat.codes
    keep = [c for c in score_cols if all_df[c].notna().sum() >= max(500, len(all_df) * 0.02)]
    features = ["market_code"] + keep
    print(f"\nusable features: {len(features)} ({len(keep)} score_* factors + market_code)")

    split = int(len(all_df) * 0.75)
    train, test = all_df.iloc[:split].copy(), all_df.iloc[split:].copy()
    print(f"TRAIN n={len(train):,}  TEST n={len(test):,}  (75/25 chronological split)")

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=15,
        reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, train_pred)
    auc_test = roc_auc_score(y_test, test_pred)
    acc_train = accuracy_score(y_train, train_pred >= 0.5)
    acc_test = accuracy_score(y_test, test_pred >= 0.5)
    print(f"\nAUC  TRAIN={auc_train:.3f}  TEST={auc_test:.3f}")
    print(f"ACCURACY (0.5 threshold)  TRAIN={acc_train:.3f}  TEST={acc_test:.3f}")
    print("(for reference: ~90%+ raw accuracy on a real win/loss task like this would itself "
          "be a red flag for leakage, not a target -- not chased here, reported as-is)")

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\ntop 15 learned feature weights (pooled, cross-market):")
    for f, v in imp.head(15).items():
        print(f"  {f:<40} {v:.4f}")

    test = test.copy()
    test["model_prob"] = test_pred
    test["market_prob"] = implied_prob(test["odds"])
    test["edge"] = test["model_prob"] - test["market_prob"]

    print("\n-- pooled model, betting on edge, minus-money only, TEST holdout -- overall and per-market --")
    for thresh in [0.0, 0.03, 0.05, 0.08]:
        sub = test[(test.odds < 0) & (test.edge > thresh)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        print(f"  ALL MARKETS edge>{thresh:<5} n={s['n']:<6} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if gate(s) else 'fail'}")

    print("\n  per-market breakdown at edge>0.03:")
    for m in MARKETS:
        sub = test[(test.odds < 0) & (test.edge > 0.03) & (test.market == m)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        print(f"    {m:<20} n={s['n']:<6} ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if gate(s) else 'fail'}")


if __name__ == "__main__":
    main()
