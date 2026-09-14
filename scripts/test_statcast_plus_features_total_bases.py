"""
One honest, single-shot test for total_bases: combine real statcast quality
features (exit velocity, barrel rate, expected-vs-actual SLG/wOBA gap --
tested standalone and negative in Addendum 19, but never combined with the
production score_* factors in one model) into the same real, production
score_* factor set Model A used, over the one window both actually exist in
(2026-05-17 -> 2026-07-28, pick_history's own entry odds -- real warehouse
odds for total_bases only run to 2026-05-24, nowhere near enough overlap
with statcast's 2026-05-20 start).

Single 75/25 chronological split, ONE edge threshold reported (edge>0.0,
no threshold search) -- reported honestly regardless of outcome, per
explicit agreement not to search for a passing cut.
"""
import pathlib
import unicodedata

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def norm_name(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def norm_statcast_name(s):
    """statcast_exit_velo/statcast_xstats store 'Last, First' -- pick_history
    uses 'First Last'. Confirmed directly: 0/528 names matched before this
    fix (different format, not different players)."""
    if s is None:
        return None
    s = str(s).strip()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return norm_name(s)


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
    if s["n"] is None or s["n"] == 0:
        return False
    if s["n"] >= MIN_GRADED:
        return s["roi"] is not None and s["roi"] > 0
    return s["lo"] is not None and s["lo"] > 0


def to_float(series):
    return pd.to_numeric(series, errors="coerce")


def main():
    con = duckdb.connect(DB, read_only=True)

    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    select_cols = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in score_cols)

    df = con.execute(f"""
        SELECT player_name, TRY_CAST(game_date AS DATE) AS game_date,
               TRY_CAST(odds AS INTEGER) AS odds,
               lower(hit) IN ('true','t','1') AS win,
               {select_cols}
        FROM pick_history
        WHERE mlb_market_type = 'batter_total_bases'
          AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchdf()
    df["name_norm"] = df["player_name"].map(norm_name)
    print(f"pick_history batter_total_bases: n={len(df):,}, {df.game_date.min()} -> {df.game_date.max()}")

    exit_velo = con.execute("""
        SELECT player_name, snapshot_date, avg_hit_speed, max_hit_speed,
               anglesweetspotpercent, brl_percent, ev95percent
        FROM statcast_exit_velo
    """).fetchdf()
    for c in ["avg_hit_speed", "max_hit_speed", "anglesweetspotpercent", "brl_percent", "ev95percent"]:
        exit_velo[c] = to_float(exit_velo[c])
    exit_velo["name_norm"] = exit_velo["player_name"].map(norm_statcast_name)
    exit_velo["snapshot_date"] = pd.to_datetime(exit_velo["snapshot_date"]).astype("datetime64[ns]")

    xstats = con.execute("""
        SELECT player_name, snapshot_date, slg, est_slg, est_slg_minus_slg_diff,
               woba, est_woba, est_woba_minus_woba_diff
        FROM statcast_xstats
    """).fetchdf()
    for c in ["slg", "est_slg", "est_slg_minus_slg_diff", "woba", "est_woba", "est_woba_minus_woba_diff"]:
        xstats[c] = to_float(xstats[c])
    xstats["name_norm"] = xstats["player_name"].map(norm_statcast_name)
    xstats["snapshot_date"] = pd.to_datetime(xstats["snapshot_date"]).astype("datetime64[ns]")
    con.close()

    df["game_date"] = pd.to_datetime(df["game_date"]).astype("datetime64[ns]")

    # Point-in-time join: most recent statcast snapshot strictly BEFORE the
    # pick's own game_date -- no leakage, same discipline as every other
    # point-in-time feature in this project.
    ev_sorted = exit_velo.sort_values("snapshot_date")
    xs_sorted = xstats.sort_values("snapshot_date")

    merged = pd.merge_asof(
        df.sort_values("game_date"),
        ev_sorted[["name_norm", "snapshot_date", "avg_hit_speed", "max_hit_speed",
                   "anglesweetspotpercent", "brl_percent", "ev95percent"]].rename(
            columns={"snapshot_date": "ev_snap"}),
        left_on="game_date", right_on="ev_snap",
        left_by="name_norm", right_by="name_norm",
        direction="backward", allow_exact_matches=False,
    )
    merged = pd.merge_asof(
        merged.sort_values("game_date"),
        xs_sorted[["name_norm", "snapshot_date", "slg", "est_slg", "est_slg_minus_slg_diff",
                   "woba", "est_woba", "est_woba_minus_woba_diff"]].rename(
            columns={"snapshot_date": "xs_snap"}),
        left_on="game_date", right_on="xs_snap",
        left_by="name_norm", right_by="name_norm",
        direction="backward", allow_exact_matches=False,
    )

    statcast_features = [
        "avg_hit_speed", "max_hit_speed", "anglesweetspotpercent", "brl_percent", "ev95percent",
        "slg", "est_slg", "est_slg_minus_slg_diff", "woba", "est_woba", "est_woba_minus_woba_diff",
    ]
    has_statcast = merged[statcast_features].notna().any(axis=1)
    print(f"rows with a point-in-time statcast match: {has_statcast.sum():,} / {len(merged):,}")
    merged = merged[has_statcast].reset_index(drop=True)

    if len(merged) < MIN_GRADED:
        print(f"INSUFFICIENT DATA after requiring a real statcast match: n={len(merged)} < {MIN_GRADED}")
        print("Honest result: this combination cannot be tested at a trustworthy sample size. Not reported as a finding.")
        return

    keep_score = [c for c in score_cols if merged[c].notna().sum() >= max(200, len(merged) * 0.05)]
    features = keep_score + statcast_features
    print(f"features used: {len(keep_score)} score_* factors + {len(statcast_features)} statcast quality features")

    merged = merged.sort_values("game_date").reset_index(drop=True)
    split = int(len(merged) * 0.75)
    train, test = merged.iloc[:split].copy(), merged.iloc[split:].copy()
    print(f"TRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=15,
        reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred >= 0.5)
    print(f"\nAUC TEST={auc:.3f}  accuracy TEST={acc:.3f}")

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\ntop 10 feature weights:")
    for f, v in imp.head(10).items():
        tag = " [statcast]" if f in statcast_features else ""
        print(f"  {f:<40} {v:.4f}{tag}")

    test = test.copy()
    test["model_prob"] = test_pred
    test["market_prob"] = implied_prob(test["odds"])
    test["edge"] = test["model_prob"] - test["market_prob"]

    # ONE pre-specified cut. No threshold search.
    sub = test[(test.odds < 0) & (test.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\n-- single pre-specified cut: edge>0.0, minus-money only --")
    print(f"n={s['n']}  WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")


if __name__ == "__main__":
    main()
