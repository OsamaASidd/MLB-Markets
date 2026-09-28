"""
Addendum 69: walk-forward validation of the Addendum 56b h2h line-movement
signal. The single 70/30 split left only n=27 qualifying bets -- too
small to draw any conclusion. Same standard second-check as every other
single-split result this session: expanding-window folds pool far more
held-out predictions (up to n=1,460 total pool) than one static split.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import gate, stat, profit, implied_prob  # noqa: E402

DB = str(ROOT / "db" / "mlb_markets.duckdb")
CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")
N_FOLDS = 6
FEATURES = ["market_prob", "movement", "n_snapshots"]


def build_pool(con, con2):
    snaps = con2.execute("""
        SELECT event_id, pick_side, odds, snapshot_time, game_time
        FROM cache_odds_snapshots_game_side
        WHERE snapshot_time <= game_time
    """).fetchdf()
    snaps["implied"] = implied_prob(snaps["odds"].values)
    snaps["odds_decimal"] = np.where(snaps["odds"] > 0, snaps["odds"] / 100 + 1, 100 / (-snaps["odds"]) + 1)

    first_t = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].transform("min")
    last_t = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].transform("max")
    opening = snaps[snaps.snapshot_time == first_t].groupby(["event_id", "pick_side"])["implied"].median().rename("open_prob")
    closing_grp = snaps[snaps.snapshot_time == last_t]
    closing_odds = closing_grp.loc[closing_grp.groupby(["event_id", "pick_side"])["odds_decimal"].idxmax()][
        ["event_id", "pick_side", "odds"]].set_index(["event_id", "pick_side"])["odds"].rename("closing_odds")
    closing_prob = closing_grp.groupby(["event_id", "pick_side"])["implied"].median().rename("close_prob")
    n_snaps = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].nunique().rename("n_snapshots")

    mv = pd.concat([opening, closing_prob, closing_odds, n_snaps], axis=1).reset_index()
    mv["movement"] = mv["close_prob"] - mv["open_prob"]

    id_map = pd.read_csv(ROOT / "data_raw" / "event_id_mapping_h2h_totals_gap.csv")[["event_id", "game_pk"]].drop_duplicates(subset=["event_id"])
    mv = mv.merge(id_map, on="event_id", how="inner")

    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")
    outcomes = con.execute("""
        SELECT game_pk, home_score, away_score, TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true AND home_score IS NOT NULL AND away_score IS NOT NULL
    """).fetchdf().drop_duplicates(subset=["game_pk"])

    pool = mv.merge(outcomes, on="game_pk", how="inner")
    pool["win"] = np.where(pool["pick_side"] == "home", pool["home_score"] > pool["away_score"],
                            pool["away_score"] > pool["home_score"])
    pool["odds"] = pool["closing_odds"]
    pool["market_prob"] = implied_prob(pool["odds"].values)
    pool = pool.dropna(subset=["movement", "win", "odds", "game_date"])
    return pool.sort_values(["game_date", "event_id", "pick_side"], kind="mergesort").reset_index(drop=True)


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    pool = build_pool(con, con2)
    print(f"pool: n={len(pool)}  date range {pool.game_date.min()} -> {pool.game_date.max()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), untuned logistic regression ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 50 or len(test_df) < 10:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train_df[FEATURES], train_df["win"].astype(int))
        t = test_df.copy()
        t["model_prob"] = model.predict_proba(t[FEATURES])[:, 1]
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("\n  edge>0.0 cut produced ZERO bets -- not evaluated")
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
