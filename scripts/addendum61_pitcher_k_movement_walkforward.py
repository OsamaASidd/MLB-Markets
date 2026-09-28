"""
Addendum 61: walk-forward validation of the Addendum 60 line-movement
signal for pitcher_strikeouts. The single pre-registered 70/30 split
(Addendum 60) only left n=36 bets in the held-out set after the edge>0.0
cut -- too small to draw any conclusion from (CI=[-20.62,32.35]). This is
the same standard second-check applied to every single-split result all
session (Addendum 47, 49, etc), run regardless of whether the first
split passed or failed -- NOT a re-roll after a bad result. Expanding-
window folds pool far more held-out bets (up to n=708 rows total,
across multiple folds) than one static split can.

Same simple, untuned logistic regression as Addendum 60 (no
hyperparameter search -- there's nothing to tune with 3 features), same
single edge>0.0 cut, official gate rule, evaluated exactly once on the
pooled held-out predictions.
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
from xgboost_individual_markets import DB, gate, stat, profit, norm_name  # noqa: E402
from addendum60_pitcher_k_line_movement import CACHE_DB, MAPPING, build_movement_features, prob_to_american  # noqa: E402

N_FOLDS = 5
FEATURES = ["market_prob", "movement", "n_snapshots"]


def build_pool(con, con2):
    mv = build_movement_features(con2)
    id_map = pd.read_csv(MAPPING)[["game_pk", "event_id"]].drop_duplicates(subset=["event_id"])
    mv = mv.merge(id_map, on="event_id", how="inner")
    mv["name_norm"] = mv["player_name"].map(norm_name)

    box = con.execute("""
        SELECT game_pk, player_name, strikeouts FROM boxscore
        WHERE position_type = 'Pitcher' AND is_starter = True
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    box = box.drop_duplicates(subset=["game_pk", "name_norm"])

    pool = mv.merge(box[["game_pk", "name_norm", "strikeouts"]], on=["game_pk", "name_norm"], how="inner")
    pool = pool.dropna(subset=["strikeouts", "closing_line"])
    pool["win"] = np.where(pool["pick_side"] == "over", pool["strikeouts"] > pool["closing_line"],
                            pool["strikeouts"] < pool["closing_line"])
    pool["market_prob"] = pool["close_prob"]
    pool = pool.dropna(subset=["movement", "win", "market_prob"])
    pool["odds"] = prob_to_american(pool["market_prob"].values)

    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")
    dates = con.execute("""
        SELECT game_pk, TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
    """).fetchdf().drop_duplicates(subset=["game_pk"])
    pool = pool.merge(dates, on="game_pk", how="left").dropna(subset=["game_date"])
    return pool.sort_values(["game_date", "game_pk", "pick_side", "name_norm"], kind="mergesort").reset_index(drop=True)


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
