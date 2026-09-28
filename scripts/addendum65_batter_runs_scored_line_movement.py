"""
Addendum 65: line-movement (CLV-direction) feature for batter_runs_scored
-- same rationale and method as Addendum 60 (pitcher_strikeouts): every
prior attempt on this market used player/team-quality features, and
Addendum 62's EDA showed those have ~zero correlation with the actual
outcome once the market's own price already reflects them. Line movement
describes MARKET BEHAVIOR itself, a genuinely different signal category
not yet tried for this market.

Data: db/cache_features.duckdb table
cache_odds_snapshots_batter_runs_scored, freshly pulled from the client's
live cache_odds_snapshots table (market key 'batter_runs_scored' --
confirmed correct, no naming mismatch this time). 1,073,108 snapshot
rows, 2026-06-20 -> 2026-09-19, 1,218 unique events, 580 unique players.

event_id -> game_pk bridge: reuses data_raw/event_id_mapping_pitcher_k_gap.csv
unmodified -- event_id is a per-GAME identifier from the Odds API, shared
across every market requested for that game, so the same mapping built
for the pitcher_strikeouts gap backfill applies here too (confirmed:
737 of 1,218 events overlap, same as pitcher_k's 738).

CRITICAL (same leakage bug found and fixed in Addendum 56/56b/60):
restrict to snapshot_time <= game_time BEFORE any aggregation.

Ground rules (same as 56/56b/60): simple logistic regression, ZERO
hyperparameter search, single pre-specified chronological 70/30 split,
single edge>0.0 cut, official gate rule, evaluated exactly once.
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
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob, norm_name  # noqa: E402
from addendum60_pitcher_k_line_movement import prob_to_american  # noqa: E402

CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")
MAPPING = ROOT / "data_raw" / "event_id_mapping_pitcher_k_gap.csv"  # game-level, market-agnostic


def build_movement_features(con2):
    snaps = con2.execute("""
        SELECT event_id, player_name, pick_side, line, odds, snapshot_time, game_time
        FROM cache_odds_snapshots_batter_runs_scored
        WHERE snapshot_time <= game_time
    """).fetchdf()
    snaps["implied"] = implied_prob(snaps["odds"].values)
    key = ["event_id", "player_name", "pick_side"]

    first_t = snaps.groupby(key)["snapshot_time"].transform("min")
    last_t = snaps.groupby(key)["snapshot_time"].transform("max")
    opening = snaps[snaps.snapshot_time == first_t].groupby(key)["implied"].median().rename("open_prob")
    closing = snaps[snaps.snapshot_time == last_t].groupby(key)["implied"].median().rename("close_prob")
    n_snaps = snaps.groupby(key)["snapshot_time"].nunique().rename("n_snapshots")
    closing_line = snaps[snaps.snapshot_time == last_t].groupby(key)["line"].median().rename("closing_line")

    mv = pd.concat([opening, closing, n_snaps, closing_line], axis=1).reset_index()
    mv["movement"] = mv["close_prob"] - mv["open_prob"]
    return mv


def build_pool(con, con2):
    mv = build_movement_features(con2)
    id_map = pd.read_csv(MAPPING)[["game_pk", "event_id"]].drop_duplicates(subset=["event_id"])
    mv = mv.merge(id_map, on="event_id", how="inner")
    mv["name_norm"] = mv["player_name"].map(norm_name)

    box = con.execute("""
        SELECT game_pk, player_name, runs_scored FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    box = box.dropna(subset=["runs_scored"]).drop_duplicates(subset=["game_pk", "name_norm"])

    pool = mv.merge(box[["game_pk", "name_norm", "runs_scored"]], on=["game_pk", "name_norm"], how="inner")
    pool = pool.dropna(subset=["runs_scored", "closing_line"])
    pool["win"] = np.where(pool["pick_side"] == "over", pool["runs_scored"] > pool["closing_line"],
                            pool["runs_scored"] < pool["closing_line"])
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

    print("building line-movement features from cache_odds_snapshots_batter_runs_scored...")
    pool = build_pool(con, con2)
    print(f"  final pool with real movement + real runs_scored outcomes: n={len(pool)}  "
          f"date range {pool.game_date.min()} -> {pool.game_date.max()}")

    if len(pool) < 100:
        print(f"\n  INSUFFICIENT DATA (n={len(pool)}) for a meaningful holdout -- UNDERPOWERED, not PASS/FAIL")
        return

    cut = int(len(pool) * 0.70)
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"\n  chronological 70/30 split: train n={len(train)} ({train.game_date.min()}->{train.game_date.max()})  "
          f"test n={len(test)} ({test.game_date.min()}->{test.game_date.max()})")

    FEATURES = ["market_prob", "movement", "n_snapshots"]
    if len(train) < 50 or len(test) < 20:
        print("  INSUFFICIENT DATA after split -- UNDERPOWERED")
        return

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train["win"].astype(int))

    t = test.copy()
    t["model_prob"] = model.predict_proba(t[FEATURES])[:, 1]
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]
    if len(sub) == 0:
        print("\n  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    coefs = dict(zip(FEATURES, model.named_steps["logisticregression"].coef_[0]))
    print(f"\n  logistic regression coefficients (standardized): {coefs}")
    print(f"\n  === RESULT (single pre-registered cut, official gate rule, evaluated ONCE) ===")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
