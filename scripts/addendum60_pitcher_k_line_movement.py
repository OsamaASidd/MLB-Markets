"""
Addendum 60: line-movement (CLV-direction) feature for pitcher_strikeouts
-- the one genuinely untried signal class for this market (every prior
attempt used player/team-quality features, which Addendum 59's EDA
showed have ZERO correlation with the actual over/under outcome, because
the sportsbook's line-setting process already absorbed them). Line
movement describes MARKET BEHAVIOR itself (which way sharp money moved
the price), not player/team quality -- same rationale as Addendum 56/56b
for totals/h2h, applied here for the first time.

Data: db/cache_features.duckdb table cache_odds_snapshots_pitcher_k,
freshly pulled from the client's live cache_odds_snapshots table (market
key 'pitcher_k', NOT 'pitcher_strikeouts' -- confirmed via a distinct-
market sample query). 266,669 snapshot rows, 2026-06-20 -> 2026-09-19,
1,219 unique events, 326 unique pitchers.

CRITICAL (same leakage bug found and fixed in Addendum 56/56b): restrict
to snapshot_time <= game_time BEFORE any aggregation -- otherwise
"closing" includes post-game settled prices.

Feature (pre-registered, single, simple, explainable -- same as 56/56b):
  movement = implied_prob(closing_odds) - implied_prob(opening_odds)
  for the SAME (event_id, player_name, pick_side), using the median
  price across bookmakers at the earliest and latest snapshot_time.

event_id -> game_pk bridge: data_raw/event_id_mapping_pitcher_k_gap.csv
(already built for the Phase 1/2 backfill covering this exact window) --
738 of the 1,219 snapshot events resolve to a known real game_pk.

Ground rules (same as 56/56b): simple logistic regression, ZERO
hyperparameter search, single pre-specified chronological 70/30 split,
single edge>0.0 cut, official gate rule, evaluated exactly once. A FAIL
is a valid, honestly-reportable outcome.
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

CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")
MAPPING = ROOT / "data_raw" / "event_id_mapping_pitcher_k_gap.csv"


def prob_to_american(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)


def build_movement_features(con2):
    snaps = con2.execute("""
        SELECT event_id, player_name, pick_side, line, odds, snapshot_time, game_time
        FROM cache_odds_snapshots_pitcher_k
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


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building line-movement features from cache_odds_snapshots_pitcher_k...")
    mv = build_movement_features(con2)
    print(f"  {len(mv)} (event_id, player, pick_side) movement rows built "
          f"({mv.event_id.nunique()} unique events, median snapshots/pick={mv.n_snapshots.median():.0f})")

    id_map = pd.read_csv(MAPPING)[["game_pk", "event_id"]].drop_duplicates(subset=["event_id"])
    mv = mv.merge(id_map, on="event_id", how="inner")
    print(f"  after mapping event_id -> game_pk (real games only): n={len(mv)}, "
          f"{mv.event_id.nunique()} unique events, {mv.game_pk.nunique()} unique games")

    mv["name_norm"] = mv["player_name"].map(norm_name)

    box = con.execute("""
        SELECT game_pk, player_name, strikeouts FROM boxscore
        WHERE position_type = 'Pitcher' AND is_starter = True
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    box = box.drop_duplicates(subset=["game_pk", "name_norm"])

    n_before = len(mv)
    pool = mv.merge(box[["game_pk", "name_norm", "strikeouts"]], on=["game_pk", "name_norm"], how="inner")
    print(f"  row-count check (boxscore merge): before={n_before} after={len(pool)} "
          f"{'OK' if len(pool) <= n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) <= n_before

    pool = pool.dropna(subset=["strikeouts", "closing_line"])
    pool["win"] = np.where(pool["pick_side"] == "over", pool["strikeouts"] > pool["closing_line"],
                            pool["strikeouts"] < pool["closing_line"])
    pool["market_prob"] = pool["close_prob"]
    pool = pool.dropna(subset=["movement", "win", "market_prob"])

    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")
    dates = con.execute("""
        SELECT game_pk, TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
    """).fetchdf().drop_duplicates(subset=["game_pk"])
    pool = pool.merge(dates, on="game_pk", how="left").dropna(subset=["game_date"])
    pool = pool.sort_values(["game_date", "game_pk", "pick_side", "name_norm"], kind="mergesort").reset_index(drop=True)

    print(f"\nfinal pool with real movement + real K outcomes: n={len(pool)}  "
          f"date range {pool.game_date.min()} -> {pool.game_date.max()}")

    # need synthetic odds for profit() -- reconstruct from close_prob (approx, same
    # convention as every other addendum: implied_prob is the vig-inclusive market price)
    pool["odds"] = prob_to_american(pool["market_prob"].values)

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
