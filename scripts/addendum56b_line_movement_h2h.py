"""
Addendum 56b: line-movement feature for h2h, same idea as Addendum 56
(totals) but for game_side. Unlike totals, no separate odds backfill was
done for h2h in this window (2026-06-20+) -- but cache_odds_snapshots_
game_side already carries real per-bookmaker odds over time, so the LATEST
snapshot per (event_id, pick_side) serves as the real closing price
directly, no extra API spend needed.

Same pre-registered feature, model, and discipline as Addendum 56:
movement = implied_prob(close) - implied_prob(open), simple logistic
regression (zero tuning), single chronological 70/30 split, single
edge>0.0 cut, official gate rule, evaluated once.
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


def to_american(d):
    return np.where(d >= 2.0, (d - 1) * 100, -100 / (d - 1))


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building closing odds + movement directly from cache_odds_snapshots_game_side...")
    # CRITICAL: restrict to strictly PRE-GAME snapshots. Verified directly
    # (this session): 84.6% of (event,side,game_time) groups in this table
    # have their latest recorded snapshot 10-12+ hours AFTER game_time --
    # post-game/settled prices, not real closing lines. Using those would
    # leak the outcome into the "closing odds" feature. This filter is the
    # fix, applied before any aggregation.
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
    # closing ODDS: best (highest decimal payout, i.e. best price) among books at the LATEST snapshot time
    closing_grp = snaps[snaps.snapshot_time == last_t]
    closing_odds = closing_grp.loc[closing_grp.groupby(["event_id", "pick_side"])["odds_decimal"].idxmax()][
        ["event_id", "pick_side", "odds"]].set_index(["event_id", "pick_side"])["odds"].rename("closing_odds")
    closing_prob = closing_grp.groupby(["event_id", "pick_side"])["implied"].median().rename("close_prob")
    n_snaps = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].nunique().rename("n_snapshots")

    mv = pd.concat([opening, closing_prob, closing_odds, n_snaps], axis=1).reset_index()
    mv["movement"] = mv["close_prob"] - mv["open_prob"]
    print(f"  {len(mv)} (event_id, pick_side) rows with closing odds + movement "
          f"({mv.event_id.nunique()} unique events, median snapshots/pick={mv.n_snapshots.median():.0f})")

    # cache_odds_snapshots uses REAL Odds-API hex event_ids, but
    # cache_mlb_historical_outcomes mostly uses SYNTHETIC "mlb_<game_pk>"
    # ids for this same window (confirmed earlier this session, the exact
    # reason the totals gap backfill needed its own event-ID discovery
    # phase) -- the two id spaces don't overlap directly. Reuse that
    # discovery mapping (built for the same games/window) to get from
    # event_id -> game_pk, then join outcomes via game_pk instead.
    id_map = pd.read_csv(ROOT / "data_raw" / "event_id_mapping_h2h_totals_gap.csv")[["event_id", "game_pk"]].drop_duplicates(subset=["event_id"])
    mv = mv.merge(id_map, on="event_id", how="inner")
    print(f"  mapped to a real game_pk via the discovery mapping: {len(mv)} rows "
          f"({mv.event_id.nunique()} events -- some snapshot events may be outside the discovered window)")

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
    print(f"  final h2h pool with real closing odds + movement: n={len(pool)}")

    pool = pool.sort_values(["game_date", "event_id", "pick_side"], kind="mergesort").reset_index(drop=True)
    if len(pool) < 100:
        print(f"  INSUFFICIENT DATA (n={len(pool)}) -- UNDERPOWERED")
        return

    cut = int(len(pool) * 0.70)
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"  chronological 70/30 split: train n={len(train)} ({train.game_date.min()}->{train.game_date.max()})  "
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
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
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
