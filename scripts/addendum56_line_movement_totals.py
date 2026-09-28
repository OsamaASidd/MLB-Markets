"""
Addendum 56: line-movement (CLV-direction) feature for totals, tested on
the one real window where it's actually available -- cache_odds_snapshots
only starts 2026-06-20, which overlaps ONLY the 935-game gap backfilled in
Addendum 50 (2026-06-25 to 2026-09-23), not the 2023-2025 historical pool.
This is a standalone, scope-limited test, not a rebuild of the full
walk-forward pipeline -- pre-registered as such before running.

WHY this is a genuinely new signal, not a rehash: every prior feature this
project has used describes TEAM/PLAYER QUALITY (Elo, bullpen, park, pitcher
stats). Line movement describes MARKET BEHAVIOR ITSELF -- how the price
moved from the earliest available snapshot to the closest-to-game-time
snapshot. This is the signal professional bettors actually track (closing
line value), confirmed via external research this session, and it has
never been used as a direct model feature in this project before (only
tested indirectly as a bias hypothesis in Addendum 42, a different
question).

Feature (pre-registered, single, simple, explainable):
  movement = implied_prob(closing_odds) - implied_prob(opening_odds)
  for the SAME side being bet, using the best (median across books, to
  reduce single-book noise) price at the earliest and latest snapshot_time
  for that (event_id, pick_side). Positive movement = the market moved
  TOWARD this side since it opened (steam/sharp money signal).

Ground rules (same as every other addendum this session): model is a
simple logistic regression (StandardScaler, default C=1.0, ZERO
hyperparameter search -- no tuning degrees of freedom), single
pre-specified chronological 70/30 split within this window, single
edge>0.0 cut, official gate rule, evaluated exactly once. A FAIL is a
valid, honestly-reportable outcome -- this is the last genuinely untried
idea this session, not a guaranteed win.
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
GAP_MAPPING = ROOT / "data_raw" / "event_id_mapping_h2h_totals_gap.csv"


def build_movement_features(con2):
    """One row per (event_id, pick_side): opening/closing implied prob and
    their difference, using the median price across bookmakers at the
    earliest and latest snapshot_time to reduce single-book noise."""
    # CRITICAL: restrict to strictly PRE-GAME snapshots -- verified this
    # session that 84.2% of (event,side,game_time) groups in this table
    # have their latest recorded snapshot 10-12+ hours AFTER game_time
    # (post-game/settled prices, not real closing lines -- confirmed via
    # the h2h version of this same table, Addendum 56b). Filtering here
    # before any aggregation to avoid the same leakage.
    snaps = con2.execute("""
        SELECT event_id, pick_side, line, odds, snapshot_time, game_time
        FROM cache_odds_snapshots_game_total
        WHERE snapshot_time <= game_time
    """).fetchdf()
    snaps["implied"] = implied_prob(snaps["odds"].values)

    first_t = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].transform("min")
    last_t = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].transform("max")
    opening = snaps[snaps.snapshot_time == first_t].groupby(["event_id", "pick_side"])["implied"].median().rename("open_prob")
    closing = snaps[snaps.snapshot_time == last_t].groupby(["event_id", "pick_side"])["implied"].median().rename("close_prob")
    n_snaps = snaps.groupby(["event_id", "pick_side"])["snapshot_time"].nunique().rename("n_snapshots")

    mv = pd.concat([opening, closing, n_snaps], axis=1).reset_index()
    mv["movement"] = mv["close_prob"] - mv["open_prob"]
    return mv


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building line-movement features from cache_odds_snapshots_game_total...")
    mv = build_movement_features(con2)
    print(f"  {len(mv)} (event_id, pick_side) movement rows built "
          f"({mv.event_id.nunique()} unique events, median snapshots/pick={mv.n_snapshots.median():.0f})")

    print("\nloading the 935-game gap-backfilled totals pool (Addendum 50 odds + real outcomes)...")
    gap = pd.read_csv(GAP_MAPPING)  # game_pk, event_id, home_team, away_team, commence_time_utc
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")
    outcomes = con.execute("""
        SELECT game_pk, home_score, away_score, TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true AND home_score IS NOT NULL AND away_score IS NOT NULL
    """).fetchdf().drop_duplicates(subset=["game_pk"])

    gap_odds_raw = []
    import json
    with open(ROOT / "data_raw" / "totals_gap_odds_cache.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            resp = rec["response"]
            if "error" in resp or "data" not in resp:
                continue
            data = resp["data"]
            for bk in data.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] != "totals":
                        continue
                    for oc in mk["outcomes"]:
                        gap_odds_raw.append({
                            "game_pk": rec["game_pk"], "event_id": rec["event_id"],
                            "point": oc["point"], "side": oc["name"].lower(), "price": oc["price"],
                        })
    raw = pd.DataFrame(gap_odds_raw)
    # primary line per game: most-quoted point
    primary = raw.groupby(["game_pk", "point"]).size().reset_index(name="n")
    primary = primary.sort_values(["game_pk", "n"], ascending=[True, False]).drop_duplicates(subset=["game_pk"])
    raw = raw.merge(primary[["game_pk", "point"]], on=["game_pk", "point"])
    best = raw.groupby(["game_pk", "event_id", "point", "side"])["price"].max().reset_index()
    best = best.pivot_table(index=["game_pk", "event_id", "point"], columns="side", values="price").reset_index()
    best = best.dropna(subset=["over", "under"])

    def to_american(d):
        return np.where(d >= 2.0, (d - 1) * 100, -100 / (d - 1))
    best["best_over_odds"] = to_american(best["over"].values)
    best["best_under_odds"] = to_american(best["under"].values)
    best = best.rename(columns={"point": "line"})

    pool = best.merge(outcomes, on="game_pk", how="inner")
    pool["total_runs"] = pool["home_score"] + pool["away_score"]
    under = pool.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
    over = pool.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
    long_pool = pd.concat([under, over]).dropna(subset=["odds"])
    print(f"  base pool (gap games, both sides, real outcomes): n={len(long_pool)}")

    n_before = len(long_pool)
    long_pool = long_pool.merge(mv, on=["event_id"], how="left", suffixes=("", "_mv"))
    # mv has pick_side, pool has 'side' -- align names for the match, then filter to matching side
    long_pool = long_pool[long_pool["side"] == long_pool["pick_side"]]
    print(f"  row-count check: before-merge n={n_before}, after merge+side-match n={len(long_pool)} "
          f"(expected: at most n_before if some picks have no movement history)")

    long_pool["market_prob"] = implied_prob(long_pool["odds"].values)
    long_pool = long_pool.dropna(subset=["movement", "win", "odds"])
    print(f"  final pool with real movement data: n={len(long_pool)}")

    # game_date already carried through from `outcomes` (cache_mlb_historical_outcomes,
    # which covers this whole gap window -- client_games does NOT, that's the reason
    # this gap needed backfilling in the first place; using it here would be the same bug).
    long_pool = long_pool.dropna(subset=["game_date"]).sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)

    if len(long_pool) < 100:
        print(f"\n  INSUFFICIENT DATA (n={len(long_pool)}) for a meaningful holdout -- reporting as UNDERPOWERED, not PASS/FAIL")
        return

    cut = int(len(long_pool) * 0.70)
    train, test = long_pool.iloc[:cut].copy(), long_pool.iloc[cut:].copy()
    print(f"\n  chronological 70/30 split: train n={len(train)} ({train.game_date.min()}->{train.game_date.max()})  "
          f"test n={len(test)} ({test.game_date.min()}->{test.game_date.max()})")

    FEATURES = ["market_prob", "movement", "n_snapshots"]
    if len(train) < 50 or len(test) < 20:
        print(f"  INSUFFICIENT DATA after split -- UNDERPOWERED")
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
