"""
Addendum 41: explainable model, genuine pre-registered chronological holdout
for batter_runs_scored (prop_type 'runs_scored' in pick_history).

WHY this run exists: Addendum 38 added real feature enrichment (batter
splits, team batting stats, team OAA -- all pulled from the client's live
system into db/cache_features.duckdb) to the pick_history-only runs_scored
market and found a near-miss with a tuned XGBoost model: edge>0.0 test
ROI=+3.59% (n=434), CI=[-3.66%, 10.83%] -- crosses zero, so still officially
FAIL, but directionally promising, and the model there was a 45+28-feature
XGBoost tuned via GridSearchCV on CV AUC. Per this project's own established
discipline (see Addendum 39, which ran the same kind of check for
batter_total_bases/batter_home_runs and found the near-miss there did NOT
replicate), the correct next step is a SEPARATE, genuinely pre-registered
run with (a) a small, hand-justified, explainable feature set instead of a
large opaque one, (b) a simple model with zero tuning degrees of freedom,
and (c) a single untouched holdout split decided and written down before any
results exist.

Why not an "older years vs newer year" split like Addendum 39: runs_scored
only has real production factors logged over one ~2.5 month window
(2026-05-17 to 2026-07-27 -- see run_pick_history_market() in
xgboost_individual_markets.py). There is no second year of this data to hold
out. Instead this uses a strict CHRONOLOGICAL 70/30 split BY DATE within
that single window -- first 70% of dates = train, last 30% = test, sorted
by (game_date, id) and cut once, never randomly shuffled across the
boundary. This is the same "sort by date, cut chronologically" convention
Addendum 39 used per explicit user direction, applied here to the only
window this market actually has.

PRE-REGISTERED FEATURE SET (decided here, before running, 7 features -- each
with an obvious real-world story for a client):
  market_prob                 -- what the market already thinks (the
                                  baseline any edge must beat), from the
                                  pick's own odds column.
  score_recent_run_form        -- the batter's own recent run-scoring form
                                  (pick_history score_*, 100% coverage) --
                                  the most direct possible signal for "does
                                  this batter score runs lately."
  score_batter_obp             -- the batter's on-base percentage (100%
                                  coverage) -- literal precondition for
                                  scoring a run (can't score without
                                  reaching base).
  score_lineup_spot            -- the batter's lineup position (100%
                                  coverage) -- leadoff/top-of-order hitters
                                  get more plate appearances and more
                                  R-scoring opportunities per game than
                                  bottom-of-order hitters.
  score_opposing_pitcher_quality -- quality of the opposing starting pitcher
                                  (100% coverage) -- a weaker opposing
                                  pitcher means more baserunners/runs for
                                  everyone in that lineup.
  ck_tb_runs_per_game          -- the batter's own team's runs-per-game
                                  (cache_team_batting_stats.runs_per_game,
                                  ASOF point-in-time join, ~100% coverage in
                                  Addendum 38) -- a batter on a high-scoring
                                  offense gets driven in more often,
                                  independent of his own skill.
  ck_vs_rhp_ops                -- the batter's own recent OPS specifically
                                  vs RHP (cache_mlb_batter_splits, ASOF
                                  point-in-time join, ~95% coverage in
                                  Addendum 38, most starters are RHP so this
                                  is the more populated/representative split)
                                  -- a second, independent read on current
                                  batter form beyond the pick_history
                                  score_* columns.

Deliberately EXCLUDED, with reasons (same discipline as Addendum 38's own
exclusions):
  - cache_mlb_team_oaa (team OAA / defensive runs-saved): Addendum 38's own
    code comment flags this as the batter's OWN team's fielding metric, not
    an opposing-defense metric (joined on base.team = the batter's team) --
    no honest "does this predict MY runs scored" story, so left out of an
    explainable feature set on purpose, even though coverage was decent.
  - score_bullpen_quality: conceptually close to score_opposing_pitcher_
    quality (both are "how good is the pitching I face"); keeping both
    would double up on the same story without adding a distinct one, so
    only the higher-signal starting-pitcher-quality column is kept, to hold
    the feature count small and each one non-redundant.
  - vs_lhp_ops: covers a materially smaller in-window population (LHP
    starters are the minority) and duplicates the batter-quality story
    vs_rhp_ops already tells; kept out to avoid two near-duplicate splits
    features.

MODEL: logistic regression (StandardScaler + LogisticRegression, default
C=1.0, max_iter=1000 only to let the solver converge -- NOT a tuning
parameter). No hyperparameter search of any kind -- zero tuning degrees of
freedom, identical discipline to Addendum 39.

SPLIT: chronological 70/30 by date, decided once, no re-splitting.

DECISION RULE (fixed here, before running): single edge>0.0 cut (odds<0,
model_prob - market_prob > 0), official gate rule from betgenius/harness/
lib/metrics.ts (n>=500 -> ROI>0 passes; n<500 -> 95% CI lower bound>0
passes). Evaluated ONCE on the untouched test split. No threshold search
folds into this headline number.

INTERPRETATION RULE (decided before running):
  - PASS on this untouched holdout -> real evidence the Addendum 38 near-miss
    replicates under a small, explainable, zero-tuning model.
  - FAIL -> the near-miss does not survive honest out-of-sample testing with
    an explainable model, same conclusion Addendum 39 reached for the other
    market's near-miss.
  - n<500 with a very wide CI (test width > 40 points) -> report as
    UNDERPOWERED, not PASS or FAIL.

Reuses DB, MIN_GRADED, gate, stat, profit, implied_prob from
xgboost_individual_markets.py unmodified (same base pick_history query/
window/win/odds logic as run_pick_history_market(), imported not
reimplemented). Does not modify that file or any other existing script.
"""
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, MIN_GRADED, gate, stat, profit, implied_prob  # noqa: E402

CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")

PROP_TYPE = "runs_scored"
MARKET_KEY = "batter_runs_scored"

# Pre-registered feature set -- fixed before this script was ever run (see
# module docstring for the full justification of each one).
SCORE_FEATURES = ["score_recent_run_form", "score_batter_obp", "score_lineup_spot",
                   "score_opposing_pitcher_quality"]
CACHE_FEATURES = ["ck_tb_runs_per_game", "ck_vs_rhp_ops"]
FEATURES = ["market_prob"] + SCORE_FEATURES + CACHE_FEATURES

JOIN_SQL = """
    ASOF LEFT JOIN cache.cache_mlb_batter_splits bs
      ON base.player_id = bs.player_id AND bs.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_team_batting_stats tb
      ON base.team = tb.team_name AND tb.snapshot_date <= base.game_date
"""
SELECT_SQL = """
    bs.vs_rhp_ops AS ck_vs_rhp_ops,
    tb.runs_per_game AS ck_tb_runs_per_game
"""


def load_pool(con):
    """Same base pick_history rows/window/win/odds logic as
    run_pick_history_market() (imported, not reimplemented) -- reused
    unmodified, plus player_id/team for joining and the pre-registered
    score_* subset, plus the two cache-table joins, point-in-time (ASOF,
    never a future snapshot relative to the pick)."""
    score_select = ", ".join(f'TRY_CAST(ph."{c}" AS DOUBLE) AS "{c}"' for c in SCORE_FEATURES)
    query = f"""
        WITH base AS (
            SELECT ph.id, TRY_CAST(ph.game_date AS DATE) AS game_date,
                   TRY_CAST(ph.odds AS INTEGER) AS odds,
                   lower(ph.hit) IN ('true','t','1') AS win,
                   TRY_CAST(ph.player_id AS BIGINT) AS player_id, ph.team,
                   {score_select}
            FROM pick_history ph
            WHERE ph.prop_type = '{PROP_TYPE}'
              AND lower(coalesce(ph.is_synthetic,'false')) NOT IN ('true','t','1')
              AND lower(coalesce(ph.voided,'false')) NOT IN ('true','t','1')
              AND ph.hit IS NOT NULL AND TRY_CAST(ph.odds AS INTEGER) IS NOT NULL
        )
        SELECT base.*, {SELECT_SQL}
        FROM base
        {JOIN_SQL}
    """
    base_n = con.execute("""
        SELECT count(*) FROM pick_history ph
        WHERE ph.prop_type = ? AND lower(coalesce(ph.is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(ph.voided,'false')) NOT IN ('true','t','1')
          AND ph.hit IS NOT NULL AND TRY_CAST(ph.odds AS INTEGER) IS NOT NULL
    """, [PROP_TYPE]).fetchone()[0]
    df = con.execute(query).fetchdf()
    df = df.dropna(subset=["game_date"]).sort_values(
        ["game_date", "id"], kind="mergesort"
    ).reset_index(drop=True)
    return df, base_n


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=" * 78)
    print("Addendum 41: explainable logistic-regression model, genuine chronological")
    print("70/30 holdout for batter_runs_scored (2026-05-17 to 2026-07-27 window,")
    print("the only window this pick_history-only market has real logged factors).")
    print("=" * 78)

    df, base_n = load_pool(con)

    # --- row-count sanity check (Addendum 37-style join-inflation guard) ---
    # Both joins are ASOF LEFT JOIN keyed on (player_id or team, snapshot_date
    # <= game_date), which by construction matches at most one row per pick
    # (the single most-recent snapshot) -- so the row count after joining
    # must equal the row count of the base query exactly. Any inflation here
    # would mean a join is fanning out to multiple snapshot rows per pick.
    print(f"\nRow-count sanity check: base pick_history pool (pre-join) n={base_n}, "
          f"post-join pool n={len(df)}")
    if len(df) != base_n:
        print(f"  *** WARNING: post-join row count ({len(df)}) != base row count "
              f"({base_n}) -- join is fanning out, investigate before trusting results ***")
    else:
        print("  OK: post-join row count matches base row count exactly -- no fan-out.")

    n_total = len(df)
    print(f"\nprop_type={PROP_TYPE}  n={n_total}")
    print(f"date range: {df['game_date'].min()} -> {df['game_date'].max()}")
    if n_total < MIN_GRADED:
        print(f"  INSUFFICIENT DATA (<{MIN_GRADED}) -- not evaluated")
        con.close()
        return

    df["market_prob"] = implied_prob(df["odds"])

    print(f"\npre-registered features ({len(FEATURES)}): {FEATURES}")
    for c in FEATURES:
        n_cov = df[c].notna().sum()
        print(f"    {c:<32} {n_cov:>6}/{n_total}  ({100.0 * n_cov / n_total:5.1f}%)")

    pool = df.dropna(subset=FEATURES + ["win", "odds", "game_date"]).copy()
    print(f"\nafter dropping rows missing any pre-registered feature: n={len(pool)} "
          f"(dropped {n_total - len(pool)})")

    # Strict chronological 70/30 split by date -- first 70% of dates = train,
    # last 30% = test, sorted (game_date, id) and cut once. Never randomly
    # shuffled across the boundary.
    pool = pool.sort_values(["game_date", "id"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.70)
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"\nfull range: {pool['game_date'].min()} -> {pool['game_date'].max()}")
    print(f"train (chronological first 70%, through {train['game_date'].max()}): n={len(train)}")
    print(f"test  (chronological last 30%, from {test['game_date'].min()}): n={len(test)}")

    if len(train) < 100 or len(test) < 20:
        print("  INSUFFICIENT DATA for a real holdout -- not evaluated")
        con.close()
        return

    X_train, y_train = train[FEATURES], train["win"].astype(int)
    X_test = test[FEATURES]

    # No hyperparameter search -- default C=1.0, fit once. This is the
    # entire point: zero tuning degrees of freedom on a sealed test.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    test = test.copy()
    test["model_prob"] = model.predict_proba(X_test)[:, 1]
    test["edge"] = test["model_prob"] - test["market_prob"]
    sub = test[(test.odds < 0) & (test.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])

    coefs = dict(zip(FEATURES, model.named_steps["logisticregression"].coef_[0]))
    print(f"\nlogistic regression coefficients (standardized features): {coefs}")

    if s["n"] is None or s["n"] == 0:
        print("  edge>0.0 cut produced ZERO bets on the test holdout -- not evaluated")
        con.close()
        return

    passed = gate(s)
    underpowered = (s["n"] < 500) and (s["lo"] is not None) and (s["hi"] is not None) and (s["hi"] - s["lo"] > 40)
    verdict = "PASS" if passed else ("UNDERPOWERED" if underpowered else "FAIL")

    print(f"\ntest holdout, edge>0.0 (single pre-registered cut, official gate rule):")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    print("\n" + "=" * 78)
    print("COMPARISON vs Addendum 38 baseline near-miss (tuned XGBoost, 45+28 features,")
    print("75/25 split, GridSearchCV-tuned):")
    print("  runs_scored: test n=434  ROI=+3.59%  CI=[-3.66%,10.83%]  FAIL (crosses zero)")
    print(f"\nAddendum 41 (this run -- explainable logistic regression, {len(FEATURES)} hand-picked")
    print("features, zero tuning, chronological 70/30 split):")
    print(f"  {MARKET_KEY}: test n={s['n']}  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    print("=" * 78)

    con.close()


if __name__ == "__main__":
    main()
