"""
Addendum 39: explainable model, genuine pre-registered temporal holdout.

WHY this run exists: after Addendum 37 (feature enrichment) produced zero
NEW confirmed passes -- batter_home_runs and batter_total_bases are both
near-misses (n>=500, ROI>0, but 95% CI still spans negative) -- the one
remaining legitimate experiment on the historical dataset is a genuinely
untouched temporal holdout: train on OLDER seasons only, evaluate ONCE on a
NEWER season the model has never seen in any form (not in CV, not in
tuning). Every prior run in this project (Addenda 27/32/34/36/37) used a
per-year 75/25 split, which means 2025 data was already partially used
during model selection for every one of those results. This run is the
first to hold out 2025 completely.

Also: the client needs to be able to understand WHY a market passes, not
just that a black-box model says so. XGBoost with 25 features is not that.
This run deliberately uses:
  - ONE simple, standard model: logistic regression (StandardScaler +
    default C=1.0, no hyperparameter search at all -- zero tuning degrees
    of freedom, which is the whole point of a sealed pre-registered test).
  - SIX hand-picked features, each with an obvious real-world story a
    client can be told directly:
      market_prob      -- what the market already thinks (the baseline
                           any edge must beat)
      elo_diff          -- which team is actually better right now
      starter_xera_diff -- opposing/both starters' real Statcast quality
                           (the single biggest signal gap Addendum 37
                           closed -- previously totally absent)
      bullpen_era_diff  -- bullpen quality, same idea
      park_factor       -- hr_factor for batter_home_runs, hits_factor for
                           batter_total_bases (the market-specific park
                           effect)
      is_dome           -- static park context (no wind/weather variance)
  - PRE-REGISTERED BEFORE RUNNING (written here, in this docstring, before
    any results existed): train ONLY on game_date.year in {2023, 2024};
    test ONCE on ALL of game_date.year == 2025, no re-splitting, no peeking
    at 2025 before this point. Single edge>0.0 cut (odds<0), official gate
    rule (n>=500 -> ROI>0; n<500 -> 95% CI lower bound>0), exactly as
    everywhere else in this project. Reported as-is, pass or fail.

Interpretation rule, decided before running (per the project's own
established asymmetric-interpretation convention):
  - PASS on this untouched holdout -> real evidence the effect replicates
    out-of-sample, meaningfully stronger than the in-sample near-miss.
  - FAIL -> the near-miss does not survive true out-of-sample testing;
    treat as FAIL, not as a near-miss anymore.
  - underpowered (n<500 AND the CI is uninformatively wide) -> report as
    underpowered, not as PASS or FAIL.

Reuses build_games_v2/build_box_ids/build_features_v2 from
addendum37_historical_enrichment.py unmodified (same corrected,
deduplicated pipeline -- see that file's fix for the pre-existing
duplicate-game_pk bug). Does not modify that file or any other existing
script.
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
from xgboost_individual_markets import gate, stat, profit  # noqa: E402
from multi_model_comparison import build_games_and_box, build_warehouse_pool  # noqa: E402
from addendum37_historical_enrichment import DB, CACHE_DB, build_games_v2, build_box_ids, build_features_v2  # noqa: E402

# Pre-registered feature set -- fixed before this script was ever run.
MARKET_PARK_FACTOR = {
    "batter_home_runs": "hr_factor",
    "batter_total_bases": "hits_factor",
}
BASE_FEATURES = ["market_prob", "elo_diff", "starter_xera_diff", "bullpen_era_diff", "is_dome"]


def explainable_holdout(market_name, stat_col, games_v2, box_ids, con2, con, box):
    park_col = MARKET_PARK_FACTOR[market_name]
    features = BASE_FEATURES + [park_col]

    pool = build_warehouse_pool(con, box, market_name, stat_col, "batter")
    pool = build_features_v2(pool, games_v2, "batter", box_ids, con2)
    pool = pool.dropna(subset=features + ["win", "odds", "game_date"]).copy()
    pool["game_date"] = pd.to_datetime(pool["game_date"])

    # Chronological 70/30 split over the full available date range (same
    # convention as split_chronological() elsewhere in this project) --
    # per explicit user direction, superseding this script's original
    # 2023-2024-train/2025-test design. Real warehouse-odds data for these
    # markets only spans 2023-05-03 to 2025-05-28 (verified directly --
    # there is no 2023-2026 range on disk for these two markets; that
    # wider range exists only for the separate pick_history-only markets).
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.70)
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"\n{market_name}")
    print(f"  pre-registered features: {features}")
    print(f"  full range: {pool['game_date'].min().date()} -> {pool['game_date'].max().date()}")
    print(f"  train (chronological first 70%, through {train['game_date'].max().date()}): n={len(train)}")
    print(f"  test  (chronological last 30%, from {test['game_date'].min().date()}): n={len(test)}")

    if len(train) < 100 or len(test) < 20:
        print("  INSUFFICIENT DATA for a real holdout -- not evaluated")
        return {"market": market_name, "verdict": "insufficient_data"}

    X_train, y_train = train[features], train["win"].astype(int)
    X_test = test[features]

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

    if s["n"] is None or s["n"] == 0:
        print("  edge>0.0 cut produced ZERO bets on the 2025 holdout -- not evaluated")
        return {"market": market_name, "verdict": "no_bets"}

    passed = gate(s)
    underpowered = (s["n"] < 500) and (s["lo"] is not None) and (s["hi"] is not None) and (s["hi"] - s["lo"] > 40)
    verdict = "PASS" if passed else ("UNDERPOWERED" if underpowered else "FAIL")

    coefs = dict(zip(features, model.named_steps["logisticregression"].coef_[0]))
    print(f"  logistic regression coefficients (standardized features): {coefs}")
    print(f"  2025 holdout, edge>0.0: n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  "
          f"CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"market": market_name, "n": s["n"], "roi": s["roi"], "lo": s["lo"], "hi": s["hi"],
            "coefs": coefs, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    print("building corrected (deduplicated) game log + Addendum 37 features...")
    games, box = build_games_and_box(con)
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    print("\n=== Addendum 39: explainable logistic-regression model, genuine 2023-2024 -> 2025 "
          "pre-registered holdout (2025 never touched before this evaluation) ===")

    results = []
    for market_name, stat_col in [("batter_home_runs", "home_runs"), ("batter_total_bases", "total_bases")]:
        results.append(explainable_holdout(market_name, stat_col, games_v2, box_ids, con2, con, box))

    con.close()
    con2.close()

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
