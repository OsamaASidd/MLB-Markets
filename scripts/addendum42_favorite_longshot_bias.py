"""
Addendum 42: favorite-longshot bias -- a different kind of signal than
"which team is better" (Elo/L10/bullpen/park, already tried and exhausted
in Addenda 9/17/32/34/37 for h2h/totals, all FAIL, underpowered on real
n after the edge>0.0 filter). Favorite-longshot bias is a documented market-
calibration effect: bettors overvalue longshots and undervalue favorites, so
the market's OWN implied probability is a biased estimator of true win
probability as a function of the probability level itself. If real and
detectable here, a recalibration of implied_prob (not a new predictive
feature) could recover positive edge on the same real closing odds already
shown insufficient for a team-quality model.

Ground rule (this project's standing discipline, restated because it
matters most here): recalibration is fit via isotonic regression on the
TRAIN split ONLY -- a fixed, non-data-snooped, standard method for
correcting exactly this kind of calibration bias, not tuned against test-set
ROI. The test split is scored exactly ONCE, after the recalibration curve is
already fixed. No threshold search, no re-splitting, no peeking.

Two-step method, both markets (h2h, totals):
  1. EXISTENCE CHECK (descriptive, on the full real closing-odds pool, no
     train/test asymmetry -- this is not model tuning, just "does the
     pattern exist at all before building anything on it"): bin every real
     bet by implied_prob decile, compare actual win rate to mean implied
     probability per bin. A real favorite-longshot bias shows underdogs
     (low implied_prob) winning LESS than implied and favorites (high
     implied_prob) winning MORE than implied -- i.e. diff = actual - implied
     increasing with implied_prob. Decision rule, fixed before looking at
     any test-set number: Spearman rank correlation between decile order
     and diff must be >= 0.6 (a real, not noisy-looking, monotonic pattern)
     for that market to proceed to step 2. This threshold is fixed here,
     before either market's correlation is computed, not chosen after
     seeing the results.
  2. IF detected for a market: fit IsotonicRegression(market_prob -> win) on
     the TRAIN split only (split_per_year, this project's existing
     per-market convention), apply to TEST split, bet where odds<0 (favorite
     side only -- the side the bias implies is underpriced) AND
     recalib_prob - market_prob > 0.0 (single pre-specified cut, same form
     as every other addendum's edge>0.0 gate check). Official gate rule
     (xgboost_individual_markets.gate: n>=500->ROI>0; else CI lower
     bound>0), evaluated once.
  IF NOT detected for a market: stop, report "no detectable bias, no model
  built" for that market -- per the task's explicit instruction not to force
  a model onto a hypothesis that doesn't hold.

Data / row-count discipline: uses the SAME corrected, deduplicated game log
(build_games_v2 from addendum37_historical_enrichment.py) that fixed the
real duplicate-game_pk fan-out bug discovered in Addendum 37 (games had
4,547 rows / 3,845 unique game_pk; merging into build_game_pool without
dedup silently ~doubled the h2h/totals pool). Reuses build_game_pool /
build_features / split_per_year from multi_model_comparison.py, and
implied_prob / gate / stat / profit from xgboost_individual_markets.py,
unmodified. Row counts are printed at every merge step below specifically
to catch a repeat of that same bug class.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import gate, stat, profit, implied_prob  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_game_pool, build_features, split_per_year,
)
from addendum37_historical_enrichment import DB, CACHE_DB, build_games_v2  # noqa: E402

N_BINS = 10
SPEARMAN_THRESHOLD = 0.6  # fixed before either market's correlation is computed


def calibration_table(pool, market_name):
    """Descriptive existence check on the FULL pool (train+test combined --
    this is not model tuning, no parameter is fit here, just a plain
    binned comparison of implied vs actual, same as any calibration-curve
    diagnostic). Returns (table_df, spearman_rho, detected: bool)."""
    df = pool[["market_prob", "win"]].copy()
    df["win"] = df["win"].astype(int)
    df["bin"] = pd.qcut(df["market_prob"], N_BINS, labels=False, duplicates="drop")
    table = df.groupby("bin").agg(
        n=("win", "size"),
        avg_implied=("market_prob", "mean"),
        actual_wr=("win", "mean"),
    ).reset_index()
    table["avg_implied_pct"] = (table["avg_implied"] * 100).round(2)
    table["actual_wr_pct"] = (table["actual_wr"] * 100).round(2)
    table["diff_pct"] = (table["actual_wr_pct"] - table["avg_implied_pct"]).round(2)

    rho, pval = spearmanr(table["bin"], table["diff_pct"])
    detected = bool(rho is not None and not np.isnan(rho) and rho >= SPEARMAN_THRESHOLD)

    # Secondary confirmatory diagnostic: logistic regression of win on
    # logit(market_prob). slope > 1 is the classic favorite-longshot-bias
    # signature (market's probability spread is compressed vs. reality).
    # Descriptive only -- not used to select or fit the betting model.
    p = df["market_prob"].clip(1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).values.reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000).fit(logit_p, df["win"])
    slope, intercept = lr.coef_[0][0], lr.intercept_[0]

    print(f"\n  [{market_name}] calibration by implied-probability decile "
          f"(full pool, n={len(df)}):")
    print(f"    {'bin':<4}{'n':<7}{'avg_implied%':<14}{'actual_wr%':<12}{'diff (actual-implied)':<10}")
    for _, r in table.iterrows():
        print(f"    {int(r['bin']):<4}{int(r['n']):<7}{r['avg_implied_pct']:<14}"
              f"{r['actual_wr_pct']:<12}{r['diff_pct']:<10}")
    print(f"    Spearman rho(bin order, diff) = {rho:.3f} (p={pval:.3g}); "
          f"pre-specified threshold for 'detected' = {SPEARMAN_THRESHOLD}")
    print(f"    confirmatory diagnostic: logistic regression win ~ logit(market_prob): "
          f"slope={slope:.3f} intercept={intercept:.3f} "
          f"(slope>1 ~ favorite-longshot-bias-shaped; slope~1,intercept~0 ~ well-calibrated)")
    print(f"    VERDICT: {'DETECTED -- proceeding to recalibration model' if detected else 'NOT DETECTED -- stopping, no model built for this market'}")

    return table, rho, detected


def recalibration_test(pool, market_name):
    """Fit isotonic regression on TRAIN only, evaluate ONCE on TEST. Bet
    odds<0 (favorite side) & recalibrated edge>0.0, single pre-specified
    cut, official gate rule. No threshold search, no re-fitting."""
    train, test = split_per_year(pool)
    print(f"\n  [{market_name}] recalibration model: n(train/test)={len(train)}/{len(test)}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train["market_prob"], train["win"].astype(int))

    test = test.copy()
    test["recalib_prob"] = iso.predict(test["market_prob"])
    test["edge"] = test["recalib_prob"] - test["market_prob"]

    sub = test[(test.odds < 0) & (test.edge > 0.0)]
    if len(sub) == 0:
        print("    edge>0.0 (odds<0) cut produced ZERO bets on the test split -- not evaluated")
        return {"market": market_name, "verdict": "no_bets", "n": 0}

    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    passed = gate(s)
    underpowered = (s["n"] < 500) and (s["lo"] is not None) and (s["hi"] is not None) and (s["hi"] - s["lo"] > 40)
    verdict = "PASS" if passed else ("UNDERPOWERED" if underpowered else "FAIL")

    print(f"    test split, recalibrated edge>0.0 & odds<0: n={s['n']}  WR={s['wr']}%  "
          f"ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"market": market_name, "n": s["n"], "wr": s["wr"], "roi": s["roi"],
            "lo": s["lo"], "hi": s["hi"], "verdict": verdict}


def build_market_pool(con, games_v2, market_name):
    pool_raw = build_game_pool(con, games_v2, market_name)
    n_raw = len(pool_raw)
    n_unique_gamepk = pool_raw["game_pk"].nunique()
    pool = build_features(pool_raw, games_v2)
    n_after = len(pool)
    print(f"  [{market_name}] row-count sanity check: build_game_pool -> n={n_raw} "
          f"(unique game_pk={n_unique_gamepk}); after build_features merge -> n={n_after} "
          f"({'OK, no fan-out' if n_after == n_raw else 'MISMATCH -- investigate'})")
    return pool


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building real team game log (existing pipeline)...")
    games, box = build_games_and_box(con)
    print(f"  raw games: n={len(games)}, unique game_pk={games['game_pk'].nunique()}")

    print("applying Addendum 37's corrected/deduplicated game log (build_games_v2) -- "
          "fixes the real duplicate-game_pk fan-out bug that inflates h2h/totals pools "
          "when merged without dedup...")
    games_v2 = build_games_v2(con, con2, games)
    print(f"  deduplicated games_v2: n={len(games_v2)}, unique game_pk={games_v2['game_pk'].nunique()}")

    print("\n=== Addendum 42: favorite-longshot bias -- h2h and totals ===")
    print("=== STEP 1: does the market's implied probability systematically "
          "miscalibrate vs. actual outcomes (descriptive, full pool, no test-set peeking)? ===")

    results = {}
    for market_name in ["h2h", "totals"]:
        pool = build_market_pool(con, games_v2, market_name)
        _, rho, detected = calibration_table(pool, market_name)
        results[market_name] = {"pool": pool, "rho": rho, "detected": detected}

    con.close()
    con2.close()

    print("\n=== STEP 2: recalibration model (isotonic regression on TRAIN only), "
          "evaluated ONCE on the untouched TEST split -- only for markets where "
          "STEP 1 detected a real pattern ===")

    final = []
    for market_name in ["h2h", "totals"]:
        r = results[market_name]
        if not r["detected"]:
            print(f"\n  [{market_name}] STEP 1 did not detect a real favorite-longshot-bias "
                  f"pattern in this data (Spearman rho={r['rho']:.3f} < {SPEARMAN_THRESHOLD}) -- "
                  f"per protocol, no recalibration model is built or tested for this market. "
                  f"This is a valid, honest stopping point, not a gap in the analysis.")
            final.append({"market": market_name, "verdict": "no bias detected -- not modeled"})
        else:
            final.append(recalibration_test(r["pool"], market_name))

    print("\n=== SUMMARY ===")
    for r in final:
        print(f"  {r}")


if __name__ == "__main__":
    main()
