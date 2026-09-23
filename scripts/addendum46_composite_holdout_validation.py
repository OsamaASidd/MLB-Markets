"""
Addendum 46: genuine out-of-sample validation of Addendum 45a's
batter_form_index composite for batter_total_bases.

WHY this run exists: Addendum 45a found batter_total_bases PASSES with the
new composite feature (n=4927, ROI=2.55%, CI=[0.18,4.92] -- the first CI
in this entire project that excludes zero) -- but using the SAME per-year
75/25 split every prior addendum on this market has used. Addendum 39
already showed this exact market, under a genuinely out-of-sample temporal
holdout (train on strictly older data, test on data the model never saw in
any form) with a simple, zero-tuning model, produces ZERO qualifying bets
-- the earlier near-miss did not replicate. Before treating Addendum 45a's
result as real, the composite feature itself must survive the same kind of
genuine holdout, not just the split already shown to overfit for this
market. This is the standard this project has applied to every other
promising result (Addenda 39, 41) -- applying it here too, not skipping it
because the number looks good.

Method (pre-registered before running, decided from the two holdout
designs already used and accepted in this project):
  (A) Older-years-train / newest-year-test (Addendum 39's original design):
      train ONLY on game_date.year in {2023, 2024}, test ONCE on ALL of
      game_date.year == 2025.
  (B) Chronological 70/30 across the full available range (the design the
      user explicitly requested for Addendum 39's re-run): sort by date,
      first 70% train, last 30% test.
  Same simple, explainable model as Addendum 39 (logistic regression,
  StandardScaler, default C=1.0, ZERO hyperparameter search -- no tuning
  degrees of freedom at all), run twice per split: once with Addendum 39's
  original 6 features, once with those 6 features PLUS batter_form_index
  (Addendum 45a's composite, reused unmodified) -- isolating the
  composite's own marginal contribution under a real holdout.
  Single edge>0.0 cut, official gate rule. Reported once, pass or fail,
  regardless of outcome -- this is the whole point of a sealed test.

Reuses build_games_v2/build_box_ids/build_features_v2 (addendum37) and
build_batter_form_index (addendum45a) UNMODIFIED. Does not modify either
file.
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
from addendum45a_batter_composite_index import build_batter_form_index, RAW_RATE_COLS  # noqa: E402

BASE_FEATURES = ["market_prob", "elo_diff", "starter_xera_diff", "bullpen_era_diff", "is_dome", "hits_factor"]


def zscore_composite(train, test):
    """Same method as Addendum 45a's add_composite: fit mean/std on TRAIN
    only, apply to both -- never fit on test."""
    train = train.copy()
    test = test.copy()
    zcols = []
    for c in RAW_RATE_COLS:
        mu, sd = train[c].mean(), train[c].std()
        zc = f"z_{c}"
        if sd and sd > 0 and not np.isnan(sd):
            train[zc] = (train[c] - mu) / sd
            test[zc] = (test[c] - mu) / sd
        else:
            train[zc] = np.nan
            test[zc] = np.nan
        zcols.append(zc)
    train["batter_form_index"] = train[zcols].mean(axis=1, skipna=True)
    test["batter_form_index"] = test[zcols].mean(axis=1, skipna=True)
    return train, test


def eval_once(train, test, features, label):
    train = train.dropna(subset=features + ["win", "odds"]).copy()
    test = test.dropna(subset=features + ["win", "odds"]).copy()
    if len(train) < 100 or len(test) < 20:
        print(f"    [{label}] INSUFFICIENT DATA: train={len(train)} test={len(test)}")
        return None

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features], train["win"].astype(int))

    t = test.copy()
    t["model_prob"] = model.predict_proba(t[features])[:, 1]
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]
    if len(sub) == 0:
        print(f"    [{label}] edge>0.0 produced ZERO bets -- not evaluated")
        return {"label": label, "verdict": "no_bets"}
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"    [{label}] n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"label": label, **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    print("building corrected game log + Addendum 37 features + Addendum 45a composite...")
    games, box = build_games_and_box(con)
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)
    form = build_batter_form_index(con)

    pool = build_warehouse_pool(con, box, "batter_total_bases", "total_bases", "batter")
    pool = build_features_v2(pool, games_v2, "batter", box_ids, con2)
    n_before = len(pool)
    form_by_gamepk = form.drop_duplicates(subset=["game_pk", "name_norm"])[["game_pk", "name_norm"] + RAW_RATE_COLS]
    pool = pool.merge(form_by_gamepk, on=["game_pk", "name_norm"], how="left")
    print(f"row-count sanity check: before={n_before}  after={len(pool)}  "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before, "fan-out detected, aborting"

    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool["year"] = pool["game_date"].dt.year

    print("\n" + "=" * 90)
    print("=== SPLIT A: train 2023-2024 only, test ALL of 2025 (Addendum 39's original design) ===")
    print("=" * 90)
    trainA = pool[pool["year"].isin([2023, 2024])].copy()
    testA = pool[pool["year"] == 2025].copy()
    trainA, testA = zscore_composite(trainA, testA)
    print(f"  train n={len(trainA)}   test n={len(testA)}")
    rA_base = eval_once(trainA, testA, BASE_FEATURES, "Split A -- baseline (no composite)")
    rA_comp = eval_once(trainA, testA, BASE_FEATURES + ["batter_form_index"], "Split A -- + batter_form_index")

    print("\n" + "=" * 90)
    print("=== SPLIT B: chronological 70/30 over the full available range ===")
    print("=" * 90)
    poolB = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    cut = int(len(poolB) * 0.70)
    trainB, testB = poolB.iloc[:cut].copy(), poolB.iloc[cut:].copy()
    trainB, testB = zscore_composite(trainB, testB)
    print(f"  full range: {poolB['game_date'].min().date()} -> {poolB['game_date'].max().date()}")
    print(f"  train n={len(trainB)} (through {trainB['game_date'].max().date()})   "
          f"test n={len(testB)} (from {testB['game_date'].min().date()})")
    rB_base = eval_once(trainB, testB, BASE_FEATURES, "Split B -- baseline (no composite)")
    rB_comp = eval_once(trainB, testB, BASE_FEATURES + ["batter_form_index"], "Split B -- + batter_form_index")

    con.close()
    con2.close()

    print("\n" + "=" * 90)
    print("=== SUMMARY: does batter_form_index survive a genuine out-of-sample holdout? ===")
    print("=" * 90)
    for r in [rA_base, rA_comp, rB_base, rB_comp]:
        print(f"  {r}")
    print("\nFor reference, Addendum 45a's in-sample-split (per-year 75/25) result was:")
    print("  baseline: n=4501 ROI=1.61 CI=[-0.9,4.13] PASS  |  + composite: n=4927 ROI=2.55 CI=[0.18,4.92] PASS")
    print("If the results above are FAIL/no_bets/UNDERPOWERED, that per-year-split PASS did not "
          "replicate under genuine holdout -- same conclusion pattern as Addendum 39.")


if __name__ == "__main__":
    main()
