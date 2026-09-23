"""
Addendum 45a: a single new engineered feature -- a composite "batter form
index" -- tested on the two still-FAILing batter markets that use a per-
batter individual model: batter_total_bases (real warehouse odds,
2023-2025) and batter_runs_scored (pick_history-only, 2026 window).

MOTIVATION: a batter's hits/RBI-scoring propensity is mechanically
correlated with total-bases/runs-scored propensity (a hitter in good form
gets more hits AND more total bases AND scores more runs). Every prior
addendum (27, 32, 34, 36, 37, 39, 41, 43a) only ever fed total_bases/
runs_scored raw game/park/pitcher-quality features -- never a derived
"how good is this batter looking right now" signal. This is NOT pooling
across markets (the client's actual constraint, restated in Addendum 27's
docstring) -- it's ONE new feature, computed from raw boxscore data (never
from another market's model prediction), added on top of each market's own
existing, individually-fit XGBoost model. hits/rbis stay untouched --
they already pass and are out of scope here.

POINT-IN-TIME SAFETY (the mandatory constraint for this addendum):
  - The composite is built ENTIRELY from `boxscore` -- each batter's own
    trailing rolling per-at-bat rates (hits, total_bases, rbi), computed
    over a fixed trailing window of their last FORM_WINDOW=15 games,
    STRICTLY BEFORE the game in question (`shift(1)` before the rolling
    window, so the current game's own line never leaks into its own
    feature -- verified explicitly below).
  - This is a raw historical STATISTIC, never a model's prediction for the
    game being evaluated and never a model trained on overlapping data --
    it is exactly the same kind of point-in-time rolling feature this
    project already trusts (Addendum 40's `opp_lineup_k_rate`, Addendum
    37's bullpen/starter rolling snapshots via merge_asof), just computed
    from the batter's own box score instead of team/pitcher tables.
  - N=15 is a single pre-specified choice (same discipline as Addendum
    40), not tuned against any ROI number -- picked once, before this
    script's results existed, and never revisited.
  - For batter_total_bases the join key is (game_pk, name_norm) -- exact
    game identity, same key every prior warehouse-odds addendum already
    uses. For batter_runs_scored (pick_history has no reliable game_pk on
    ~48% of rows, but player_id is ~96% populated) the join key is
    (player_id, game_date); where a player has more than one boxscore row
    on the same calendar date (doubleheader), the EARLIEST game_pk's
    rolling value is used deterministically -- the most conservative
    choice, since it is guaranteed to exclude even same-day action.
  - Combination method: each of the 3 raw rolling rates is z-scored using
    TRAIN-SPLIT-ONLY mean/std (identical discipline to this project's own
    established precedent -- multi_model_comparison.py's SimpleImputer/
    StandardScaler, fit on train, applied to train+test, never fit on
    test rows) and simple-averaged (equal weights) into ONE scalar,
    `batter_form_index`. A simple, explainable average was picked because
    it is the most transparent combination method available and requires
    no additional tuned parameters; alternative combination methods
    (weighted average, PCA, etc.) were NOT grid-searched against test ROI
    -- doing so would be exactly the kind of method search this project's
    guardrails prohibit (see HANDOFF.md).
  - Missing composite (e.g. a player's MLB debut with zero prior games in
    `boxscore`) is left as NaN, never imputed to look like average form --
    handled natively by XGBoost's `missing=np.nan`, the same convention
    already used for every other feature throughout this project.

MODEL/SELECTION DISCIPLINE (ground rule 2): ONE model family (XGBoost),
ONE pre-specified hyperparameter grid (PARAM_GRID/FIXED_PARAMS imported
UNMODIFIED from tuned_xgboost_4_markets.py), selection via CV-averaged ROC
AUC on the TRAINING split only (5-fold StratifiedKFold + GridSearchCV) --
ROI is never touched during selection. For each market TWO models are fit
with the identical grid/CV/split -- baseline features only, and baseline +
`batter_form_index` -- so the new feature's marginal CV-AUC contribution is
isolated and reported honestly, pass or fail either way. Each model's test
split is scored exactly ONCE, at the single pre-specified edge>0.0 cut, via
the project's official gate() (n>=500 -> ROI>0; else 95% CI lower bound>0,
verified against betgenius/harness/lib/metrics.ts).

DATA PIPELINE REUSE (ground rule: reuse, don't reinvent):
  - batter_total_bases: build_games_v2 / build_box_ids / build_features_v2
    / FEATURES_V2 imported UNMODIFIED from addendum37_historical_enrichment.py
    (the current best-tried feature set for this market), plus
    build_games_and_box / build_warehouse_pool / split_per_year imported
    UNMODIFIED from multi_model_comparison.py. Same per-year 75/25 split
    this market has used since Addendum 32/37.
  - batter_runs_scored: load_pool / FEATURES imported UNMODIFIED from
    addendum41_runs_scored_holdout.py (the current best-tried explainable
    feature set for this market -- market_prob + 4 pre-registered score_*
    columns + 2 cache-table ASOF columns). Same chronological 70/30 split
    Addendum 41 used. Model family changes from Addendum 41's zero-tuning
    logistic regression to XGBoost here, per this addendum's own explicit
    instruction to standardize on XGBoost/tuned_xgboost_4_markets.py's grid
    for both markets -- flagged explicitly so the comparison against
    Addendum 41's number is read as "different model, not an apples-to-
    apples re-run."

Every merge's row count is printed before/after (ground rule 3 -- the exact
bug class already found and fixed in Addenda 37/40).
"""
import os
import pathlib
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import (  # noqa: E402
    DB, MIN_GRADED, gate, stat, profit, norm_name, implied_prob,
)
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_warehouse_pool, split_per_year,
)
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, build_games_v2, build_box_ids, build_features_v2, FEATURES_V2,
    BASELINES as TB_BASELINES,
)
from addendum41_runs_scored_holdout import (  # noqa: E402
    FEATURES as RS_FEATURES, load_pool as rs_load_pool,
)
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

FORM_WINDOW = 15  # single pre-specified trailing-game window, not tuned against ROI.
BATTER_POS = ["Catcher", "Hitter", "Infielder", "Outfielder"]
RAW_RATE_COLS = ["roll_hit_rate", "roll_tb_rate", "roll_rbi_rate"]

# Addendum 41's headline number, cited for direct comparison (not re-derived
# here) -- different model family (zero-tuning logistic regression there vs
# XGBoost here), flagged explicitly in the final summary.
RS_BASELINE = {"desc": "Addendum 41 explainable logistic regression (7 features, "
                       "zero tuning, chronological 70/30)",
               "n": 462, "roi": 0.89, "lo": -6.13, "hi": 7.91, "verdict": "FAIL"}


def build_batter_form_index(con):
    """Point-in-time rolling batter form index, computed directly from
    `boxscore` (spans 2020-07-23 to 2026-07-27 -- covers both markets'
    windows with real trailing history). For every (player_id, game_pk)
    batter row: three rolling per-at-bat rates over the player's own
    trailing FORM_WINDOW games STRICTLY BEFORE this game (shift(1) applied
    before the rolling window -- the current game's own at_bats/hits/
    total_bases/rbi never enter its own feature)."""
    b = con.execute(f"""
        SELECT player_id, game_pk, game_date, player_name,
               at_bats, hits, total_bases, rbi
        FROM boxscore
        WHERE position_type IN ({",".join("'" + p + "'" for p in BATTER_POS)})
    """).fetchdf()
    n_raw = len(b)
    b["game_date"] = pd.to_datetime(b["game_date"])
    b["game_pk"] = pd.to_numeric(b["game_pk"], errors="coerce")
    b = b.dropna(subset=["player_id", "game_pk", "game_date"]).copy()
    b["player_id"] = b["player_id"].astype("int64")
    b["game_pk"] = b["game_pk"].astype("int64")
    b["name_norm"] = b["player_name"].map(norm_name)
    for c in ["at_bats", "hits", "total_bases", "rbi"]:
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0)

    before = len(b)
    b = b.drop_duplicates(subset=["player_id", "game_pk"]).reset_index(drop=True)
    if len(b) != before:
        print(f"  form-index build: dropped {before - len(b)} exact (player_id, game_pk) "
              f"duplicate boxscore rows before computing rolling windows")

    b = b.sort_values(["player_id", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    def _roll(g):
        ab = g["at_bats"].shift(1).rolling(FORM_WINDOW, min_periods=1).sum()
        hits = g["hits"].shift(1).rolling(FORM_WINDOW, min_periods=1).sum()
        tb = g["total_bases"].shift(1).rolling(FORM_WINDOW, min_periods=1).sum()
        rbi = g["rbi"].shift(1).rolling(FORM_WINDOW, min_periods=1).sum()
        return pd.DataFrame({
            "roll_hit_rate": np.where(ab > 0, hits / ab, np.nan),
            "roll_tb_rate": np.where(ab > 0, tb / ab, np.nan),
            "roll_rbi_rate": np.where(ab > 0, rbi / ab, np.nan),
        }, index=g.index)

    rolled = b.groupby("player_id", group_keys=False).apply(_roll)
    form = pd.concat([b[["player_id", "game_pk", "game_date", "name_norm"]], rolled], axis=1)
    print(f"  batter form index: {n_raw} raw boxscore batter rows -> {len(form)} point-in-time "
          f"rolling rows (trailing {FORM_WINDOW}-game window, strictly-prior via shift(1) "
          f"before the rolling sum -- current game's own stats never enter its own feature)")
    return form


def add_composite(train, test, note):
    """z-scores the 3 raw point-in-time rolling rates using TRAIN-split-only
    mean/std (fit on train, applied to train AND test -- same discipline as
    multi_model_comparison.py's SimpleImputer/StandardScaler) and averages
    them (equal weights) into ONE scalar feature, batter_form_index."""
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
    print(f"  {note} batter_form_index coverage: train {train['batter_form_index'].notna().mean() * 100:.1f}%  "
          f"test {test['batter_form_index'].notna().mean() * 100:.1f}%  (missing = NaN, handled "
          f"natively by XGBoost, same convention as every other feature)")
    return train, test


def fit_cv_and_test(train, test, features, market_name, tag):
    X_train, y_train = train[features], train["win"].astype(int)
    X_test = test[features]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID,
        scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same established fix as
    # tuned_xgboost_4_markets.py / addendum37_historical_enrichment.py).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, evaluated once
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    print(f"  [{market_name} / {tag}]")
    print(f"    n(train/test)={len(train)}/{len(test)}  CV_AUC={search.best_score_:.4f}  "
          f"best_params={search.best_params_}")
    print(f"    edge>0.0 test (single pre-specified cut, official gate): n={s['n']}  "
          f"ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"market": market_name, "tag": tag, "n_train": len(train), "n_test": len(test),
            "cv_auc": round(search.best_score_, 4), "best_params": search.best_params_,
            **s, "verdict": verdict}


def run_total_bases(con, con2, form):
    print("\n" + "=" * 78)
    print("batter_total_bases -- real warehouse odds (2023-2025), per-year 75/25 split")
    print("baseline pipeline: build_games_v2/build_box_ids/build_features_v2/FEATURES_V2, "
          "imported unmodified from addendum37_historical_enrichment.py")
    print("=" * 78)

    games, box = build_games_and_box(con)
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    pool = build_warehouse_pool(con, box, "batter_total_bases", "total_bases", "batter")
    pool = build_features_v2(pool, games_v2, "batter", box_ids, con2)

    n_before = len(pool)
    form_by_gamepk = form.drop_duplicates(subset=["game_pk", "name_norm"])[
        ["game_pk", "name_norm"] + RAW_RATE_COLS]
    pool = pool.merge(form_by_gamepk, on=["game_pk", "name_norm"], how="left")
    ok = len(pool) == n_before
    print(f"\n  row-count sanity check (composite merge, LEFT join on game_pk+name_norm): "
          f"before={n_before}  after={len(pool)}  {'OK -- no fan-out' if ok else '*** WARNING: FAN-OUT, investigate ***'}")

    if len(pool) < MIN_GRADED:
        print(f"  n={len(pool)} INSUFFICIENT DATA (<{MIN_GRADED})")
        return None, None

    train, test = split_per_year(pool)
    if len(train) < 50 or len(test) < 50:
        print("  INSUFFICIENT DATA after split")
        return None, None
    train, test = add_composite(train, test, note="[batter_total_bases]")

    r_base = fit_cv_and_test(train, test, FEATURES_V2, "batter_total_bases", "baseline (FEATURES_V2, no composite)")
    r_comp = fit_cv_and_test(train, test, FEATURES_V2 + ["batter_form_index"],
                              "batter_total_bases", "FEATURES_V2 + batter_form_index")
    return r_base, r_comp


def run_runs_scored(con, form):
    print("\n" + "=" * 78)
    print("batter_runs_scored -- pick_history-only, 2026 window, chronological 70/30 split")
    print("baseline pipeline: load_pool/FEATURES, imported unmodified from "
          "addendum41_runs_scored_holdout.py; model changed to XGBoost per this addendum's spec")
    print("=" * 78)

    df, base_n = rs_load_pool(con)
    df["market_prob"] = implied_prob(df["odds"])
    df["game_date"] = pd.to_datetime(df["game_date"])

    n_before = len(df)
    form_sorted = form.sort_values(["player_id", "game_date", "game_pk"], kind="mergesort")
    # doubleheader safety: keep the EARLIEST game_pk's rolling value per
    # (player_id, game_date) -- the most conservative choice, guaranteed to
    # exclude even same-calendar-day action from the composite.
    form_by_date = form_sorted.drop_duplicates(subset=["player_id", "game_date"], keep="first")[
        ["player_id", "game_date"] + RAW_RATE_COLS]
    df = df.merge(form_by_date, on=["player_id", "game_date"], how="left")
    ok = len(df) == n_before
    print(f"\n  row-count sanity check (composite merge, LEFT join on player_id+game_date): "
          f"before={n_before}  after={len(df)}  {'OK -- no fan-out' if ok else '*** WARNING: FAN-OUT, investigate ***'}")

    pool = df.dropna(subset=RS_FEATURES + ["win", "odds", "game_date"]).copy()
    print(f"  after dropping rows missing any pre-registered baseline feature: n={len(pool)} "
          f"(dropped {n_before - len(pool)}; batter_form_index raw rates are NOT required "
          f"non-null here -- missing is handled natively by XGBoost, same as every other feature)")

    if len(pool) < MIN_GRADED:
        print(f"  n={len(pool)} INSUFFICIENT DATA (<{MIN_GRADED})")
        return None, None

    pool = pool.sort_values(["game_date", "id"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.70)
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"  train (chronological first 70%, through {train['game_date'].max()}): n={len(train)}")
    print(f"  test  (chronological last 30%, from {test['game_date'].min()}): n={len(test)}")
    if len(train) < 100 or len(test) < 20:
        print("  INSUFFICIENT DATA for a real holdout")
        return None, None

    train, test = add_composite(train, test, note="[batter_runs_scored]")

    r_base = fit_cv_and_test(train, test, RS_FEATURES, "batter_runs_scored", "baseline (Addendum 41 FEATURES, XGBoost, no composite)")
    r_comp = fit_cv_and_test(train, test, RS_FEATURES + ["batter_form_index"],
                              "batter_runs_scored", "Addendum 41 FEATURES + batter_form_index (XGBoost)")
    return r_base, r_comp


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("=" * 78)
    print("Addendum 45a: composite point-in-time 'batter form index' feature, tested on")
    print("batter_total_bases and batter_runs_scored -- each still gets its OWN individual")
    print("model, own gate evaluation, own pass/fail verdict. Composite built ONLY from")
    print(f"trailing {FORM_WINDOW}-game boxscore history, strictly prior to each game.")
    print("=" * 78)

    form = build_batter_form_index(con)

    tb_base, tb_comp = run_total_bases(con, con2, form)
    con2.close()

    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")
    rs_base, rs_comp = run_runs_scored(con, form)

    con.close()

    print("\n" + "=" * 78)
    print("SUMMARY: CV AUC and test gate result, with vs without the composite feature")
    print("=" * 78)

    for label, base, comp, prior in [
        ("batter_total_bases", tb_base, tb_comp, TB_BASELINES["batter_total_bases"]),
        ("batter_runs_scored", rs_base, rs_comp, RS_BASELINE),
    ]:
        print(f"\n{label}")
        print(f"  prior best result [{prior['desc']}]: n={prior['n']}  ROI={prior['roi']}  "
              f"CI=[{prior['lo']},{prior['hi']}]  {prior['verdict']}")
        if base is None or comp is None:
            print("  Addendum 45a: could not be evaluated (insufficient data -- see above)")
            continue
        print(f"  Addendum 45a baseline (no composite):   CV_AUC={base['cv_auc']}  "
              f"test n={base['n']}  ROI={base['roi']}  CI=[{base['lo']},{base['hi']}]  {base['verdict']}")
        print(f"  Addendum 45a + batter_form_index:        CV_AUC={comp['cv_auc']}  "
              f"test n={comp['n']}  ROI={comp['roi']}  CI=[{comp['lo']},{comp['hi']}]  {comp['verdict']}")
        delta = round(comp['cv_auc'] - base['cv_auc'], 4)
        print(f"  marginal CV-AUC contribution of batter_form_index: {delta:+.4f}")

    print("\n" + "=" * 78)
    print("Both markets reported as-is, pass or fail, single pre-specified grid/cut/feature -- "
          "same discipline as every prior addendum. A new feature that improves CV AUC but "
          "still fails the gate is an honest FAIL, not grounds to keep searching.")
    print("=" * 78)


if __name__ == "__main__":
    main()
