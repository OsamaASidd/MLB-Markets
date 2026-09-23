"""
Addendum 45b: a single new engineered feature -- a starting pitcher's own
"dominance index" -- tried on BOTH pitcher_strikeouts (real warehouse odds,
2023-2025) and pitcher_outs (pick_history-only, 2026 window). Each market
still gets its own model, its own CV-AUC-only selection, its own single
edge>0.0 cut, its own official-gate verdict -- this is one new feature
tried on two already-established pipelines, not a pooled model.

MOTIVATION: a starting pitcher's strikeout rate and how deep he works into
games are mechanically driven by the same underlying "how dominant is this
pitcher right now" latent quality -- missing more bats also means fewer
balls in play, more efficient innings, and (all else equal) more outs
recorded before a manager pulls him. Neither pitcher_strikeouts (Addenda 9,
17, 34, 37, 40, 43a) nor pitcher_outs (Addenda 27, 30, 34, 38, 43b) has ever
combined a pitcher's own recent K-rate trend with their own recent
innings-depth trend into one composite. That's the new idea tested here.

COMPOSITE CONSTRUCTION (point-in-time-safe by construction):
For every real starting-pitcher outing in `boxscore` (position_type=
'Pitcher', is_starter=True, strikeouts and innings_pitched both present,
2020-07-23 to 2026-07-27 coverage -- spans both markets' test windows),
compute, per (player_id, game_pk, game_date):
  - k_per9_roll   = (sum of strikeouts over that pitcher's own last N=5
                     starts STRICTLY BEFORE this game_date) /
                    (sum of outs-recorded over those same 5 starts) * 27
  - outs_per_start_roll = mean outs-recorded over those same 5 starts
Both are computed with `.shift(1)` applied BEFORE the rolling window, so the
current start is never included in its own trailing average -- the very
first tracked start of a pitcher's career (in this dataset) correctly gets
NaN (there is no real trailing history yet), not a leaked/zero value.
innings_pitched -> outs via ip_to_outs(), imported unmodified from
xgboost_individual_markets.py (the same helper used for bullpen fatigue
elsewhere in this project). N=5 is a single pre-specified choice (a
starting pitcher makes ~30 starts/season vs. a batter's ~150+ games, so a
much shorter trailing window than the batter-side L10/L15 conventions this
project already uses) -- NOT tuned against any ROI number below.

Both rolling rates are then z-scored (population mean/std over the full
rolling table, itself built with zero use of any game's own outcome or any
future game) and averaged into ONE composite, `pitcher_dominance_index`.
Feeding this into XGBoost specifically: tree splits are invariant to any
monotonic (here, affine) rescaling of a single input feature, so the exact
reference population used for the z-score has NO effect on the tuned
model's splits, predictions, or downstream ROI -- the z-score is included
purely so the composite is interpretable as a single signed "dominance"
score (positive = missing more bats AND going deeper than this pitcher's
own recent baseline), not because it changes what XGBoost learns.

Why this is point-in-time-safe: every input to the composite for a given
game is drawn exclusively from that SAME pitcher's own starts with a
game_date strictly before the game being predicted -- never the same game,
never a future game, never another pitcher's data. This mirrors the exact
discipline already established for opp_lineup_k_rate (Addendum 40) and the
bullpen/starter enrichment (Addendum 37), just applied to the pitcher's own
trailing form instead of an opponent's or a team's.

MANDATORY SANITY CHECKS (Addendum 37/40's own discovered bug class -- a
duplicate join key silently fanning out the row count): row counts printed
before/after every merge below; a >1% unexpected change raises immediately
rather than being trusted silently. The (player_id, game_pk) dedup below is
the SAME pre-existing data-quality issue Addendum 37 (`games`, 4,547 vs.
3,845 unique game_pk) and Addendum 40 (`team_game`, 52 duplicate rows) both
independently found and fixed the same way (keep the earliest game_date per
duplicated key) -- not a new bug, the same one recurring in a third table
built from the same underlying boxscore source.

METHOD, market by market:
  pitcher_strikeouts -- reuses build_games_v2 / build_box_ids /
    build_features_v2 / FEATURES_V2 from addendum37_historical_enrichment.py
    UNMODIFIED (imported, not edited), same per-year 75/25 split, same
    PARAM_GRID/FIXED_PARAMS (tuned_xgboost_4_markets.py) CV-AUC-selection
    pattern already used by Addendum 37/40 (reuses Addendum 40's own
    tune_and_evaluate() function directly, unmodified, since it already
    accepts an arbitrary `features` list and always uses split_per_year).
    Two configurations run on the IDENTICAL pool/train/test rows: (a)
    FEATURES_V2 alone (baseline, reproduces Addendum 37's own number), (b)
    FEATURES_V2 + pitcher_dominance_index -- isolates the new feature's
    marginal contribution on an apples-to-apples split.
  pitcher_outs -- reuses Addendum 38's pick_history base query, ASOF cache
    joins (PITCHER_OUTS_JOIN/PITCHER_OUTS_SELECT, load_enriched_pool, all
    imported unmodified from addendum38_pickhistory_enrichment.py), same
    score_*/cache coverage floor (>=200 rows or >=5%, whichever larger).
    Split changed to chronological 70/30 (Addendum 41/43b's convention,
    per this task's explicit instruction), NOT Addendum 38's original
    75/25, so this script's own baseline number will differ slightly from
    Addendum 38's committed 75/25 result -- both are printed side by side,
    difference flagged, not hidden. Two configurations on the IDENTICAL
    pool/train/test rows: (a) score_*+cache features alone (baseline), (b)
    + pitcher_dominance_index.

Every result reported, pass or fail, single pre-specified edge>0.0 cut,
official gate() rule (xgboost_individual_markets.py, verified against the
client's real betgenius/harness/lib/metrics.ts). No threshold search folds
into either headline verdict.
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
    DB, MIN_GRADED, gate, stat, profit, norm_name, ip_to_outs, implied_prob,
)
from multi_model_comparison import build_games_and_box, build_warehouse_pool, split_per_year  # noqa: E402
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    build_games_v2, build_box_ids, build_features_v2, FEATURES_V2, CACHE_DB, BASELINES,
)
from addendum38_pickhistory_enrichment import (  # noqa: E402
    PITCHER_OUTS_JOIN, PITCHER_OUTS_SELECT, load_enriched_pool,
)

N_STARTS = 5  # pre-specified rolling window (starts, not games) -- NOT tuned against ROI

# Cited, already-committed baselines for direct before/after context.
BASELINE_STRIKEOUTS_ADD37 = {
    "desc": "Addendum 37 FEATURES_V2 (bullpen+starters+park), no pitcher_dominance_index",
    "cv_auc": 0.541, "n": 360, "roi": -3.81, "verdict": "FAIL",
}
BASELINE_OUTS_ADD38 = {
    "desc": "Addendum 38 score_*+cache features, 75/25 split (this script uses 70/30 per task spec)",
    "n_train": 598, "n_test": 200, "cv_auc": 0.517, "n": 45, "roi": -18.09,
    "lo": -45.21, "hi": 9.03, "verdict": "FAIL",
}


# ---------------------------------------------------------------------------
# The composite feature itself -- built once, shared by both markets.
# ---------------------------------------------------------------------------

def build_pitcher_dominance_index(con, n_starts=N_STARTS):
    """Point-in-time composite: rolling K-per-9 and rolling outs-per-start
    over a starting pitcher's own last N=5 starts strictly before the
    current game_date, z-scored and averaged. See module docstring for the
    full point-in-time justification."""
    bx = con.execute("""
        SELECT player_id, game_pk, game_date, strikeouts, innings_pitched
        FROM boxscore
        WHERE position_type = 'Pitcher' AND is_starter = True
          AND strikeouts IS NOT NULL AND innings_pitched IS NOT NULL
    """).fetchdf()
    bx["game_date"] = pd.to_datetime(bx["game_date"])
    bx["outs"] = bx["innings_pitched"].map(ip_to_outs)
    n_raw = len(bx)

    # Same pre-existing (player_id, game_pk) -> >1 game_date data issue
    # Addendum 37 (`games`) and Addendum 40 (`team_game`) both independently
    # found and fixed in this exact same underlying boxscore source -- kept
    # earliest game_date per pair, same convention.
    bx = bx.sort_values(["player_id", "game_pk", "game_date"], kind="mergesort")
    dup_before = bx.duplicated(subset=["player_id", "game_pk"], keep=False).sum()
    bx = bx.drop_duplicates(subset=["player_id", "game_pk"], keep="first").reset_index(drop=True)
    if dup_before:
        print(f"  [pitcher_dominance_index dedup] {dup_before} duplicate (player_id, game_pk) rows "
              f"found (pre-existing game_pk/game_date data issue, same class as Addendum 37/40's own "
              f"dedups) -- kept earliest game_date per pair, {n_raw} -> {len(bx)} rows")
    dup = bx.duplicated(subset=["player_id", "game_pk"]).sum()
    assert dup == 0, f"build_pitcher_dominance_index: {dup} duplicate (player_id, game_pk) rows before rolling"

    bx = bx.sort_values(["player_id", "game_date", "game_pk"], kind="mergesort")

    def roll(grp):
        grp = grp.sort_values(["game_date", "game_pk"], kind="mergesort")
        k_roll = grp["strikeouts"].shift(1).rolling(n_starts, min_periods=1).sum()
        outs_roll_sum = grp["outs"].shift(1).rolling(n_starts, min_periods=1).sum()
        outs_roll_mean = grp["outs"].shift(1).rolling(n_starts, min_periods=1).mean()
        grp = grp.copy()
        grp["k_per9_roll"] = np.where(outs_roll_sum > 0, k_roll / outs_roll_sum * 27.0, np.nan)
        grp["outs_per_start_roll"] = outs_roll_mean
        return grp

    bx = bx.groupby("player_id", group_keys=False).apply(roll)
    out = bx[["player_id", "game_pk", "game_date", "k_per9_roll", "outs_per_start_roll"]].reset_index(drop=True)

    dup2 = out.duplicated(subset=["player_id", "game_pk"]).sum()
    assert dup2 == 0, f"build_pitcher_dominance_index: {dup2} duplicate (player_id, game_pk) rows after rolling"

    for col in ["k_per9_roll", "outs_per_start_roll"]:
        mu, sd = out[col].mean(), out[col].std(ddof=0)
        out[col + "_z"] = (out[col] - mu) / sd

    # skipna=False: composite requires BOTH components to have a real
    # trailing estimate (they're computed from the identical window of the
    # identical starts, so this only differs when outs_roll_sum was 0 for
    # one but not the other -- an edge case, not the common case).
    out["pitcher_dominance_index"] = out[["k_per9_roll_z", "outs_per_start_roll_z"]].mean(axis=1, skipna=False)

    cov = round(100 * out["pitcher_dominance_index"].notna().mean(), 1)
    print(f"  pitcher_dominance_index table: {len(out)} (player_id, game_pk) starter-rows, "
          f"N={n_starts}-start trailing window, {cov}% non-null overall "
          f"(NaN = pitcher's first tracked start(s), correctly no trailing history yet)")
    return out


def merge_dominance_asof(df, composite):
    """Attaches pitcher_dominance_index by player_id via the most recent
    row in `composite` with game_date <= this row's game_date (merge_asof,
    direction='backward') -- same pattern as Addendum 37's own bullpen
    asof() join. Guaranteed to preserve row count exactly (merge_asof
    matches at most one right row per left row, and rows with a missing
    player_id are handled separately below rather than dropped) -- asserted
    at the end anyway, per this project's own sanity-check discipline.
    player_id arrives as a pandas nullable Int64 from pick_history's
    TRY_CAST(... AS BIGINT) vs. boxscore's native int64 in `composite` --
    both coerced to float64 here so merge_asof's dtype check doesn't choke
    on the mismatch (values themselves are small integer IDs, exactly
    representable as float64, so no precision is lost)."""
    n0 = len(df)
    d = pd.DataFrame({
        "_idx": df.index,
        # .astype("float64") (not pd.to_numeric, which is a no-op on an
        # already-numeric dtype and was the actual bug on the first attempt
        # here) forces BOTH sides to the identical numpy float64, since
        # pick_history's player_id arrives as a pandas nullable Int64
        # (TRY_CAST(...AS BIGINT)) while boxscore's is a plain numpy int64
        # -- merge_asof's dtype check otherwise rejects the two as
        # incompatible even though the underlying values match.
        "player_id": df["player_id"].astype("float64"),
        # explicit datetime64[ns]: pandas 2.x infers datetime64[us] for
        # values that came from DuckDB's DATE type (pick_history) vs. the
        # datetime64[ns] pandas infers when parsing boxscore's VARCHAR
        # game_date -- merge_asof's dtype check rejects the unit mismatch
        # the same way it rejected the player_id dtype mismatch above.
        "game_date": pd.to_datetime(df["game_date"]).astype("datetime64[ns]").values,
    })
    comp = composite[["player_id", "game_date", "pitcher_dominance_index"]].copy()
    comp["player_id"] = comp["player_id"].astype("float64")
    comp["game_date"] = pd.to_datetime(comp["game_date"]).astype("datetime64[ns]")

    has_pid = d["player_id"].notna()
    d_valid = d[has_pid].sort_values("game_date")
    comp_sorted = comp.sort_values("game_date")

    merged = pd.merge_asof(d_valid, comp_sorted, on="game_date", by="player_id", direction="backward")

    result = pd.Series(np.nan, index=df.index, dtype="float64")
    result.loc[merged["_idx"].values] = merged["pitcher_dominance_index"].values
    n_missing_pid = int((~has_pid).sum())
    if n_missing_pid:
        print(f"    ({n_missing_pid} rows had no player_id -- pitcher_dominance_index left NaN "
              f"for those, same missing=np.nan handling as every other feature in this project)")
    assert len(result) == n0, (
        f"merge_dominance_asof: row count changed {n0} -> {len(result)} -- investigate")
    return result.values


# ---------------------------------------------------------------------------
# pitcher_strikeouts (real warehouse odds, 2023-2025, Addendum 37 pipeline)
# ---------------------------------------------------------------------------

def tune_and_evaluate_strikeouts(config_name, pool, features, split_mode="per_year"):
    """Same GridSearchCV/StratifiedKFold/PARAM_GRID/FIXED_PARAMS CV-AUC-
    selection discipline as Addendum 37/40 (structurally identical to
    Addendum 40's own tune_and_evaluate(), reimplemented locally here only
    because that function hardcodes a coverage print for its OWN feature,
    opp_lineup_k_rate -- not reusable as-is for a different new feature)."""
    if len(pool) < 500:
        print(f"  {config_name}: n={len(pool)} INSUFFICIENT DATA (<500)")
        return None
    train, test = split_per_year(pool)
    if len(train) < 50 or len(test) < 50:
        print(f"  {config_name}: INSUFFICIENT DATA after split")
        return None

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same fix used throughout this
    # project's addenda).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    print(f"  {config_name}")
    print(f"    features ({len(features)}): {features}")
    print(f"    n(train/test)={len(train)}/{len(test)}")
    print(f"    best CV AUC={search.best_score_:.3f}  params={search.best_params_}")
    print(f"    edge>0.0 bets: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"config": config_name, "n_train": len(train), "n_test": len(test),
            "cv_auc": round(search.best_score_, 3), "best_params": search.best_params_,
            **s, "verdict": verdict}


def run_pitcher_strikeouts(con, con2, dominance):
    print("\n" + "=" * 78)
    print("pitcher_strikeouts (real warehouse odds, Addendum 37 pipeline)")
    print("=" * 78)

    print("building real team game log, Elo, L10, bullpen fatigue, park factors "
          "(existing pipeline, unmodified import)...")
    games, box = build_games_and_box(con)
    print("extending with Addendum 37 historical features (unmodified import): point-in-time "
          "bullpen quality, both-starters + opposing-starter statcast quality, park dims...")
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    pool = build_warehouse_pool(con, box, "pitcher_strikeouts", "strikeouts", "pitcher")
    n0 = len(pool)
    pool = build_features_v2(pool, games_v2, "pitcher", box_ids, con2)
    n1 = len(pool)
    print(f"  [row-count check] raw warehouse pool: n={n0}  ->  after build_features_v2 "
          f"(Addendum 37, unmodified): n={n1} ({100.0 * (n1 - n0) / n0:+.1f}%)")
    print(f"    (this drop is build_features_v2's own established, unmodified INNER join onto "
          f"games_v2 -- reproduces Addendum 40's own identical n=18779 -> 12679 for this exact "
          f"pool, not a new bug introduced here; the strict no-fan-out checks below apply to the "
          f"NEW merges this script adds on top: player_id attach and pitcher_dominance_index attach)")

    pool = pool.merge(box_ids[["game_pk", "name_norm", "player_id"]], on=["game_pk", "name_norm"], how="left")
    n2 = len(pool)
    print(f"  [row-count check] after attaching player_id via box_ids: n={n2}")
    if n2 != n1:
        raise RuntimeError(f"row-count changed attaching player_id ({n1} -> {n2}) -- box_ids "
                            f"is supposed to be deduplicated by (game_pk, name_norm); investigate")

    pool = pool.merge(dominance[["player_id", "game_pk", "pitcher_dominance_index"]],
                       on=["player_id", "game_pk"], how="left")
    n3 = len(pool)
    print(f"  [row-count check] after attaching pitcher_dominance_index (exact merge on "
          f"[player_id, game_pk], composite table deduplicated by construction): n={n3}")
    if n3 != n2:
        raise RuntimeError(f"row-count changed attaching pitcher_dominance_index ({n2} -> {n3}) -- "
                            f"composite table has a duplicate (player_id, game_pk) key; investigate")

    cov = round(100 * pool["pitcher_dominance_index"].notna().mean(), 1)
    print(f"  pitcher_dominance_index coverage in this pool: {cov}%")

    results = []
    print("\n--- Configuration (a): FEATURES_V2 alone (baseline, reproduces Addendum 37) ---")
    r_a = tune_and_evaluate_strikeouts("(a) FEATURES_V2 (no composite)", pool, FEATURES_V2, "per_year")
    if r_a:
        results.append(r_a)

    print("\n--- Configuration (b): FEATURES_V2 + pitcher_dominance_index ---")
    features_b = FEATURES_V2 + ["pitcher_dominance_index"]
    r_b = tune_and_evaluate_strikeouts("(b) FEATURES_V2 + pitcher_dominance_index", pool, features_b, "per_year")
    if r_b:
        results.append(r_b)

    return results


# ---------------------------------------------------------------------------
# pitcher_outs (pick_history-only, 2026 window, Addendum 38 pipeline)
# ---------------------------------------------------------------------------

def tune_and_evaluate_pickhistory(config_name, df, features, cut_frac=0.70):
    """Same GridSearchCV/StratifiedKFold/PARAM_GRID/FIXED_PARAMS CV-AUC-
    selection discipline as every other addendum, chronological 70/30 split
    (Addendum 41/43b's convention, per this task's explicit instruction --
    NOT Addendum 38's original 75/25)."""
    n_total = len(df)
    if n_total < MIN_GRADED:
        print(f"  {config_name}: n={n_total} INSUFFICIENT DATA (<{MIN_GRADED})")
        return None

    df = df.sort_values(["game_date", "id"], kind="mergesort").reset_index(drop=True)
    cut = int(n_total * cut_frac)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    if len(train) < 100 or len(test) < 20:
        print(f"  {config_name}: INSUFFICIENT DATA after split")
        return None

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (same fix used throughout this
    # project's addenda).
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]

    t = test.copy()
    t["model_prob"] = test_pred
    t["market_prob"] = implied_prob(t["odds"])
    t["edge"] = t["model_prob"] - t["market_prob"]
    sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"

    print(f"  {config_name}")
    print(f"    features ({len(features)}): {features}")
    print(f"    n(train/test)={len(train)}/{len(test)}")
    print(f"    best CV AUC={search.best_score_:.3f}  params={search.best_params_}")
    print(f"    edge>0.0 bets: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"config": config_name, "n_train": len(train), "n_test": len(test),
            "cv_auc": round(search.best_score_, 3), "best_params": search.best_params_,
            **s, "verdict": verdict}


def run_pitcher_outs(con, dominance):
    print("\n" + "=" * 78)
    print("pitcher_outs (pick_history-only, 2026 window, Addendum 38 pipeline)")
    print("=" * 78)

    df, score_cols = load_enriched_pool(con, "pitcher_outs", PITCHER_OUTS_JOIN, PITCHER_OUTS_SELECT)
    n0 = len(df)
    print(f"  [row-count check] pick_history pool + Addendum 38 ASOF cache joins (unmodified "
          f"import): n={n0}")

    df["pitcher_dominance_index"] = merge_dominance_asof(df, dominance)
    n1 = len(df)
    print(f"  [row-count check] after attaching pitcher_dominance_index (merge_asof by player_id, "
          f"direction='backward'): n={n1}")
    if n1 != n0:
        raise RuntimeError(f"row-count changed attaching pitcher_dominance_index ({n0} -> {n1}) -- "
                            f"investigate before trusting anything downstream")

    cov = round(100 * df["pitcher_dominance_index"].notna().mean(), 1)
    print(f"  pitcher_dominance_index coverage in this pool: {cov}%")

    new_cols = [c for c in df.columns if c.startswith("ck_")]
    min_cov = max(200, n0 * 0.05)
    score_keep = [c for c in score_cols if df[c].notna().sum() >= min_cov]
    cache_keep = [c for c in new_cols if df[c].notna().sum() >= min_cov]
    baseline_features = score_keep + cache_keep
    print(f"  baseline features kept (coverage >= {min_cov:.0f} rows): "
          f"{len(score_keep)}/{len(score_cols)} score_* + {len(cache_keep)}/{len(new_cols)} cache "
          f"= {len(baseline_features)} total")

    results = []
    print("\n--- Configuration (a): score_*+cache features alone (baseline, 70/30 split) ---")
    r_a = tune_and_evaluate_pickhistory("(a) score_*+cache (no composite)", df, baseline_features)
    if r_a:
        results.append(r_a)

    print("\n--- Configuration (b): score_*+cache + pitcher_dominance_index ---")
    features_b = baseline_features + ["pitcher_dominance_index"]
    r_b = tune_and_evaluate_pickhistory("(b) score_*+cache + pitcher_dominance_index", df, features_b)
    if r_b:
        results.append(r_b)

    return results


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=" * 78)
    print("Addendum 45b: pitcher dominance index (rolling K-per-9 + rolling outs-per-start, "
          f"N={N_STARTS} trailing starts, z-scored composite) -- new feature tried on both "
          "pitcher_strikeouts and pitcher_outs, each market's own model/gate/verdict.")
    print("=" * 78)

    print("\nbuilding pitcher_dominance_index from real boxscore starts (2020-2026 coverage, "
          "point-in-time by construction)...")
    dominance = build_pitcher_dominance_index(con, N_STARTS)

    strikeouts_results = run_pitcher_strikeouts(con, con2, dominance)
    outs_results = run_pitcher_outs(con, dominance)

    con.close()
    con2.close()

    print("\n" + "=" * 78)
    print("SUMMARY: Addendum 45b, pitcher_dominance_index, both markets, isolated marginal effect")
    print("=" * 78)

    print("\npitcher_strikeouts")
    print(f"  BEFORE [{BASELINE_STRIKEOUTS_ADD37['desc']}]: CV_AUC={BASELINE_STRIKEOUTS_ADD37['cv_auc']}  "
          f"n={BASELINE_STRIKEOUTS_ADD37['n']}  ROI={BASELINE_STRIKEOUTS_ADD37['roi']}  "
          f"{BASELINE_STRIKEOUTS_ADD37['verdict']}")
    for r in strikeouts_results:
        print(f"  {r['config']}: n(train/test)={r['n_train']}/{r['n_test']}  CV_AUC={r['cv_auc']}  "
              f"test n={r['n']} ROI={r['roi']} CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    print("\npitcher_outs")
    b = BASELINE_OUTS_ADD38
    print(f"  BEFORE [{b['desc']}]: n(train/test)={b['n_train']}/{b['n_test']}  CV_AUC={b['cv_auc']}  "
          f"n={b['n']}  ROI={b['roi']}  CI=[{b['lo']},{b['hi']}]  {b['verdict']}")
    for r in outs_results:
        print(f"  {r['config']}: n(train/test)={r['n_train']}/{r['n_test']}  CV_AUC={r['cv_auc']}  "
              f"test n={r['n']} ROI={r['roi']} CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    all_results = strikeouts_results + outs_results
    passes = [r for r in all_results if r["verdict"] == "PASS"]
    print(f"\n{len(passes)} of {len(all_results)} configurations pass the official gate rule. "
          f"Reported as-is, pass or fail, single pre-specified N/grid/cut -- same discipline as "
          f"every prior addendum. A FAIL here is a valid, reportable result, not grounds to "
          f"search further windows, combination methods, or thresholds against this test set.")


if __name__ == "__main__":
    main()
