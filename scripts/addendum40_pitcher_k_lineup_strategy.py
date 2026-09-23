"""
Addendum 40: opposing-lineup strikeout-rate feature for pitcher_strikeouts.

Motivation (per HANDOFF.md's open-research framing, not a rescue attempt):
Addendum 37 added opposing-pitcher-quality and bullpen features to
pitcher_strikeouts and still only cleared CV AUC=0.541 / test ROI=-3.81%
(n=360, FAIL) -- see reports/MILESTONE_1_GATE_REPORT.md's addenda trail.
No strikeout-specific opposing-BATTER signal has ever been used for this
market, despite this project already having real per-player boxscore K
data (2023-2026) sitting in the warehouse. A starting pitcher's strikeout
total depends heavily on how strikeout-prone the batting lineup he's
facing is -- that's genuinely new information, not a re-cut of existing
features, and it's self-computable from `boxscore` with no new external
data pull.

New feature: `opp_lineup_k_rate` -- for the team BATTING against the
pitcher in this game (the lineup the pitcher's Ks are drawn from), their
point-in-time rolling strikeout rate (sum(batter_strikeouts) /
sum(plate_appearances)) over that team's most recent N=15 games strictly
BEFORE this game's date. N=15 is a single pre-specified choice (a
standard "recent form" window, matching the spirit of this project's
existing L10 features) -- not grid-searched against test ROI, per the
task's ground rule.

Two isolated configurations reported, pass or fail, no cherry-picking:
  (a) FEATURES_V2 (Addendum 37's full enrichment: bullpen quality, both
      starters' statcast quality, opposing-starter statcast, park
      dimensions) + opp_lineup_k_rate.
  (b) The ORIGINAL baseline features only (side_code, market_prob,
      elo_diff, l10_diff, fatigue_diff, runs_factor, hr_factor, k_factor,
      hits_factor) + opp_lineup_k_rate -- isolates whether this specific
      new feature helps on its own, without Addendum 37's other additions
      riding along.

Method is the exact same disciplined tuning/selection/gate methodology as
Addendum 37 / 34 (scripts/tuned_xgboost_4_markets.py):
  - ONE model family: XGBoost.
  - ONE pre-specified hyperparameter grid: PARAM_GRID/FIXED_PARAMS
    imported as-is from tuned_xgboost_4_markets.py, not widened.
  - Selection metric: CV-averaged ROC AUC on the TRAIN split only
    (5-fold StratifiedKFold via GridSearchCV). ROI/gate is never touched
    during selection.
  - Refit once, score once on the untouched test split, single
    pre-specified edge>0.0 cut (odds<0), official gate rule (gate() in
    xgboost_individual_markets.py: n>=500 -> ROI>0; else CI lower
    bound>0).
  - Same per-year 75/25 split pitcher_strikeouts already uses (split_per_year,
    per multi_model_comparison.py / Addendum 37).
  - Both configurations reported, pass or fail. Test set touched exactly
    once per configuration, at the end.

Reuses build_games_v2 / build_box_ids / build_features_v2 / FEATURES_V2
from addendum37_historical_enrichment.py, and FEATURES / build_games_and_box /
build_warehouse_pool / build_features / split_per_year from
multi_model_comparison.py, and PARAM_GRID / FIXED_PARAMS from
tuned_xgboost_4_markets.py -- all imported unmodified, none edited.

Join-fan-out sanity check (per Addendum 37's own discovered bug, where a
duplicate-key merge caused a ~200x row explosion and spuriously high CV
AUC before being caught): every merge this script performs to attach
opp_lineup_k_rate is keyed on columns already deduplicated upstream
(box_ids by [game_pk, name_norm], the game/team-id context by [game_pk],
the rolling K-rate table by [team_id, game_pk]) -- pool row counts are
printed before/after each merge below and asserted not to grow beyond a
tiny epsilon (rounding/float artifacts aside), consistent with a clean
many-to-one join.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
import joblib  # noqa: E402
import os  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from xgboost_individual_markets import DB, gate, stat, profit, norm_name  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    FEATURES, build_games_and_box, build_warehouse_pool, build_features,
    split_per_year,
)
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    build_games_v2, build_box_ids, build_features_v2, FEATURES_V2, CACHE_DB,
)

N_GAMES = 15  # pre-specified rolling window -- NOT tuned against test ROI

# Addendum 37's own cited pre-enrichment baseline, quoted here for direct
# before/after context (not re-derived -- see that script's BASELINES dict).
BASELINE_ADD37 = {"desc": "Addendum 37 FEATURES_V2 (bullpen+starters+park), no opp_lineup_k_rate",
                   "n": 360, "roi": -3.81, "verdict": "FAIL"}


def build_team_k_rate(con, n_games=N_GAMES):
    """Point-in-time team batting strikeout rate: rolling
    sum(batter_strikeouts) / sum(plate_appearances) over the team's most
    recent n_games games STRICTLY BEFORE the current game (shift(1) before
    the rolling window -- no lookahead). Computed directly from `boxscore`
    (2020-2026 coverage, no new data pull). Rows missing either
    batter_strikeouts or plate_appearances are excluded from both sums so
    the numerator/denominator stay consistent per game."""
    bx = con.execute("""
        SELECT game_pk, game_date, team_id, batter_strikeouts, plate_appearances
        FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
          AND batter_strikeouts IS NOT NULL AND plate_appearances IS NOT NULL
    """).fetchdf()
    bx["game_date"] = pd.to_datetime(bx["game_date"])

    team_game = (bx.groupby(["team_id", "game_pk", "game_date"], as_index=False)
                   .agg(team_k=("batter_strikeouts", "sum"), team_pa=("plate_appearances", "sum")))

    # Same pre-existing game_pk/game_date data-quality issue Addendum 37
    # documented for `games` (a handful of game_pk values carry >1
    # game_date, here always the very next calendar day, with IDENTICAL
    # team_k/team_pa -- a duplicate pull of the same game under two dates,
    # not a real doubleheader). Verified: 26 such (team_id, game_pk) pairs,
    # 2023 season only, stats identical across the duplicate date. Deduped
    # by keeping the EARLIEST game_date per (team_id, game_pk), same
    # keep="first"-after-ascending-sort convention build_games_v2 already
    # uses for the analogous `games` table dedup.
    team_game = team_game.sort_values(["team_id", "game_pk", "game_date"], kind="mergesort")
    dup_before = team_game.duplicated(subset=["team_id", "game_pk"], keep=False).sum()
    team_game = team_game.drop_duplicates(subset=["team_id", "game_pk"], keep="first").reset_index(drop=True)
    if dup_before:
        print(f"    [team_k_rate dedup] {dup_before} duplicate (team_id, game_pk) rows found "
              f"(pre-existing game_pk/game_date data issue, same class as Addendum 37's `games` "
              f"dedup) -- kept earliest game_date per pair, {len(team_game)} rows remain")

    dup = team_game.duplicated(subset=["team_id", "game_pk"]).sum()
    assert dup == 0, f"build_team_k_rate: {dup} duplicate (team_id, game_pk) rows before rolling"

    team_game = team_game.sort_values(["team_id", "game_date", "game_pk"], kind="mergesort")

    def roll(grp):
        grp = grp.sort_values(["game_date", "game_pk"], kind="mergesort")
        k_roll = grp["team_k"].shift(1).rolling(n_games, min_periods=1).sum()
        pa_roll = grp["team_pa"].shift(1).rolling(n_games, min_periods=1).sum()
        grp = grp.copy()
        grp["opp_lineup_k_rate"] = k_roll / pa_roll
        return grp

    team_game = team_game.groupby("team_id", group_keys=False).apply(roll)
    out = team_game[["team_id", "game_pk", "opp_lineup_k_rate"]].reset_index(drop=True)
    dup2 = out.duplicated(subset=["team_id", "game_pk"]).sum()
    assert dup2 == 0, f"build_team_k_rate: {dup2} duplicate (team_id, game_pk) rows after rolling"
    return out


def add_opp_lineup_k_rate(pool, game_ctx, box_ids, team_k_rate, label=""):
    """Attaches opp_lineup_k_rate -- the rolling K-rate of the LINEUP
    BATTING against the pitcher in this game (the opposing team relative
    to the pitcher whose strikeout prop this row is), not the pitcher's
    own team. Every merge is on keys already verified/deduplicated
    upstream (box_ids: [game_pk, name_norm]; game_ctx: [game_pk];
    team_k_rate: [team_id, game_pk]) -- row counts printed before/after
    each merge as a join-fan-out sanity check."""
    n0 = len(pool)

    pool = pool.merge(box_ids[["game_pk", "name_norm", "team_id"]],
                       on=["game_pk", "name_norm"], how="left")
    n1 = len(pool)

    pool = pool.merge(game_ctx, on="game_pk", how="left")
    n2 = len(pool)

    pool["opp_team_id"] = np.where(pool["team_id"] == pool["home_team_id"],
                                    pool["away_team_id"], pool["home_team_id"])

    pool = pool.merge(team_k_rate.rename(columns={"team_id": "opp_team_id"}),
                       on=["opp_team_id", "game_pk"], how="left")
    n3 = len(pool)

    print(f"    [join-fan-out check{(' ' + label) if label else ''}] "
          f"n0={n0} -> +box_ids(pitcher team_id)={n1} -> +game_ctx(home/away)={n2} "
          f"-> +team_k_rate(opp_lineup_k_rate)={n3}")
    for stage, before, after in [("box_ids", n0, n1), ("game_ctx", n1, n2), ("team_k_rate", n2, n3)]:
        if before > 0 and after > before * 1.10:
            raise RuntimeError(f"join-fan-out detected at {stage}: {before} -> {after} rows "
                                f"(>10% growth) -- investigate before trusting any result")
    assert n0 == n3, f"row count changed across left-join chain: {n0} -> {n3} (should be identical for left joins on deduplicated keys)"

    return pool


def tune_and_evaluate(config_name, pool, features, split_mode):
    if len(pool) < 500:
        print(f"  {config_name}: n={len(pool)} INSUFFICIENT DATA (<500)")
        return None
    train, test = split_per_year(pool)  # pitcher_strikeouts always uses the per-year split
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
    # Python -- threading backend instead (same fix as tuned_xgboost_4_markets.py
    # / addendum37, and matches this project's standing "always parallelize,
    # threading backend as fallback on this sandboxed Windows Python" note).
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

    coverage = round(100 * pool["opp_lineup_k_rate"].notna().mean(), 1)

    print(f"  {config_name}")
    print(f"    features ({len(features)}): {features}")
    print(f"    n(train/test)={len(train)}/{len(test)}")
    print(f"    opp_lineup_k_rate coverage (% non-null, full pool): {coverage}%")
    print(f"    best CV AUC={search.best_score_:.3f}  params={search.best_params_}")
    print(f"    edge>0.0 bets: n={s['n']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {verdict}")

    return {"config": config_name, "n_train": len(train), "n_test": len(test),
            "cv_auc": search.best_score_, "best_params": search.best_params_,
            "coverage": coverage, **s, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)

    print("building real team game log, Elo, L10, bullpen fatigue, park factors (existing pipeline)...")
    games, box = build_games_and_box(con)

    print("extending with Addendum 37 historical features (unmodified import): point-in-time "
          "bullpen quality, both-starters + opposing-starter statcast quality, park dims...")
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    print(f"\ncomputing new Addendum 40 feature: opp_lineup_k_rate (rolling N={N_GAMES}-game "
          f"team batting K-rate, point-in-time, no lookahead, computed directly from boxscore)...")
    team_k_rate = build_team_k_rate(con, N_GAMES)
    print(f"  team_k_rate table: {len(team_k_rate)} (team_id, game_pk) rows, "
          f"{round(100 * team_k_rate['opp_lineup_k_rate'].notna().mean(), 1)}% non-null overall")

    # game_ctx: one row per game_pk with home/away team_id -- built from the
    # BASE `games` table (build_team_game_log's own output), deduplicated by
    # game_pk to avoid the same pre-existing duplicate-game_pk issue Addendum
    # 37 documented and fixed for games_v2 (4,547 rows / 3,845 unique game_pk).
    game_ctx = games.drop_duplicates(subset=["game_pk"])[["game_pk", "home_team_id", "away_team_id"]]

    print("\n=== Addendum 40: opp_lineup_k_rate added to pitcher_strikeouts, two isolated "
          "configurations, same tuning discipline as Addendum 34/37 ===")

    all_results = []

    # --- Configuration (a): FEATURES_V2 (Addendum 37 full enrichment) + opp_lineup_k_rate ---
    print("\n--- Configuration (a): FEATURES_V2 + opp_lineup_k_rate ---")
    pool_a = build_warehouse_pool(con, box, "pitcher_strikeouts", "strikeouts", "pitcher")
    print(f"  raw warehouse pool (pitcher_strikeouts, over+under rows): n={len(pool_a)}")
    pool_a = build_features_v2(pool_a, games_v2, "pitcher", box_ids, con2)
    print(f"  after build_features_v2 (Addendum 37, unmodified): n={len(pool_a)}")
    pool_a = add_opp_lineup_k_rate(pool_a, game_ctx, box_ids, team_k_rate, label="(config a)")
    features_a = FEATURES_V2 + ["opp_lineup_k_rate"]
    r_a = tune_and_evaluate("(a) FEATURES_V2 + opp_lineup_k_rate", pool_a, features_a, "per_year")
    if r_a:
        all_results.append(r_a)

    # --- Configuration (b): ORIGINAL baseline features + opp_lineup_k_rate only ---
    print("\n--- Configuration (b): baseline features + opp_lineup_k_rate only (isolated) ---")
    pool_b = build_warehouse_pool(con, box, "pitcher_strikeouts", "strikeouts", "pitcher")
    print(f"  raw warehouse pool (pitcher_strikeouts, over+under rows): n={len(pool_b)}")
    pool_b = build_features(pool_b, games)
    print(f"  after build_features (original baseline, unmodified): n={len(pool_b)}")
    pool_b = add_opp_lineup_k_rate(pool_b, game_ctx, box_ids, team_k_rate, label="(config b)")
    features_b = FEATURES + ["opp_lineup_k_rate"]
    r_b = tune_and_evaluate("(b) baseline FEATURES + opp_lineup_k_rate", pool_b, features_b, "per_year")
    if r_b:
        all_results.append(r_b)

    con.close()
    con2.close()

    print("\n=== SUMMARY: pitcher_strikeouts, opp_lineup_k_rate feature, both configurations ===")
    print(f"\nBEFORE [Addendum 37, {BASELINE_ADD37['desc']}]: n={BASELINE_ADD37['n']} "
          f"ROI={BASELINE_ADD37['roi']}  {BASELINE_ADD37['verdict']}")
    for r in all_results:
        print(f"\n{r['config']}")
        print(f"  n(train/test)={r['n_train']}/{r['n_test']}  CV_AUC={r['cv_auc']:.3f}  "
              f"opp_lineup_k_rate coverage={r['coverage']}%")
        print(f"  test: n={r['n']}  ROI={r['roi']}  CI=[{r['lo']},{r['hi']}]  {r['verdict']}")

    passes = [r for r in all_results if r["verdict"] == "PASS"]
    print(f"\n{len(passes)} of {len(all_results)} configurations pass the official gate rule. "
          f"Reported as-is, pass or fail, single pre-specified grid/cut/N -- same discipline as "
          f"every prior addendum. A FAIL here is a valid, reportable result, not grounds to "
          f"search further thresholds or feature cuts against this test set.")


if __name__ == "__main__":
    main()
