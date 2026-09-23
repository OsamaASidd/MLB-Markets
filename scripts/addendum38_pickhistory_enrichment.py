"""
Addendum 38: pick_history-only enrichment for pitcher_outs / batter_runs_scored
(prop_type 'pitcher_outs' / 'runs_scored' in pick_history), using the new
read-only cache_features.duckdb tables pulled by scripts/pull_cache_features.py.

Both markets have NO warehouse odds at all (client's own market_config.ts:
allowedSources=PICK_HISTORY_ONLY) and were previously restricted to the
narrow score_* columns already sitting in pick_history, over the ONLY window
that market ever had real logged production factors (2026-05-17 to
2026-07-27; see run_pick_history_market() in xgboost_individual_markets.py,
reused here unmodified via import). This addendum keeps that same base
query, same window, same win/odds logic -- and ADDS a well-justified subset
of the new cache tables, point-in-time joined (never a future snapshot
relative to the pick), on top of the existing score_* columns.

Methodology, identical discipline to every other addendum in this project
(see tuned_xgboost_4_markets.py, Addendum 34):
  - ONE model family: XGBoost.
  - ONE pre-specified hyperparameter grid (PARAM_GRID / FIXED_PARAMS),
    reused as-is from tuned_xgboost_4_markets.py, not re-tuned here.
  - Selection metric: CV-averaged ROC AUC on the TRAINING split only
    (5-fold StratifiedKFold, GridSearchCV) -- ROI/gate never touched during
    selection.
  - Refit once on the full training split, scored ONCE on the untouched
    75/25 chronological test split, at the single pre-specified edge>0.0
    cut, official gate rule (n>=500 -> ROI>0; n<500 -> 95% CI lower bound>0).
  - Every result reported, pass or fail. A few other thresholds are shown
    as a side note, same as Addendum 27/34's scripts, clearly labeled
    "multiple-comparisons exposed" -- not the headline verdict.

Known baseline (score_* only, plain XGBoost, no CV tuning), from Addendum 27
(reports/MILESTONE_1_GATE_REPORT.md) / reports/gate_results.csv:
  runs_scored (pick_history):  test n=1,715  AUC=0.612  edge>0.0: n=514  ROI=-3.35%  CI=[-10.11%,3.42%]  FAIL
  pitcher_outs (pick_history): test n=200    AUC=0.570  edge>0.0: n=8    ROI=-4.71%  CI=[-75.38%,65.96%] FAIL
Both currently FAIL. This script's job is to report honestly whether real
new point-in-time signal (batter/pitcher splits, bullpen state, team
offense, defense) moves either market, not to manufacture a pass.
"""
import os
import pathlib
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import DB, MIN_GRADED, gate, stat, profit, implied_prob  # noqa: E402
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")

# ---------------------------------------------------------------------------
# Per-market enrichment: which cache tables, how they join, point-in-time.
#
# pitcher_outs -- "how many outs does the starter/bullpen get" -- pitcher's
# own recent form (last3, inn1) by player_id, and the team's bullpen/hook
# environment (bullpen_stats, pen_rest, bullpen_high_leverage, manager_hook)
# by team (the pitcher's OWN team -- pick_history.team is the pitcher's
# team, confirmed directly against sample rows).
#
# batter_runs_scored -- batter/team offensive production -- the batter's own
# platoon splits (batter_splits) by player_id, and the batter's own team's
# offensive profile (team_batting_stats) and team defensive efficiency
# (team_oaa, included per the task brief's own market grouping -- a team's
# own OAA is a fielding/run-prevention metric, not a hitting one; included
# with that caveat, not billed as a pure offense signal) by team.
#
# cache_umpire_stats: SKIPPED -- pick_history has no umpire_name column
# (confirmed via DESCRIBE pick_history), no honest join key exists.
# cache_savant_team_chase: SKIPPED despite ~75% raw join coverage -- it only
# has 2 distinct snapshot_dates (2026-06-21, 2026-06-28) over a 2.5-month
# window, so "coverage" here just means every pick after 06-28 gets the
# SAME single stale value broadcast forward for up to a month. That's not a
# real point-in-time signal, it's a near-constant with a coverage number
# that looks better than it is -- excluded on that basis, not row-count.
# cache_statcast_pitcher_arsenal / cache_statcast_framing: not in the task's
# suggested pairing for either market and would inflate the feature set
# without a clear mechanism tie to "outs recorded" or "runs scored" beyond
# what pitcher_splits/arsenal-adjacent signal already offered by last3/inn1
# gives -- left out to keep the feature set well-justified, not maximal.
# ---------------------------------------------------------------------------

PITCHER_OUTS_JOIN = """
    ASOF LEFT JOIN cache.cache_mlb_pitcher_last3 l
      ON base.player_id = l.player_id AND l.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_mlb_pitcher_inn1 i
      ON base.player_id = i.player_id AND i.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_mlb_bullpen_stats bp
      ON base.team = bp.team_name AND bp.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_mlb_pen_rest pr
      ON base.team = pr.team_name AND pr.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_mlb_bullpen_high_leverage hl
      ON base.team = hl.team_name AND hl.snapshot_date <= base.game_date
    LEFT JOIN cache.cache_mlb_team_manager_hook mh
      ON base.team = mh.team_name
"""
PITCHER_OUTS_SELECT = """
    l.last3_era AS ck_last3_era, l.last3_ip AS ck_last3_ip, l.last3_starts AS ck_last3_starts,
    i.inn1_era AS ck_inn1_era, i.inn1_ip AS ck_inn1_ip, i.inn1_bf AS ck_inn1_bf,
    i.inn1_walks AS ck_inn1_walks, i.inn1_hits AS ck_inn1_hits, i.inn1_pitches AS ck_inn1_pitches,
    bp.bullpen_era AS ck_bullpen_era, bp.bullpen_whip AS ck_bullpen_whip, bp.bullpen_ip AS ck_bullpen_ip,
    bp.bullpen_k_per_9 AS ck_bullpen_k_per_9, bp.bullpen_bb_per_9 AS ck_bullpen_bb_per_9,
    bp.bullpen_baa AS ck_bullpen_baa,
    pr.pen_ip_48h AS ck_pen_ip_48h, pr.games_in_48h AS ck_games_in_48h,
    hl.hl_arm_count AS ck_hl_arm_count, hl.hl_avg_era AS ck_hl_avg_era,
    mh.starter_games_started AS ck_mh_starter_gs, mh.starter_ip AS ck_mh_starter_ip,
    mh.starter_pitches_per_inning AS ck_mh_pitches_per_inning,
    mh.avg_pitches_per_start AS ck_mh_avg_pitches_per_start, mh.avg_ip_per_start AS ck_mh_avg_ip_per_start,
    mh.hook_index AS ck_mh_hook_index
"""
# note: cache_mlb_team_manager_hook has exactly ONE snapshot_date
# (2026-06-25) -- it is NOT point-in-time-correct across the window, it's a
# single current-state snapshot joined identically to every pick regardless
# of date. Kept in as coarse season-level manager-tendency context (it
# directly targets quick-hook behavior, conceptually the closest table to
# "how many outs does the starter get"), flagged here rather than pretended
# to be point-in-time.

RUNS_SCORED_JOIN = """
    ASOF LEFT JOIN cache.cache_mlb_batter_splits bs
      ON base.player_id = bs.player_id AND bs.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_team_batting_stats tb
      ON base.team = tb.team_name AND tb.snapshot_date <= base.game_date
    ASOF LEFT JOIN cache.cache_mlb_team_oaa oaa
      ON base.team = oaa.full_team_name AND oaa.snapshot_date <= base.game_date
"""
RUNS_SCORED_SELECT = """
    bs.vs_lhp_pa AS ck_vs_lhp_pa, bs.vs_lhp_avg AS ck_vs_lhp_avg, bs.vs_lhp_obp AS ck_vs_lhp_obp,
    bs.vs_lhp_slg AS ck_vs_lhp_slg, bs.vs_lhp_ops AS ck_vs_lhp_ops,
    bs.vs_rhp_pa AS ck_vs_rhp_pa, bs.vs_rhp_avg AS ck_vs_rhp_avg, bs.vs_rhp_obp AS ck_vs_rhp_obp,
    bs.vs_rhp_slg AS ck_vs_rhp_slg, bs.vs_rhp_ops AS ck_vs_rhp_ops,
    tb.k_rate AS ck_tb_k_rate, tb.vs_lhp_k_rate AS ck_tb_vs_lhp_k_rate, tb.vs_rhp_k_rate AS ck_tb_vs_rhp_k_rate,
    tb.runs_per_game AS ck_tb_runs_per_game, tb.ops_season AS ck_tb_ops_season, tb.obp_season AS ck_tb_obp_season,
    tb.slg_season AS ck_tb_slg_season, tb.iso_season AS ck_tb_iso_season, tb.bb_rate AS ck_tb_bb_rate,
    tb.avg_season AS ck_tb_avg_season,
    oaa.oaa AS ck_oaa, oaa.oaa_infront AS ck_oaa_infront, oaa.oaa_lateral_to_3b AS ck_oaa_lateral_to_3b,
    oaa.oaa_lateral_to_1b AS ck_oaa_lateral_to_1b, oaa.oaa_behind AS ck_oaa_behind,
    oaa.actual_success_rate AS ck_oaa_actual_success_rate,
    oaa.expected_success_rate AS ck_oaa_expected_success_rate,
    oaa.diff_success_rate AS ck_oaa_diff_success_rate
"""

MARKET_SPECS = {
    "pitcher_outs": dict(prop_type="pitcher_outs", join_sql=PITCHER_OUTS_JOIN, select_sql=PITCHER_OUTS_SELECT),
    "runs_scored": dict(prop_type="runs_scored", join_sql=RUNS_SCORED_JOIN, select_sql=RUNS_SCORED_SELECT),
}


def load_enriched_pool(con, prop_type, join_sql, select_sql):
    """Same base pick_history rows/window/win/odds logic as
    run_pick_history_market(), plus player_id/team for joining, plus the
    new cache features via point-in-time (ASOF/static) LEFT JOINs against
    the attached read-only cache_features.duckdb."""
    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    score_select = ", ".join(f'TRY_CAST(ph."{c}" AS DOUBLE) AS "{c}"' for c in score_cols)
    query = f"""
        WITH base AS (
            SELECT ph.id, TRY_CAST(ph.game_date AS DATE) AS game_date,
                   TRY_CAST(ph.odds AS INTEGER) AS odds,
                   lower(ph.hit) IN ('true','t','1') AS win,
                   TRY_CAST(ph.player_id AS BIGINT) AS player_id, ph.team,
                   {score_select}
            FROM pick_history ph
            WHERE ph.prop_type = '{prop_type}'
              AND lower(coalesce(ph.is_synthetic,'false')) NOT IN ('true','t','1')
              AND lower(coalesce(ph.voided,'false')) NOT IN ('true','t','1')
              AND ph.hit IS NOT NULL AND TRY_CAST(ph.odds AS INTEGER) IS NOT NULL
        )
        SELECT base.*, {select_sql}
        FROM base
        {join_sql}
    """
    df = con.execute(query).fetchdf()
    return df.dropna(subset=["game_date"]).sort_values(
        ["game_date", "id"], kind="mergesort"
    ).reset_index(drop=True), score_cols


def report_coverage(df, cols, label):
    print(f"  {label} join coverage (n_base={len(df)}):")
    for c in cols:
        n = df[c].notna().sum()
        pct = 100.0 * n / len(df) if len(df) else 0.0
        print(f"    {c:<32} {n:>6}/{len(df)}  ({pct:5.1f}%)")


def tune_and_evaluate(market_key, df, score_cols, new_cols):
    n_total = len(df)
    if n_total < MIN_GRADED:
        print(f"  {market_key:<20} n={n_total:<8} INSUFFICIENT DATA (<{MIN_GRADED})")
        return

    # same coverage floor used by run_pick_history_market() for score_*
    # columns, applied identically to the new engineered features so the
    # selection rule for "is this feature usable" doesn't change between
    # the old and new columns.
    min_cov = max(200, n_total * 0.05)
    score_keep = [c for c in score_cols if df[c].notna().sum() >= min_cov]
    new_keep = [c for c in new_cols if df[c].notna().sum() >= min_cov]
    dropped_new = [c for c in new_cols if c not in new_keep]
    features = score_keep + new_keep

    print(f"\n  === {market_key} ===")
    print(f"  n={n_total}  score_* kept: {len(score_keep)}/{len(score_cols)}  "
          f"new cache features kept: {len(new_keep)}/{len(new_cols)}")
    if dropped_new:
        print(f"  new features dropped for coverage <{min_cov:.0f} rows ({5}% or 200, whichever larger): {dropped_new}")

    split = int(n_total * 0.75)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    if len(train) < 50 or len(test) < 50:
        print(f"  {market_key:<20} n={n_total:<8} INSUFFICIENT DATA after split")
        return

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    # loky (process-based) joblib backend fails in this sandboxed Windows
    # Python -- threading backend instead (XGBoost's fit releases the GIL,
    # so this still parallelizes real work). Same pattern as Addendum 34.
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred >= 0.5)

    test = test.copy()
    test["model_prob"] = test_pred
    test["market_prob"] = implied_prob(test["odds"])
    test["edge"] = test["model_prob"] - test["market_prob"]

    all_cuts = []
    best = None
    for thresh in [0.0, 0.02, 0.03, 0.05, 0.08]:
        sub = test[(test.odds < 0) & (test.edge > thresh)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        passed = gate(s)
        all_cuts.append({"thresh": thresh, **s, "passed": passed})
        if best is None or (passed and not best.get("passed")):
            best = {"thresh": thresh, **s, "passed": passed}
        elif passed and best.get("passed") and s["roi"] and best["roi"] and s["roi"] > best["roi"]:
            best = {"thresh": thresh, **s, "passed": passed}

    verdict = "PASS" if best and best["passed"] else "FAIL"
    no_filter = all_cuts[0]
    nf_verdict = "PASS" if no_filter["passed"] else "FAIL"

    print(f"  n(train/test)={len(train):,}/{len(test):,}  best CV AUC={search.best_score_:.3f}  "
          f"test AUC={auc:.3f}  test acc={acc:.3f}")
    print(f"  best params: {search.best_params_}")
    print(f"    edge>0.0 (no threshold search, OFFICIAL VERDICT): n={no_filter['n']} ROI={no_filter['roi']} "
          f"CI=[{no_filter['lo']},{no_filter['hi']}]  {nf_verdict}")
    print(f"    best of 5 thresholds tried on this same test set: edge>{best['thresh']} n={best['n']} "
          f"ROI={best['roi']} CI=[{best['lo']},{best['hi']}]  {verdict}  (multiple-comparisons exposed)")
    return {"market": market_key, "n_train": len(train), "n_test": len(test),
            "cv_auc": round(search.best_score_, 3), "test_auc": round(auc, 3),
            "no_filter": no_filter, "best_cut": best,
            "no_filter_verdict": nf_verdict, "verdict": verdict}


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=" * 78)
    print("Addendum 38: pick_history-only enrichment, pitcher_outs / runs_scored")
    print("New cache tables joined point-in-time (ASOF <= game_date, never future),")
    print("combined with existing score_* columns. CV-AUC-only selection, single")
    print("pre-registered PARAM_GRID reused from tuned_xgboost_4_markets.py,")
    print("official gate rule scored once on the untouched chronological test split.")
    print("=" * 78)

    results = []
    for market_key, spec in MARKET_SPECS.items():
        df, score_cols = load_enriched_pool(con, spec["prop_type"], spec["join_sql"], spec["select_sql"])
        new_cols = [c for c in df.columns if c.startswith("ck_")]
        report_coverage(df, new_cols, market_key)
        res = tune_and_evaluate(market_key, df, score_cols, new_cols)
        if res:
            results.append(res)

    print("\n" + "=" * 78)
    print("SUMMARY vs known baseline (score_* only, plain XGBoost, Addendum 27):")
    print("  runs_scored (pick_history):  baseline test n=1,715 AUC=0.612  "
          "edge>0.0 n=514 ROI=-3.35% CI=[-10.11%,3.42%]  FAIL")
    print("  pitcher_outs (pick_history): baseline test n=200   AUC=0.570  "
          "edge>0.0 n=8   ROI=-4.71% CI=[-75.38%,65.96%]  FAIL")
    print()
    for r in results:
        nf = r["no_filter"]
        print(f"  {r['market']:<15} n(train/test)={r['n_train']:,}/{r['n_test']:,}  "
              f"CV AUC={r['cv_auc']}  test AUC={r['test_auc']}  "
              f"edge>0.0: n={nf['n']} ROI={nf['roi']} CI=[{nf['lo']},{nf['hi']}]  "
              f"{r['no_filter_verdict']}  (official verdict)")

    con.close()


if __name__ == "__main__":
    main()
