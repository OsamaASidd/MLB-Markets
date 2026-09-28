"""
Addendum 88: two-stage "hurdle" model for batter_runs_scored -- a
genuinely different statistical formulation from every prior attempt
(XGBoost/GLM/NB/line-movement all modeled the single binary "scored a
run" outcome directly). This decomposes it into the actual causal
structure of the game:

    P(scored) = P(reached base at least once) * P(scored | reached base)

A player can't score without first reaching base -- collapsing both
steps into one binary target (as every prior model did) forces one
function to learn two different processes at once (getting on base:
speed/OBP/patience-driven; scoring once on: lineup protection/teammates'
power-driven). Splitting them lets each stage use the SAME existing
point-in-time features (Elo/L10/fatigue/park, all already corrected of
the two fan-out bugs from Addendum 62) but fit its own relationship.

  Stage 1: P(reached_base | features)          -- fit on ALL rows
  Stage 2: P(scored | reached_base=1, features) -- fit ONLY on the
           subset who actually reached base that game

reached_base derived directly from existing boxscore columns:
  reached_base = (hits + batter_walks + batter_hbp) > 0

Both stages: CV-AUC-selected XGBoost (PARAM_GRID_WIDE/FIXED_PARAMS,
unmodified from Addendum 49), walk-forward expanding folds, single
edge>0.0 cut against the de-vigged market_prob, gate touched exactly
once on the combined P(scored) = P(reach)*P(score|reach).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum62_batter_runs_scored_eda import build_pool as build_base_pool  # noqa: E402

N_FOLDS = 6
FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor"]


def build_hurdle_pool(con):
    """Reuses Addendum 62's corrected batter_runs_scored pool (both
    fan-out bugs fixed) but rebuilds reached_base from the raw box score
    columns directly -- pick_main_line already collapsed to one row per
    (game, player) so we go back to the raw boxscore join for the
    reached-base components."""
    pool = build_base_pool(con)
    box = con.execute("""
        SELECT game_pk, player_name, hits, batter_walks, batter_hbp FROM boxscore
    """).fetchdf()
    from xgboost_individual_markets import norm_name
    box["name_norm"] = box["player_name"].map(norm_name)
    box = box.drop_duplicates(subset=["game_pk", "name_norm"])
    n0 = len(pool)
    pool = pool.merge(box[["game_pk", "name_norm", "hits", "batter_walks", "batter_hbp"]],
                       on=["game_pk", "name_norm"], how="left")
    print(f"  row-count check (reached-base component merge): before={n0} after={len(pool)} "
          f"{'OK' if len(pool) == n0 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n0
    pool["reached_base"] = ((pool["hits"].fillna(0) + pool["batter_walks"].fillna(0) + pool["batter_hbp"].fillna(0)) > 0).astype(int)
    return pool


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building batter_runs_scored pool with reached_base decomposition...")
    pool = build_hurdle_pool(con)
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"  pool: n={len(pool)}  reached_base rate={pool['reached_base'].mean():.3f}  "
          f"scored-given-reached rate={pool[pool.reached_base==1]['win'].mean() if 'win' in pool.columns else float('nan'):.3f}")

    # 'win' in the original pool is side-specific (over/under); the hurdle
    # model needs the underlying binary "did they score" target regardless
    # of side -- reconstruct from over/under rows via runs_scored>0.
    scored_rows = pool[pool.side == "over"].copy()
    scored_rows["scored"] = scored_rows["win"].astype(int)  # over wins iff runs_scored>0.5, i.e. scored>=1

    fold_id = pd.qcut(np.arange(len(scored_rows)), N_FOLDS, labels=False)
    scored_rows["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), two-stage hurdle model ===")
    for f in range(1, N_FOLDS):
        train_df = scored_rows[scored_rows["fold"] < f]
        test_df = scored_rows[scored_rows["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        # Stage 1: P(reached_base) on ALL train rows
        X1, y1 = train_df[FEATURES], train_df["reached_base"]
        model1, cv1, params1 = cv_select(X1, y1)
        print(f"  fold {f} stage1 (reached_base): CV_AUC={cv1:.4f}  params={params1}")

        # Stage 2: P(scored | reached_base=1) on train rows who reached base
        train_reached = train_df[train_df.reached_base == 1]
        X2, y2 = train_reached[FEATURES], train_reached["scored"]
        model2, cv2, params2 = cv_select(X2, y2)
        print(f"  fold {f} stage2 (scored|reached): CV_AUC={cv2:.4f}  params={params2}")

        p_reach = model1.predict_proba(test_df[FEATURES])[:, 1]
        p_score_given_reach = model2.predict_proba(test_df[FEATURES])[:, 1]
        p_scored = p_reach * p_score_given_reach

        over_p = implied_prob(test_df["best_over_odds"].values)
        under_p = implied_prob(test_df["best_under_odds"].values)
        market_prob_over_devig = over_p / (over_p + under_p)

        over_rows = pd.DataFrame({
            "game_date": test_df["game_date"].values, "odds": test_df["best_over_odds"].values,
            "win": test_df["scored"].values.astype(bool), "edge": p_scored - market_prob_over_devig,
        })
        under_rows = pd.DataFrame({
            "game_date": test_df["game_date"].values, "odds": test_df["best_under_odds"].values,
            "win": ~test_df["scored"].values.astype(bool), "edge": (1 - p_scored) - (1 - market_prob_over_devig),
        })
        all_bets.append(pd.concat([over_rows, under_rows], ignore_index=True))

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
