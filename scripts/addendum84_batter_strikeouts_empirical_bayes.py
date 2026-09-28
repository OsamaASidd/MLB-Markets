"""
Addendum 84: empirical-Bayes per-player shrinkage feature for
batter_strikeouts -- genuinely different from every population-level
model tried so far (XGBoost, GLM, NB), which all fit ONE set of global
coefficients for every player. This lets each individual batter carry
their own small deviation from the market's pricing, estimated from
their own prior history and shrunk toward zero based on how much data
they have (a player with 300 prior graded bets gets more weight on their
own history than one with 5).

Point-in-time safe: for each (player, bet) row, the shrunk effect uses
ONLY that player's STRICTLY PRIOR resolved bets (shift before any
expanding aggregation, same discipline as Addendum 58/68).

  raw_effect_i   = mean(actual_win - market_prob) over player i's prior bets
  shrunk_effect_i = raw_effect_i * n_i / (n_i + k)      (k = shrinkage strength)

k is treated as a hyperparameter, selected via CV-AUC on training data
only (never ROI) alongside the existing feature set -- not hand-picked
to make the result look better.

Reuses Addendum 75's corrected pool (both real fan-out bugs + the leaked
starter feature already fixed) as the base; only adds this one new
feature and re-runs the walk-forward.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum49_h2h_walkforward import PARAM_GRID_WIDE, FIXED_PARAMS  # noqa: E402
from addendum75_batter_strikeouts_market_offset_glm import main as _unused  # noqa: E402  (import guard only)

N_FOLDS = 6
K_GRID = [5, 10, 20, 40, 80, 160]
BASE_FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor",
                 "bullpen_era_diff", "bullpen_k9_diff", "opp_sp_roll_era", "opp_sp_roll_ip",
                 "lf_distance", "cf_distance", "rf_distance", "lf_wall_height", "cf_wall_height",
                 "rf_wall_height", "cf_compass_degrees", "is_dome", "side_code", "market_prob"]


def build_pool_with_eb_feature(con, con2):
    """Rebuilds the exact Addendum 75 pool (imported pieces, not
    duplicated logic) and adds the point-in-time EB shrinkage feature."""
    import numpy as np
    from xgboost_individual_markets import norm_name, implied_prob
    from addendum37_historical_enrichment import add_bullpen_features, add_park_features, build_box_ids
    from addendum48_h2h_totals_fuller_gamelog import build_fuller_games
    from addendum55c_novig_batter_markets import build_market_dataset_both_sides
    from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era

    box_bk = con.execute("""
        SELECT game_pk, player_name, batter_strikeouts, is_starter, position_type FROM boxscore
    """).fetchdf()
    box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
    cache_path = ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl"
    pool = build_market_dataset_both_sides(cache_path, "batter_strikeouts", "batter_strikeouts", box_bk,
                                            ["Catcher", "Hitter", "Infielder", "Outfielder"], starter_only=False)

    games_v3, id_to_name = build_fuller_games(con, con2)
    n0 = len(games_v3)
    games_v3 = add_bullpen_features(con2, games_v3, id_to_name)
    assert len(games_v3) == n0
    games_v3 = add_park_features(con, con2, games_v3)
    assert len(games_v3) == n0
    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    games_v3 = games_v3.merge(op, on="game_pk", how="left")
    assert len(games_v3) == n0
    games_v3["game_date"] = pd.to_datetime(games_v3["game_date"])
    box_ids = build_box_ids(con)

    GAME_LEVEL_COLS = ["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                        "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor",
                        "bullpen_era_diff", "bullpen_k9_diff",
                        "lf_distance", "cf_distance", "rf_distance", "lf_wall_height", "cf_wall_height",
                        "rf_wall_height", "cf_compass_degrees", "is_dome"]
    n0p = len(pool)
    pool = pool.merge(games_v3[GAME_LEVEL_COLS], on="game_pk", how="inner")
    pool["name_norm"] = pool["tiebreak"]
    n1 = len(pool)
    pool = pool.merge(box_ids[["game_pk", "name_norm", "team_id"]], on=["game_pk", "name_norm"], how="left")
    ctx = games_v3.drop_duplicates(subset=["game_pk"])[
        ["game_pk", "home_team_id", "away_team_id", "home_starter_id", "away_starter_id"]]
    pool = pool.merge(ctx, on="game_pk", how="left")
    pool["opp_starter_id"] = pd.Series(np.where(pool["team_id"] == pool["home_team_id"],
                                        pool["away_starter_id"], pool["home_starter_id"]), index=pool.index).astype("Int64")
    assert len(pool) == n0p

    roll = build_starter_rolling_era(con)
    roll_o = roll.rename(columns={"player_id": "opp_starter_id", "sp_roll_era": "opp_sp_roll_era", "sp_roll_ip": "opp_sp_roll_ip"})
    roll_o["opp_starter_id"] = roll_o["opp_starter_id"].astype("Int64")
    n2 = len(pool)
    pool = pool.merge(roll_o[["opp_starter_id", "game_pk", "opp_sp_roll_era", "opp_sp_roll_ip"]],
                       on=["opp_starter_id", "game_pk"], how="left")
    assert len(pool) == n2

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["game_date"] = pd.to_datetime(pool["game_date"])

    over_p = implied_prob(pool["best_over_odds"].values)
    under_p = implied_prob(pool["best_under_odds"].values)
    total = over_p + under_p
    pool["market_prob"] = np.where(pool["side"].values == "over", over_p / total, under_p / total)
    pool = pool.dropna(subset=["win", "odds", "market_prob"])
    return pool.sort_values(["game_date", "game_pk", "name_norm", "side"], kind="mergesort").reset_index(drop=True)


def add_eb_feature(pool, k):
    """Point-in-time: for each row, uses only that player's STRICTLY
    PRIOR resolved bets (shift(1) before expanding mean)."""
    pool = pool.copy()
    pool["_win_int"] = pool["win"].astype(int)
    pool["_resid"] = pool["_win_int"] - pool["market_prob"]
    g = pool.groupby("name_norm")
    pool["_prior_mean_resid"] = g["_resid"].transform(lambda s: s.shift(1).expanding().mean())
    pool["_prior_n"] = g["_resid"].transform(lambda s: s.shift(1).expanding().count())
    pool["eb_effect"] = pool["_prior_mean_resid"].fillna(0) * (pool["_prior_n"].fillna(0) / (pool["_prior_n"].fillna(0) + k))
    return pool.drop(columns=["_win_int", "_resid", "_prior_mean_resid", "_prior_n"])


def cv_select_with_k(train_df):
    """Select both XGBoost hyperparameters AND the EB shrinkage constant k
    via CV-AUC on training data only -- k is a real hyperparameter here,
    not hand-picked."""
    from sklearn.metrics import roc_auc_score
    best = None
    for k in K_GRID:
        train_k = add_eb_feature(train_df, k)
        features = BASE_FEATURES + ["eb_effect"]
        X, y = train_k[features], train_k["win"].astype(int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        search = GridSearchCV(xgb.XGBClassifier(**FIXED_PARAMS), PARAM_GRID_WIDE, scoring="roc_auc", cv=cv, n_jobs=4)
        import joblib
        with joblib.parallel_backend("threading"):
            search.fit(X, y)
        if best is None or search.best_score_ > best[0]:
            best = (search.best_score_, k, search.best_estimator_, search.best_params_)
    return best


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building batter_strikeouts pool with point-in-time EB player-shrinkage feature...")
    pool = build_pool_with_eb_feature(con, con2)
    print(f"  pool: n={len(pool)}  unique players={pool.name_norm.nunique()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), XGBoost + EB player effect ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df_raw = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df_raw)}")
        if len(train_df) < 300 or len(test_df_raw) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        cv_auc, best_k, model, params = cv_select_with_k(train_df)
        print(f"  fold {f}: CV_AUC={cv_auc:.4f}  best_k={best_k}  params={params}")

        # recompute eb_effect for train+test together (fold<=f, chronological
        # order restored via sort_index since pool's index IS row order) so
        # test rows see all STRICTLY PRIOR history, still point-in-time safe
        idx = train_df.index.union(test_df_raw.index)
        combined_eb = add_eb_feature(pool.loc[idx].sort_index(), best_k)
        test_eb = combined_eb.loc[test_df_raw.index]

        features = BASE_FEATURES + ["eb_effect"]
        test_pred = model.predict_proba(test_eb[features])[:, 1]
        t = test_eb.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

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
