"""
Addendum 78: Negative Binomial count regression for pitcher_outs -- a
genuinely different architecture from every prior attempt (XGBoost
classification, logistic regression, market-offset ridge GLM), all of
which collapsed the actual outs-recorded count down to a binary
over/under label before modeling. This models the real count directly:

    outs_recorded ~ NegativeBinomial(mu = exp(X @ coef), alpha)

then derives P(over line) / P(under line) from the fitted count
distribution's survival/CDF at the line, same pattern as Addendum 43b's
Poisson approach for batter_runs_scored. Negative Binomial (not plain
Poisson) because outs recorded is heavily overdispersed: mean=15.24,
variance=21.35 (verified directly) -- Poisson assumes variance=mean,
NB estimates the extra dispersion via MLE.

Uses the SAME corrected, point-in-time feature set as Addendum 74/70
(fixed starter-quality rolling ERA, real bullpen/park enrichment) --
only the model family changes.

Walk-forward: expanding-window folds, single fixed model form (no
hyperparameter grid -- NB's dispersion is estimated by MLE, not tuned),
pooled held-out bets, official gate evaluated exactly once.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, norm_name, implied_prob, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features, add_park_features, TEAM_NAME_ALIAS  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from test_backfilled_markets import parse_cache, decimal_to_american  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402

N_FOLDS = 5
FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor",
            "bullpen_era_diff", "bullpen_k9_diff", "sp_roll_era_diff", "sp_roll_ip_diff",
            "lf_distance", "cf_distance", "rf_distance", "lf_wall_height", "cf_wall_height",
            "rf_wall_height", "cf_compass_degrees", "is_dome"]


def build_unique_starts_pool(con, con2):
    """One row per real (game, starting pitcher) with the actual outs
    count -- not the doubled over/under bet-row format used everywhere
    else, since a count regression needs one observation per real event."""
    box = con.execute("""
        SELECT game_pk, player_name, team_id, outs, outs AS pitcher_outs_stat, position_type, is_starter FROM boxscore
        WHERE position_type = 'Pitcher' AND is_starter = True AND outs IS NOT NULL
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    box = box.drop_duplicates(subset=["game_pk", "name_norm"])

    po_cache = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"
    odds = parse_cache(po_cache, "pitcher_outs")
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds = odds.dropna(subset=["game_pk"])
    odds["game_pk"] = odds["game_pk"].astype("int64")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    odds["best_over_odds"] = odds["best_over_dec"].map(decimal_to_american)
    odds["best_under_odds"] = odds["best_under_dec"].map(decimal_to_american)
    odds_unique = odds.drop_duplicates(subset=["game_pk", "name_norm"])[
        ["game_pk", "name_norm", "line", "best_over_odds", "best_under_odds", "commence_time"]
    ]

    pool = odds_unique.merge(box[["game_pk", "name_norm", "team_id", "outs"]], on=["game_pk", "name_norm"], how="inner")
    print(f"  unique real starts with real odds + real outcome: n={len(pool)}")

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

    roll = build_starter_rolling_era(con)
    roll_h = roll.rename(columns={"player_id": "home_starter_id", "sp_roll_era": "home_sp_roll_era", "sp_roll_ip": "home_sp_roll_ip"})
    roll_h["home_starter_id"] = roll_h["home_starter_id"].astype("Int64")
    games_v3 = games_v3.merge(roll_h[["home_starter_id", "game_pk", "home_sp_roll_era", "home_sp_roll_ip"]],
                               on=["home_starter_id", "game_pk"], how="left")
    roll_a = roll.rename(columns={"player_id": "away_starter_id", "sp_roll_era": "away_sp_roll_era", "sp_roll_ip": "away_sp_roll_ip"})
    roll_a["away_starter_id"] = roll_a["away_starter_id"].astype("Int64")
    games_v3 = games_v3.merge(roll_a[["away_starter_id", "game_pk", "away_sp_roll_era", "away_sp_roll_ip"]],
                               on=["away_starter_id", "game_pk"], how="left")
    assert len(games_v3) == n0
    games_v3["sp_roll_era_diff"] = games_v3["away_sp_roll_era"] - games_v3["home_sp_roll_era"]
    games_v3["sp_roll_ip_diff"] = games_v3["home_sp_roll_ip"] - games_v3["away_sp_roll_ip"]
    games_v3["elo_diff"] = (games_v3["home_elo"] + 24) - games_v3["away_elo"]
    games_v3["l10_diff"] = games_v3["home_l10"] - games_v3["away_l10"]
    games_v3["fatigue_diff"] = games_v3["home_fatigue"] - games_v3["away_fatigue"]

    game_cols = ["game_pk", "game_date"] + FEATURES
    n_before = len(pool)
    pool = pool.merge(games_v3.drop_duplicates(subset=["game_pk"])[game_cols], on="game_pk", how="inner")
    print(f"  row-count check (game-log merge): before={n_before} after={len(pool)} "
          f"{'OK' if len(pool) <= n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) <= n_before

    over_p = implied_prob(pool["best_over_odds"].values)
    under_p = implied_prob(pool["best_under_odds"].values)
    total = over_p + under_p
    pool["fair_over_prob"] = over_p / total
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool.dropna(subset=["outs", "line"]).sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


def fit_negbinom(X_train, y_train):
    X_c = sm.add_constant(X_train, has_constant="add")
    model = sm.NegativeBinomialP(y_train, X_c, p=2)
    return model.fit(disp=0, maxiter=200)


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building unique real-starts pool for count regression...")
    pool = build_unique_starts_pool(con, con2)
    print(f"  final pool: n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")
    print(f"  outs distribution: mean={pool['outs'].mean():.2f}  var={pool['outs'].var():.2f}  (overdispersed: var >> mean)")

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    pool_feat = pd.DataFrame(imputer.fit_transform(pool[FEATURES]), columns=FEATURES, index=pool.index)
    pool = pd.concat([pool.drop(columns=FEATURES), pool_feat], axis=1)

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), Negative Binomial count regression ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        try:
            res = fit_negbinom(train_df[FEATURES], train_df["outs"])
        except Exception as e:
            print(f"  fold {f}: fit failed ({e}), skipped")
            continue

        X_test_c = sm.add_constant(test_df[FEATURES], has_constant="add")
        mu = np.clip(res.predict(X_test_c), 0.5, None)
        alpha = res.params.get("alpha", 0.1) if hasattr(res.params, "get") else res.params[-1]
        alpha = max(float(alpha), 1e-4)
        n_param = 1 / alpha
        p_param = n_param / (n_param + mu)
        k = np.floor(test_df["line"].values)
        p_over = 1 - nbinom.cdf(k, n_param, p_param)
        model_prob_over = p_over

        for side, model_p, odds_col in [("over", model_prob_over, "best_over_odds"), ("under", 1 - model_prob_over, "best_under_odds")]:
            win = (test_df["outs"].values > test_df["line"].values) if side == "over" else (test_df["outs"].values < test_df["line"].values)
            odds_vals = test_df[odds_col].values
            market_p = test_df["fair_over_prob"].values if side == "over" else (1 - test_df["fair_over_prob"].values)
            edge = model_p - market_p
            all_bets.append(pd.DataFrame({
                "game_date": test_df["game_date"].values, "odds": odds_vals, "win": win, "edge": edge,
            }))

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
