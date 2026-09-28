"""
Addendum 83: joint run-distribution model for h2h + totals, instead of
two independent binary classifiers -- genuinely different from every
prior attempt on these two markets. Fits home_runs ~ Poisson(mu_home)
and away_runs ~ Poisson(mu_away) from the SAME clean point-in-time
feature set (Addendum 68), then derives BOTH markets' probabilities from
one shared model:

  P(home win)   = P(home_runs > away_runs)   -- double sum over the
                  joint (independent) Poisson pmf, truncated to 0-25 runs
  P(total>line) = 1 - PoissonCDF(floor(line), mu_home + mu_away)
                  -- sum of independent Poissons is Poisson(mu_home+mu_away)

Simplification (disclosed): independent Poisson, not a full bivariate/
correlated count model -- a legitimate first step per the task, with
correlation left as a further refinement if this baseline shows promise.

Same walk-forward discipline: fit on train only, gate touched once per
market on the pooled held-out set.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.impute import SimpleImputer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS, TEAM_NAME_ALIAS  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402
from addendum37_historical_enrichment import add_bullpen_features, add_park_features  # noqa: E402

N_FOLDS = 5
FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor",
            "bullpen_era_diff", "bullpen_k9_diff", "sp_roll_era_diff", "sp_roll_ip_diff",
            "lf_distance", "cf_distance", "rf_distance", "lf_wall_height", "cf_wall_height",
            "rf_wall_height", "cf_compass_degrees", "is_dome"]
MAX_RUNS = 25


def build_games_with_scores(con, con2):
    games, id_to_name = build_fuller_games(con, con2)
    n0 = len(games)
    games = add_bullpen_features(con2, games, id_to_name)
    assert len(games) == n0
    games = add_park_features(con, con2, games)
    assert len(games) == n0

    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    games = games.merge(op, on="game_pk", how="left")
    assert len(games) == n0

    roll = build_starter_rolling_era(con)
    roll_h = roll.rename(columns={"player_id": "home_starter_id", "sp_roll_era": "home_sp_roll_era", "sp_roll_ip": "home_sp_roll_ip"})
    roll_h["home_starter_id"] = roll_h["home_starter_id"].astype("Int64")
    games = games.merge(roll_h[["home_starter_id", "game_pk", "home_sp_roll_era", "home_sp_roll_ip"]],
                         on=["home_starter_id", "game_pk"], how="left")
    roll_a = roll.rename(columns={"player_id": "away_starter_id", "sp_roll_era": "away_sp_roll_era", "sp_roll_ip": "away_sp_roll_ip"})
    roll_a["away_starter_id"] = roll_a["away_starter_id"].astype("Int64")
    games = games.merge(roll_a[["away_starter_id", "game_pk", "away_sp_roll_era", "away_sp_roll_ip"]],
                         on=["away_starter_id", "game_pk"], how="left")
    assert len(games) == n0
    games["sp_roll_era_diff"] = games["away_sp_roll_era"] - games["home_sp_roll_era"]
    games["sp_roll_ip_diff"] = games["home_sp_roll_ip"] - games["away_sp_roll_ip"]
    games["elo_diff"] = (games["home_elo"] + 24) - games["away_elo"]
    games["l10_diff"] = games["home_l10"] - games["away_l10"]
    games["fatigue_diff"] = games["home_fatigue"] - games["away_fatigue"]
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def load_odds(con):
    h2h = con.execute("""
        SELECT g.game_pk, co.market_key, co.best_over_odds AS odds
        FROM client_closing_odds co JOIN client_games g ON co.event_id = g.event_id
        WHERE co.market_key IN ('h2h__home', 'h2h__away') AND g.game_pk IS NOT NULL AND g.game_pk != ''
    """).fetchdf()
    h2h["game_pk"] = pd.to_numeric(h2h["game_pk"], errors="coerce")
    h2h["odds"] = pd.to_numeric(h2h["odds"], errors="coerce")
    h2h = h2h.dropna(subset=["game_pk", "odds"])
    h2h["game_pk"] = h2h["game_pk"].astype("int64")
    h2h_wide = h2h.pivot_table(index="game_pk", columns="market_key", values="odds", aggfunc="first").reset_index()
    h2h_wide = h2h_wide.dropna(subset=["h2h__home", "h2h__away"])
    h2h_wide["away_prob_devig"] = implied_prob(h2h_wide["h2h__away"]) / (
        implied_prob(h2h_wide["h2h__home"]) + implied_prob(h2h_wide["h2h__away"]))

    totals = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'totals' AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    totals["game_pk"] = pd.to_numeric(totals["game_pk"], errors="coerce")
    totals["line"] = pd.to_numeric(totals["line"], errors="coerce")
    totals["best_over_odds"] = pd.to_numeric(totals["best_over_odds"], errors="coerce")
    totals["best_under_odds"] = pd.to_numeric(totals["best_under_odds"], errors="coerce")
    totals = totals.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    totals["game_pk"] = totals["game_pk"].astype("int64")
    totals["over_prob"] = implied_prob(totals["best_over_odds"])
    totals["under_prob"] = implied_prob(totals["best_under_odds"])
    tot_total = totals["over_prob"] + totals["under_prob"]
    totals["over_prob_devig"] = totals["over_prob"] / tot_total

    return h2h_wide, totals


def p_home_win(mu_home, mu_away, max_runs=MAX_RUNS):
    a = np.arange(0, max_runs + 1)
    ph = poisson.pmf(a, mu_home)
    pa = poisson.pmf(a, mu_away)
    # P(home > away) = sum_{h} ph[h] * sum_{a<h} pa[a]
    cum_pa = np.cumsum(pa) - pa  # P(away < h) for each h
    return float(np.sum(ph * cum_pa))


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building game log + point-in-time features...")
    games = build_games_with_scores(con, con2)
    h2h_odds, totals_odds = load_odds(con)

    pool = games.merge(h2h_odds[["game_pk", "away_prob_devig", "h2h__away"]], on="game_pk", how="inner")
    pool = pool.merge(totals_odds[["game_pk", "line", "best_over_odds", "best_under_odds", "over_prob_devig"]], on="game_pk", how="inner")
    pool = pool.dropna(subset=["home_runs_", "away_runs_"]).sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    print(f"  pool with both real h2h+totals odds and real scores: n={len(pool)}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    h2h_bets, totals_bets = [], []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), joint independent-Poisson model ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train_df[FEATURES])
        X_train = sm.add_constant(X_train, has_constant="add")
        home_model = sm.GLM(train_df["home_runs_"], X_train, family=sm.families.Poisson()).fit()
        away_model = sm.GLM(train_df["away_runs_"], X_train, family=sm.families.Poisson()).fit()

        X_test = imputer.transform(test_df[FEATURES])
        X_test = sm.add_constant(X_test, has_constant="add")
        mu_home = np.clip(home_model.predict(X_test), 0.3, None)
        mu_away = np.clip(away_model.predict(X_test), 0.3, None)

        for i, (_, row) in enumerate(test_df.iterrows()):
            mh, ma = mu_home[i], mu_away[i]
            p_home = p_home_win(mh, ma)
            p_away = 1 - p_home
            market_p_away = row["away_prob_devig"]
            edge_away = p_away - market_p_away
            h2h_bets.append({"game_date": row["game_date"], "odds": row["h2h__away"], "win": row["away_runs_"] > row["home_runs_"], "edge": edge_away})

            mu_total = mh + ma
            line = row["line"]
            p_over = 1 - poisson.cdf(np.floor(line), mu_total)
            total_runs = row["home_runs_"] + row["away_runs_"]
            market_p_over = row["over_prob_devig"]
            for side, model_p, odds_col, win in [
                ("over", p_over, "best_over_odds", total_runs > line),
                ("under", 1 - p_over, "best_under_odds", total_runs < line),
            ]:
                market_p = market_p_over if side == "over" else (1 - market_p_over)
                totals_bets.append({"game_date": row["game_date"], "odds": row[odds_col], "win": win, "edge": model_p - market_p})

    for name, bets in [("h2h", h2h_bets), ("totals", totals_bets)]:
        pooled = pd.DataFrame(bets)
        print(f"\n=== {name} pooled walk-forward held-out bets: n={len(pooled)} ===")
        sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
        if len(sub) == 0:
            print(f"  {name}: edge>0.0 cut produced ZERO bets -- not evaluated")
            continue
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        verdict = "PASS" if gate(s) else "FAIL"
        print(f"{name} edge>0.0, official gate rule, evaluated ONCE:")
        print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
