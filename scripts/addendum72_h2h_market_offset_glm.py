"""
Addendum 72: market-offset ridge-logistic GLM for h2h, modeled on the
client's own live production architecture (Fernando GLM,
_shared/scoring_mlb_hits_model.ts, described by the client):

    logit P(win) = logit(no-vig market price) [fixed offset, weight 1.0]
                   + intercept + sum(coef_i * feature_i)   [ridge-penalized]

WHY this is worth trying separately from every prior XGBoost attempt:
Addendum 59/62/67's EDA repeatedly found the market already well-
calibrated and most raw features carrying near-zero MARGINAL information
once the market's own price is accounted for. A generic flexible model
(XGBoost) has to spend capacity re-deriving the market's own signal from
the raw features before it can even start looking for a genuine residual
-- that's real variance/overfitting risk with no offsetting benefit here.
The market-offset structure removes that: it starts already anchored at
the market's price and only asks whether the clean, point-in-time
features (Addendum 68's corrected elo/bullpen/L10/starter-rolling-ERA/
park set) explain any SMALL, ridge-regularized deviation from it. This is
exactly the shape of problem this architecture is designed for.

De-vig: h2h has BOTH sides' real closing odds in client_closing_odds
(h2h__home AND h2h__away, both n=7,088) -- properly de-vigged here
(fair_prob = implied(this_side) / (implied(home)+implied(away))), unlike
every prior h2h addendum which only ever used the raw single-side price.

Same walk-forward discipline as every other addendum: ridge-alpha
selected via CV-AUC on TRAIN only (never ROI), expanding-window folds,
gate evaluated exactly once on the pooled held-out set.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, add_bullpen_features, add_park_features  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS, TEAM_NAME_ALIAS  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_starter_rolling_era  # noqa: E402

N_FOLDS = 5
ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "bullpen_era_diff", "bullpen_k9_diff",
            "sp_roll_era_diff", "sp_roll_ip_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor",
            "lf_distance", "cf_distance", "rf_distance", "lf_wall_height", "cf_wall_height",
            "rf_wall_height", "cf_compass_degrees", "is_dome"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_devigged_pool(con, con2):
    print("building fuller game log + clean point-in-time features (Addendum 48/68)...")
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
    assert len(games) == n0, "starter rolling ERA merge fan-out -- STOP"
    games["sp_roll_era_diff"] = games["away_sp_roll_era"] - games["home_sp_roll_era"]
    games["sp_roll_ip_diff"] = games["home_sp_roll_ip"] - games["away_sp_roll_ip"]

    print("\npulling BOTH sides' real closing odds for de-vig...")
    odds = con.execute("""
        SELECT g.game_pk, co.market_key, co.best_over_odds AS odds
        FROM client_closing_odds co
        JOIN client_games g ON co.event_id = g.event_id
        WHERE co.market_key IN ('h2h__home', 'h2h__away') AND g.game_pk IS NOT NULL AND g.game_pk != ''
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds = odds.dropna(subset=["game_pk", "odds"])
    odds["game_pk"] = odds["game_pk"].astype("int64")
    wide = odds.pivot_table(index="game_pk", columns="market_key", values="odds", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["h2h__home", "h2h__away"])
    wide["home_prob_raw"] = implied_prob(wide["h2h__home"].values)
    wide["away_prob_raw"] = implied_prob(wide["h2h__away"].values)
    total = wide["home_prob_raw"] + wide["away_prob_raw"]
    wide["away_prob_devig"] = wide["away_prob_raw"] / total
    wide["hold"] = total - 1.0
    print(f"  {len(wide)} games with both real h2h sides quoted, mean overround={round(100*(total.mean()-1),2)}%")

    n_before = len(wide)
    pool = wide.merge(games, on="game_pk", how="inner")
    print(f"  row-count check (game-log merge): before={n_before} after={len(pool)} "
          f"{'OK' if len(pool) <= n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) <= n_before

    pool["win"] = pool["away_runs_"] > pool["home_runs_"]
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = pool["away_prob_devig"]
    pool["odds"] = pool["h2h__away"]
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool.dropna(subset=["win", "market_prob"]).sort_values(
        ["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


def cv_select_alpha(X_train, y_train, offset_train):
    """CV-AUC-only ridge alpha selection (never ROI). 5-fold, mean held-out
    AUC per alpha, best alpha wins."""
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = {a: [] for a in ALPHA_GRID}
    for tr_idx, va_idx in cv.split(X_train):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]
        otr, ova = offset_train[tr_idx], offset_train[va_idx]
        Xtr_c = sm.add_constant(Xtr, has_constant="add")
        Xva_c = sm.add_constant(Xva, has_constant="add")
        for a in ALPHA_GRID:
            try:
                res = sm.GLM(ytr, Xtr_c, family=sm.families.Binomial(), offset=otr).fit_regularized(alpha=a, L1_wt=0.0)
                pred = res.predict(Xva_c, offset=ova)
                scores[a].append(roc_auc_score(yva, pred))
            except Exception:
                scores[a].append(np.nan)
    mean_scores = {a: np.nanmean(v) for a, v in scores.items()}
    best_alpha = max(mean_scores, key=mean_scores.get)
    return best_alpha, mean_scores[best_alpha]


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    pool = build_devigged_pool(con, con2)
    print(f"\nfinal de-vigged h2h pool: n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), market-offset ridge-logistic GLM ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f].copy()
        test_df = pool[pool["fold"] == f].copy()
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_raw = imputer.fit_transform(train_df[FEATURES])
        X_train = scaler.fit_transform(X_train_raw)
        y_train = train_df["win"].astype(int).values
        offset_train = logit(train_df["market_prob"].values)

        best_alpha, cv_auc = cv_select_alpha(X_train, y_train, offset_train)
        print(f"  fold {f}: CV_AUC={cv_auc:.4f}  best_ridge_alpha={best_alpha}")

        X_train_c = sm.add_constant(X_train, has_constant="add")
        final_model = sm.GLM(y_train, X_train_c, family=sm.families.Binomial(), offset=offset_train).fit_regularized(
            alpha=best_alpha, L1_wt=0.0)

        X_test_raw = imputer.transform(test_df[FEATURES])
        X_test = scaler.transform(X_test_raw)
        X_test_c = sm.add_constant(X_test, has_constant="add")
        offset_test = logit(test_df["market_prob"].values)
        test_pred = final_model.predict(X_test_c, offset=offset_test)

        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge", "model_prob", "market_prob", "hold"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")

    from sklearn.metrics import log_loss
    y_all = pooled["win"].astype(int)
    ll_model = log_loss(y_all, pooled["model_prob"].clip(1e-6, 1 - 1e-6))
    ll_market = log_loss(y_all, pooled["market_prob"].clip(1e-6, 1 - 1e-6))
    print(f"\n  DIAGNOSTIC: log-loss on full held-out set (n={len(pooled)}) -- "
          f"model={ll_model:.4f}  market(de-vigged close)={ll_market:.4f}  "
          f"{'model beats close' if ll_model < ll_market else 'market close still better'} "
          f"(delta={ll_market - ll_model:+.4f}, positive = model better)")

    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    print(f"  DIAGNOSTIC: mean hold (vig) on the {len(sub)} qualifying bets = {round(100*sub['hold'].mean(), 2)}%")
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
