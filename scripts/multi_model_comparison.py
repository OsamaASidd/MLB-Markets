"""
Pre-registered multi-model comparison on the 8 FAIL markets (Addendum 32).

Method locked in BEFORE running, in HANDOFF.md, specifically so the result
can't be cherry-picked after the fact:
  - Same real point-in-time features already used throughout this audit
    (Elo, L10, bullpen fatigue, park factors) -- no new, untested features.
  - Same train/test split each market already used for its published
    XGBoost result (per-year 75/25 for the 5 warehouse/game markets,
    single chronological 75/25 for the 3 backfilled markets) -- reused
    exactly, not re-cut to find a friendlier split.
  - 5 models per market: XGBoost (already the published baseline),
    LightGBM, HistGradientBoostingClassifier, RandomForest, and a plain
    logistic regression baseline. Not real Azure AutoML -- this project
    has no Azure subscription/credentials, so this is the disclosed
    substitute: a small, standard multi-algorithm comparison instead.
  - ONE pre-specified cut (edge>0.0, odds<0), the official gate rule,
    applied once per model per market. No threshold search folded into
    this pass -- if that's wanted later it's a separate, separately-
    labeled run, same as every other addendum in this project.
  - Every model's result for every market gets reported, pass or fail.

Reuses pool-building/feature code paths already published and verified in
xgboost_individual_markets.py (5 markets) and test_backfilled_markets.py
(3 markets) -- does not modify either file, so their already-reported
numbers stay exactly as published.
"""
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from xgboost_individual_markets import (  # noqa: E402
    DB, build_team_game_log, build_elo_l10, build_bullpen_fatigue,
    pick_main_line, norm_name, gate, stat, profit, implied_prob,
)
from test_backfilled_markets import build_market_dataset  # noqa: E402

FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
            "runs_factor", "hr_factor", "k_factor", "hits_factor"]


def build_games_and_box(con):
    games = build_team_game_log(con)
    elo_l10 = build_elo_l10(games)
    fatigue = build_bullpen_fatigue(con)
    games = games.merge(elo_l10, on="game_pk")
    games["home_fatigue"] = games.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    games["away_fatigue"] = games.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)

    parks = con.execute("SELECT * FROM ballpark_factors").fetchdf()
    for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
        parks[c] = pd.to_numeric(parks[c], errors="coerce")
    park_by_event = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    park_by_event = park_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    event_to_gamepk = con.execute("SELECT event_id, game_pk FROM client_games").fetchdf()
    event_to_gamepk["game_pk"] = pd.to_numeric(event_to_gamepk["game_pk"], errors="coerce")
    park_by_gamepk = park_by_event.merge(event_to_gamepk, on="event_id", how="inner")[
        ["game_pk", "runs_factor", "hr_factor", "k_factor", "hits_factor"]]
    games = games.merge(park_by_gamepk, on="game_pk", how="left")

    box = con.execute("""
        SELECT game_pk, player_name, hits, total_bases, rbi, home_runs, strikeouts, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    return games, box


def build_warehouse_pool(con, box, market_key, stat_col, kind):
    odds = con.execute(f"""
        SELECT co.event_id, co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.player_name
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = '{market_key}' AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
    odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    odds = odds.dropna(subset=["game_pk"])
    odds = pick_main_line(odds, ["game_pk", "name_norm"])

    if kind == "batter":
        box_k = box[box.position_type.isin(["Catcher", "Hitter", "Infielder", "Outfielder"])]
    else:
        box_k = box[(box.position_type == "Pitcher") & (box.is_starter == True)]
    m = odds.merge(box_k[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
    m = m.dropna(subset=[stat_col])
    under = m.copy(); under["win"] = under[stat_col] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = under["name_norm"]
    over = m.copy(); over["win"] = over[stat_col] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = over["name_norm"]
    return pd.concat([under, over]).dropna(subset=["odds"])


def build_game_pool(con, games, market_key):
    if market_key == "h2h":
        odds = con.execute("""
            SELECT co.game_pk, co.best_over_odds AS odds
            FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                  JOIN client_games g ON co.event_id = g.event_id) co
            WHERE co.market_key = 'h2h__away' AND co.game_pk IS NOT NULL AND co.game_pk != ''
        """).fetchdf()
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
        odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
        odds = odds.dropna(subset=["game_pk"])
        m = odds.merge(games[["game_pk", "home_runs_", "away_runs_"]], on="game_pk", how="inner")
        m["win"] = m["away_runs_"] > m["home_runs_"]; m["side"] = "away"; m["tiebreak"] = ""
        return m.dropna(subset=["odds"])
    else:  # totals (spreads already passes -- not in this run)
        odds = con.execute("""
            SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds
            FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                  JOIN client_games g ON co.event_id = g.event_id) co
            WHERE co.market_key = 'totals' AND co.game_pk IS NOT NULL AND co.game_pk != ''
        """).fetchdf()
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
        odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
        odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
        odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
        odds = odds.dropna(subset=["game_pk"])
        odds = pick_main_line(odds, ["game_pk"], over_odds_col="best_over_odds")
        m = odds.merge(games[["game_pk", "home_runs_", "away_runs_"]], on="game_pk", how="inner")
        m["total_runs"] = m["home_runs_"] + m["away_runs_"]
        under = m.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = ""
        over = m.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = ""
        return pd.concat([under, over]).dropna(subset=["odds"])


def build_features(pool, games):
    pool = pool.merge(games[["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                              "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]],
                       on="game_pk", how="inner")
    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    return pool


def split_per_year(pool):
    train_parts, test_parts = [], []
    for _, grp in pool.groupby(pool.game_date.dt.year):
        grp = grp.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort")
        cut = int(len(grp) * 0.75)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def split_chronological(pool):
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.75)
    return pool.iloc[:cut].copy(), pool.iloc[cut:].copy()


def make_models():
    return {
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
            reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_lambda=3.0, random_state=0, n_jobs=1, verbose=-1,
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.03, max_iter=300, l2_regularization=3.0, random_state=0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=0, n_jobs=1,
        ),
        "logistic_regression": LogisticRegression(max_iter=1000),
    }


def evaluate_market(pool, games, market_name, split_mode):
    pool = build_features(pool, games)
    if len(pool) < 500:
        print(f"  {market_name}: n={len(pool)} INSUFFICIENT DATA (<500)")
        return []
    train, test = (split_per_year(pool) if split_mode == "per_year" else split_chronological(pool))
    if len(train) < 50 or len(test) < 50:
        print(f"  {market_name}: INSUFFICIENT DATA after split")
        return []

    X_train_raw, y_train = train[FEATURES], train["win"].astype(int)
    X_test_raw, y_test = test[FEATURES], test["win"].astype(int)

    # NaN handling: XGBoost/HistGBM take NaN natively; RF/LogReg cannot, so
    # impute with the TRAIN median only (no test-set leakage into the fit).
    imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    X_train_imp = pd.DataFrame(imputer.transform(X_train_raw), columns=FEATURES, index=X_train_raw.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=FEATURES, index=X_test_raw.index)
    scaler = StandardScaler().fit(X_train_imp)
    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    results = []
    for name, model in make_models().items():
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            test_pred = model.predict_proba(X_test_scaled)[:, 1]
        elif name == "random_forest":
            model.fit(X_train_imp, y_train)
            test_pred = model.predict_proba(X_test_imp)[:, 1]
        else:
            model.fit(X_train_raw, y_train)
            test_pred = model.predict_proba(X_test_raw)[:, 1]

        auc = roc_auc_score(y_test, test_pred)
        acc = accuracy_score(y_test, test_pred >= 0.5)
        t = test.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        sub = t[(t.odds < 0) & (t.edge > 0.0)]  # single pre-specified cut, no threshold search
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        verdict = "PASS" if gate(s) else "FAIL"
        print(f"  {market_name:<20} {name:<20} AUC={auc:.3f} acc={acc:.3f} "
              f"n={s['n']:<6} ROI={s['roi']} CI=[{s['lo']},{s['hi']}]  {verdict}")
        results.append({"market": market_name, "model": name, "auc": round(auc, 3), "acc": round(acc, 3), **s, "verdict": verdict})
    return results


def main():
    con = duckdb.connect(DB, read_only=True)
    print("building real team game log, Elo, L10, bullpen fatigue, park factors...")
    games, box = build_games_and_box(con)

    all_results = []
    print("\n=== 5 warehouse/game FAIL markets x 5 models, single pre-specified cut (edge>0.0) ===")
    for market_key, stat_col, kind in [
        ("batter_total_bases", "total_bases", "batter"),
        ("batter_home_runs", "home_runs", "batter"),
        ("pitcher_strikeouts", "strikeouts", "pitcher"),
    ]:
        pool = build_warehouse_pool(con, box, market_key, stat_col, kind)
        all_results += evaluate_market(pool, games, market_key, "per_year")
    for market_key in ["h2h", "totals"]:
        pool = build_game_pool(con, games, market_key)
        all_results += evaluate_market(pool, games, market_key, "per_year")

    print("\n=== 3 real-backfilled FAIL markets x 5 models, single pre-specified cut (edge>0.0) ===")
    backfill_specs = [
        ("pitcher_outs", ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl", "pitcher_outs_stat",
         ["Pitcher"], True, "outs AS pitcher_outs_stat"),
        ("batter_runs_scored", ROOT / "data_raw" / "runs_scored_odds_cache.jsonl", "runs_scored",
         ["Catcher", "Hitter", "Infielder", "Outfielder"], False, "runs_scored"),
        ("batter_strikeouts", ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl", "batter_strikeouts",
         ["Catcher", "Hitter", "Infielder", "Outfielder"], False, "batter_strikeouts"),
    ]
    for market_name, cache_path, stat_col, position_filter, starter_only, select_col in backfill_specs:
        if not cache_path.exists():
            print(f"  {market_name}: cache file missing ({cache_path.name}), skipping")
            continue
        box_bk = con.execute(f"""
            SELECT game_pk, player_name, {select_col}, is_starter, position_type FROM boxscore
        """).fetchdf()
        box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
        pool = build_market_dataset(cache_path, market_name, stat_col, box_bk, position_filter, starter_only)
        all_results += evaluate_market(pool, games, market_name, "chronological")

    con.close()

    print("\n=== SUMMARY: any PASS across 8 markets x 5 models, single pre-specified cut ===")
    passes = [r for r in all_results if r["verdict"] == "PASS"]
    if not passes:
        print(f"  None. {len(all_results)} market/model combinations tested, zero passes on the "
              f"pre-specified cut. XGBoost was already the strongest tabular learner tried in "
              f"prior addenda -- this confirms it wasn't leaving a materially better model on "
              f"the table among standard alternatives.")
    else:
        for r in passes:
            print(f"  PASS: {r['market']} / {r['model']}  n={r['n']} ROI={r['roi']} CI=[{r['lo']},{r['hi']}]")
        print(f"  {len(passes)} of {len(all_results)} market/model combinations passed. Reported "
              f"as-is -- a single model passing among 5 tried, on one of 8 markets, is exactly "
              f"the multiple-comparisons exposure flagged in HANDOFF.md's guardrails, not "
              f"grounds to declare that market fixed without independent confirmation.")


if __name__ == "__main__":
    main()
