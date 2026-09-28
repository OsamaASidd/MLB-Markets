"""
Addendum 57: integrate the freshly-backfilled pitcher_strikeouts real odds
(2,842 real games, 2025-05-28 to 2026-09-23 gap) with the existing
2023-2025 real-odds pool, extend the game log the same way Addendum 50 did
for totals, and re-test via walk-forward validation.

Same disciplined methodology as every prior addendum: CV-AUC-only model
selection on training data, official gate touched exactly once on the
full pooled held-out set, row-count sanity checks throughout.
"""
import json
import pathlib
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob, norm_name, pick_main_line  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, TEAM_NAME_ALIAS, add_bullpen_features, add_starter_game_features, add_park_features, build_box_ids  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS  # noqa: E402
from addendum49_h2h_walkforward import PARAM_GRID_WIDE, FIXED_PARAMS, cv_select  # noqa: E402

GAP_CACHE = ROOT / "data_raw" / "pitcher_k_gap_odds_cache.jsonl"
N_FOLDS = 6


def decimal_to_american(d):
    if d is None or d <= 1.0:
        return np.nan
    return (d - 1) * 100 if d >= 2.0 else -100 / (d - 1)


def parse_gap_odds():
    """Best (MAX decimal) price per side, per (game_pk, player), at the
    primary (most-quoted) line for that player."""
    rows = []
    with open(GAP_CACHE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            resp = rec["response"]
            if "error" in resp or "data" not in resp:
                continue
            data = resp["data"]
            by_player = {}  # player -> point -> {"over":[], "under":[]}
            for bk in data.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] != "pitcher_strikeouts":
                        continue
                    for oc in mk["outcomes"]:
                        player = oc.get("description")
                        if not player:
                            continue
                        pt = oc["point"]
                        side = oc["name"].lower()
                        by_player.setdefault(player, {}).setdefault(pt, {"over": [], "under": []})
                        by_player[player][pt][side].append(oc["price"])
            for player, points in by_player.items():
                primary_pt = max(points.items(), key=lambda kv: (len(kv[1]["over"]) + len(kv[1]["under"]), -kv[0]))[0]
                overs = points[primary_pt]["over"]
                unders = points[primary_pt]["under"]
                if not overs or not unders:
                    continue
                rows.append({
                    "game_pk": rec["game_pk"], "player_name": player, "line": primary_pt,
                    "best_over_odds": decimal_to_american(max(overs)),
                    "best_under_odds": decimal_to_american(max(unders)),
                })
    df = pd.DataFrame(rows)
    print(f"  parsed {len(df)} (game, pitcher) rows with usable pitcher_strikeouts odds from gap backfill")
    return df


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building Addendum 48 fuller game log (through 2026-06-25)...")
    games_v2, id_to_name = build_fuller_games(con, con2)
    name_to_id = {v: k for k, v in id_to_name.items()}

    print("\nextending game log with cache_mlb_historical_outcomes through 2026-09-23...")
    existing_pks = set(games_v2.game_pk)
    raw = con.execute("""
        SELECT game_pk, home_team, away_team, home_score, away_score,
               TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND TRY_CAST(commence_time AS DATE) > DATE '2026-06-25'
    """).fetchdf()
    raw = raw[~raw.game_pk.isin(existing_pks)]
    raw = raw[~raw.home_team.isin(EXCLUDE_TEAMS) & ~raw.away_team.isin(EXCLUDE_TEAMS)]
    raw["home_team"] = raw["home_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["away_team"] = raw["away_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["home_team_id"] = raw["home_team"].map(name_to_id)
    raw["away_team_id"] = raw["away_team"].map(name_to_id)
    raw = raw.dropna(subset=["home_team_id", "away_team_id", "game_date"]).drop_duplicates(subset=["game_pk"])
    raw["home_team_id"] = raw["home_team_id"].astype("int64")
    raw["away_team_id"] = raw["away_team_id"].astype("int64")
    raw["home_runs_"] = raw["home_score"].astype(float)
    raw["away_runs_"] = raw["away_score"].astype(float)
    raw["event_id"] = None
    games_ext = pd.concat([games_v2, raw[["game_pk", "game_date", "home_team_id", "away_team_id",
                                           "home_runs_", "away_runs_", "event_id"]]], ignore_index=True)
    games_ext = games_ext.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    assert games_ext.game_pk.nunique() == len(games_ext), "duplicate game_pk after extension -- STOP"

    from xgboost_individual_markets import build_elo_l10, build_bullpen_fatigue
    elo_l10 = build_elo_l10(games_ext[["game_pk", "home_team_id", "away_team_id", "home_runs_", "away_runs_", "game_date"]])
    games_ext = games_ext.drop(columns=[c for c in ["home_elo", "away_elo", "home_l10", "away_l10"] if c in games_ext.columns])
    games_ext = games_ext.merge(elo_l10, on="game_pk", how="inner")
    assert games_ext.game_pk.nunique() == len(games_ext), "elo merge changed row count -- STOP"

    fatigue = build_bullpen_fatigue(con)
    games_ext["home_fatigue"] = games_ext.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    games_ext["away_fatigue"] = games_ext.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)
    if "runs_factor" not in games_ext.columns:
        for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
            games_ext[c] = np.nan
    print(f"  extended game log: n={len(games_ext)}  date range {games_ext.game_date.min()} -> {games_ext.game_date.max()}")

    print("\nAddendum 37 enrichment (bullpen quality, starter quality, park dims)...")
    games_ext = add_bullpen_features(con2, games_ext, id_to_name)
    games_ext = add_starter_game_features(con, con2, games_ext)
    games_ext = add_park_features(con, con2, games_ext)

    print("\nparsing freshly-backfilled pitcher_strikeouts odds...")
    gap_odds = parse_gap_odds()

    print("\nloading existing 2023-2025 real warehouse odds...")
    existing_odds = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.player_name
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'pitcher_strikeouts' AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    existing_odds["game_pk"] = pd.to_numeric(existing_odds["game_pk"], errors="coerce")
    for c in ["line", "best_over_odds", "best_under_odds"]:
        existing_odds[c] = pd.to_numeric(existing_odds[c], errors="coerce")
    existing_odds = existing_odds.dropna(subset=["game_pk"])
    existing_odds["game_pk"] = existing_odds["game_pk"].astype("int64")
    existing_odds["name_norm"] = existing_odds["player_name"].map(norm_name)
    existing_odds = pick_main_line(existing_odds, ["game_pk", "name_norm"])

    gap_odds["game_pk"] = gap_odds["game_pk"].astype("int64")
    gap_odds["name_norm"] = gap_odds["player_name"].map(norm_name)
    gap_odds = pick_main_line(gap_odds, ["game_pk", "name_norm"])

    overlap = set(existing_odds.game_pk) & set(gap_odds.game_pk)
    print(f"  existing (2023-2025): {len(existing_odds)} rows | gap-backfilled: {len(gap_odds)} rows | overlapping game_pks: {len(overlap)}")
    gap_odds_clean = gap_odds[~gap_odds.game_pk.isin(set(existing_odds.game_pk))]
    all_odds = pd.concat([existing_odds[["game_pk", "name_norm", "line", "best_over_odds", "best_under_odds"]],
                           gap_odds_clean[["game_pk", "name_norm", "line", "best_over_odds", "best_under_odds"]]],
                          ignore_index=True)
    print(f"  combined pitcher_strikeouts odds pool: n={len(all_odds)} (game, pitcher) rows")

    box = con.execute("SELECT game_pk, player_name, strikeouts, is_starter, position_type FROM boxscore WHERE position_type='Pitcher' AND is_starter=True").fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    m = all_odds.merge(box[["game_pk", "name_norm", "strikeouts"]], on=["game_pk", "name_norm"], how="inner").dropna(subset=["strikeouts"])
    under = m.copy(); under["win"] = under["strikeouts"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = under["name_norm"]
    over = m.copy(); over["win"] = over["strikeouts"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = over["name_norm"]
    pool = pd.concat([under, over]).dropna(subset=["odds"])
    n_before = len(pool)
    print(f"  pool matched to real boxscore strikeout outcomes: n={n_before}")

    pool = pool.merge(games_ext, on="game_pk", how="inner")
    print(f"  row-count sanity check (game-log merge): before={n_before}  after={len(pool)}  "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"  final pool: n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")

    FEATURES = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                "runs_factor", "hr_factor", "k_factor", "hits_factor",
                "bullpen_era_diff", "bullpen_k9_diff", "starter_xera_diff", "starter_era_diff",
                "starter_xba_diff", "lf_distance", "cf_distance", "rf_distance",
                "lf_wall_height", "cf_wall_height", "rf_wall_height", "cf_compass_degrees", "is_dome"]

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds) ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        X_train, y_train = train_df[FEATURES], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[FEATURES])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    print("\nFor reference:")
    print("  Addendum 52 (fuller game log, no gap backfill) walk-forward: n=310  ROI=4.74%  CI=[-6.0,15.48]  FAIL")
    print("  Addendum 55b (de-vigged, no gap backfill) walk-forward: n=1071  ROI=-0.45%  CI=[-6.17,5.26]  FAIL")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
