"""
Addendum 50: integrate the freshly-backfilled totals odds (937 real games,
2026-05-25 to 2026-09-23, data_raw/totals_gap_odds_cache.jsonl) into the
fuller game log, extending it past Addendum 48's 2026-06-25 ceiling using
cache_mlb_historical_outcomes directly (it already carries real completed
games all the way to 2026-09-23 -- confirmed this session -- so no new
data source is needed for the game-outcome side, only the odds side
needed backfilling).

Re-runs the SAME walk-forward validation as Addendum 49 (expanding-window
folds, CV-AUC hyperparameter selection on training data only, ROI touched
exactly once on the full pooled held-out set) -- now with real 2026 summer
data included, not just through late May.

Best price per side at the primary line (closest-to-even-implied-
probability, same convention as pick_main_line elsewhere in this project)
taken as MAX decimal price across bookmakers, converted to American odds
(same convention as every other odds table in this project).
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob, norm_name  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, TEAM_NAME_ALIAS  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS  # noqa: E402
from addendum49_h2h_walkforward import attach_enrichment, cv_select, PARAM_GRID_WIDE  # noqa: E402

GAP_CACHE = ROOT / "data_raw" / "totals_gap_odds_cache.jsonl"
N_FOLDS = 6


def decimal_to_american(d):
    if d is None or d <= 1.0:
        return np.nan
    if d >= 2.0:
        return (d - 1) * 100
    return -100 / (d - 1)


def parse_gap_odds():
    """Best (MAX decimal) price per side at the primary (most-quoted) line,
    per game -- same 'best across books' convention as client_closing_odds
    elsewhere in this project."""
    rows = []
    with open(GAP_CACHE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            resp = rec["response"]
            if "error" in resp or "data" not in resp:
                continue
            data = resp["data"]
            books = data.get("bookmakers", [])
            point_prices = {}  # point -> {"over": [decimals], "under": [decimals]}
            for bk in books:
                for mk in bk.get("markets", []):
                    if mk["key"] != "totals":
                        continue
                    for oc in mk["outcomes"]:
                        pt = oc["point"]
                        side = oc["name"].lower()
                        point_prices.setdefault(pt, {"over": [], "under": []})
                        point_prices[pt][side].append(oc["price"])
            if not point_prices:
                continue
            # primary line: the one quoted by the most books (ties broken by lowest point)
            primary_pt = max(point_prices.items(), key=lambda kv: (len(kv[1]["over"]) + len(kv[1]["under"]), -kv[0]))[0]
            overs = point_prices[primary_pt]["over"]
            unders = point_prices[primary_pt]["under"]
            if not overs or not unders:
                continue
            rows.append({
                "game_pk": rec["game_pk"],
                "line": primary_pt,
                "best_over_odds": decimal_to_american(max(overs)),
                "best_under_odds": decimal_to_american(max(unders)),
            })
    df = pd.DataFrame(rows)
    print(f"  parsed {len(df)} games with usable totals odds from {GAP_CACHE.name}")
    return df


def build_extended_games(con, con2, games_v2, id_to_name):
    """Extend Addendum 48's games_v3 (through 2026-06-25) with real games
    from cache_mlb_historical_outcomes for dates after that, up through
    2026-09-23 -- same cleaning (exclude exhibitions, alias, name->id map)
    as build_fuller_games, applied to just the new slice."""
    name_to_id = {v: k for k, v in id_to_name.items()}
    existing_pks = set(games_v2.game_pk)

    raw = con.execute("""
        SELECT game_pk, home_team, away_team, home_score, away_score,
               TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND TRY_CAST(commence_time AS DATE) > DATE '2026-06-25'
          AND TRY_CAST(commence_time AS DATE) <= DATE '2026-09-23'
    """).fetchdf()
    raw = raw[~raw.game_pk.isin(existing_pks)]
    raw = raw[~raw.home_team.isin(EXCLUDE_TEAMS) & ~raw.away_team.isin(EXCLUDE_TEAMS)]
    raw["home_team"] = raw["home_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["away_team"] = raw["away_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["home_team_id"] = raw["home_team"].map(name_to_id)
    raw["away_team_id"] = raw["away_team"].map(name_to_id)
    n_before = len(raw)
    raw = raw.dropna(subset=["home_team_id", "away_team_id", "game_date"]).drop_duplicates(subset=["game_pk"])
    print(f"  extension slice (2026-06-26 -> 2026-09-23): {n_before} candidate games -> {len(raw)} usable")
    raw["home_team_id"] = raw["home_team_id"].astype("int64")
    raw["away_team_id"] = raw["away_team_id"].astype("int64")
    raw["home_runs_"] = raw["home_score"].astype(float)
    raw["away_runs_"] = raw["away_score"].astype(float)
    raw["event_id"] = None  # not needed downstream for totals (odds already resolved directly by game_pk)

    combined = pd.concat([games_v2, raw[["game_pk", "game_date", "home_team_id", "away_team_id",
                                          "home_runs_", "away_runs_", "event_id"]]], ignore_index=True)
    combined = combined.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    assert combined.game_pk.nunique() == len(combined), "duplicate game_pk after extension -- STOP"

    # Recompute Elo/L10/park/bullpen features over the FULL extended log --
    # more accurate than before since the schedule is now more complete.
    from xgboost_individual_markets import build_elo_l10, build_bullpen_fatigue
    elo_l10 = build_elo_l10(combined[["game_pk", "home_team_id", "away_team_id", "home_runs_", "away_runs_", "game_date"]])
    combined = combined.drop(columns=[c for c in ["home_elo", "away_elo", "home_l10", "away_l10"] if c in combined.columns])
    combined = combined.merge(elo_l10, on="game_pk", how="inner")
    assert combined.game_pk.nunique() == len(combined), "elo merge changed row count -- STOP"

    fatigue = build_bullpen_fatigue(con)
    combined["home_fatigue"] = combined.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    combined["away_fatigue"] = combined.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)

    if "runs_factor" not in combined.columns:
        combined["runs_factor"] = np.nan
        combined["hr_factor"] = np.nan
        combined["k_factor"] = np.nan
        combined["hits_factor"] = np.nan

    return combined


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building Addendum 48 fuller game log (through 2026-06-25)...")
    games_v2, id_to_name = build_fuller_games(con, con2)

    print("\nextending game log with cache_mlb_historical_outcomes through 2026-09-23...")
    games_ext = build_extended_games(con, con2, games_v2, id_to_name)
    print(f"  extended game log: n={len(games_ext)}  date range {games_ext.game_date.min()} -> {games_ext.game_date.max()}")

    print("\nparsing freshly-backfilled totals odds...")
    gap_odds = parse_gap_odds()

    # existing totals odds (through ~2026-05-24) via client_closing_odds
    existing_odds = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'totals' AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    existing_odds["game_pk"] = pd.to_numeric(existing_odds["game_pk"], errors="coerce")
    existing_odds["line"] = pd.to_numeric(existing_odds["line"], errors="coerce")
    existing_odds["best_over_odds"] = pd.to_numeric(existing_odds["best_over_odds"], errors="coerce")
    existing_odds["best_under_odds"] = pd.to_numeric(existing_odds["best_under_odds"], errors="coerce")
    existing_odds = existing_odds.dropna(subset=["game_pk"])
    existing_odds["game_pk"] = existing_odds["game_pk"].astype("int64")

    from xgboost_individual_markets import pick_main_line
    existing_odds = pick_main_line(existing_odds, ["game_pk"], over_odds_col="best_over_odds")

    overlap = set(existing_odds.game_pk) & set(gap_odds.game_pk)
    print(f"  existing odds: {len(existing_odds)} games | gap-backfilled odds: {len(gap_odds)} games | overlap: {len(overlap)}")
    gap_odds_clean = gap_odds[~gap_odds.game_pk.isin(set(existing_odds.game_pk))]
    all_odds = pd.concat([existing_odds[["game_pk", "line", "best_over_odds", "best_under_odds"]],
                           gap_odds_clean], ignore_index=True)
    print(f"  combined totals odds pool: n={len(all_odds)} games")

    m = all_odds.merge(games_ext, on="game_pk", how="inner")
    m["total_runs"] = m["home_runs_"] + m["away_runs_"]
    under = m.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = ""
    over = m.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = ""
    pool = pd.concat([under, over]).dropna(subset=["odds"])
    n_before = len(pool)
    print(f"  totals pool (pre-enrichment): n={n_before}")

    pool = attach_enrichment(con, con2, pool, id_to_name)
    print(f"  row-count sanity check: before={n_before}  after={len(pool)}  {'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
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
    print("  Addendum 48 (fuller game log, no gap backfill) single-split: n=234  ROI=-9.38%  CI=[-23.35,4.59]  FAIL")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
