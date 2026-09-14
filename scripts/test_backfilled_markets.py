"""
Honest individual-market test for pitcher_outs and batter_runs_scored using
the real historical odds just backfilled from The Odds API (previously
zero/near-zero real warehouse coverage in this project -- confirmed the gap
was the client's own backfill pipeline, not a provider limitation).

Same discipline as every other market in this project: real point-in-time
features (Elo, L10, bullpen fatigue, park factors), single 75/25
chronological split, the corrected official gate rule (n>=500 -> ROI>0;
n<500 -> CI lower bound>0), and both the single pre-specified cut
(edge>0.0, no threshold search) and, separately and clearly labeled, a
best-of-5 search flagged as multiple-comparisons exposed -- exactly the
same distinction already applied to the other individual markets.
"""
import json
import pathlib
import sys
from collections import deque

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from xgboost_individual_markets import (
    norm_name, implied_prob, profit, stat, gate,
    build_team_game_log, build_elo_l10, build_bullpen_fatigue,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def decimal_to_american(dec):
    if dec is None or (isinstance(dec, float) and np.isnan(dec)) or dec <= 1.0:
        return None
    if dec >= 2.0:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def parse_cache(cache_path, market_key):
    """One row per (event_id, player_name): consensus line (mode across
    books), best price at that line for over and under separately."""
    rows = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            resp = rec["response"]
            if "error" in resp or not resp.get("data"):
                continue
            data = resp["data"]
            event_id = rec["event_id"]
            game_pk = rec["game_pk"]
            commence_time = data.get("commence_time")

            per_player = {}  # player_name -> {"lines": [...], "over": [(line,dec,book)], "under": [...]}
            for bm in data.get("bookmakers", []):
                book = bm["key"]
                for m in bm.get("markets", []):
                    if m["key"] != market_key:
                        continue
                    for o in m.get("outcomes", []):
                        player = o.get("description")
                        if player is None:
                            continue
                        side = o["name"].lower()
                        pt = o.get("point")
                        price = o.get("price")
                        if pt is None or price is None:
                            continue
                        d = per_player.setdefault(player, {"over": [], "under": []})
                        d[side].append((pt, price, book))

            for player, d in per_player.items():
                all_lines = [p for p, _, _ in d["over"]] + [p for p, _, _ in d["under"]]
                if not all_lines:
                    continue
                consensus_line = pd.Series(all_lines).mode().iloc[0]
                over_at_line = [(p, price, book) for p, price, book in d["over"] if p == consensus_line]
                under_at_line = [(p, price, book) for p, price, book in d["under"] if p == consensus_line]
                best_over = max(over_at_line, key=lambda x: x[1]) if over_at_line else None
                best_under = max(under_at_line, key=lambda x: x[1]) if under_at_line else None
                rows.append({
                    "event_id": event_id, "game_pk": game_pk, "commence_time": commence_time,
                    "player_name": player, "line": consensus_line,
                    "best_over_dec": best_over[1] if best_over else None,
                    "best_under_dec": best_under[1] if best_under else None,
                })
    return pd.DataFrame(rows)


def build_market_dataset(cache_path, market_key, stat_col, box, position_filter, starter_only):
    odds = parse_cache(cache_path, market_key)
    print(f"  parsed {len(odds):,} real (event,player) rows from cache")
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    odds["best_over_odds"] = odds["best_over_dec"].map(decimal_to_american)
    odds["best_under_odds"] = odds["best_under_dec"].map(decimal_to_american)
    odds = odds.dropna(subset=["game_pk"])

    box_f = box[box.position_type.isin(position_filter)]
    if starter_only:
        box_f = box_f[box_f.is_starter == True]
    m = odds.merge(box_f[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
    m = m.dropna(subset=[stat_col])
    print(f"  {len(m):,} rows matched to a real box-score outcome")

    under = m.copy(); under["win"] = under[stat_col] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = under["name_norm"]
    over = m.copy(); over["win"] = over[stat_col] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = over["name_norm"]
    both = pd.concat([under, over]).dropna(subset=["odds"])
    return both[["market_key", "game_pk", "commence_time", "odds", "win", "side", "tiebreak"]] if "market_key" in both.columns else both.assign(market_key=market_key)[["market_key", "game_pk", "commence_time", "odds", "win", "side", "tiebreak"]]


def fit_and_report(pool, games, market_name):
    pool = pool.merge(games[["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                              "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]],
                       on="game_pk", how="inner")
    if len(pool) < MIN_GRADED:
        print(f"  INSUFFICIENT DATA: n={len(pool)} < {MIN_GRADED}")
        return None

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])
    features = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                "runs_factor", "hr_factor", "k_factor", "hits_factor"]

    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    split = int(len(pool) * 0.75)
    train, test = pool.iloc[:split].copy(), pool.iloc[split:].copy()
    print(f"  TRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
        reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred >= 0.5)
    print(f"  AUC TEST={auc:.3f}  accuracy TEST={acc:.3f}")

    test = test.copy()
    test["model_prob"] = test_pred
    test["edge"] = test["model_prob"] - test["market_prob"]

    sub0 = test[(test.odds < 0) & (test.edge > 0.0)]
    profits0 = sub0.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s0 = stat(profits0, sub0["win"])
    v0 = "PASS" if gate(s0) else "FAIL"
    print(f"  single pre-specified cut (edge>0.0): n={s0['n']} ROI={s0['roi']} CI=[{s0['lo']},{s0['hi']}]  {v0}")

    best = None
    for thresh in [0.0, 0.02, 0.03, 0.05, 0.08]:
        sub = test[(test.odds < 0) & (test.edge > thresh)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        passed = gate(s)
        if best is None or (passed and not best.get("passed")):
            best = {"thresh": thresh, **s, "passed": passed}
        elif passed and best.get("passed") and s["roi"] and best["roi"] and s["roi"] > best["roi"]:
            best = {"thresh": thresh, **s, "passed": passed}
    vbest = "PASS" if best["passed"] else "FAIL"
    print(f"  best-of-5 cut (multiple-comparisons exposed): edge>{best['thresh']} n={best['n']} "
          f"ROI={best['roi']} CI=[{best['lo']},{best['hi']}]  {vbest}")

    return {"market": market_name, "auc": round(auc, 3), "acc": round(acc, 3),
            "no_filter": s0, "no_filter_verdict": v0, "best_cut": best, "best_verdict": vbest}


def main():
    con = duckdb.connect(DB, read_only=True)
    print("building real team game log, Elo, L10, bullpen fatigue from 2023-2026 client data...")
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
        SELECT game_pk, player_name, runs_scored, outs AS pitcher_outs_stat, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    con.close()

    results = []
    print("\n=== pitcher_outs (real backfilled odds) ===")
    po_cache = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"
    pool = build_market_dataset(po_cache, "pitcher_outs", "pitcher_outs_stat", box,
                                 ["Pitcher"], starter_only=True)
    r = fit_and_report(pool, games, "pitcher_outs")
    if r:
        results.append(r)

    print("\n=== batter_runs_scored (real backfilled odds) ===")
    rs_cache = ROOT / "data_raw" / "runs_scored_odds_cache.jsonl"
    pool2 = build_market_dataset(rs_cache, "batter_runs_scored", "runs_scored", box,
                                  ["Catcher", "Hitter", "Infielder", "Outfielder"], starter_only=False)
    r2 = fit_and_report(pool2, games, "runs_scored")
    if r2:
        results.append(r2)

    print("\n=== batter_strikeouts (real backfilled odds -- never testable before this) ===")
    bk_cache = ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl"
    con2 = duckdb.connect(DB, read_only=True)
    box_bk = con2.execute("""
        SELECT game_pk, player_name, batter_strikeouts, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    con2.close()
    box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
    pool3 = build_market_dataset(bk_cache, "batter_strikeouts", "batter_strikeouts", box_bk,
                                  ["Catcher", "Hitter", "Infielder", "Outfielder"], starter_only=False)
    r3 = fit_and_report(pool3, games, "batter_strikeouts")
    if r3:
        results.append(r3)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['market']:<20} no-filter: {r['no_filter_verdict']:<6}  best-of-5: {r['best_verdict']}")


if __name__ == "__main__":
    main()
