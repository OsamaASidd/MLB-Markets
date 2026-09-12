"""
Model B: pooled, all-markets, all-4-real-years (2023-2026) model, using
features I can build myself across that whole span (the production
150-factor set doesn't exist before 2026-05-17 -- see Model A for that).

Features: market type, odds-implied probability, team Elo (rebuilt here
from real client box scores across 2023-2026 -- NOT the external 2014-2019
dataset used in Addendum 5), team L10 form, real ballpark factors, real
bullpen fatigue (relief outs, last 2 days) -- all point-in-time correct.

Markets included: batter_hits, batter_total_bases, batter_rbis,
batter_home_runs, pitcher_strikeouts, h2h, spreads, totals -- the 8 with
real odds-warehouse coverage. batter_runs_scored and pitcher_outs excluded
(zero warehouse rows, a real provider gap, not a choice).

Split: 75/25 done chronologically WITHIN each calendar year separately,
then all four years' 75% pooled into one TRAIN set and all four years'
25% pooled into one TEST set -- per your spec, not a single global cutoff.
"""
import pathlib
import unicodedata
from collections import defaultdict, deque

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500

K, HFA, REGRESS = 20, 24, 1.0 / 3.0
BASE_RATING = 1500.0


def norm_name(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def ip_to_outs(ip_str):
    if ip_str is None:
        return 0
    try:
        whole = int(float(ip_str))
        frac = round(float(ip_str) - whole, 1)
        return whole * 3 + {0.0: 0, 0.1: 1, 0.2: 2}.get(frac, 0)
    except (ValueError, TypeError):
        return 0


def profit(win, odds):
    if not win:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_prob(odds):
    return np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100))


def stat(profits, wins):
    n = len(profits)
    if n == 0:
        return {"n": 0, "roi": None, "lo": None, "hi": None, "wr": None}
    m = profits.mean() * 100
    se = profits.std(ddof=1) / np.sqrt(n) * 100 if n > 1 else np.nan
    return {"n": n, "wr": round(wins.mean() * 100, 1), "roi": round(m, 2),
            "lo": round(m - 1.96 * se, 2) if not np.isnan(se) else None,
            "hi": round(m + 1.96 * se, 2) if not np.isnan(se) else None}


def gate(s):
    return s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0


def build_team_game_log(con):
    """One row per (team_id, game_pk): real runs scored/allowed, home/away, date."""
    team_runs = con.execute("""
        SELECT game_pk, team_id, sum(runs_scored) AS team_runs
        FROM boxscore WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
        GROUP BY 1,2 HAVING sum(runs_scored) IS NOT NULL
    """).fetchdf()
    side_map = con.execute("""
        SELECT g.game_pk, l.team_side, b.team_id, count(*) n
        FROM lineups l JOIN client_games g ON l.event_id = g.event_id
        JOIN boxscore b ON b.player_id = l.player_id AND b.game_pk = g.game_pk
        GROUP BY 1,2,3
    """).fetchdf()
    side_map["game_pk"] = pd.to_numeric(side_map["game_pk"], errors="coerce")
    side_map = side_map.sort_values("n", ascending=False).drop_duplicates(subset=["game_pk", "team_side"])
    dates = con.execute("SELECT DISTINCT game_pk, game_date FROM boxscore").fetchdf()
    dates["game_date"] = pd.to_datetime(dates["game_date"])

    home = side_map[side_map.team_side == "home"][["game_pk", "team_id"]].rename(columns={"team_id": "home_team_id"})
    away = side_map[side_map.team_side == "away"][["game_pk", "team_id"]].rename(columns={"team_id": "away_team_id"})
    g = dates.merge(home, on="game_pk").merge(away, on="game_pk")
    g = g.merge(team_runs.rename(columns={"team_id": "home_team_id", "team_runs": "home_runs_"}),
                on=["game_pk", "home_team_id"], how="inner")
    g = g.merge(team_runs.rename(columns={"team_id": "away_team_id", "team_runs": "away_runs_"}),
                on=["game_pk", "away_team_id"], how="inner")
    return g.sort_values("game_date").reset_index(drop=True)


def build_elo_l10(games):
    """Point-in-time Elo + L10 per game, from real 2023-2026 results."""
    elo, l10 = {}, {}
    current_season = None
    rows = []
    for _, g in games.iterrows():
        season = g["game_date"].year
        if current_season is not None and season != current_season:
            for t in elo:
                elo[t] += REGRESS * (BASE_RATING - elo[t])
        current_season = season

        h, a = g["home_team_id"], g["away_team_id"]
        elo.setdefault(h, BASE_RATING)
        elo.setdefault(a, BASE_RATING)
        l10.setdefault(h, deque(maxlen=10))
        l10.setdefault(a, deque(maxlen=10))

        eh, ea = elo[h], elo[a]
        l10h = sum(l10[h]) if l10[h] else 5
        l10a = sum(l10[a]) if l10[a] else 5

        rows.append({"game_pk": g["game_pk"], "home_elo": eh, "away_elo": ea,
                      "home_l10": l10h, "away_l10": l10a})

        home_win = 1 if g["home_runs_"] > g["away_runs_"] else 0
        exp_home = 1.0 / (1.0 + 10 ** ((ea - (eh + HFA)) / 400.0))
        elo[h] = eh + K * (home_win - exp_home)
        elo[a] = ea + K * ((1 - home_win) - (1 - exp_home))
        l10[h].append(home_win)
        l10[a].append(1 - home_win)
    return pd.DataFrame(rows)


def build_bullpen_fatigue(con):
    relief = con.execute("""
        SELECT game_pk, game_date, team_id, innings_pitched
        FROM boxscore WHERE position_type='Pitcher' AND is_starter=False AND innings_pitched IS NOT NULL
    """).fetchdf()
    relief["outs"] = relief["innings_pitched"].map(ip_to_outs)
    relief["game_date"] = pd.to_datetime(relief["game_date"])
    daily = relief.groupby(["team_id", "game_date"])["outs"].sum().reset_index()
    fatigue = {}
    for team_id, grp in daily.sort_values("game_date").groupby("team_id"):
        dts, outs = grp["game_date"].tolist(), grp["outs"].tolist()
        for i, d in enumerate(dts):
            fatigue[(team_id, d)] = sum(o for dd, o in zip(dts, outs) if 0 < (d - dd).days <= 2)
    return fatigue


def main():
    con = duckdb.connect(DB, read_only=True)
    print("building real team game log, Elo, L10, bullpen fatigue from 2023-2026 client data...")
    games = build_team_game_log(con)
    elo_l10 = build_elo_l10(games)
    fatigue = build_bullpen_fatigue(con)
    games = games.merge(elo_l10, on="game_pk")
    print(f"real games with full feature set: {len(games):,}, {games.game_date.min().date()} -> {games.game_date.max().date()}")

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

    def pick_main_line(df, group_cols, over_odds_col="best_over_odds"):
        """Many player-props here are offered as a full alt-line ladder (e.g.
        total_bases 0.5/1.5/2.5/3.5/4.5 all for the same real player-game).
        Sorting by line and taking the lowest (or highest) grabs an extreme
        alt line every time -- confirmed: 131,453 of ~146,000 deduped
        total_bases rows landed on the 0.5 alt line, not the book's real
        50/50 main line -- which is a mispriced corner of the market, not
        representative pricing, and explains an impossibly large apparent
        edge. Instead pick, per group, the single line whose OVER price is
        closest to a true coin-flip (|implied_prob - 0.5| smallest) -- that
        is the book's actual primary line."""
        df = df.copy()
        df["_over_prob"] = implied_prob(df[over_odds_col])
        df["_dist_even"] = (df["_over_prob"] - 0.5).abs()
        return df.sort_values("_dist_even").groupby(group_cols, as_index=False).first().drop(columns=["_over_prob", "_dist_even"])

    def load_batter_market(market_key, stat_col):
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
        box_b = box[box.position_type.isin(["Catcher", "Hitter", "Infielder", "Outfielder"])]
        m = odds.merge(box_b[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
        m = m.dropna(subset=[stat_col])
        under = m.copy(); under["win"] = under[stat_col] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
        over = m.copy(); over["win"] = over[stat_col] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
        both = pd.concat([under, over]).dropna(subset=["odds"])
        both["market"] = market_key
        return both[["market", "game_pk", "odds", "win", "side"]]

    def load_pitcher_market(market_key, stat_col):
        odds = con.execute(f"""
            SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.player_name
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
        box_p = box[(box.position_type == "Pitcher") & (box.is_starter == True)]
        m = odds.merge(box_p[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
        m = m.dropna(subset=[stat_col])
        under = m.copy(); under["win"] = under[stat_col] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
        over = m.copy(); over["win"] = over[stat_col] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
        both = pd.concat([under, over]).dropna(subset=["odds"])
        both["market"] = market_key
        return both[["market", "game_pk", "odds", "win", "side"]]

    def load_game_market(market_key):
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
            m["win"] = m["away_runs_"] > m["home_runs_"]
            m["side"] = "away"; m["market"] = "h2h"
            return m.dropna(subset=["odds"])[["market", "game_pk", "odds", "win", "side"]]
        if market_key == "spreads":
            odds = con.execute("""
                SELECT co.game_pk, co.line, co.best_over_odds AS odds
                FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                      JOIN client_games g ON co.event_id = g.event_id) co
                WHERE co.market_key = 'spreads__away' AND co.game_pk IS NOT NULL AND co.game_pk != ''
            """).fetchdf()
            odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
            odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
            odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
            odds = odds.dropna(subset=["game_pk"])
            # one row per real game -- multiple spread lines per game (10,794
            # rows / 7,440 real games) were otherwise all counted separately
            odds = pick_main_line(odds, ["game_pk"], over_odds_col="odds")
            m = odds.merge(games[["game_pk", "home_runs_", "away_runs_"]], on="game_pk", how="inner")
            m["win"] = (m["away_runs_"] + m["line"]) > m["home_runs_"]
            m["side"] = "away"; m["market"] = "spreads"
            return m.dropna(subset=["odds"])[["market", "game_pk", "odds", "win", "side"]]
        if market_key == "totals":
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
            # one row per real game -- multiple total lines per game (13,818
            # rows / 7,440 real games) were otherwise all counted separately
            odds = pick_main_line(odds, ["game_pk"], over_odds_col="best_over_odds")
            m = odds.merge(games[["game_pk", "home_runs_", "away_runs_"]], on="game_pk", how="inner")
            m["total_runs"] = m["home_runs_"] + m["away_runs_"]
            under = m.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
            over = m.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
            both = pd.concat([under, over]).dropna(subset=["odds"])
            both["market"] = "totals"
            return both[["market", "game_pk", "odds", "win", "side"]]

    print("loading real picks for all 8 markets with warehouse coverage...")
    parts = [
        load_batter_market("batter_hits", "hits"),
        load_batter_market("batter_total_bases", "total_bases"),
        load_batter_market("batter_rbis", "rbi"),
        load_batter_market("batter_home_runs", "home_runs"),
        load_pitcher_market("pitcher_strikeouts", "strikeouts"),
        load_game_market("h2h"),
        load_game_market("spreads"),
        load_game_market("totals"),
    ]
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.merge(games[["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                              "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]],
                       on="game_pk", how="inner")
    print(f"\npooled multi-year dataset: n={len(pool):,}, {pool.game_date.min().date()} -> {pool.game_date.max().date()}")
    print(pool.groupby("market").size().sort_values(ascending=False))
    print(pool.groupby(pool.game_date.dt.year).size())

    pool["market_code"] = pool["market"].astype("category").cat.codes
    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])

    features = ["market_code", "side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                "runs_factor", "hr_factor", "k_factor", "hits_factor"]

    # per-year 75/25 chronological split, pooled across years
    train_parts, test_parts = [], []
    for year, grp in pool.groupby(pool.game_date.dt.year):
        grp = grp.sort_values("game_date")
        cut = int(len(grp) * 0.75)
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    print(f"\nTRAIN n={len(train):,}  TEST n={len(test):,}  (75/25 within each of 2023/2024/2025/2026, pooled)")

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
        reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0,
    )
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    print(f"\nAUC  TRAIN={roc_auc_score(y_train, train_pred):.3f}  TEST={roc_auc_score(y_test, test_pred):.3f}")
    print(f"ACCURACY (0.5)  TRAIN={accuracy_score(y_train, train_pred>=0.5):.3f}  TEST={accuracy_score(y_test, test_pred>=0.5):.3f}")

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nlearned feature weights (multi-year, cross-market):")
    for f, v in imp.items():
        print(f"  {f:<20} {v:.4f}")

    test = test.copy()
    test["model_prob"] = test_pred
    test["edge"] = test["model_prob"] - test["market_prob"]
    print("\n-- betting on edge, minus-money only, TEST holdout (all years pooled) --")
    for thresh in [0.0, 0.03, 0.05, 0.08]:
        sub = test[(test.odds < 0) & (test.edge > thresh)]
        profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
        s = stat(profits, sub["win"])
        print(f"  edge>{thresh:<5} n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if gate(s) else 'fail'}")

    con.close()


if __name__ == "__main__":
    main()
