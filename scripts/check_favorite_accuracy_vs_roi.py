"""
Direct test of "accuracy" as a target, separate from ROI: for every pooled
market, bet the side the market's own price already favors (market_prob >
0.5) -- zero modeling required -- and report accuracy vs. ROI side by side.

Purpose: show precisely why chasing a raw accuracy number (e.g. "make this
above 80%") is not the same request as "make this profitable," and can be
achieved trivially in a market that is already a confirmed loser
(home_runs -- see Addendum 17/18).
"""
import duckdb
import numpy as np
import pandas as pd

from xgboost_pooled_B_multiyear import (
    norm_name, implied_prob, build_team_game_log, build_elo_l10,
)

DB = "db/mlb_markets.duckdb"


def pick_main_line(df, group_cols, over_odds_col="best_over_odds"):
    df = df.copy()
    df["_over_prob"] = implied_prob(df[over_odds_col])
    df["_dist_even"] = (df["_over_prob"] - 0.5).abs()
    return df.sort_values("_dist_even").groupby(group_cols, as_index=False).first().drop(columns=["_over_prob", "_dist_even"])


def profit(win, odds):
    if not win:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def main():
    con = duckdb.connect(DB, read_only=True)
    games = build_team_game_log(con)
    elo_l10 = build_elo_l10(games)
    games = games.merge(elo_l10, on="game_pk")

    box = con.execute("""
        SELECT game_pk, player_name, hits, total_bases, rbi, home_runs, strikeouts, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

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
            odds = pick_main_line(odds, ["game_pk"], over_odds_col="best_over_odds")
            m = odds.merge(games[["game_pk", "home_runs_", "away_runs_"]], on="game_pk", how="inner")
            m["total_runs"] = m["home_runs_"] + m["away_runs_"]
            under = m.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
            over = m.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
            both = pd.concat([under, over]).dropna(subset=["odds"])
            both["market"] = "totals"
            return both[["market", "game_pk", "odds", "win", "side"]]

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
    pool["market_prob"] = implied_prob(pool["odds"])

    fav = pool[pool["market_prob"] > 0.5].copy()
    fav["profit"] = fav.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    print("== always bet the side the market itself favors (market_prob>0.5) -- zero modeling, per market ==")
    summary = fav.groupby("market").agg(
        n=("win", "size"),
        accuracy=("win", "mean"),
        roi_pct=("profit", lambda s: s.mean() * 100),
    ).sort_values("accuracy", ascending=False)
    print(summary)
    print()
    print(f"overall (all markets pooled): accuracy={fav['win'].mean():.4f} n={len(fav)} roi%={fav['profit'].mean()*100:.2f}")

    con.close()


if __name__ == "__main__":
    main()
