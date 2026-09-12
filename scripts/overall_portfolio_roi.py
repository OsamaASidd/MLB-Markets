"""
Single blended ROI across the client's ENTIRE default policy, on the whole
real 2023-2026 data available (all 4 seasons, every market the policy
covers, unconditional -- exactly what Addendum 9/14 tested per-market,
pooled into one number).

This is NOT the same question as "does any one market pass the gate" --
it answers "if someone flat-staked every pick this policy would generate
across all of it, what's the actual blended return." Reuses the exact same
grading logic as grade_client_odds_warehouse.py (same joins, same profit
formula) so the per-row profit values are identical to what's already been
reported; this script just pools them into one portfolio-level statistic
with its own honest CI instead of only reporting each market separately.
"""
import pathlib
import unicodedata
import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def norm_name(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def profit(win, odds):
    if pd.isna(win) or pd.isna(odds):
        return np.nan
    if not win:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def stat(df):
    d = df.dropna(subset=["profit"])
    n = len(d)
    if n == 0:
        return {"n": 0, "wr": None, "roi": None, "lo": None, "hi": None}
    m = d["profit"].mean() * 100
    se = d["profit"].std(ddof=1) / np.sqrt(n) * 100
    return {"n": n, "wr": round(d["win"].mean() * 100, 1), "roi": round(m, 2),
            "lo": round(m - 1.96 * se, 2), "hi": round(m + 1.96 * se, 2)}


def main():
    con = duckdb.connect(DB, read_only=True)
    box = con.execute("""
        SELECT game_pk, player_name, hits, total_bases FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    def load_batter(market_key, stat_col):
        odds = con.execute(f"""
            SELECT co.game_pk, co.line, co.best_under_odds, co.season, co.player_name
            FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                  JOIN client_games g ON co.event_id = g.event_id) co
            WHERE co.market_key = '{market_key}'
        """).fetchdf()
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
        odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
        odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
        odds["name_norm"] = odds["player_name"].map(norm_name)
        m = odds.merge(box[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
        m = m.dropna(subset=[stat_col, "line", "best_under_odds"])
        m["win"] = m[stat_col] < m["line"]
        m["profit"] = m.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)
        return m[["profit", "win"]]

    hits = load_batter("batter_hits", "hits")
    tb = load_batter("batter_total_bases", "total_bases")

    team_runs = con.execute("""
        SELECT game_pk, team_id, sum(runs_scored) AS team_runs
        FROM boxscore WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
        GROUP BY 1,2 HAVING sum(runs_scored) IS NOT NULL
    """).fetchdf()
    game_scores = team_runs.groupby("game_pk")["team_runs"].sum().reset_index(name="total_runs")
    odds_tot = con.execute("""
        SELECT co.game_pk, co.line, co.best_under_odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'totals'
    """).fetchdf()
    odds_tot["game_pk"] = pd.to_numeric(odds_tot["game_pk"], errors="coerce")
    odds_tot["line"] = pd.to_numeric(odds_tot["line"], errors="coerce")
    odds_tot["best_under_odds"] = pd.to_numeric(odds_tot["best_under_odds"], errors="coerce")
    totals = odds_tot.merge(game_scores, on="game_pk", how="inner")
    totals["win"] = totals["total_runs"] < totals["line"]
    totals["profit"] = totals.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)
    totals = totals[["profit", "win"]]

    side_map = con.execute("""
        SELECT g.game_pk, l.team_side, b.team_id, count(*) n
        FROM lineups l JOIN client_games g ON l.event_id = g.event_id
        JOIN boxscore b ON b.player_id = l.player_id AND b.game_pk = g.game_pk
        GROUP BY 1,2,3
    """).fetchdf()
    side_map["game_pk"] = pd.to_numeric(side_map["game_pk"], errors="coerce")
    side_map = side_map.sort_values("n", ascending=False).drop_duplicates(subset=["game_pk", "team_side"])
    home_ids = side_map[side_map.team_side == "home"][["game_pk", "team_id"]].rename(columns={"team_id": "home_team_id"})
    away_ids = side_map[side_map.team_side == "away"][["game_pk", "team_id"]].rename(columns={"team_id": "away_team_id"})
    team_scores = team_runs.merge(home_ids, on="game_pk").merge(away_ids, on="game_pk")
    home_res = team_scores[team_scores.team_id == team_scores.home_team_id][["game_pk", "team_runs"]].rename(columns={"team_runs": "home_runs_"})
    away_res = team_scores[team_scores.team_id == team_scores.away_team_id][["game_pk", "team_runs"]].rename(columns={"team_runs": "away_runs_"})
    game_res = home_res.merge(away_res, on="game_pk")

    odds_h2h = con.execute("""
        SELECT co.game_pk, co.best_over_odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'h2h__away'
    """).fetchdf()
    odds_h2h["game_pk"] = pd.to_numeric(odds_h2h["game_pk"], errors="coerce")
    odds_h2h["best_over_odds"] = pd.to_numeric(odds_h2h["best_over_odds"], errors="coerce")
    h2h = odds_h2h.merge(game_res, on="game_pk", how="inner")
    h2h["win"] = h2h["away_runs_"] > h2h["home_runs_"]
    h2h["profit"] = h2h.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)
    h2h = h2h[["profit", "win"]]

    odds_spr = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'spreads__away'
    """).fetchdf()
    odds_spr["game_pk"] = pd.to_numeric(odds_spr["game_pk"], errors="coerce")
    odds_spr["line"] = pd.to_numeric(odds_spr["line"], errors="coerce")
    odds_spr["best_over_odds"] = pd.to_numeric(odds_spr["best_over_odds"], errors="coerce")
    spreads = odds_spr.merge(game_res, on="game_pk", how="inner")
    spreads["win"] = (spreads["away_runs_"] + spreads["line"]) > spreads["home_runs_"]
    spreads["profit"] = spreads.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)
    spreads = spreads[["profit", "win"]]

    parts = {"hits->under": hits, "total_bases->under": tb, "totals->under": totals,
              "h2h->away": h2h, "spreads->away": spreads}

    print("=" * 90)
    print("PER-MARKET (whole real data, all 4 seasons, client's default policy)")
    print("=" * 90)
    all_rows = []
    for label, df in parts.items():
        s = stat(df)
        print(f"  {label:<22} n={s['n']:<8} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]")
        all_rows.append(df)

    pooled = pd.concat(all_rows, ignore_index=True)
    s = stat(pooled)
    print("\n" + "=" * 90)
    print("BLENDED PORTFOLIO -- every market/side in the client's default policy, pooled")
    print("=" * 90)
    print(f"  n={s['n']:,}  win rate={s['wr']}%  ROI={s['roi']}%  95% CI=[{s['lo']}%, {s['hi']}%]")
    passed = s['n'] >= MIN_GRADED and s['lo'] is not None and s['lo'] > 0
    print(f"  gate (n>=500, CI lower bound>0): {'PASS' if passed else 'FAIL'}")

    con.close()


if __name__ == "__main__":
    main()
