"""
New hypothesis (sourced from sports-betting research, not yet tested in
this audit): a team's bullpen, having thrown heavy relief innings in the
last 1-2 days, is more fatigued and should allow more runs -- relevant to
`totals` (already a confirmed FAIL market) and `runs_scored`.

Real data used: boxscore (already in this repo's db, now spanning all 4
real seasons 2023-2026) -- relief innings pitched = innings_pitched for
Pitcher rows where is_starter=False, summed per team per day. Point-in-time
correct: only games strictly before the game in question count toward its
fatigue figure.

Same discipline as everywhere else: split by season, gate = n>=500 and
95% CI lower bound > 0.
"""
import pathlib
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


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

    relief = con.execute("""
        SELECT game_pk, game_date, team_id, innings_pitched
        FROM boxscore WHERE position_type='Pitcher' AND is_starter=False
          AND innings_pitched IS NOT NULL
    """).fetchdf()
    relief["outs"] = relief["innings_pitched"].map(ip_to_outs)
    relief["game_date"] = pd.to_datetime(relief["game_date"])
    daily = relief.groupby(["team_id", "game_date"])["outs"].sum().reset_index()

    # rolling relief-outs thrown in the prior 2 calendar days per team, point-in-time
    daily = daily.sort_values(["team_id", "game_date"])
    fatigue_lookup = {}
    for team_id, grp in daily.groupby("team_id"):
        dates = grp["game_date"].tolist()
        outs = grp["outs"].tolist()
        for i, d in enumerate(dates):
            window_outs = sum(o for dd, o in zip(dates, outs) if 0 < (d - dd).days <= 2)
            fatigue_lookup[(team_id, d)] = window_outs

    team_runs = con.execute("""
        SELECT game_pk, team_id, sum(runs_scored) AS team_runs
        FROM boxscore WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
        GROUP BY 1,2 HAVING sum(runs_scored) IS NOT NULL
    """).fetchdf()
    game_scores = team_runs.groupby("game_pk")["team_runs"].sum().reset_index(name="total_runs")

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

    games = con.execute("SELECT DISTINCT game_pk, game_date FROM boxscore").fetchdf()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.merge(home_ids, on="game_pk").merge(away_ids, on="game_pk").merge(game_scores, on="game_pk")

    games["home_fatigue"] = games.apply(lambda r: fatigue_lookup.get((r.home_team_id, r.game_date), 0), axis=1)
    games["away_fatigue"] = games.apply(lambda r: fatigue_lookup.get((r.away_team_id, r.game_date), 0), axis=1)
    games["combined_fatigue"] = games["home_fatigue"] + games["away_fatigue"]

    print(f"real games with fatigue features: {len(games):,}")
    print("\ncorr(combined bullpen fatigue [outs, last 2 days], total_runs):",
          round(np.corrcoef(games["combined_fatigue"], games["total_runs"])[0, 1], 3))

    odds = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.season
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'totals'
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
    odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
    m = odds.merge(games[["game_pk", "total_runs", "combined_fatigue"]], on="game_pk", how="inner")
    m = m.dropna(subset=["total_runs", "line", "best_over_odds"])

    print(f"\nmatched to real totals odds: {len(m):,}")

    high_fatigue = m[m.combined_fatigue >= m.combined_fatigue.quantile(0.75)].copy()
    high_fatigue["win"] = high_fatigue["total_runs"] > high_fatigue["line"]
    high_fatigue["profit"] = high_fatigue.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)

    print("\n-- bet OVER when combined bullpen fatigue is in the top quartile (both teams' last-2-day relief workload) --")
    for season in [2023, 2024, 2025, 2026, None]:
        sub = high_fatigue if season is None else high_fatigue[high_fatigue.season == season]
        s = stat(sub)
        passed = s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0
        label = "ALL 2023-2026" if season is None else f"season {season}"
        print(f"  {label:<16} n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")

    con.close()


if __name__ == "__main__":
    main()
