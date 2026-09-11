"""
Build point-in-time team power rankings (Elo) and rolling form (last-10-game
record) over the real 2014-2019 MLB game dataset (historical_game_odds,
14,784 games -- see load_historical_2014_2019.py).

Point-in-time discipline: every feature attached to a game is computed ONLY
from games strictly before it (Elo pre-game rating, L10 record from the 10
games prior). No feature ever sees its own game's result or anything after
it -- that would be lookahead bias, the single easiest way to fake an edge.

Elo method: standard logistic Elo, K=20 (a conventional value for MLB -- high
game count per team per season means a lower K than NFL-style is standard),
home-field advantage baked in as +24 Elo points on the home team's rating
when computing win expectancy (widely-used constant for MLB, e.g. FiveThirtyEight's
model). Ratings reset partially toward 1500 at each season boundary
(1/3 regression to the mean) since rosters turn over -- also standard practice,
prevents one team's rating from drifting unboundedly across years.

Output: `game_features` table with elo_home_pre, elo_away_pre, elo_diff
(home + HFA - away), l10_home_wins, l10_away_wins, l10_diff, plus the
original odds/scores, ready for strategy exploration.
"""
import pathlib
from collections import deque

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

K = 20
HFA = 24
REGRESS_TO_MEAN = 1.0 / 3.0
BASE_RATING = 1500.0


def main():
    con = duckdb.connect(DB)
    games = con.execute("""
        SELECT date, game, season, home_abbrev, away_abbrev, home_score, away_score,
               home_open_ml, away_open_ml, home_close_ml, away_close_ml,
               home_run_line, away_run_line, home_run_line_odds, away_run_line_odds,
               open_ou_line, open_ou_odds, close_ou_line, close_ou_odds
        FROM historical_game_odds
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date, game
    """).fetchdf()

    elo = {}
    l10 = {}  # team -> deque of last 10 results (1=win, 0=loss)
    current_season = None
    rows = []

    for _, g in games.iterrows():
        season = g["season"]
        if current_season is not None and season != current_season:
            # season turnover: regress everyone toward the mean
            for t in elo:
                elo[t] = elo[t] + REGRESS_TO_MEAN * (BASE_RATING - elo[t])
        current_season = season

        h, a = g["home_abbrev"], g["away_abbrev"]
        elo.setdefault(h, BASE_RATING)
        elo.setdefault(a, BASE_RATING)
        l10.setdefault(h, deque(maxlen=10))
        l10.setdefault(a, deque(maxlen=10))

        elo_home_pre, elo_away_pre = elo[h], elo[a]
        l10_home_wins = sum(l10[h]) if l10[h] else None
        l10_home_games = len(l10[h])
        l10_away_wins = sum(l10[a]) if l10[a] else None
        l10_away_games = len(l10[a])

        rows.append({
            **g.to_dict(),
            "elo_home_pre": elo_home_pre,
            "elo_away_pre": elo_away_pre,
            "elo_diff": (elo_home_pre + HFA) - elo_away_pre,
            "l10_home_wins": l10_home_wins,
            "l10_home_games": l10_home_games,
            "l10_away_wins": l10_away_wins,
            "l10_away_games": l10_away_games,
        })

        # update AFTER recording pre-game features
        home_win = 1 if g["home_score"] > g["away_score"] else 0
        expected_home = 1.0 / (1.0 + 10 ** (((elo_away_pre) - (elo_home_pre + HFA)) / 400.0))
        elo[h] = elo_home_pre + K * (home_win - expected_home)
        elo[a] = elo_away_pre + K * ((1 - home_win) - (1 - expected_home))
        l10[h].append(home_win)
        l10[a].append(1 - home_win)

    import pandas as pd
    out = pd.DataFrame(rows)
    con.execute("DROP TABLE IF EXISTS game_features")
    con.register("out_df", out)
    con.execute("CREATE TABLE game_features AS SELECT * FROM out_df")
    con.unregister("out_df")
    con.execute("CHECKPOINT")
    print(f"built game_features: {len(out):,} rows")
    print(out[["date", "home_abbrev", "away_abbrev", "elo_home_pre", "elo_away_pre",
               "l10_home_wins", "l10_home_games"]].tail(10).to_string())
    con.close()


if __name__ == "__main__":
    main()
