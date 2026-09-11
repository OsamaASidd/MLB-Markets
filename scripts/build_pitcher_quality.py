"""
Point-in-time starting-pitcher quality signal, real 2024-2026 data.

Baseball-specific hypothesis (flagged in Addendum 5): a single starter drives
far more single-game variance in MLB than overall team strength, so team-level
Elo underselling the sport doesn't mean pitcher-level signal is equally weak.

Data (all real, already in this repo's db, pulled from the production
warehouse in Milestone 1):
  - boxscore: real per-player-per-game stats, 2024-04-01 to 2026-07-27
  - opposing_pitcher: real starter identity (home/away) per game_pk

Method:
  1. Derive real final team scores per game_pk from boxscore position-player
     rows (Catcher/Hitter/Infielder/Outfielder runs_scored) -- cross-checked
     against the opposing team's pitcher-runs-allowed sum, which must match
     exactly (it's the same runs, counted two ways). This is NOT modeled or
     estimated; it's the actual boxscore, reconstructed correctly.
  2. Identify each game's home/away team_id by matching the starter's
     player_id (from opposing_pitcher) to their team_id in boxscore.
  3. For each starter, walk their starts in date order and compute a rolling
     runs-allowed-per-9-innings (RA/9) over their **prior** starts only
     (window sizes tested: last 3, last 5, last 8) -- point-in-time, the
     start being predicted is never included in its own rolling average.
  4. Output `pitcher_game_features`: one row per game with each side's
     starter, their pre-game rolling RA/9 at each window size, and the
     real final score / winner.
"""
import pathlib
from collections import defaultdict, deque

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

WINDOWS = [3, 5, 8]


def main():
    con = duckdb.connect(DB)

    team_scores = con.execute("""
        SELECT game_pk, team_id, sum(runs_scored) AS team_runs
        FROM boxscore
        WHERE position_type IN ('Catcher', 'Hitter', 'Infielder', 'Outfielder')
        GROUP BY 1, 2
        HAVING sum(runs_scored) IS NOT NULL
    """).fetchdf()

    starts = con.execute("""
        SELECT player_id, player_name, game_pk, game_date, team_id,
               TRY_CAST(innings_pitched AS DOUBLE) AS ip, TRY_CAST(pitcher_earned_runs AS DOUBLE) AS er
        FROM boxscore
        WHERE position_type = 'Pitcher' AND is_starter = True
          AND innings_pitched IS NOT NULL AND pitcher_earned_runs IS NOT NULL
        ORDER BY game_date, game_pk
    """).fetchdf()
    # innings_pitched is stored as e.g. 5.3 meaning 5 and 1/3 innings (baseball notation, not decimal)
    whole = starts["ip"].astype(int)
    frac = (starts["ip"] - whole).round(1)
    starts["ip_real"] = whole + frac.map({0.0: 0.0, 0.1: 1 / 3, 0.2: 2 / 3}).fillna(0.0)

    matchups = con.execute("""
        SELECT o.game_pk, o.home_starter_id, o.away_starter_id
        FROM opposing_pitcher o
        WHERE o.home_starter_id IS NOT NULL AND o.away_starter_id IS NOT NULL
    """).fetchdf()

    # rolling RA/9 per pitcher, point-in-time
    history = defaultdict(lambda: {w: deque(maxlen=w) for w in WINDOWS})
    pre_game_ra9 = {}  # (player_id, game_pk) -> {window: ra9_or_None}

    for _, s in starts.iterrows():
        pid, gpk = s["player_id"], s["game_pk"]
        pre = {}
        for w in WINDOWS:
            h = history[pid][w]
            if len(h) < w:
                pre[w] = None
            else:
                total_ip = sum(x[0] for x in h)
                total_er = sum(x[1] for x in h)
                pre[w] = round(total_er / total_ip * 9, 3) if total_ip > 0 else None
        pre_game_ra9[(pid, gpk)] = pre
        for w in WINDOWS:
            history[pid][w].append((s["ip_real"], s["er"]))

    rows = []
    ts_lookup = {(r.game_pk, r.team_id): r.team_runs for r in team_scores.itertuples()}
    game_dates = dict(zip(starts["game_pk"], starts["game_date"]))
    for _, m in matchups.iterrows():
        gpk, hp, ap = m["game_pk"], m["home_starter_id"], m["away_starter_id"]
        h_start = starts[(starts.player_id == hp) & (starts.game_pk == gpk)]
        a_start = starts[(starts.player_id == ap) & (starts.game_pk == gpk)]
        if h_start.empty or a_start.empty:
            continue
        h_team = h_start.iloc[0]["team_id"]
        a_team = a_start.iloc[0]["team_id"]
        home_runs = ts_lookup.get((gpk, h_team))
        away_runs = ts_lookup.get((gpk, a_team))
        if home_runs is None or away_runs is None:
            continue
        row = {
            "game_pk": gpk, "game_date": h_start.iloc[0]["game_date"],
            "home_starter_id": hp, "away_starter_id": ap,
            "home_score": home_runs, "away_score": away_runs,
            "home_win": int(home_runs > away_runs),
        }
        for w in WINDOWS:
            row[f"home_ra9_l{w}"] = pre_game_ra9.get((hp, gpk), {}).get(w)
            row[f"away_ra9_l{w}"] = pre_game_ra9.get((ap, gpk), {}).get(w)
        rows.append(row)

    out = pd.DataFrame(rows)
    con.execute("DROP TABLE IF EXISTS pitcher_game_features")
    con.register("out_df", out)
    con.execute("CREATE TABLE pitcher_game_features AS SELECT * FROM out_df")
    con.unregister("out_df")
    con.execute("CHECKPOINT")
    print(f"built pitcher_game_features: {len(out):,} rows")
    for w in WINDOWS:
        n_have_both = out[[f"home_ra9_l{w}", f"away_ra9_l{w}"]].notna().all(axis=1).sum()
        print(f"  window={w}: {n_have_both:,} games with both starters having a full rolling window")
    con.close()


if __name__ == "__main__":
    main()
