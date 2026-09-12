"""
Backfill real 2023 box scores from the free MLB Stats API (no key, no
credit cost) so the 2023 season of the client's real odds warehouse
(already loaded, 641,974 rows, currently ungraded) can finally be graded
alongside 2024-2026 -- extending Addendum 9/10 to all 4 real seasons
instead of 3.

Appends into the existing `boxscore` table, same schema as the 2024-2026
rows already there (position_type, is_starter, at_bats, hits, home_runs,
total_bases, rbi, plate_appearances, innings_pitched, pitches_thrown,
strikeouts, walks, batters_faced, pitcher_runs, pitcher_earned_runs,
runs_scored, outs, batter_strikeouts, batter_walks, batter_hbp,
batter_sac_flies) so every existing grading script keeps working unchanged.

Resumable: skips game_pks already present in boxscore.
"""
import pathlib
import time
import urllib.request
import json

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

COLS = ["player_id", "game_pk", "game_date", "team_id", "player_name", "position_type",
        "is_starter", "batting_order_slot", "at_bats", "hits", "home_runs", "total_bases",
        "rbi", "plate_appearances", "innings_pitched", "pitches_thrown", "strikeouts",
        "walks", "batters_faced", "pitcher_runs", "pitcher_earned_runs", "runs_scored",
        "outs", "batter_strikeouts", "batter_walks", "batter_hbp", "batter_sac_flies"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_boxscore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 2:
                log(f"  FAILED game_pk={game_pk}: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))


def parse_team(team_side_data, game_pk, game_date):
    team_id = team_side_data["team"]["id"]
    rows = []
    for pid, p in team_side_data.get("players", {}).items():
        person = p["person"]
        pos = p.get("position", {})
        stats = p.get("stats", {})
        bat, pit = stats.get("batting", {}), stats.get("pitching", {})
        if not bat and not pit:
            continue  # didn't play
        bo = p.get("battingOrder")
        is_starter = bool(pit) and pit.get("gamesStarted") == 1
        if bat:
            is_starter = is_starter or (bo is not None and int(bo) % 100 == 0)
        rows.append({
            "player_id": person["id"], "game_pk": game_pk, "game_date": game_date,
            "team_id": team_id, "player_name": person["fullName"],
            "position_type": pos.get("type"), "is_starter": is_starter,
            "batting_order_slot": int(bo) // 100 if bo else None,
            "at_bats": bat.get("atBats"), "hits": bat.get("hits"),
            "home_runs": bat.get("homeRuns"), "total_bases": bat.get("totalBases"),
            "rbi": bat.get("rbi"), "plate_appearances": bat.get("plateAppearances"),
            "innings_pitched": pit.get("inningsPitched"),
            "pitches_thrown": pit.get("numberOfPitches") or pit.get("pitchesThrown"),
            "strikeouts": pit.get("strikeOuts"), "walks": pit.get("baseOnBalls"),
            "batters_faced": pit.get("battersFaced"), "pitcher_runs": pit.get("runs"),
            "pitcher_earned_runs": pit.get("earnedRuns"), "runs_scored": bat.get("runs"),
            "outs": pit.get("outs"), "batter_strikeouts": bat.get("strikeOuts"),
            "batter_walks": bat.get("baseOnBalls"), "batter_hbp": bat.get("hitByPitch"),
            "batter_sac_flies": bat.get("sacFlies"),
        })
    return rows


def main():
    con = duckdb.connect(DB)
    games = con.execute("""
        SELECT DISTINCT TRY_CAST(game_pk AS BIGINT) AS game_pk, game_date
        FROM client_games WHERE season = '2023' AND game_pk IS NOT NULL
    """).fetchdf()
    done = set(con.execute("SELECT DISTINCT game_pk FROM boxscore WHERE game_date < '2024-01-01'").fetchdf()["game_pk"])
    todo = games[~games.game_pk.isin(done)]
    log(f"2023 games total: {len(games)}, already done: {len(done)}, remaining: {len(todo)}")

    buffer = []
    total_new = 0
    for i, (_, g) in enumerate(todo.iterrows()):
        data = fetch_boxscore(g.game_pk)
        if data is None:
            continue
        for side in ["home", "away"]:
            buffer.extend(parse_team(data["teams"][side], g.game_pk, g.game_date))
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            if buffer:
                df = pd.DataFrame(buffer, columns=COLS)
                con.register("df_v", df)
                col_list = ", ".join(f'"{c}"' for c in COLS)
                con.execute(f"INSERT INTO boxscore ({col_list}) SELECT {col_list} FROM df_v")
                con.unregister("df_v")
                total_new += len(buffer)
                buffer = []
            log(f"  {i+1}/{len(todo)} games processed, {total_new:,} boxscore rows inserted so far")

    con.execute("CHECKPOINT")
    n = con.execute("SELECT count(*) FROM boxscore WHERE game_date < '2024-01-01'").fetchone()[0]
    log(f"DONE. 2023 boxscore rows now in db: {n:,}")
    con.close()


if __name__ == "__main__":
    main()
