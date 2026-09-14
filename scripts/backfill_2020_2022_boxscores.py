"""
Backfill real 2020-2022 MLB box scores from the free MLB Stats API.

Unlike the 2023 backfill, client_games has no rows before 2023-05-03, so
the list of real games to fetch comes from MLB's own public schedule
endpoint instead (sportId=1 = MLB), not from client data. Real season
date ranges used (regular season only, per MLB's own published schedule):
  2020: 2020-07-23 -> 2020-09-27 (COVID-shortened 60-game season)
  2021: 2021-04-01 -> 2021-10-03
  2022: 2022-04-07 -> 2022-10-05

Same schema/columns as backfill_2023_boxscores.py so every existing
grading script keeps working unchanged. Resumable: skips game_pks
already present in boxscore.
"""
import pathlib
import threading
import time
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
CONCURRENCY = 10

SEASON_RANGES = [
    ("2020-07-23", "2020-09-27"),
    ("2021-04-01", "2021-10-03"),
    ("2022-04-07", "2022-10-05"),
]

COLS = ["player_id", "game_pk", "game_date", "team_id", "player_name", "position_type",
        "is_starter", "batting_order_slot", "at_bats", "hits", "home_runs", "total_bases",
        "rbi", "plate_appearances", "innings_pitched", "pitches_thrown", "strikeouts",
        "walks", "batters_faced", "pitcher_runs", "pitcher_earned_runs", "runs_scored",
        "outs", "batter_strikeouts", "batter_walks", "batter_hbp", "batter_sac_flies"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_schedule(start_date, end_date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":  # regular season only, no spring/playoffs
                continue
            games.append({
                "game_pk": g["gamePk"],
                "game_date": d["date"],
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "commence_time_utc": g.get("gameDate"),
            })
    return games


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
            continue
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

    all_games = []
    for start, end in SEASON_RANGES:
        log(f"fetching real MLB schedule {start} -> {end}...")
        games = fetch_schedule(start, end)
        log(f"  {len(games)} real regular-season games")
        all_games.extend(games)

    games_df = pd.DataFrame(all_games).drop_duplicates(subset=["game_pk"])
    log(f"total real 2020-2022 games: {len(games_df):,}")

    # Save the schedule (with home/away team names + commence time) for
    # Phase 2 (odds-provider event matching) to use -- this is the only
    # place that real event-to-game_pk link can come from for this period,
    # since client_games has no rows before 2023.
    schedule_path = ROOT / "data_raw" / "mlb_schedule_2020_2022.csv"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(schedule_path, index=False)
    log(f"saved schedule to {schedule_path}")

    done = set(con.execute("""
        SELECT DISTINCT game_pk FROM boxscore
        WHERE game_date >= '2020-01-01' AND game_date < '2023-01-01'
    """).fetchdf()["game_pk"])
    todo = games_df[~games_df.game_pk.isin(done)]
    log(f"already done: {len(done)}, remaining: {len(todo)}")

    buffer = []
    total_new = 0
    done_count = 0
    lock = threading.Lock()
    todo_list = list(todo.itertuples(index=False))

    def fetch_one(g):
        return g, fetch_boxscore(g.game_pk)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(fetch_one, g) for g in todo_list]
        for fut in as_completed(futures):
            g, data = fut.result()
            with lock:
                done_count += 1
                if data is not None:
                    for side in ["home", "away"]:
                        buffer.extend(parse_team(data["teams"][side], g.game_pk, g.game_date))
                if done_count % 200 == 0 or done_count == len(todo_list):
                    if buffer:
                        df = pd.DataFrame(buffer, columns=COLS)
                        con.register("df_v", df)
                        col_list = ", ".join(f'"{c}"' for c in COLS)
                        con.execute(f"INSERT INTO boxscore ({col_list}) SELECT {col_list} FROM df_v")
                        con.unregister("df_v")
                        total_new += len(buffer)
                        buffer = []
                    log(f"  {done_count}/{len(todo_list)} games processed, {total_new:,} boxscore rows inserted so far")

    con.execute("CHECKPOINT")
    n = con.execute("SELECT count(*) FROM boxscore WHERE game_date >= '2020-01-01' AND game_date < '2023-01-01'").fetchone()[0]
    log(f"DONE. 2020-2022 boxscore rows now in db: {n:,}")
    con.close()


if __name__ == "__main__":
    main()
