"""
Forward-only real-odds collector, API-Sports.io MLB (`API_SPORTS_KEY` in
.env, Pro plan). This is a genuine gap-filler, not a backfill: API-Sports'
`/odds` endpoint only retains roughly the last 2-3 days of games (confirmed
empirically -- 2021/2022 real games and every 2026 date before 2026-09-14
returned zero odds rows; the bet-type catalog does include the exact props
this project needs -- Pitcher Outs (75), Pitcher Strikeouts (78), Player
Runs (73), RBIs (77), etc. -- but there is no historical archive to pull
from it). The only way to get real history out of this source is to start
capturing it ourselves, going forward, the same way any real odds warehouse
gets built.

Storage is intentionally a SEPARATE local db file (db/api_sports_odds.duckdb,
git-ignored), not the committed db/mlb_markets.duckdb -- this table grows
every time the collector runs and shouldn't bloat a committed binary. It can
be merged/joined into the main warehouse later once there's enough real
volume to be worth analyzing.

Run daily (or more often, to catch line movement -- every pull is a new
timestamped snapshot, nothing is overwritten):
    python scripts/collect_api_sports_odds.py                 # today, US/Eastern
    python scripts/collect_api_sports_odds.py --date 2026-09-20
"""
import argparse
import datetime
import os
import pathlib
import sys
import time

import duckdb
import requests
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("API_SPORTS_KEY")
if not API_KEY:
    sys.exit("API_SPORTS_KEY not set -- put it in .env (git-ignored), never hardcode it here.")

BASE = "https://v1.baseball.api-sports.io"
DB = str(ROOT / "db" / "api_sports_odds.duckdb")
LEAGUE_ID = 1  # MLB
HEADERS = {"x-apisports-key": API_KEY}


def call(path, params=None, retries=3):
    for attempt in range(retries):
        r = requests.get(BASE + path, headers=HEADERS, params=params or {}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            print(f"  API error on {path} {params}: {data['errors']}")
        return data
    return {"response": []}


def ensure_schema(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS api_sports_games (
            game_id BIGINT,
            game_date TIMESTAMP,
            season INTEGER,
            status_short VARCHAR,
            home_team_id BIGINT,
            home_team_name VARCHAR,
            away_team_id BIGINT,
            away_team_name VARCHAR,
            home_score INTEGER,
            away_score INTEGER,
            pulled_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS api_sports_odds (
            game_id BIGINT,
            bookmaker_id INTEGER,
            bookmaker_name VARCHAR,
            bet_id INTEGER,
            bet_name VARCHAR,
            value VARCHAR,
            odd DOUBLE,
            pulled_at TIMESTAMP
        )
    """)


def collect_for_date(con, date_str, season):
    now = datetime.datetime.now(datetime.timezone.utc)
    games = call("/games", {"league": LEAGUE_ID, "season": season, "date": date_str}).get("response", [])
    print(f"{date_str}: {len(games)} games")

    game_rows = []
    for g in games:
        game_rows.append((
            g["id"], g["game"]["date"] if "game" in g else g.get("date"), season,
            g["status"]["short"], g["teams"]["home"]["id"], g["teams"]["home"]["name"],
            g["teams"]["away"]["id"], g["teams"]["away"]["name"],
            (g.get("scores", {}).get("home") or {}).get("total"),
            (g.get("scores", {}).get("away") or {}).get("total"),
            now,
        ))
    if game_rows:
        con.executemany(
            "INSERT INTO api_sports_games VALUES (?,?,?,?,?,?,?,?,?,?,?)", game_rows
        )

    odds_rows = []
    for g in games:
        gid = g["id"]
        odds_resp = call("/odds", {"game": gid}).get("response", [])
        for row in odds_resp:
            for bk in row.get("bookmakers", []):
                for bet in bk.get("bets", []):
                    for v in bet.get("values", []):
                        try:
                            odd_val = float(v["odd"])
                        except (TypeError, ValueError):
                            continue
                        odds_rows.append((
                            gid, bk["id"], bk["name"], bet["id"], bet["name"],
                            v["value"], odd_val, now,
                        ))
        time.sleep(0.15)  # stay well under the daily/rate limits

    if odds_rows:
        con.executemany(
            "INSERT INTO api_sports_odds VALUES (?,?,?,?,?,?,?,?)", odds_rows
        )
    print(f"  -> {len(game_rows)} games, {len(odds_rows)} odds rows captured this pull")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD, default = today (UTC)")
    ap.add_argument("--season", type=int, default=None, help="default = year of --date")
    args = ap.parse_args()

    date_str = args.date or datetime.date.today().isoformat()
    season = args.season or int(date_str[:4])

    pathlib.Path(DB).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB)
    ensure_schema(con)
    collect_for_date(con, date_str, season)
    total_games = con.execute("SELECT count(DISTINCT game_id) FROM api_sports_games").fetchone()[0]
    total_odds = con.execute("SELECT count(*) FROM api_sports_odds").fetchone()[0]
    print(f"\nwarehouse totals so far: {total_games} distinct games, {total_odds} odds rows, in {DB}")
    con.close()


if __name__ == "__main__":
    main()
