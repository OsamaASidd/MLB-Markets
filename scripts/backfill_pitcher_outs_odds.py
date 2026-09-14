"""
Real historical pitcher_outs odds backfill via The Odds API's historical
endpoint -- the one legitimate fix for pitcher_outs' actual blocker (sample
size: as few as 8-200 graded test picks in every prior cut in this audit).

Confirmed directly before running at scale: this market IS available from
this provider historically (client_games.event_id matches Odds API event
IDs exactly -- verified against a real 2024-06-01 game). The prior "no
warehouse odds" situation for pitcher_outs was the CLIENT's own backfill
pipeline never having pulled it, not a provider gap.

Cost: 10 credits per event (one market requested per call). ~7,691 real
games with box scores * 10 = ~77K credits, against a 4.7M-credit remaining
balance on a live, shared production API key -- checked and kept modest
deliberately, since this is real money regardless of the balance size.

Snapshot picked at commence_time - 30min (a real, bettable pre-game price,
never after commence -- no leakage). Saves raw responses to a local cache
file as it goes so a partial run can resume without re-spending credits on
already-fetched games.
"""
import os
import json
import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone

import duckdb
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
CACHE_PATH = ROOT / "data_raw" / "pitcher_outs_odds_cache.jsonl"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("ODDS_API_KEY not set -- put it in .env (git-ignored), never hardcode it here.")

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"


def load_cache():
    """event_id -> raw API response, from whatever's already been fetched."""
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["event_id"]] = rec
    return cache


def fetch_one(event_id, snapshot_iso):
    url = f"{BASE}/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "pitcher_outs",
        "date": snapshot_iso,
    }
    r = requests.get(url, params=params, timeout=20)
    remaining = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}, remaining
    return r.json(), remaining


def main():
    con = duckdb.connect(DB, read_only=True)
    games = con.execute("""
        SELECT g.event_id, g.game_pk, g.game_date, g.commence_time_utc
        FROM client_games g
        WHERE CAST(g.game_pk AS BIGINT) IN (
                SELECT DISTINCT game_pk FROM boxscore
                WHERE position_type='Pitcher' AND is_starter=True
              )
          AND g.event_id IS NOT NULL
          AND g.commence_time_utc IS NOT NULL
        ORDER BY g.game_date
    """).fetchdf()
    con.close()
    print(f"[backfill] {len(games):,} real games to fetch pitcher_outs odds for")

    cache = load_cache()
    print(f"[backfill] {len(cache):,} already cached from a prior run")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fetched_this_run = 0
    errors = 0
    last_remaining = None

    with open(CACHE_PATH, "a", encoding="utf-8") as out:
        for i, row in games.iterrows():
            event_id = row["event_id"]
            if event_id in cache:
                continue
            commence = datetime.fromisoformat(row["commence_time_utc"].replace(" ", "T"))
            if commence.tzinfo is None:
                commence = commence.replace(tzinfo=timezone.utc)
            snapshot = commence - timedelta(minutes=30)
            snapshot_iso = snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")

            data, remaining = fetch_one(event_id, snapshot_iso)
            last_remaining = remaining
            rec = {
                "event_id": event_id,
                "game_pk": row["game_pk"],
                "game_date": row["game_date"],
                "snapshot_iso": snapshot_iso,
                "response": data,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            fetched_this_run += 1
            if "error" in data:
                errors += 1

            if fetched_this_run % 200 == 0:
                print(f"[backfill] ...{fetched_this_run} fetched this run "
                      f"({errors} errors), credits remaining: {last_remaining}")
            time.sleep(0.15)  # gentle pace against a shared live key

    print(f"[backfill] done: {fetched_this_run} fetched this run, {errors} errors, "
          f"credits remaining: {last_remaining}")


if __name__ == "__main__":
    main()
