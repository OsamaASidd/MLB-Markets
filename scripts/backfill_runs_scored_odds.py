"""
Real historical batter_runs_scored odds backfill via The Odds API's
historical endpoint. Same justification as pitcher_outs: confirmed directly
that this provider DOES offer batter_runs_scored historically (seen
alongside pitcher_outs in the same test event-odds response) -- the "zero
warehouse rows" finding earlier in this audit was the client's own backfill
pipeline never having pulled it, not a provider gap. Every prior runs_scored
test in this project used pick_history (the production system's own picks,
n=1,715-19,727 depending on source) rather than an independent real-odds
backtest -- this is the first time that's actually possible.

Cost: 10 credits/event, same ~7,691 real games as pitcher_outs -- ~77K
credits, against a 4.7M balance on a live shared key.
"""
import os
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import duckdb
import requests

CONCURRENCY = 10  # same rationale as backfill_pitcher_outs_odds.py -- the
                   # sequential run was network-latency-bound, not rate-
                   # limit-bound; same total credits spent either way.

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
CACHE_PATH = ROOT / "data_raw" / "runs_scored_odds_cache.jsonl"

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
        "markets": "batter_runs_scored",
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
                WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
              )
          AND g.event_id IS NOT NULL
          AND g.commence_time_utc IS NOT NULL
        ORDER BY g.game_date
    """).fetchdf()
    con.close()
    print(f"[backfill] {len(games):,} real games to fetch batter_runs_scored odds for")

    cache = load_cache()
    print(f"[backfill] {len(cache):,} already cached from a prior run")

    todo = []
    for _, row in games.iterrows():
        if row["event_id"] in cache:
            continue
        commence = datetime.fromisoformat(row["commence_time_utc"].replace(" ", "T"))
        if commence.tzinfo is None:
            commence = commence.replace(tzinfo=timezone.utc)
        snapshot_iso = (commence - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        todo.append((row["event_id"], row["game_pk"], row["game_date"], snapshot_iso))
    print(f"[backfill] {len(todo):,} remaining to fetch, {CONCURRENCY} concurrent workers")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fetched_this_run = 0
    errors = 0
    last_remaining = None
    write_lock = threading.Lock()
    out = open(CACHE_PATH, "a", encoding="utf-8")

    def work(item):
        event_id, game_pk, game_date, snapshot_iso = item
        data, remaining = fetch_one(event_id, snapshot_iso)
        return event_id, game_pk, game_date, snapshot_iso, data, remaining

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(work, item) for item in todo]
            for fut in as_completed(futures):
                event_id, game_pk, game_date, snapshot_iso, data, remaining = fut.result()
                last_remaining = remaining
                rec = {
                    "event_id": event_id,
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "snapshot_iso": snapshot_iso,
                    "response": data,
                }
                with write_lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    fetched_this_run += 1
                    if "error" in data:
                        errors += 1
                    if fetched_this_run % 200 == 0:
                        print(f"[backfill] ...{fetched_this_run}/{len(todo)} fetched this run "
                              f"({errors} errors), credits remaining: {last_remaining}")
    finally:
        out.close()

    print(f"[backfill] done: {fetched_this_run} fetched this run, {errors} errors, "
          f"credits remaining: {last_remaining}")


if __name__ == "__main__":
    main()
