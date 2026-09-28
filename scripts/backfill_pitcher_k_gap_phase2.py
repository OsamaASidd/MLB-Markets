"""
Phase 2: fetch real historical pitcher_strikeouts odds for the 2,851 games
resolved in Phase 1 (2025-05-28 to 2026-09-23 gap). Same pattern as
backfill_pitcher_outs_odds.py: snapshot at commence_time - 30min, 10
credits per event (one market per call), resumable JSONL cache.
"""
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

CONCURRENCY = 10

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAPPING_PATH = ROOT / "data_raw" / "event_id_mapping_pitcher_k_gap.csv"
CACHE_PATH = ROOT / "data_raw" / "pitcher_k_gap_odds_cache.jsonl"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("ODDS_API_KEY not set")

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
    params = {"apiKey": API_KEY, "regions": "us", "markets": "pitcher_strikeouts", "date": snapshot_iso}
    r = requests.get(url, params=params, timeout=20)
    remaining = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}, remaining
    return r.json(), remaining


def main():
    games = pd.read_csv(MAPPING_PATH)
    print(f"[backfill] {len(games):,} real games to fetch pitcher_strikeouts odds for")

    cache = load_cache()
    print(f"[backfill] {len(cache):,} already cached from a prior run")

    todo = []
    for _, row in games.iterrows():
        if row["event_id"] in cache:
            continue
        commence = datetime.fromisoformat(str(row["commence_time_utc"]).replace("Z", "+00:00"))
        if commence.tzinfo is None:
            commence = commence.replace(tzinfo=timezone.utc)
        snapshot_iso = (commence - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        todo.append((row["event_id"], row["game_pk"], snapshot_iso))
    print(f"[backfill] {len(todo):,} remaining to fetch, {CONCURRENCY} concurrent workers")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fetched_this_run, errors, last_remaining = 0, 0, None
    write_lock = threading.Lock()
    out = open(CACHE_PATH, "a", encoding="utf-8")

    def work(item):
        event_id, game_pk, snapshot_iso = item
        data, remaining = fetch_one(event_id, snapshot_iso)
        return event_id, game_pk, snapshot_iso, data, remaining

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(work, item) for item in todo]
            for fut in as_completed(futures):
                event_id, game_pk, snapshot_iso, data, remaining = fut.result()
                last_remaining = remaining
                rec = {"event_id": event_id, "game_pk": game_pk, "snapshot_iso": snapshot_iso, "response": data}
                with write_lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    fetched_this_run += 1
                    if "error" in data:
                        errors += 1
                    if fetched_this_run % 200 == 0:
                        print(f"[backfill] ...{fetched_this_run}/{len(todo)} fetched this run "
                              f"({errors} errors), credits remaining: {last_remaining}", flush=True)
    finally:
        out.close()

    print(f"[backfill] done: {fetched_this_run} fetched this run, {errors} errors, "
          f"credits remaining: {last_remaining}")


if __name__ == "__main__":
    main()
