"""
Phase 2: fetch real historical `totals` odds for the 937 games resolved in
Phase 1 (data_raw/event_id_mapping_h2h_totals_gap.csv -- the 2026-05-25 to
2026-09-23 gap, doubleheader-collision rows already dropped for data
integrity).

Same pattern as backfill_pitcher_outs_odds.py: snapshot at
commence_time - 30min (a real, bettable pre-game price, never after
commence -- no leakage), 10 credits per event (one market requested per
call), incremental JSONL cache so a partial run resumes without
re-spending credits on already-fetched games.

h2h explicitly excluded from this pass -- Addendum 49's walk-forward
result (n=1,114, ROI=-0.37%, decisively negative at a sample size past the
n>=500 threshold) already answered that market conclusively; spending more
credits on it isn't expected to change the answer.
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
MAPPING_PATH = ROOT / "data_raw" / "event_id_mapping_h2h_totals_gap.csv"
CACHE_PATH = ROOT / "data_raw" / "totals_gap_odds_cache.jsonl"

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
    params = {"apiKey": API_KEY, "regions": "us", "markets": "totals", "date": snapshot_iso}
    r = requests.get(url, params=params, timeout=20)
    remaining = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}, remaining
    return r.json(), remaining


def main():
    games = pd.read_csv(MAPPING_PATH)
    print(f"[backfill] {len(games):,} real games to fetch totals odds for")

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
    fetched_this_run = 0
    errors = 0
    last_remaining = None
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
                    if fetched_this_run % 100 == 0:
                        print(f"[backfill] ...{fetched_this_run}/{len(todo)} fetched this run "
                              f"({errors} errors), credits remaining: {last_remaining}", flush=True)
    finally:
        out.close()

    print(f"[backfill] done: {fetched_this_run} fetched this run, {errors} errors, "
          f"credits remaining: {last_remaining}")


if __name__ == "__main__":
    main()
