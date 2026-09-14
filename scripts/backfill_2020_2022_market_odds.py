"""
Phase 3: real historical odds backfill for 2020-2022, for the same three
markets already backfilled for 2023-2026 (pitcher_outs, batter_runs_scored,
batter_strikeouts). Source of real games: data_raw/event_id_mapping_2020_2022.csv
(Phase 2's output), joined to boxscore (Phase 1's output) to keep only games
we can actually grade.

One script, market passed as an arg, since the only differences from the
existing 2023-2026 per-market scripts are the market string, the boxscore
filter, and the cache filename -- everything else (snapshot timing, request
shape, resume-by-cache, concurrency) is identical by design so Phase 4 can
treat both eras' caches the same way.

Usage:
    python3 scripts/backfill_2020_2022_market_odds.py pitcher_outs
    python3 scripts/backfill_2020_2022_market_odds.py batter_runs_scored
    python3 scripts/backfill_2020_2022_market_odds.py batter_strikeouts

Cost: 10 credits per event per market, same as the 2023-2026 backfills.
Snapshot picked at commence_time - 30min (real, bettable pre-game price,
no leakage). Saves raw responses to a market-specific cache file as it goes
so a partial run can resume without re-spending credits.
"""
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import requests

CONCURRENCY = 10

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MAPPING_PATH = ROOT / "data_raw" / "event_id_mapping_2020_2022.csv"

MARKET_CONFIG = {
    "pitcher_outs": {
        "api_market": "pitcher_outs",
        "boxscore_filter": "position_type='Pitcher' AND is_starter=True",
        "cache_name": "pitcher_outs_odds_cache_2020_2022.jsonl",
    },
    "batter_runs_scored": {
        "api_market": "batter_runs_scored",
        "boxscore_filter": "position_type IN ('Catcher','Hitter','Infielder','Outfielder')",
        "cache_name": "runs_scored_odds_cache_2020_2022.jsonl",
    },
    "batter_strikeouts": {
        "api_market": "batter_strikeouts",
        "boxscore_filter": "position_type IN ('Catcher','Hitter','Infielder','Outfielder')",
        "cache_name": "batter_strikeouts_odds_cache_2020_2022.jsonl",
    },
}

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("ODDS_API_KEY not set -- put it in .env (git-ignored), never hardcode it here.")

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"


def load_cache(cache_path):
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["event_id"]] = rec
    return cache


def fetch_one(event_id, snapshot_iso, api_market):
    url = f"{BASE}/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": api_market,
        "date": snapshot_iso,
    }
    r = requests.get(url, params=params, timeout=20)
    remaining = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}, remaining
    return r.json(), remaining


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MARKET_CONFIG:
        sys.exit(f"usage: python3 {sys.argv[0]} <{'|'.join(MARKET_CONFIG)}>")
    market = sys.argv[1]
    cfg = MARKET_CONFIG[market]
    cache_path = ROOT / "data_raw" / cfg["cache_name"]

    if not MAPPING_PATH.exists():
        sys.exit(f"missing {MAPPING_PATH} -- run build_event_id_mapping_2020_2022.py first")
    mapping = pd.read_csv(MAPPING_PATH)
    print(f"[backfill-2020-2022:{market}] {len(mapping):,} mapped real games available")

    con = duckdb.connect(DB, read_only=True)
    graded_pks = set(con.execute(f"""
        SELECT DISTINCT game_pk FROM boxscore
        WHERE {cfg['boxscore_filter']}
          AND game_date >= '2020-01-01' AND game_date < '2023-01-01'
    """).fetchdf()["game_pk"])
    con.close()

    mapping = mapping[mapping["game_pk"].isin(graded_pks)]
    print(f"[backfill-2020-2022:{market}] {len(mapping):,} of those have real box-score coverage for this market")

    cache = load_cache(cache_path)
    print(f"[backfill-2020-2022:{market}] {len(cache):,} already cached from a prior run")

    todo = []
    for _, row in mapping.iterrows():
        if row["event_id"] in cache:
            continue
        commence = datetime.fromisoformat(str(row["commence_time_utc"]).replace("Z", "+00:00"))
        if commence.tzinfo is None:
            commence = commence.replace(tzinfo=timezone.utc)
        snapshot_iso = (commence - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        todo.append((row["event_id"], row["game_pk"], row["game_date"], snapshot_iso))
    print(f"[backfill-2020-2022:{market}] {len(todo):,} remaining to fetch, {CONCURRENCY} concurrent workers")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_this_run = 0
    errors = 0
    last_remaining = None
    write_lock = threading.Lock()
    out = open(cache_path, "a", encoding="utf-8")

    def work(item):
        event_id, game_pk, game_date, snapshot_iso = item
        data, remaining = fetch_one(event_id, snapshot_iso, cfg["api_market"])
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
                        print(f"[backfill-2020-2022:{market}] ...{fetched_this_run}/{len(todo)} fetched this run "
                              f"({errors} errors), credits remaining: {last_remaining}")
    finally:
        out.close()

    print(f"[backfill-2020-2022:{market}] done: {fetched_this_run} fetched this run, {errors} errors, "
          f"credits remaining: {last_remaining}")


if __name__ == "__main__":
    main()
