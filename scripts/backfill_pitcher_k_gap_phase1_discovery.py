"""
Phase 1: resolve real Odds API event_id for the pitcher_strikeouts real-odds
gap (2025-05-28 to 2026-09-23, 4,359 real games -- 1,946 already have a
real event_id via client_games, 2,413 need fresh discovery here).

Same pattern as backfill_h2h_totals_odds_phase1_discovery.py: one
events-list call per unique real game date, match by normalized
(home_team, away_team), consume each candidate event once (doubleheader-
safe, matching by closest commence_time -- same fix applied there).
Raw per-day responses cached (data_raw/events_list_raw_cache.jsonl,
shared with that script -- reprocessing a day already fetched costs
nothing).

Cost: ~1 credit per unique day queried (an events-list call, not a full
odds call).

Writes data_raw/event_id_mapping_pitcher_k_gap.csv (game_pk, event_id,
commence_time_utc) -- UNION of existing real ids (from client_games) and
newly discovered ones, ready for phase 2.
"""
import json
import os
import pathlib
import sys
import time
import unicodedata
from datetime import datetime, timezone

import duckdb
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
CACHE_DB = str(ROOT / "db" / "cache_features.duckdb")
OUT_PATH = ROOT / "data_raw" / "event_id_mapping_pitcher_k_gap.csv"
RAW_CACHE_PATH = ROOT / "data_raw" / "events_list_raw_cache.jsonl"  # shared cache w/ the h2h/totals discovery script

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("ODDS_API_KEY not set -- put it in .env")

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"
GAP_START = "2025-05-28"
GAP_END = "2026-09-23"


def norm_team(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def load_raw_cache():
    cache = {}
    if RAW_CACHE_PATH.exists():
        with open(RAW_CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["query_date"]] = rec["events"]
    return cache


def fetch_events_for_day(date_str, raw_cache, raw_out):
    if date_str in raw_cache:
        return raw_cache[date_str], "cached"
    params = {"apiKey": API_KEY, "date": f"{date_str}T23:30:00Z"}
    r = requests.get(BASE, params=params, timeout=20)
    remaining = r.headers.get("x-requests-remaining")
    events = r.json().get("data", []) if r.status_code == 200 else []
    if r.status_code != 200:
        print(f"    HTTP {r.status_code} for {date_str}: {r.text[:150]}")
    raw_out.write(json.dumps({"query_date": date_str, "events": events}) + "\n")
    raw_out.flush()
    return events, remaining


def closest_match(candidates, target_commence):
    if not candidates:
        return None
    def dist(e):
        try:
            et = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        except Exception:
            return float("inf")
        tt = target_commence
        if tt.tzinfo is None:
            tt = tt.replace(tzinfo=timezone.utc)
        return abs((et - tt).total_seconds())
    return min(candidates, key=dist)


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    all_games = con.execute(f"""
        SELECT DISTINCT o.game_pk, o.home_team, o.away_team,
               o.commence_time AT TIME ZONE 'UTC' AS commence_time,
               date_trunc('day', o.commence_time AT TIME ZONE 'UTC') AS game_date,
               g.event_id AS existing_event_id
        FROM cache.cache_mlb_historical_outcomes o
        LEFT JOIN client_games g ON o.event_id = g.event_id AND g.event_id NOT LIKE 'mlb_%'
        WHERE o.game_completed = true
          AND o.commence_time BETWEEN TIMESTAMP '{GAP_START}' AND TIMESTAMP '{GAP_END}'
    """).fetchdf()
    all_games["home_norm"] = all_games["home_team"].map(norm_team)
    all_games["away_norm"] = all_games["away_team"].map(norm_team)
    con.close()

    have = all_games[all_games.existing_event_id.notna()].drop_duplicates(subset=["game_pk"])
    need = all_games[all_games.existing_event_id.isna()].drop_duplicates(subset=["game_pk"])
    print(f"[discovery] {len(all_games)} total games, {len(have)} already have a real event_id, "
          f"{len(need)} need discovery, {need.game_date.nunique()} unique dates to query")

    raw_cache = load_raw_cache()
    print(f"[discovery] {len(raw_cache)} dates' raw events already cached locally")

    out_rows = [{"game_pk": r.game_pk, "event_id": r.existing_event_id, "commence_time_utc": r.commence_time.isoformat()}
                for r in have.itertuples()]
    resolved, unresolved = len(out_rows), 0

    unique_dates = sorted(need.game_date.dt.date.astype(str).unique())
    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_out = open(RAW_CACHE_PATH, "a", encoding="utf-8")
    try:
        for i, date_str in enumerate(unique_dates):
            day_games = need[need.game_date.dt.date.astype(str) == date_str]
            events, remaining = fetch_events_for_day(date_str, raw_cache, raw_out)
            pool = {}
            for e in events:
                key = (norm_team(e["home_team"]), norm_team(e["away_team"]))
                pool.setdefault(key, []).append(e)
            for g in day_games.itertuples():
                key = (g.home_norm, g.away_norm)
                candidates = pool.get(key, [])
                e = closest_match(candidates, g.commence_time)
                if e:
                    candidates.remove(e)
                    out_rows.append({"game_pk": g.game_pk, "event_id": e["id"], "commence_time_utc": e["commence_time"]})
                    resolved += 1
                else:
                    unresolved += 1
            if (i + 1) % 30 == 0 or i == len(unique_dates) - 1:
                print(f"[discovery] ...{i + 1}/{len(unique_dates)} dates processed, "
                      f"{resolved} resolved (cumulative), {unresolved} unresolved so far, "
                      f"credits remaining: {remaining}", flush=True)
                pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
            time.sleep(0.15)
    finally:
        raw_out.close()

    df = pd.DataFrame(out_rows).drop_duplicates(subset=["game_pk"])
    dupe = df.duplicated("event_id").sum()
    df.to_csv(OUT_PATH, index=False)
    print(f"[discovery] DONE. {len(df)} games resolved total, {unresolved} unresolved, "
          f"{dupe} duplicate event_ids (should be near 0) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
