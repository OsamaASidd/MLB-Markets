"""
Phase 1: resolve real Odds API event_id + commence_time for the h2h/totals
backfill gap (2026-05-25 to 2026-09-23, ~1,568 real games -- verified via
cache_mlb_historical_outcomes, which has home_team/away_team/game_pk/
commence_time for every one of them but only a SYNTHETIC "mlb_<game_pk>"
placeholder id for almost all of them post-2026-06, not a real Odds API
event_id -- confirmed empirically before writing this: only 21 of 1,568
games in this window already have a genuine (non-"mlb_"-prefixed) event_id
stored locally).

Same pattern as build_event_id_mapping_2020_2022.py (Phase 2 of that
backfill): one events-list call per unique real game date, match by
normalized (home_team, away_team). Cost: ~1 credit per unique date (an
events-list call, not a full odds call) -- ~120 real dates in this window.

Writes data_raw/event_id_mapping_h2h_totals_gap.csv. Does NOT touch
db/mlb_markets.duckdb. Resumable: re-run picks up from whatever dates are
already in the output CSV.
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
OUT_PATH = ROOT / "data_raw" / "event_id_mapping_h2h_totals_gap.csv"
RAW_CACHE_PATH = ROOT / "data_raw" / "events_list_raw_cache.jsonl"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("ODDS_API_KEY not set -- put it in .env (git-ignored), never hardcode it here.")

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"
GAP_START = "2026-05-25"
GAP_END = "2026-09-23"


def norm_team(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


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
    """Cached: a date already fetched (even under the old buggy matching
    logic) is never re-queried -- only the MATCHING logic changed, so we
    can reprocess for free from the raw cache."""
    if date_str in raw_cache:
        return raw_cache[date_str], "cached"
    params = {"apiKey": API_KEY, "date": f"{date_str}T23:30:00Z"}
    r = requests.get(BASE, params=params, timeout=20)
    remaining = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        print(f"    HTTP {r.status_code} for {date_str}: {r.text[:150]}")
        events = []
    else:
        events = r.json().get("data", [])
    raw_out.write(json.dumps({"query_date": date_str, "events": events}) + "\n")
    raw_out.flush()
    return events, remaining


def closest_match(candidates, target_commence):
    """Pick the candidate event whose commence_time is closest to the
    local game's own recorded commence_time -- distinguishes doubleheader
    games (same team pair, same date, different start times) instead of
    silently colliding on the same event for both."""
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
    games = con.execute(f"""
        SELECT DISTINCT game_pk, home_team, away_team,
               TRY_CAST(commence_time AS TIMESTAMP) AS commence_time,
               date_trunc('day', commence_time AT TIME ZONE 'UTC') AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true
          AND commence_time BETWEEN TIMESTAMP '{GAP_START}' AND TIMESTAMP '{GAP_END}'
    """).fetchdf()
    con.close()
    games["home_norm"] = games["home_team"].map(norm_team)
    games["away_norm"] = games["away_team"].map(norm_team)
    print(f"[discovery] {len(games):,} real games in the gap, "
          f"{games.game_date.nunique()} unique dates")

    raw_cache = load_raw_cache()
    print(f"[discovery] {len(raw_cache)} dates' raw events already cached locally -- "
          f"reprocessing those for free, only genuinely new dates cost credits")

    # Full reprocess every time (cheap: dedup by game_pk at the end) -- the
    # doubleheader-safe matching logic needs every date's full candidate
    # list, not just previously-unresolved games, since two local games may
    # need to split one date's event list between them.
    games = games.drop_duplicates(subset=["game_pk"]).reset_index(drop=True)
    unique_dates = sorted(games.game_date.dt.date.astype(str).unique())
    out_rows = []
    resolved = 0
    unresolved = 0
    last_remaining = None

    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_out = open(RAW_CACHE_PATH, "a", encoding="utf-8")
    try:
        for i, date_str in enumerate(unique_dates):
            day_games = games[games.game_date.dt.date.astype(str) == date_str]
            events, remaining = fetch_events_for_day(date_str, raw_cache, raw_out)
            if remaining != "cached":
                last_remaining = remaining

            # candidate pool keyed by team pair, each event usable ONCE --
            # consumed on match so a doubleheader's second game gets the
            # OTHER event, not a duplicate of the first.
            pool = {}
            for e in events:
                key = (norm_team(e["home_team"]), norm_team(e["away_team"]))
                pool.setdefault(key, []).append(e)

            for _, g in day_games.iterrows():
                key = (g.home_norm, g.away_norm)
                candidates = pool.get(key, [])
                e = closest_match(candidates, g.commence_time)
                if e:
                    candidates.remove(e)  # consume -- next doubleheader game gets a different one
                    out_rows.append({
                        "game_pk": g.game_pk, "query_date": date_str,
                        "home_team": g.home_team, "away_team": g.away_team,
                        "event_id": e["id"], "commence_time_utc": e["commence_time"],
                    })
                    resolved += 1
                else:
                    unresolved += 1

            if (i + 1) % 20 == 0 or i == len(unique_dates) - 1:
                print(f"[discovery] ...{i + 1}/{len(unique_dates)} dates processed, "
                      f"{resolved} resolved, {unresolved} unresolved so far, "
                      f"credits remaining: {last_remaining}", flush=True)
                pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
            time.sleep(0.15 if remaining != "cached" else 0)
    finally:
        raw_out.close()

    df = pd.DataFrame(out_rows)
    dupe_check = df.duplicated("event_id").sum()
    pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
    print(f"[discovery] DONE. {resolved} games resolved, {unresolved} unresolved, "
          f"{dupe_check} duplicate event_ids remaining (should be 0) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
