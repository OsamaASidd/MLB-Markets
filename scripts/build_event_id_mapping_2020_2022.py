"""
Phase 2: map real 2020-2022 MLB games (from the schedule Phase 1 just
pulled) to their real Odds API event_id, so Phase 3 can pull odds for a
specific real game the same way the 2023-2026 backfills did via
client_games.event_id -- except client_games has no rows before 2023, so
this has to come from the Odds API's own historical event listings
instead, matched by (home_team, away_team, game_date).

Does NOT touch db/mlb_markets.duckdb -- reads the schedule CSV Phase 1
wrote, writes its own output CSV. Runs independently/in parallel with the
box-score backfill.

Cost: ~1 credit per unique game_date queried (an events-list call, not a
full odds call) -- roughly 400-450 real days across the three seasons.
"""
import csv
import json
import os
import pathlib
import sys
import time
import unicodedata
from datetime import datetime

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEDULE_PATH = ROOT / "data_raw" / "mlb_schedule_2020_2022.csv"
OUT_PATH = ROOT / "data_raw" / "event_id_mapping_2020_2022.csv"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("ODDS_API_KEY not set -- put it in .env (git-ignored), never hardcode it here.")

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"


def norm_team(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def wait_for_schedule(timeout_s=600):
    waited = 0
    while not SCHEDULE_PATH.exists() and waited < timeout_s:
        time.sleep(5)
        waited += 5
    if not SCHEDULE_PATH.exists():
        sys.exit(f"schedule file never appeared at {SCHEDULE_PATH} after {timeout_s}s")


def fetch_events_for_day(date_str):
    """One events-list call for a given real day (evening UTC snapshot,
    covers essentially all that day's US game times)."""
    url = f"{BASE}"
    params = {"apiKey": API_KEY, "date": f"{date_str}T23:30:00Z"}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return []
    return r.json().get("data", [])


def main():
    print("[mapping] waiting for schedule CSV from Phase 1...")
    wait_for_schedule()
    schedule = pd.read_csv(SCHEDULE_PATH)
    schedule["home_norm"] = schedule["home_team"].map(norm_team)
    schedule["away_norm"] = schedule["away_team"].map(norm_team)
    print(f"[mapping] {len(schedule):,} real games in schedule")

    unique_dates = sorted(schedule["game_date"].unique())
    print(f"[mapping] {len(unique_dates)} unique real game dates to query")

    resolved = 0
    unresolved = 0
    out_rows = []
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, date_str in enumerate(unique_dates):
        day_games = schedule[schedule.game_date == date_str]
        events = fetch_events_for_day(date_str)
        events_by_key = {}
        for e in events:
            k = (norm_team(e["home_team"]), norm_team(e["away_team"]))
            events_by_key[k] = e

        for _, g in day_games.iterrows():
            key = (g.home_norm, g.away_norm)
            e = events_by_key.get(key)
            if e:
                out_rows.append({
                    "game_pk": g.game_pk, "game_date": g.game_date,
                    "home_team": g.home_team, "away_team": g.away_team,
                    "event_id": e["id"], "commence_time_utc": e["commence_time"],
                })
                resolved += 1
            else:
                unresolved += 1

        if (i + 1) % 50 == 0 or i == len(unique_dates) - 1:
            print(f"[mapping] ...{i+1}/{len(unique_dates)} days queried, "
                  f"{resolved} games resolved, {unresolved} unresolved", flush=True)
            pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
        time.sleep(0.1)

    pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
    print(f"[mapping] DONE. {resolved} real games mapped to a real event_id, "
          f"{unresolved} unresolved (saved to {OUT_PATH})")


if __name__ == "__main__":
    main()
