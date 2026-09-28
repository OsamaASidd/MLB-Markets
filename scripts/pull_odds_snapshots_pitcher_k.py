"""
Scoped pull of cache_odds_snapshots for market='pitcher_k' (the client's
market key for pitcher_strikeouts prop snapshots -- confirmed via a
distinct-market sample query; NOT 'pitcher_strikeouts', which returns
zero rows in this table). Same day-chunked pattern as
pull_odds_snapshots_scoped.py (that script never pulled this market).

Writes db/cache_features.duckdb table `cache_odds_snapshots_pitcher_k`.
"""
import os
import pathlib
import time

import duckdb
import pandas as pd
import psycopg2

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_OUT = str(ROOT / "db" / "cache_features.duckdb")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DSN = (
    f"host={os.environ['DB_HOST']} port={os.environ['DB_PORT']} dbname={os.environ['DB_NAME']} "
    f"user={os.environ['DB_USER']} password={os.environ['DB_PASSWORD']} "
    f"sslmode=verify-full sslrootcert={os.environ['DB_SSLROOTCERT']} "
    "keepalives=1 keepalives_idle=20 keepalives_interval=10 keepalives_count=5"
)

MARKET = "pitcher_k"
START = "2026-06-20"
END = "2026-09-24"  # exclusive, through today


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    pg = psycopg2.connect(DSN)
    out = duckdb.connect(DB_OUT)

    days = pd.date_range(START, END, freq="D", inclusive="left")
    frames = []
    for d in days:
        day_str = d.strftime("%Y-%m-%d")
        next_str = (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            df = pd.read_sql(
                "SELECT sport, event_id, game_date, game_time, home_team, away_team, "
                "market, prop_type, player_name, pick_side, line, odds, bookmaker, "
                "snapshot_time, source_name FROM cache_odds_snapshots "
                "WHERE market = %(market)s AND snapshot_time >= %(d0)s AND snapshot_time < %(d1)s",
                pg, params={"market": MARKET, "d0": day_str, "d1": next_str},
            )
        except Exception as e:
            pg.rollback()
            log(f"  {day_str}: ERROR {e}")
            continue
        frames.append(df)
        if len(frames) % 20 == 0:
            log(f"  ...{len(frames)}/{len(days)} days pulled, running total {sum(len(f) for f in frames):,} rows")

    full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.execute("CREATE OR REPLACE TABLE cache_odds_snapshots_pitcher_k AS SELECT * FROM full_df")
    log(f"DONE. {len(full_df):,} rows -> table cache_odds_snapshots_pitcher_k "
        f"({full_df['event_id'].nunique() if len(full_df) else 0} unique events, "
        f"{full_df['player_name'].nunique() if len(full_df) else 0} unique players)")

    pg.close()
    out.close()


if __name__ == "__main__":
    main()
