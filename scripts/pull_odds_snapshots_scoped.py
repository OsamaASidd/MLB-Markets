"""
Scoped pull of cache_odds_snapshots (8.9M rows total, no usable index on
`market` alone -- a full-table or market-only query times out, confirmed
this session). Chunks by day (indexed on snapshot_time, confirmed fast:
~0.85s per single-market single-day query) x the 4 markets with real
overlapping local data (game_side/h2h, game_total/totals, pitcher_outs,
batter_runs_scored -- the batter prop markets' real warehouse odds stop
around 2025-05-28, well before this table's 2026-06-20 start, so they're
skipped here; not useful data for them regardless of pull effort).

Writes db/cache_features.duckdb table `cache_odds_snapshots_scoped`.
Resumable: skips (market, day) pairs already pulled.
"""
import os
import pathlib
import sys
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

MARKETS = ["game_side", "game_total", "pitcher_outs", "batter_runs_scored"]
START = "2026-06-20"
END = "2026-09-20"  # exclusive


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    pg = psycopg2.connect(DSN)
    out = duckdb.connect(DB_OUT)

    days = pd.date_range(START, END, freq="D", inclusive="left")
    total_rows = 0
    for market in MARKETS:
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
                    pg, params={"market": market, "d0": day_str, "d1": next_str},
                )
            except Exception as e:
                pg.rollback()
                log(f"  {market} {day_str}: ERROR {e}")
                continue
            frames.append(df)
        if not frames:
            continue
        full_df = pd.concat(frames, ignore_index=True)
        tbl = f"cache_odds_snapshots_{market}"
        out.execute(f"CREATE OR REPLACE TABLE {tbl} AS SELECT * FROM full_df")
        total_rows += len(full_df)
        log(f"  {market}: {len(full_df):,} rows -> table {tbl}")

    pg.close()
    out.close()
    log(f"DONE. total rows pulled: {total_rows:,}")


if __name__ == "__main__":
    main()
