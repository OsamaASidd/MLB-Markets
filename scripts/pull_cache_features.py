"""
One-time pull of the client's Supabase cache_* feature tables into a local
DuckDB file, for building richer per-market models on the 8 FAIL markets.

Same read-only credential/connection pattern already used throughout this
project (see pull_pick_history.py). Read-only role (harness_readonly),
SELECT only, small-batch friendly (connection limit on this role is 10 --
this script uses exactly one connection, run it alone, not alongside other
scripts hitting the same DB).

Writes db/cache_features.duckdb with one table per source table, 1:1 schema.
"""
import os
import pathlib
import time

import duckdb
import pandas as pd
import psycopg2
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DSN = (
    f"host={os.environ['DB_HOST']} port={os.environ['DB_PORT']} dbname={os.environ['DB_NAME']} "
    f"user={os.environ['DB_USER']} password={os.environ['DB_PASSWORD']} "
    f"sslmode=verify-full sslrootcert={os.environ['DB_SSLROOTCERT']} "
    "keepalives=1 keepalives_idle=20 keepalives_interval=10 keepalives_count=5"
)
OUT_DB = str(ROOT / "db" / "cache_features.duckdb")

TABLES = [
    "cache_statcast_pitcher_arsenal",
    "cache_statcast_batters_pull_rate",
    "cache_mlb_park_dimensions",
    "cache_mlb_ballpark_orientation",
    "cache_statcast_batters_contact_rate",
    "cache_statcast_batters_sprint_speed",
    "cache_statcast_framing",
    "cache_mlb_batter_splits",
    "cache_mlb_pitcher_splits",
    "cache_mlb_bullpen_stats",
    "cache_mlb_pitcher_last3",
    "cache_mlb_pen_rest",
    "cache_mlb_bullpen_high_leverage",
    "cache_mlb_pitcher_inn1",
    "cache_savant_team_chase",
    "cache_mlb_team_manager_hook",
    "cache_mlb_team_oaa",
    "cache_umpire_stats",
    "cache_team_batting_stats",
    "cache_mlb_historical_odds_snapshot",
    "cache_odds_snapshots",
    "props_cache",
    "mlb_pitcher_season_stats",
    "cache_mlb_historical_outcomes",
    "cache_mlb_historical_bullpen",
    "cache_mlb_historical_pitcher_statcast",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    pg = psycopg2.connect(DSN)
    out = duckdb.connect(OUT_DB)

    summary = []
    for t in TABLES:
        try:
            df = pd.read_sql(f"SELECT * FROM {t}", pg)
        except Exception as e:
            pg.rollback()
            log(f"  {t}: SKIPPED ({e})")
            summary.append((t, "ERROR", str(e)[:120]))
            continue
        out.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM df")
        log(f"  {t}: {len(df):,} rows, {len(df.columns)} cols")
        summary.append((t, len(df), len(df.columns)))

    pg.close()
    out.close()

    log(f"\nwrote {OUT_DB}")
    log("\nSUMMARY")
    for row in summary:
        log(f"  {row}")


if __name__ == "__main__":
    main()
