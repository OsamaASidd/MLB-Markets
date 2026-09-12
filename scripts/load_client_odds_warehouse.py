"""
Load the client-provided real closing-odds export (harness_out/MLB_Odds_API_2023_2026.xlsx)
into db/mlb_markets.duckdb. This is the actual cache_mlb_historical_odds warehouse,
exported read-only, zero API credits spent -- real closing best-price odds
(American, MAX across books) for every MLB market, 2023-05 through 2026-05-24.

Streams via openpyxl read-only mode (fast for millions of rows) rather than
pandas.read_excel (which loads the whole sheet into memory column-by-column
and is much slower at this size).
"""
import pathlib
import time
import openpyxl
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
XLSX = ROOT / "harness_out" / "MLB_Odds_API_2023_2026.xlsx"
DB = str(ROOT / "db" / "mlb_markets.duckdb")

ODDS_COLS = ["event_id", "game_date", "home_team", "away_team", "market_key",
             "player_name", "line", "best_over_odds", "best_under_odds",
             "books_at_close", "closing_snapshot_utc"]
GAMES_COLS = ["event_id", "game_pk", "game_date", "season", "commence_time_utc",
              "home_team", "away_team", "odds_backfill_status",
              "outcomes_backfill_status", "markets_with_odds", "odds_rows"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stream_sheet_to_duckdb(wb, sheet_name, cols, con, table, batch_size=50_000):
    ws = wb[sheet_name]
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute("CREATE TABLE " + table + " (" + ", ".join(f'"{c}" VARCHAR' for c in cols) + ")")
    rows_iter = ws.iter_rows(values_only=True)
    next(rows_iter)  # header
    batch = []
    total = 0
    for row in rows_iter:
        batch.append([str(v) if v is not None else None for v in row[: len(cols)]])
        if len(batch) >= batch_size:
            con.executemany(f"INSERT INTO {table} VALUES ({', '.join(['?'] * len(cols))})", batch)
            total += len(batch)
            log(f"  {table}: {total:,}")
            batch = []
    if batch:
        con.executemany(f"INSERT INTO {table} VALUES ({', '.join(['?'] * len(cols))})", batch)
        total += len(batch)
    log(f"DONE {table}: {total:,} rows")
    return total


def main():
    log(f"opening {XLSX} ({XLSX.stat().st_size / 1e6:.0f} MB)")
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    con = duckdb.connect(DB)

    stream_sheet_to_duckdb(wb, "Games", GAMES_COLS, con, "client_games")

    all_odds = []
    for year in [2023, 2024, 2025, 2026]:
        sheet = f"ClosingOdds_{year}"
        con.execute(f"DROP TABLE IF EXISTS closing_odds_{year}")
        n = stream_sheet_to_duckdb(wb, sheet, ODDS_COLS, con, f"closing_odds_{year}")
        all_odds.append((year, n))

    con.execute("DROP VIEW IF EXISTS client_closing_odds")
    union_sql = " UNION ALL ".join(f"SELECT *, {y} AS season FROM closing_odds_{y}" for y, _ in all_odds)
    con.execute(f"CREATE VIEW client_closing_odds AS {union_sql}")
    total = con.execute("SELECT count(*) FROM client_closing_odds").fetchone()[0]
    log(f"client_closing_odds view: {total:,} rows across {len(all_odds)} seasons")

    con.execute("CHECKPOINT")
    con.close()
    wb.close()


if __name__ == "__main__":
    main()
