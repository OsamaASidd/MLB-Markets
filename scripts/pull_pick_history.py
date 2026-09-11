"""
Refresh pick_history (the source of truth for the gate) from the production
Supabase warehouse into the local db/mlb_markets.duckdb file.

Read-only credential (harness_readonly), small-batch keyset pagination —
same pattern already proven in projects/local_analysis/pull_ph.py. Always
does a full drop + re-pull (the table is all-VARCHAR and cheap enough to
refetch in one pass; keyset order guarantees no dupes/gaps mid-pull).
"""
import os, time, pathlib
import psycopg2
import pandas as pd
import duckdb
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DSN = (
    f"host={os.environ['DB_HOST']} port={os.environ['DB_PORT']} dbname={os.environ['DB_NAME']} "
    f"user={os.environ['DB_USER']} password={os.environ['DB_PASSWORD']} "
    f"sslmode=verify-full sslrootcert={os.environ['DB_SSLROOTCERT']} "
    "keepalives=1 keepalives_idle=20 keepalives_interval=10 keepalives_count=5"
)
DB = str(ROOT / "db" / "mlb_markets.duckdb")
PAGE = 600


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ddb = duckdb.connect(DB)
    ddb.execute("CREATE TABLE IF NOT EXISTS _loaded(unit VARCHAR, rows BIGINT, done_at VARCHAR)")
    ddb.execute("DROP TABLE IF EXISTS pick_history")
    ddb.execute("DELETE FROM _loaded WHERE unit='pick_history'")

    conn = psycopg2.connect(DSN)
    conn.set_session(readonly=True)
    with conn.cursor() as sc:
        sc.execute("SET statement_timeout=0")

    last_ct, last_id, total, created = None, None, 0, False
    while True:
        with conn.cursor() as c:
            if last_ct is None:
                c.execute("SELECT * FROM pick_history WHERE sport='mlb' ORDER BY created_at, id LIMIT %s", (PAGE,))
            else:
                c.execute(
                    "SELECT * FROM pick_history WHERE sport='mlb' AND (created_at, id) > (%s, %s) "
                    "ORDER BY created_at, id LIMIT %s",
                    (last_ct, last_id, PAGE),
                )
            rows = c.fetchall()
            cols = [d[0] for d in c.description]
        if not rows:
            break
        df = pd.DataFrame(rows, columns=cols)
        df = df.apply(lambda col: col.map(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)))
        if not created:
            ddb.execute("CREATE TABLE pick_history (" + ", ".join(f'"{c}" VARCHAR' for c in cols) + ")")
        ddb.register("df_v", df)
        ddb.execute("INSERT INTO pick_history SELECT * FROM df_v")
        ddb.unregister("df_v")
        created = True
        ci, ii = cols.index("created_at"), cols.index("id")
        last_ct, last_id = rows[-1][ci], rows[-1][ii]
        total += len(rows)
        if total % 12000 < PAGE:
            log(f"  pick_history: {total:,}")
        if len(rows) < PAGE:
            break

    ddb.execute("INSERT INTO _loaded VALUES ('pick_history', ?, ?)", [total, time.strftime("%Y-%m-%d %H:%M:%S")])
    log(f"DONE pick_history: {total:,} rows")
    conn.close()
    ddb.close()


if __name__ == "__main__":
    main()
