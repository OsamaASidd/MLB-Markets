"""
Build a compact db/mlb_markets.duckdb from the raw warehouse mirror.
Drops the `odds` table (only a stale 3-week 2023 fragment from an
interrupted historical pull, and not used by the gate — pick_history
already carries the entry odds for every graded pick) so the shipped
db stays small. Everything else needed for grading + point-in-time
context is kept.
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = str(ROOT / "db" / "_raw_mirror.duckdb")
DST = str(ROOT / "db" / "mlb_markets.duckdb")

KEEP_TABLES = [
    "pick_history",
    "boxscore",
    "events",
    "lineups",
    "weather",
    "opposing_pitcher",
    "player_metadata",
    "ballpark_factors",
    "algorithm_weights",
    "statcast_xstats",
    "statcast_exit_velo",
]


# These pick_history columns are large free-text/JSON blobs (LLM narrative,
# a rendered factor breakdown, a raw scoring-inputs JSON dump) that together
# account for ~800MB of the ~980MB warehouse mirror. None are read by the
# gate (which only needs prop_type/side/odds/confidence/hit/CLV/dates) so
# they're dropped to keep the shipped db small; the numeric score_* factor
# columns used for projection work are all kept.
DROP_PICK_HISTORY_COLS = {"ai_analysis", "breakdown", "scoring_inputs"}


def main():
    dst = duckdb.connect(DST)
    dst.execute(f"ATTACH '{SRC}' AS src (READ_ONLY)")
    for t in KEEP_TABLES:
        dst.execute(f"DROP TABLE IF EXISTS {t}")
        if t == "pick_history":
            cols = [r[0] for r in dst.execute("DESCRIBE src.pick_history").fetchall()]
            keep_cols = ", ".join(f'"{c}"' for c in cols if c not in DROP_PICK_HISTORY_COLS)
            dst.execute(f"CREATE TABLE {t} AS SELECT {keep_cols} FROM src.{t}")
        else:
            dst.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
        n = dst.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{t:<20} {n:,} rows")
    dst.execute("DETACH src")
    dst.execute("CHECKPOINT")
    dst.close()


if __name__ == "__main__":
    main()
