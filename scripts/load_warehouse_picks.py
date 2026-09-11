"""
Mine the harness's own historical-odds-warehouse backtest CSVs
(projects/Betting/betgenius/harness/out/*.csv) for their row-level "picks"
sections. These score real historical odds across real bookmakers against
real graded outcomes, going back to 2023 for pitcher_strikeouts — a much
larger and longer sample than the 2.5-month production pick_history log.

Loads them into db/mlb_markets.duckdb as `warehouse_picks`, tagged by
source file/window so overlapping windows across model-variant files
(e.g. the total_bases m6_* files, which all share one window with
different confidence scores) can be told apart from genuinely distinct
non-overlapping time windows (e.g. hits 2024-06 / 2025-04-25 / 2026-04-25).
"""
import csv
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS_OUT = pathlib.Path(r"C:\Sports\projects\Betting\betgenius\harness\out")
DB = str(ROOT / "db" / "mlb_markets.duckdb")

# Common fields kept across all harness CSV versions -- the schema gained
# extra columns (win_prob, edge_vs_implied, ...) between some milestone runs,
# so each file's own header (the "section,event_id,..." line) is read to
# map by name rather than assuming a fixed position, and only this common
# subset -- present in every version -- is kept.
OUT_COLS = [
    "market", "source_file", "event_id", "player_id", "player_name", "commence_time",
    "pick_side", "line", "entry_odds", "entry_bookmaker", "closing_odds",
    "closing_bookmaker", "confidence", "confidence_pre_cap", "projected_stat",
    "actual_stat", "hit", "voided", "void_reason", "clv_pct", "unit_profit",
]

# One file per market/window kept — chosen for largest N and, where multiple
# non-overlapping real time windows exist (batter_hits), each window is kept
# so they union into one longer multi-year sample instead of one 1-month slice.
SOURCES = [
    ("pitcher_strikeouts", "pitcher_strikeouts_2023-05-03_to_2026-05-24.csv"),
    ("batter_hits", "batter_hits_2024-06-01_to_2024-06-02.csv"),
    ("batter_hits", "batter_hits_oos_2025-04-25_to_2025-05-24.csv"),
    ("batter_hits", "batter_hits_2026-04-25_to_2026-05-24.csv"),
    ("batter_total_bases", "batter_total_bases_gate_m5.csv"),
    ("batter_rbis", "batter_rbis_gate_m5.csv"),
    ("batter_home_runs", "batter_home_runs_gate_m5.csv"),
]


KEEP = OUT_COLS[2:]  # the per-row fields to pull out of each file, by name


def extract_picks(path):
    header = None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("section,event_id"):
                header = next(csv.reader([line.rstrip("\n")]))[1:]  # drop leading "section"
            elif line.startswith("picks,"):
                vals = next(csv.reader([line.rstrip("\n")]))[1:]
                rows.append(dict(zip(header, vals)))
    return rows


def main():
    con = duckdb.connect(DB)
    con.execute("DROP TABLE IF EXISTS warehouse_picks")
    con.execute("CREATE TABLE warehouse_picks (" + ", ".join(f'"{c}" VARCHAR' for c in OUT_COLS) + ")")
    total = 0
    for market, fname in SOURCES:
        path = HARNESS_OUT / fname
        rows = extract_picks(path)
        clean = [[market, fname] + [r.get(k) for k in KEEP] for r in rows]
        if clean:
            con.executemany(
                f"INSERT INTO warehouse_picks VALUES ({', '.join(['?'] * len(OUT_COLS))})", clean
            )
        print(f"{market:<22} {fname:<55} +{len(clean):,}")
        total += len(clean)
    print(f"\nTOTAL loaded: {total:,}")
    con.execute("CHECKPOINT")
    con.close()


if __name__ == "__main__":
    main()
