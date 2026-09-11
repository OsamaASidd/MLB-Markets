"""
Cross-source validation: before any sweep_warehouse.py candidate can be
called a real, deployable edge, it has to also show up in live production
pick_history (what the deployed system actually recommended and how it
actually resolved) -- not just in the historical-odds-warehouse hindsight
sample, which can see the full field of every line ever offered and is
therefore exposed to selection effects the live system never faces.

Run after sweep_warehouse.py surfaces a candidate. Currently hardcoded to
the batter_hits "under 0.5" candidate found in this milestone; extend the
CHECKS list for future candidates.
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

CHECKS = [
    {
        "label": "batter_hits: under 0.5 (predict batter goes hitless), any price",
        "warehouse_where": "market='batter_hits' AND side='under' AND line=0.5",
        "production_where": "prop_type='hits' AND side='under' AND line=0.5",
    },
]


def stat_warehouse(con, where):
    con.execute("""
        CREATE OR REPLACE TEMP VIEW wp AS
        SELECT market, lower(pick_side) side, TRY_CAST(entry_odds AS INTEGER) odds,
          TRY_CAST(line AS DOUBLE) line, TRY_CAST(confidence AS INTEGER) confidence,
          lower(hit) IN ('true','t','1') win,
          COALESCE(TRY_CAST(unit_profit AS DOUBLE),
            CASE WHEN lower(hit) IN ('true','t','1')
                 THEN CASE WHEN TRY_CAST(entry_odds AS INTEGER) > 0
                           THEN TRY_CAST(entry_odds AS INTEGER) / 100.0
                           ELSE 100.0 / abs(TRY_CAST(entry_odds AS INTEGER)) END
                 ELSE -1 END) profit
        FROM warehouse_picks
        WHERE lower(coalesce(voided, 'false')) NOT IN ('true', 't', '1')
          AND hit IS NOT NULL AND TRY_CAST(entry_odds AS INTEGER) IS NOT NULL;
    """)
    return con.execute(f"""
        SELECT count(*), round(avg(win::int)*100,1), round(avg(profit)*100,2),
               round((avg(profit)-1.96*stddev_samp(profit)/sqrt(count(*)))*100,2),
               round((avg(profit)+1.96*stddev_samp(profit)/sqrt(count(*)))*100,2)
        FROM wp WHERE {where}
    """).fetchone()


def stat_production(con, where):
    con.execute("""
        CREATE OR REPLACE TEMP VIEW ph AS
        SELECT prop_type, lower(pick_side) side, TRY_CAST(odds AS INTEGER) odds,
          TRY_CAST(line AS DOUBLE) line, TRY_CAST(confidence AS INTEGER) confidence,
          lower(hit) IN ('true','t','1') win,
          CASE WHEN lower(hit) IN ('true','t','1')
               THEN CASE WHEN TRY_CAST(odds AS INTEGER) > 0
                         THEN TRY_CAST(odds AS INTEGER) / 100.0
                         ELSE 100.0 / abs(TRY_CAST(odds AS INTEGER)) END
               ELSE -1 END profit
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL
          AND lower(coalesce(is_synthetic, 'false')) NOT IN ('true', 't', '1')
          AND lower(coalesce(voided, 'false')) NOT IN ('true', 't', '1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL;
    """)
    return con.execute(f"""
        SELECT count(*), round(avg(win::int)*100,1), round(avg(profit)*100,2),
               round((avg(profit)-1.96*stddev_samp(profit)/sqrt(count(*)))*100,2),
               round((avg(profit)+1.96*stddev_samp(profit)/sqrt(count(*)))*100,2)
        FROM ph WHERE {where}
    """).fetchone()


def main():
    con = duckdb.connect(DB, read_only=True)
    for c in CHECKS:
        print(f"\n{'='*90}\n{c['label']}\n{'='*90}")
        wn, wwr, wroi, wlo, whi = stat_warehouse(con, c["warehouse_where"])
        pn, pwr, proi, plo, phi = stat_production(con, c["production_where"])
        print(f"  warehouse hindsight : n={wn:<7} WR={wwr:<6} ROI={wroi:<8} CI=[{wlo},{whi}]")
        print(f"  live production     : n={pn:<7} WR={pwr:<6} ROI={proi:<8} CI=[{plo},{phi}]")
        agree = (wroi is not None and proi is not None and wroi > 0 and proi > 0)
        print(f"  VERDICT: {'confirmed in production' if agree else 'NOT confirmed -- do not ship'}")
    con.close()


if __name__ == "__main__":
    main()
