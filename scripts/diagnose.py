"""
Per-market root-cause diagnostics, run after gate.py shows a FAIL.
For each market: side mix (favorite/dog or over/under), odds distribution,
confidence calibration (actual win rate by confidence bucket vs what the
score implies), and CLV capture rate (fraction of picks with a recorded
closing-line value — needed to trust the CLV column at all).
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

MARKETS = [
    "pitcher_strikeouts", "hits", "total_bases", "rbis", "home_runs",
    "runs_scored", "spreads", "h2h", "totals", "pitcher_outs",
]


def main():
    con = duckdb.connect(DB, read_only=True)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW ph AS
        SELECT
          prop_type,
          lower(pick_side)                         AS side,
          TRY_CAST(odds AS INTEGER)                AS odds,
          TRY_CAST(clv_pct AS DOUBLE)               AS clv_pct,
          TRY_CAST(confidence AS INTEGER)           AS confidence,
          lower(hit) IN ('true','t','1')            AS win,
          TRY_CAST(game_date AS DATE)               AS game_date
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL
          AND lower(coalesce(is_synthetic, 'false')) NOT IN ('true', 't', '1')
          AND lower(coalesce(voided, 'false'))       NOT IN ('true', 't', '1')
          AND hit IS NOT NULL
          AND TRY_CAST(odds AS INTEGER) IS NOT NULL;
    """)

    for m in MARKETS:
        n = con.execute(f"SELECT count(*) FROM ph WHERE prop_type='{m}'").fetchone()[0]
        if n == 0:
            continue
        print(f"\n{'='*92}\n{m}  (n={n:,})\n{'='*92}")

        print("-- side mix --")
        for side, cnt, wr, avg_odds in con.execute(f"""
            SELECT side, count(*), round(avg(win::int)*100,1), round(avg(odds),0)
            FROM ph WHERE prop_type='{m}' GROUP BY 1 ORDER BY 2 DESC
        """).fetchall():
            print(f"   {side:<10} n={cnt:<7} WR={wr:<6} avg_odds={avg_odds}")

        print("-- odds bucket --")
        for bucket, cnt, wr, roi in con.execute(f"""
            SELECT CASE WHEN odds >= 100 THEN 'plus'
                        WHEN odds >= -150 THEN '-100..-149'
                        WHEN odds >= -250 THEN '-150..-249'
                        ELSE '<=-250' END AS bucket,
                   count(*), round(avg(win::int)*100,1),
                   round(avg(CASE WHEN win THEN (CASE WHEN odds>0 THEN odds/100.0 ELSE 100.0/abs(odds) END) ELSE -1 END)*100,2)
            FROM ph WHERE prop_type='{m}' GROUP BY 1 ORDER BY 2 DESC
        """).fetchall():
            print(f"   {bucket:<12} n={cnt:<7} WR={wr:<6} ROI={roi}")

        print("-- confidence calibration (bucketed actual win rate vs bucket midpoint) --")
        for lo, hi, cnt, wr in con.execute(f"""
            SELECT (confidence/10)*10 AS bucket_lo, (confidence/10)*10+9 AS bucket_hi,
                   count(*), round(avg(win::int)*100,1)
            FROM ph WHERE prop_type='{m}' AND confidence IS NOT NULL
            GROUP BY 1,2 ORDER BY 1
        """).fetchall():
            if cnt < 20:
                continue
            print(f"   conf {lo}-{hi:<6} n={cnt:<7} actual_WR={wr}")

        clv_n, clv_covered = con.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE clv_pct IS NOT NULL) FROM ph WHERE prop_type='{m}'
        """).fetchone()
        pct = round(100 * clv_covered / clv_n, 1) if clv_n else 0
        print(f"-- CLV capture: {clv_covered:,}/{clv_n:,} ({pct}%) --")

    con.close()


if __name__ == "__main__":
    main()
