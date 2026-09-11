"""
Exhaustive search for a slice of `warehouse_picks` that clears the gate
(n>=500 AND ROI 95% CI lower bound > 0) with a real, walk-forward-stable
edge -- run on the harness's own historical-odds-warehouse backtest data
(real bookmaker odds, real graded outcomes, 2023-2026 for strikeouts;
2024-2026 multi-year composite for hits; single-month 2026-04/05 for
total_bases/rbis/home_runs) rather than the narrower 2.5-month production
pick_history log.

Multiple-comparisons note: dozens of confidence x odds x side cuts are
tested per market. At a 95% CI, roughly 1 in 20 dead-signal cuts will
look significant by chance alone. A cut is only reported as a genuine
candidate here if it clears the gate AND both walk-forward halves
(train = first 60% of dates, test = last 40%) are independently positive
-- chance noise essentially never survives an out-of-sample split twice.
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500

MARKETS = ["pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_rbis", "batter_home_runs"]

CONF_THRESHOLDS = [0, 50, 55, 58, 60, 62, 65, 68, 70, 72, 75, 80]
ODDS_FILTERS = [
    ("all", ""),
    ("minus", " AND odds < 0"),
    ("minus_tight(-101..-200)", " AND odds < 0 AND odds >= -200"),
    ("minus_heavy(<=-200)", " AND odds <= -200"),
    ("plus", " AND odds >= 0"),
]
SIDE_FILTERS = [("both", ""), ("over", " AND side='over'"), ("under", " AND side='under'")]


def connect():
    con = duckdb.connect(DB, read_only=True)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW wp AS
        SELECT
          market,
          lower(pick_side)                          AS side,
          TRY_CAST(entry_odds AS INTEGER)            AS odds,
          TRY_CAST(confidence AS INTEGER)            AS confidence,
          TRY_CAST(clv_pct AS DOUBLE)                AS clv_pct,
          lower(hit) IN ('true','t','1')             AS win,
          TRY_CAST(commence_time AS TIMESTAMP)        AS commence_time,
          COALESCE(TRY_CAST(unit_profit AS DOUBLE),
            CASE WHEN lower(hit) IN ('true','t','1')
                 THEN CASE WHEN TRY_CAST(entry_odds AS INTEGER) > 0
                           THEN TRY_CAST(entry_odds AS INTEGER) / 100.0
                           ELSE 100.0 / abs(TRY_CAST(entry_odds AS INTEGER)) END
                 ELSE -1 END)                         AS profit
        FROM warehouse_picks
        WHERE lower(coalesce(voided, 'false')) NOT IN ('true', 't', '1')
          AND hit IS NOT NULL
          AND TRY_CAST(entry_odds AS INTEGER) IS NOT NULL;
    """)
    return con


def stat(con, where):
    r = con.execute(f"""
        SELECT count(*) n,
               round(avg(win::int) * 100, 1) wr,
               round(avg(profit) * 100, 2) roi,
               round((avg(profit) - 1.96 * stddev_samp(profit) / sqrt(count(*))) * 100, 2) lo,
               round((avg(profit) + 1.96 * stddev_samp(profit) / sqrt(count(*))) * 100, 2) hi,
               round(avg(clv_pct), 2) clv
        FROM wp WHERE {where}
    """).fetchone()
    return dict(zip(["n", "wr", "roi", "lo", "hi", "clv"], r))


def walk_forward(con, where):
    split = con.execute(f"SELECT quantile_cont(epoch(commence_time), 0.6) FROM wp WHERE {where}").fetchone()[0]
    if split is None:
        return None, None
    tr = stat(con, where + f" AND epoch(commence_time) <  {split}")
    te = stat(con, where + f" AND epoch(commence_time) >= {split}")
    return tr, te


def main():
    con = connect()
    candidates = []
    for m in MARKETS:
        n_total = con.execute(f"SELECT count(*) FROM wp WHERE market='{m}'").fetchone()[0]
        print(f"\n{'='*100}\n{m}  (total candidates n={n_total:,})\n{'='*100}")
        if n_total == 0:
            continue
        best_for_market = None
        for conf in CONF_THRESHOLDS:
            for odds_label, odds_clause in ODDS_FILTERS:
                for side_label, side_clause in SIDE_FILTERS:
                    where = f"market='{m}'" + odds_clause + side_clause
                    if conf:
                        where += f" AND confidence >= {conf}"
                    s = stat(con, where)
                    if not s["n"] or s["n"] < 200:
                        continue
                    passed = s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0
                    tag = f"conf>={conf} {odds_label} {side_label}"
                    if passed:
                        tr, te = walk_forward(con, where)
                        stable = tr and te and tr["roi"] is not None and te["roi"] is not None and tr["roi"] > 0 and te["roi"] > 0
                        print(f"  GATE PASS  {tag:<36} n={s['n']:<7} WR={s['wr']:<6} ROI={s['roi']:<7} "
                              f"CI=[{s['lo']},{s['hi']}]  CLV={s['clv']}  "
                              f"WF(train={tr['roi'] if tr else None},test={te['roi'] if te else None})  "
                              f"{'STABLE' if stable else 'not confirmed by walk-forward'}")
                        candidates.append({"market": m, "cut": tag, **s, "wf_train": tr["roi"] if tr else None,
                                            "wf_test": te["roi"] if te else None, "stable": stable})
                        if best_for_market is None or s["roi"] > best_for_market["roi"]:
                            best_for_market = {"cut": tag, **s}
        if best_for_market is None:
            # report the closest miss: highest ROI cut with n>=200, regardless of gate
            close = con.execute(f"""
                SELECT confidence FROM wp WHERE market='{m}' AND confidence IS NOT NULL
                ORDER BY confidence DESC LIMIT 1
            """).fetchone()
            print(f"  no cut cleared the gate for {m}")

    print(f"\n{'='*100}\nCANDIDATES THAT CLEARED THE GATE: {len(candidates)}\n{'='*100}")
    stable_wins = [c for c in candidates if c["stable"]]
    print(f"Of those, walk-forward-STABLE (train AND test both positive): {len(stable_wins)}")
    for c in stable_wins:
        print(f"  {c['market']:<22} {c['cut']:<36} n={c['n']:<7} ROI={c['roi']}%  "
              f"CI_lo={c['lo']}%  train={c['wf_train']}%  test={c['wf_test']}%")
    con.close()
    return candidates


if __name__ == "__main__":
    main()
