"""
Canonical MLB market gate — run on the full local pick_history sample.

Gate rule (per client spec, MATTHEW_REQUIREMENTS.md / harness metrics.ts):
  PASS iff n >= 500 graded picks AND the 95% CI lower bound on ROI-after-vig > 0.
  n < 500 can never PASS outright (not enough evidence), even with positive
  point-estimate ROI.

Grades every MLB market on:
  - "all"                     -> the full sample, no filtering (the actual ask
                                  for this milestone: run the FULL sample)
  - "minus-money"              -> excludes plus-money (juice-heavy long-shot) bets
  - "conf>=60 minus-money"     -> the production "recommendable pick" cut
    (matches isRecommendablePick() in harness/lib/metrics.ts: confidence>=60,
    favorite side only)

Also runs a temporal walk-forward split (train = first 60% of dates, test =
last 40%) on each market's best cut, to catch an edge that is in-sample only.

Writes reports/gate_results.csv (one row per market x cut) and prints a
human-readable summary to stdout.
"""
import csv
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
OUT_CSV = ROOT / "reports" / "gate_results.csv"

MIN_GRADED = 500

MARKETS = [
    "pitcher_strikeouts", "hits", "total_bases", "rbis", "home_runs",
    "runs_scored", "spreads", "h2h", "totals", "pitcher_outs",
]

CUTS = [
    ("all", ""),
    ("minus-money", " AND odds<0"),
    ("conf>=60 minus-money", " AND odds<0 AND confidence>=60"),
]


def connect():
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
          TRY_CAST(game_date AS DATE)               AS game_date,
          CASE WHEN lower(hit) IN ('true','t','1')
               THEN CASE WHEN TRY_CAST(odds AS INTEGER) > 0
                         THEN TRY_CAST(odds AS INTEGER) / 100.0
                         ELSE 100.0 / abs(TRY_CAST(odds AS INTEGER)) END
               ELSE -1 END                          AS profit
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL
          AND lower(coalesce(is_synthetic, 'false')) NOT IN ('true', 't', '1')
          AND lower(coalesce(voided, 'false'))       NOT IN ('true', 't', '1')
          AND hit IS NOT NULL
          AND TRY_CAST(odds AS INTEGER) IS NOT NULL;
    """)
    return con


def stat(con, where):
    r = con.execute(f"""
        SELECT count(*) n,
               round(avg(win::int) * 100, 1)  wr,
               round(avg(profit)   * 100, 2)  roi,
               round((avg(profit) - 1.96 * stddev_samp(profit) / sqrt(count(*))) * 100, 2) roi_lo,
               round((avg(profit) + 1.96 * stddev_samp(profit) / sqrt(count(*))) * 100, 2) roi_hi,
               round(avg(clv_pct), 2) clv
        FROM ph WHERE {where}
    """).fetchone()
    return dict(zip(["n", "wr", "roi", "roi_lo", "roi_hi", "clv"], r))


def gate(n, roi_lo):
    return n is not None and n >= MIN_GRADED and roi_lo is not None and roi_lo > 0


def main():
    con = connect()
    tot = con.execute("SELECT count(*), min(game_date), max(game_date) FROM ph").fetchone()
    print("=" * 92)
    print("MLB MARKET GATE — full local pick_history sample")
    print(f"graded MLB picks: {tot[0]:,}   window: {tot[1]} -> {tot[2]}")
    print(f"gate rule: n >= {MIN_GRADED} AND ROI 95% CI lower bound > 0")
    print("=" * 92)

    rows_out = []
    market_verdict = {}  # market -> best cut that passed, or None

    for m in MARKETS:
        n_total = con.execute(f"SELECT count(*) FROM ph WHERE prop_type='{m}'").fetchone()[0]
        print(f"\n--- {m}  (n={n_total:,}) ---")
        if n_total == 0:
            market_verdict[m] = ("FAIL", "no_data", None)
            continue
        best_pass = None
        for label, extra in CUTS:
            where = f"prop_type='{m}'{extra}"
            s = stat(con, where)
            passed = gate(s["n"], s["roi_lo"])
            verdict = "PASS" if passed else "FAIL"
            print(f"  {label:<24} n={s['n']:<7} WR={s['wr']:<6} ROI={s['roi']:<8} "
                  f"CI=[{s['roi_lo']},{s['roi_hi']}]  CLV={s['clv']}  {verdict}")
            rows_out.append({"market": m, "cut": label, **s, "verdict": verdict})
            if passed and best_pass is None:
                best_pass = label
        market_verdict[m] = ("PASS", best_pass, None) if best_pass else ("FAIL", None, None)

    print("\n" + "=" * 92)
    print("WALK-FORWARD (train=first 60% of dates, test=last 40%) on the 'conf>=60 minus-money' cut")
    print("=" * 92)
    for m in MARKETS:
        base = f"prop_type='{m}' AND odds<0 AND confidence>=60"
        n_cut = con.execute(f"SELECT count(*) FROM ph WHERE {base}").fetchone()[0]
        if n_cut < 20:
            continue
        split = con.execute(f"SELECT quantile_cont(epoch(game_date), 0.6) FROM ph WHERE {base}").fetchone()[0]
        tr = stat(con, base + f" AND epoch(game_date) <  {split}")
        te = stat(con, base + f" AND epoch(game_date) >= {split}")
        print(f"\n--- {m} ---")
        print(f"  TRAIN n={tr['n']:<6} ROI={tr['roi']:<8} CI=[{tr['roi_lo']},{tr['roi_hi']}]")
        print(f"  TEST  n={te['n']:<6} ROI={te['roi']:<8} CI=[{te['roi_lo']},{te['roi_hi']}]")

    print("\n" + "=" * 92)
    print("VERDICT SUMMARY")
    print("=" * 92)
    for m in MARKETS:
        v, cut, _ = market_verdict[m]
        tag = f"PASS on '{cut}'" if v == "PASS" else "FAIL"
        print(f"  {m:<20} {tag}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["market", "cut", "n", "wr", "roi", "roi_lo", "roi_hi", "clv", "verdict"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {OUT_CSV}")
    con.close()


if __name__ == "__main__":
    main()
