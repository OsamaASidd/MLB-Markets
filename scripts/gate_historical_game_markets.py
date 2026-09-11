"""
Test simple, generic, mechanical game-market strategies against 6 real MLB
seasons (2014-2019, 14,784 games — see load_historical_2014_2019.py).

Important scope note: this dataset predates the client's model (which only
has real picks from 2026 onward), so it CANNOT backtest "does his algorithm
show an edge" that far back. What it CAN do, honestly, is check whether
simple, interpretable, low-degrees-of-freedom rules (home/away, favorite/dog,
run-line side, over/under, line-movement direction) show a real, generalized
edge on a genuinely large sample -- the opposite end of the risk spectrum
from the 150-factor reweight in Addendum 2, which is exactly the point: a
strategy this simple has nowhere to hide an overfit.

Gate: n>=500, ROI 95% CI lower bound > 0. Walk-forward: train=2014-2017,
test=2018-2019 (a genuine multi-year holdout, not a random split).
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def american_profit(win, odds):
    # SQL expression built inline per-strategy below; this docstring documents the rule:
    # win  -> odds>0 ? odds/100 : 100/abs(odds)
    # lose -> -1
    pass


def connect():
    con = duckdb.connect(DB, read_only=True)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW g AS
        SELECT *,
          CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS home_win,
          (home_score + away_score) AS total_runs,
          TRY_CAST(regexp_replace(CAST(home_run_line AS VARCHAR), '\\+', '') AS DOUBLE) AS home_rl,
          TRY_CAST(regexp_replace(CAST(away_run_line AS VARCHAR), '\\+', '') AS DOUBLE) AS away_rl
        FROM historical_game_odds
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL;
    """)
    return con


def profit_expr(win_col, odds_col):
    return (f"CASE WHEN {win_col} THEN "
            f"(CASE WHEN {odds_col} > 0 THEN {odds_col}/100.0 ELSE 100.0/abs({odds_col}) END) "
            f"ELSE -1 END")


STRATEGIES = [
    # label, eligibility_sql (which games this bet applies to), win_condition_sql, odds_col
    ("ML: always home",       "1=1",                                            "home_win = 1",  "home_close_ml"),
    ("ML: always away",       "1=1",                                            "home_win = 0",  "away_close_ml"),
    ("ML: always favorite",   "1=1",                                            "(home_close_ml < away_close_ml AND home_win=1) OR (away_close_ml < home_close_ml AND home_win=0)", "CASE WHEN home_close_ml < away_close_ml THEN home_close_ml ELSE away_close_ml END"),
    ("ML: always underdog",   "1=1",                                            "(home_close_ml > away_close_ml AND home_win=1) OR (away_close_ml > home_close_ml AND home_win=0)", "CASE WHEN home_close_ml > away_close_ml THEN home_close_ml ELSE away_close_ml END"),
    ("RL: always home",       "1=1",                                            "(home_score + home_rl) > away_score", "home_run_line_odds"),
    ("RL: always away",       "1=1",                                            "(away_score + away_rl) > home_score", "away_run_line_odds"),
    ("Total: always over",    "1=1",                                            "total_runs > close_ou_line", "close_ou_odds"),
    ("Total: always under",   "1=1",                                            "total_runs < close_ou_line", "close_ou_odds"),
    ("ML: line moved toward home, bet home",  "home_close_ml < home_open_ml",   "home_win = 1",  "home_close_ml"),
    ("ML: line moved toward away, bet away",  "away_close_ml < away_open_ml",   "home_win = 0",  "away_close_ml"),
    ("Total: line moved up, bet over",        "close_ou_line > open_ou_line",   "total_runs > close_ou_line", "close_ou_odds"),
    ("Total: line moved down, bet under",     "close_ou_line < open_ou_line",   "total_runs < close_ou_line", "close_ou_odds"),
]


def stat(con, win_cond, odds_col, eligibility="1=1", extra_where="1=1"):
    profit = profit_expr(f"({win_cond})", odds_col)
    r = con.execute(f"""
        SELECT count(*) n,
               round(avg(({win_cond})::int) * 100, 1) wr,
               round(avg({profit}) * 100, 2) roi,
               round((avg({profit}) - 1.96 * stddev_samp({profit}) / sqrt(count(*))) * 100, 2) lo,
               round((avg({profit}) + 1.96 * stddev_samp({profit}) / sqrt(count(*))) * 100, 2) hi
        FROM g WHERE {odds_col} IS NOT NULL AND ({eligibility}) AND ({extra_where})
    """).fetchone()
    return dict(zip(["n", "wr", "roi", "lo", "hi"], r))


def main():
    con = connect()
    total = con.execute("SELECT count(*), min(date), max(date) FROM g").fetchone()
    print("=" * 100)
    print(f"GENERIC GAME-MARKET STRATEGIES ON 6 REAL MLB SEASONS  (n={total[0]:,}, {total[1]} -> {total[2]})")
    print(f"gate: n>={MIN_GRADED}, ROI 95% CI lower bound > 0")
    print("=" * 100)

    passers = []
    for label, elig, win_cond, odds_col in STRATEGIES:
        s = stat(con, win_cond, odds_col, elig)
        passed = s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0
        print(f"  {label:<42} n={s['n']:<6} WR={s['wr']:<6} ROI={s['roi']:<8} CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")
        if passed:
            passers.append((label, elig, win_cond, odds_col, s))

    print("\n" + "=" * 100)
    print("WALK-FORWARD on anything that passed: TRAIN=2014-2017, TEST=2018-2019")
    print("=" * 100)
    for label, elig, win_cond, odds_col, s in passers:
        tr = stat(con, win_cond, odds_col, elig, "season <= 2017")
        te = stat(con, win_cond, odds_col, elig, "season >= 2018")
        print(f"\n--- {label} ---")
        print(f"  TRAIN (2014-2017) n={tr['n']:<6} ROI={tr['roi']:<8} CI=[{tr['lo']},{tr['hi']}]")
        print(f"  TEST  (2018-2019) n={te['n']:<6} ROI={te['roi']:<8} CI=[{te['lo']},{te['hi']}]")
        stable = tr["roi"] is not None and te["roi"] is not None and tr["roi"] > 0 and te["roi"] > 0
        print(f"  {'STABLE -- survives multi-year holdout' if stable else 'NOT stable -- does not survive holdout'}")

    if not passers:
        print("\n  none of the 12 generic strategies cleared the gate on the full 6-year sample.")
    con.close()


if __name__ == "__main__":
    main()
