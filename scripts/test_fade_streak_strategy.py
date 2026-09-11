"""
Price out the mean-reversion pattern found in Addendum 5 as an actual bet,
not just a win-rate observation.

Strategy being tested: "fade recent form when it disagrees with the market"
  A) Hot underdog fade: team is hot (L10 >= 7) but market has them as the
     underdog -> bet the OTHER team (the market's favorite) instead.
  B) Cold favorite hold: team is cold (L10 <= 3) but market still favors
     them -> bet that favorite anyway (don't fade a favorite just because
     they're cold). Compared against the baseline "always bet favorite"
     ROI (-4.24%, Addendum 4) to see whether the cold-streak condition adds
     anything beyond generically backing favorites.

Real odds, real ROI, same gate (n>=500, ROI 95% CI lower bound > 0), same
discipline: fit thresholds on TRAIN (2014-2017), the only thing that counts
is what happens on TEST (2018-2019), a true holdout.
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def profit_expr(win_col, odds_col):
    return (f"CASE WHEN {win_col} THEN "
            f"(CASE WHEN {odds_col} > 0 THEN {odds_col}/100.0 ELSE 100.0/abs({odds_col}) END) "
            f"ELSE -1 END")


def stat(con, win_cond, odds_col, eligibility, season_where):
    profit = profit_expr(f"({win_cond})", odds_col)
    r = con.execute(f"""
        WITH g AS (
            SELECT *, (home_score > away_score) AS home_win FROM game_features
        )
        SELECT count(*) n,
               round(avg(({win_cond})::int) * 100, 1) wr,
               round(avg({profit}) * 100, 2) roi,
               round((avg({profit}) - 1.96 * stddev_samp({profit}) / sqrt(count(*))) * 100, 2) lo,
               round((avg({profit}) + 1.96 * stddev_samp({profit}) / sqrt(count(*))) * 100, 2) hi
        FROM g
        WHERE l10_home_games = 10 AND l10_away_games = 10
          AND {odds_col} IS NOT NULL AND ({eligibility}) AND ({season_where})
    """).fetchone()
    return dict(zip(["n", "wr", "roi", "lo", "hi"], r))


def gate(s):
    return s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0


def report(con, label, win_cond, odds_col, eligibility):
    print(f"\n--- {label} ---")
    all_s = stat(con, win_cond, odds_col, eligibility, "1=1")
    tr = stat(con, win_cond, odds_col, eligibility, "season <= 2017")
    te = stat(con, win_cond, odds_col, eligibility, "season >= 2018")
    print(f"  FULL 2014-2019  n={all_s['n']:<6} WR={all_s['wr']:<6} ROI={all_s['roi']:<8} CI=[{all_s['lo']},{all_s['hi']}]  {'PASS' if gate(all_s) else 'fail'}")
    print(f"  TRAIN 2014-2017 n={tr['n']:<6} WR={tr['wr']:<6} ROI={tr['roi']:<8} CI=[{tr['lo']},{tr['hi']}]  {'PASS' if gate(tr) else 'fail'}")
    print(f"  TEST  2018-2019 n={te['n']:<6} WR={te['wr']:<6} ROI={te['roi']:<8} CI=[{te['lo']},{te['hi']}]  {'PASS' if gate(te) else 'fail'} <- true holdout")
    stable = tr["roi"] is not None and te["roi"] is not None and tr["roi"] > 0 and te["roi"] > 0
    print(f"  verdict: {'STABLE across train/test' if stable else 'NOT stable -- fails on holdout or reverses sign'}")
    return te, stable


def main():
    con = duckdb.connect(DB, read_only=True)

    print("=" * 95)
    print("STRATEGY A: fade the hot underdog -- bet the favorite instead")
    print("=" * 95)
    # home team hot+underdog -> bet AWAY (the favorite) to win, using away_close_ml
    report(con, "Home hot underdog (L10_home>=7, home priced +) -> bet AWAY (favorite)",
           "(home_score < away_score)", "away_close_ml",
           "l10_home_wins >= 7 AND home_close_ml > 0")
    # away team hot+underdog -> bet HOME (the favorite) to win, using home_close_ml
    report(con, "Away hot underdog (L10_away>=7, away priced +) -> bet HOME (favorite)",
           "home_win = 1", "home_close_ml",
           "l10_away_wins >= 7 AND away_close_ml > 0")
    # combined (union of both -- same bet direction: back the favorite when the dog is hot)
    print("\n--- COMBINED: back the favorite whenever the underdog is hot (either side) ---")
    combined_win = "(l10_home_wins>=7 AND home_close_ml>0 AND home_win=0) OR (l10_away_wins>=7 AND away_close_ml>0 AND home_win=1)"
    combined_odds = "CASE WHEN l10_home_wins>=7 AND home_close_ml>0 THEN away_close_ml ELSE home_close_ml END"
    combined_elig = "(l10_home_wins>=7 AND home_close_ml>0) OR (l10_away_wins>=7 AND away_close_ml>0)"
    report(con, "COMBINED fade-hot-underdog", combined_win, combined_odds, combined_elig)

    print("\n" + "=" * 95)
    print("STRATEGY B: back the favorite even when they're cold -- vs. baseline 'always bet favorite'")
    print("=" * 95)
    print("  (baseline, no condition, Addendum 4: FULL sample ROI = -4.24%, all seasons)")
    report(con, "Home cold favorite (L10_home<=3, home priced -) -> bet HOME anyway",
           "home_win = 1", "home_close_ml",
           "l10_home_wins <= 3 AND home_close_ml < 0")
    report(con, "Away cold favorite (L10_away<=3, away priced -) -> bet AWAY anyway",
           "(home_score < away_score)", "away_close_ml",
           "l10_away_wins <= 3 AND away_close_ml < 0")
    print("\n--- COMBINED: back the favorite whenever THEY are cold (either side) ---")
    combined_win_b = "(l10_home_wins<=3 AND home_close_ml<0 AND home_win=1) OR (l10_away_wins<=3 AND away_close_ml<0 AND home_win=0)"
    combined_odds_b = "CASE WHEN l10_home_wins<=3 AND home_close_ml<0 THEN home_close_ml ELSE away_close_ml END"
    combined_elig_b = "(l10_home_wins<=3 AND home_close_ml<0) OR (l10_away_wins<=3 AND away_close_ml<0)"
    report(con, "COMBINED cold-favorite-hold", combined_win_b, combined_odds_b, combined_elig_b)

    con.close()


if __name__ == "__main__":
    main()
