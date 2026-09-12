"""
New custom strategy: bet directly with real ballpark factors -- over on
total_bases/home_runs at hitter-friendly parks (top hr_factor), under at
pitcher-friendly parks (bottom hr_factor). ballpark_factors table loaded
but never used as an actual strategy until now (only descriptively).

Real data: ballpark_factors (30 real parks, hr_factor/runs_factor/hits_factor)
joined to weather.venue_name (matches park_name format directly) joined to
events (event_id) joined to client_games (game_pk) joined to real closing
odds + real box scores. Full 2023-2026 real warehouse where the join holds.
"""
import pathlib
import unicodedata
import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def norm_name(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def profit(win, odds):
    if pd.isna(win) or pd.isna(odds):
        return np.nan
    if not win:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def stat(df):
    d = df.dropna(subset=["profit"])
    n = len(d)
    if n == 0:
        return {"n": 0, "wr": None, "roi": None, "lo": None, "hi": None}
    m = d["profit"].mean() * 100
    se = d["profit"].std(ddof=1) / np.sqrt(n) * 100
    return {"n": n, "wr": round(d["win"].mean() * 100, 1), "roi": round(m, 2),
            "lo": round(m - 1.96 * se, 2), "hi": round(m + 1.96 * se, 2)}


def gate(s):
    return s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0


def main():
    con = duckdb.connect(DB, read_only=True)

    park_by_event = con.execute("""
        SELECT DISTINCT w.event_id, w.venue_name FROM weather w
    """).fetchdf()
    parks = con.execute("SELECT * FROM ballpark_factors").fetchdf()
    for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
        parks[c] = pd.to_numeric(parks[c], errors="coerce")
    park_map = park_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    print(f"real games with a matched park factor: {len(park_map):,} of {len(park_by_event):,} weather rows")

    box = con.execute("""
        SELECT game_pk, player_name, home_runs, total_bases FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    for market_key, stat_col, factor_col in [("batter_home_runs", "home_runs", "hr_factor"),
                                                ("batter_total_bases", "total_bases", "runs_factor")]:
        print(f"\n{'='*95}\n{market_key} -- park-factor strategy (factor col: {factor_col})\n{'='*95}")
        odds = con.execute(f"""
            SELECT co.event_id, co.game_pk, co.line, co.best_over_odds, co.best_under_odds,
                   co.season, co.player_name
            FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
                  JOIN client_games g ON co.event_id = g.event_id) co
            WHERE co.market_key = '{market_key}'
        """).fetchdf()
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
        odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
        odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
        odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
        odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
        odds["name_norm"] = odds["player_name"].map(norm_name)
        odds = odds.merge(park_map[["event_id", factor_col]], on="event_id", how="inner")

        m = odds.merge(box[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
        m = m.dropna(subset=[stat_col, factor_col])
        print(f"  matched to real park factor + real outcome: {len(m):,}")
        if len(m) < 100:
            continue

        hi = m[factor_col].quantile(0.75)
        lo = m[factor_col].quantile(0.25)

        over_hi = m[(m[factor_col] >= hi)].dropna(subset=["best_over_odds"]).copy()
        over_hi["win"] = over_hi[stat_col] > over_hi["line"]
        over_hi["profit"] = over_hi.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)

        under_lo = m[(m[factor_col] <= lo)].dropna(subset=["best_under_odds"]).copy()
        under_lo["win"] = under_lo[stat_col] < under_lo["line"]
        under_lo["profit"] = under_lo.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)

        for label, df in [("bet OVER at hitter parks (top quartile factor)", over_hi),
                            ("bet UNDER at pitcher parks (bottom quartile factor)", under_lo)]:
            print(f"\n  -- {label} --")
            for sn in [2023, 2024, 2025, 2026, None]:
                sub = df if sn is None else df[df.season == sn]
                s = stat(sub)
                passed = gate(s)
                lbl = "ALL" if sn is None else str(sn)
                print(f"    {lbl:<6} n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")

    con.close()


if __name__ == "__main__":
    main()
