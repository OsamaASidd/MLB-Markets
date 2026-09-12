"""
Apply the same fine-grained line-value / odds-band search that found the
`batter_hits` lever (Addendum 10) to every other market that failed the
blanket-policy check in Addendum 9/14/15 -- home_runs, rbis, runs_scored
(batter markets), and totals (game market) -- on the full real 2023-2026
warehouse. Same discipline: gate = n>=500 and CI lower bound>0, and a cut
only gets called real if it's not a one-season fluke.
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


def load_batter(con, box, market_key, stat_col):
    odds = con.execute(f"""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.season, co.player_name
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
    m = odds.merge(box[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
    return m.dropna(subset=[stat_col])


def sweep_batter(con, box, market_key, stat_col, label):
    print(f"\n{'='*100}\n{label} ({market_key}) -- sweep by side x odds band\n{'='*100}")
    m = load_batter(con, box, market_key, stat_col)
    for side, odds_col in [("under", "best_under_odds"), ("over", "best_over_odds")]:
        d = m.dropna(subset=[odds_col]).copy()
        d["win"] = (d[stat_col] < d["line"]) if side == "under" else (d[stat_col] > d["line"])
        d["profit"] = d.apply(lambda r: profit(r["win"], r[odds_col]), axis=1)
        print(f"\n  -- side={side} --")
        bands = [("<=-250", lambda o: o <= -250), ("-249..-150", lambda o: -250 < o <= -150),
                  ("-149..-101", lambda o: -150 < o <= -101), ("plus", lambda o: o >= 100)]
        for blabel, cond in bands:
            sub = d[d[odds_col].apply(cond)]
            seasons = {}
            ok_seasons = 0
            for sn in [2023, 2024, 2025, 2026]:
                s = stat(sub[sub.season == sn])
                seasons[sn] = s
                if s["n"] >= 200 and s["roi"] is not None and s["roi"] > 0:
                    ok_seasons += 1
            pooled = stat(sub)
            passed = gate(pooled)
            flag = "  <-- POOLED PASSES GATE" if passed else ""
            print(f"    odds {blabel:<12} pooled n={pooled['n']:<7} ROI={pooled['roi']}  CI=[{pooled['lo']},{pooled['hi']}]"
                  f"  (positive in {ok_seasons}/4 seasons with n>=200){flag}")


def sweep_totals(con):
    print(f"\n{'='*100}\ntotals -- sweep by side x total-line value\n{'='*100}")
    team_runs = con.execute("""
        SELECT game_pk, team_id, sum(runs_scored) AS team_runs
        FROM boxscore WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
        GROUP BY 1,2 HAVING sum(runs_scored) IS NOT NULL
    """).fetchdf()
    game_scores = team_runs.groupby("game_pk")["team_runs"].sum().reset_index(name="total_runs")
    odds = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.season
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'totals'
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
    odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
    odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
    m = odds.merge(game_scores, on="game_pk", how="inner")

    for side, odds_col in [("under", "best_under_odds"), ("over", "best_over_odds")]:
        d = m.dropna(subset=[odds_col, "line", "total_runs"]).copy()
        d["win"] = (d["total_runs"] < d["line"]) if side == "under" else (d["total_runs"] > d["line"])
        d["profit"] = d.apply(lambda r: profit(r["win"], r[odds_col]), axis=1)
        print(f"\n  -- side={side} --")
        for line_lo, line_hi, blabel in [(0, 7.5, "<=7.5"), (7.5, 8.5, "8-8.5"), (8.5, 9.5, "9-9.5"), (9.5, 20, ">=10")]:
            sub = d[(d.line > line_lo) & (d.line <= line_hi)]
            pooled = stat(sub)
            passed = gate(pooled)
            flag = "  <-- POOLED PASSES GATE" if passed else ""
            print(f"    total line {blabel:<8} pooled n={pooled['n']:<7} ROI={pooled['roi']}  CI=[{pooled['lo']},{pooled['hi']}]{flag}")


def main():
    con = duckdb.connect(DB, read_only=True)
    box = con.execute("""
        SELECT game_pk, player_name, home_runs, rbi, runs_scored FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    sweep_batter(con, box, "batter_home_runs", "home_runs", "home_runs")
    sweep_batter(con, box, "batter_rbis", "rbi", "rbis")
    # NOTE: runs_scored has no market_key in client_closing_odds at all (a known
    # provider gap -- The Odds API never reliably offered this prop, so the
    # warehouse has zero rows for it; only testable via production pick_history,
    # already done in Addendum 1/9). Not swept here because there's no real
    # odds data to sweep.
    sweep_totals(con)
    con.close()


if __name__ == "__main__":
    main()
