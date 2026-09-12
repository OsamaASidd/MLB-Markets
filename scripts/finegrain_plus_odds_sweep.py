"""
The Addendum 17 sweep tested "over, plus-money" for home_runs/rbis as one
big bucket and found it catastrophic (-46%/-26%). That bucket spans +100
to +99999 -- a genuinely awful long tail (Addendum 1 found median +19,900
odds on some home_run overs) could be masking a decent narrow band. Break
it into fine bins and check honestly, same gate discipline as everywhere
else, before concluding there's truly nothing recoverable in these sides.
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


def sweep(con, box, market_key, stat_col, label):
    print(f"\n{'='*100}\n{label} -- OVER side, fine-grained plus-odds bins\n{'='*100}")
    odds = con.execute(f"""
        SELECT co.game_pk, co.line, co.best_over_odds, co.season, co.player_name
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = '{market_key}'
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
    odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    m = odds.merge(box[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
    m = m.dropna(subset=[stat_col, "best_over_odds"])
    m["win"] = m[stat_col] > m["line"]
    m["profit"] = m.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)
    m = m[m.best_over_odds >= 100]

    bins = [(100, 150), (150, 200), (200, 300), (300, 400), (400, 500),
            (500, 700), (700, 1000), (1000, 1500), (1500, 3000), (3000, 999999)]
    for lo, hi in bins:
        sub = m[(m.best_over_odds >= lo) & (m.best_over_odds < hi)]
        s = stat(sub)
        passed = gate(s)
        flag = "  <-- PASSES GATE" if passed else ""
        hi_label = str(hi) if hi < 999999 else "+"
        print(f"  odds [{lo},{hi_label})   n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]{flag}")


def main():
    con = duckdb.connect(DB, read_only=True)
    box = con.execute("""
        SELECT game_pk, player_name, home_runs, rbi FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    sweep(con, box, "batter_home_runs", "home_runs", "home_runs")
    sweep(con, box, "batter_rbis", "rbi", "rbis")
    con.close()


if __name__ == "__main__":
    main()
