"""
Addendum 9 found hits->under and total_bases->under flat (~0% ROI) at full
market scale, and totals/h2h/spreads/strikeouts clearly negative. This
script searches for real, validated sub-segments (specific line values,
odds bands) that might carry an edge the unconditional "bet every line"
test washes out -- same discipline as everywhere else: fit on 2024, confirm
on 2025 as a true holdout, and only report something as real if BOTH
seasons independently clear the gate in the same direction. 2026 (partial
season, smaller n) used as a second confirmation where sample allows.

This is deliberately the same kind of search that produced the "under 0.5"
false positive earlier in this project -- the difference this time is a
much bigger real sample (100K+ vs ~1,400) and a strict two-season
replication requirement before anything is called real.
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
    se = d["profit"].std(ddof=1) / np.sqrt(n) * 100 if n > 1 else np.nan
    return {"n": n, "wr": round(d["win"].mean() * 100, 1), "roi": round(m, 2),
            "lo": round(m - 1.96 * se, 2) if not np.isnan(se) else None,
            "hi": round(m + 1.96 * se, 2) if not np.isnan(se) else None}


def gate(s):
    return s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0


def load_boxscore(con):
    df = con.execute("""
        SELECT game_pk, player_name, hits, total_bases FROM boxscore
        WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
    """).fetchdf()
    df["name_norm"] = df["player_name"].map(norm_name)
    return df


def load_market(con, market_key, stat_col, box):
    odds = con.execute(f"""
        SELECT co.game_pk, co.line, co.best_under_odds, co.season, co.player_name
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = '{market_key}' AND co.season IN ('2024','2025','2026')
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
    odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    m = odds.merge(box[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
    m = m.dropna(subset=[stat_col, "line", "best_under_odds"])
    m["win"] = m[stat_col] < m["line"]
    m["profit"] = m.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)
    return m


def sweep_line_and_odds(df, label):
    print(f"\n{'='*100}\n{label}\n{'='*100}")
    print("-- by LINE value --")
    for line_val in sorted(df["line"].dropna().unique()):
        sub = df[df.line == line_val]
        s24 = stat(sub[sub.season == 2024])
        s25 = stat(sub[sub.season == 2025])
        s26 = stat(sub[sub.season == 2026])
        if s24["n"] < 200 and s25["n"] < 200:
            continue
        pass24, pass25 = gate(s24), gate(s25)
        replicated = pass24 and pass25 and s24["roi"] is not None and s25["roi"] is not None and s24["roi"] > 0 and s25["roi"] > 0
        flag = "  <-- REPLICATED both seasons" if replicated else ""
        print(f"  line={line_val:<6} 2024: n={s24['n']:<7} ROI={s24['roi']}  |  "
              f"2025: n={s25['n']:<7} ROI={s25['roi']}  |  2026: n={s26['n']:<6} ROI={s26['roi']}{flag}")

    print("\n-- by ODDS band --")
    bands = [("<=-250", lambda o: o <= -250), ("-249..-150", lambda o: -250 < o <= -150),
             ("-149..-101", lambda o: -150 < o <= -101), ("plus (>=100)", lambda o: o >= 100)]
    for label_b, cond in bands:
        sub = df[df.best_under_odds.apply(cond)]
        s24 = stat(sub[sub.season == 2024])
        s25 = stat(sub[sub.season == 2025])
        s26 = stat(sub[sub.season == 2026])
        pass24, pass25 = gate(s24), gate(s25)
        replicated = pass24 and pass25 and s24["roi"] is not None and s25["roi"] is not None and s24["roi"] > 0 and s25["roi"] > 0
        flag = "  <-- REPLICATED both seasons" if replicated else ""
        print(f"  odds {label_b:<12} 2024: n={s24['n']:<7} ROI={s24['roi']}  |  "
              f"2025: n={s25['n']:<7} ROI={s25['roi']}  |  2026: n={s26['n']:<6} ROI={s26['roi']}{flag}")


def main():
    con = duckdb.connect(DB, read_only=True)
    box = load_boxscore(con)

    hits = load_market(con, "batter_hits", "hits", box)
    sweep_line_and_odds(hits, "batter_hits -> under, by line and odds band")

    tb = load_market(con, "batter_total_bases", "total_bases", box)
    sweep_line_and_odds(tb, "batter_total_bases -> under, by line and odds band")

    con.close()


if __name__ == "__main__":
    main()
