import pathlib
import unicodedata
import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")


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


def stat(d):
    d = d.dropna(subset=["profit"])
    n = len(d)
    if n == 0:
        return {"n": 0, "wr": None, "roi": None, "lo": None, "hi": None}
    m = d["profit"].mean() * 100
    se = d["profit"].std(ddof=1) / np.sqrt(n) * 100
    return {"n": n, "wr": round(d["win"].mean() * 100, 1), "roi": round(m, 2),
            "lo": round(m - 1.96 * se, 2), "hi": round(m + 1.96 * se, 2)}


con = duckdb.connect(DB, read_only=True)
box = con.execute("""
    SELECT game_pk, player_name, hits FROM boxscore
    WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
""").fetchdf()
box["name_norm"] = box["player_name"].map(norm_name)

odds = con.execute("""
    SELECT co.game_pk, co.line, co.best_under_odds, co.season, co.player_name
    FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
          JOIN client_games g ON co.event_id = g.event_id) co
    WHERE co.market_key = 'batter_hits' AND co.season IN ('2023','2024','2025','2026')
""").fetchdf()
odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
odds["name_norm"] = odds["player_name"].map(norm_name)
m = odds.merge(box[["game_pk", "name_norm", "hits"]], on=["game_pk", "name_norm"], how="inner")
m = m.dropna(subset=["hits", "line", "best_under_odds"])
m["win"] = m["hits"] < m["line"]
m["profit"] = m.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)

MIN_GRADED = 500

print("hits -> under, odds -249..-101 (merged moderate band), full CI, by season:")
sub = m[(m.best_under_odds <= -101) & (m.best_under_odds >= -249)]
for season in [2023, 2024, 2025, 2026, None]:
    s = stat(sub if season is None else sub[sub.season == season])
    passed = s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0
    label = "ALL 2023-2026" if season is None else f"season {season}"
    print(f"  {label:<16} n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")

print("\nsplit into the two original bands separately, full CI:")
for lo_b, hi_b, label in [(-249, -150, "-249..-150"), (-149, -101, "-149..-101")]:
    sub2 = m[(m.best_under_odds <= hi_b) & (m.best_under_odds >= lo_b)]
    print(f"\n  band {label}:")
    for season in [2023, 2024, 2025, 2026]:
        s = stat(sub2[sub2.season == season])
        passed = s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0
        print(f"    season {season}: n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")

# also check line breakdown within this odds band -- does it hold for both 0.5 and 1.5?
print("\nmoderate-odds band, split by line value:")
for line_val in [0.5, 1.5]:
    sub3 = sub[sub.line == line_val]
    print(f"\n  line={line_val}:")
    for season in [2023, 2024, 2025, 2026]:
        s = stat(sub3[sub3.season == season])
        passed = s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0
        print(f"    season {season}: n={s['n']:<7} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")

con.close()
