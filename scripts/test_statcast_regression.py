"""
New custom strategy: Statcast expected-vs-actual batting average as a
regression-to-the-mean signal. Real sabermetric concept, never tested
anywhere in this project -- statcast_xstats (loaded but unused until now)
gives each player's actual BA vs. Statcast's contact-quality-based expected
BA (est_ba). A large positive gap (actual >> expected) means a player is
running hotter than their underlying quality of contact supports --
classically a "due to cool off" signal. A large negative gap is the
opposite (unlucky, due to warm up).

Hypothesis: bet UNDER on hits/total_bases props for "hot" (positive-gap)
batters, OVER for "cold" (negative-gap) batters.

Point-in-time discipline: for each pick, use the most recent statcast
snapshot strictly BEFORE that pick's game_date (statcast_xstats has ~67
snapshot dates, 2026-05-20 to 2026-07-28 -- this only overlaps the
production pick_history window, not the full 2023-2026 warehouse, so this
test necessarily runs on the smaller real production sample).

Train = first 60% of dates, test = last 40%, same as everywhere else.
"""
import pathlib
import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
MIN_GRADED = 500


def statcast_name_to_normal(s):
    # "Wood, James" -> "james wood"
    if s is None or "," not in s:
        return s.lower().strip() if s else s
    last, first = s.split(",", 1)
    return f"{first.strip()} {last.strip()}".lower()


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

    xstats = con.execute("""
        SELECT player_name, snapshot_date, TRY_CAST(ba AS DOUBLE) ba,
               TRY_CAST(est_ba AS DOUBLE) est_ba
        FROM statcast_xstats WHERE ba IS NOT NULL AND est_ba IS NOT NULL
    """).fetchdf()
    xstats["name_norm"] = xstats["player_name"].map(statcast_name_to_normal)
    xstats["snapshot_date"] = pd.to_datetime(xstats["snapshot_date"])
    xstats = xstats.sort_values("snapshot_date")

    picks = con.execute("""
        SELECT prop_type, lower(pick_side) AS side, TRY_CAST(odds AS INTEGER) AS odds,
               TRY_CAST(game_date AS DATE) AS game_date, player_name,
               lower(hit) IN ('true','t','1') AS win
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL AND prop_type IN ('hits','total_bases')
          AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchdf()
    picks["game_date"] = pd.to_datetime(picks["game_date"])

    def normalize_pick_name(s):
        import unicodedata
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.strip().lower()
    picks["name_norm"] = picks["player_name"].map(normalize_pick_name)

    # point-in-time as-of join: most recent snapshot strictly before game_date, per player
    xstats_sorted = xstats.sort_values(["name_norm", "snapshot_date"])
    merged_rows = []
    for name, grp in picks.groupby("name_norm"):
        xg = xstats_sorted[xstats_sorted.name_norm == name]
        if xg.empty:
            continue
        for _, pick in grp.iterrows():
            prior = xg[xg.snapshot_date < pick.game_date]
            if prior.empty:
                continue
            row = prior.iloc[-1]
            merged_rows.append({**pick.to_dict(), "ba": row.ba, "est_ba": row.est_ba})

    m = pd.DataFrame(merged_rows)
    print(f"picks matched to a prior real statcast xBA snapshot: {len(m):,}")
    if len(m) < 100:
        print("too few matches -- stopping honestly rather than drawing conclusions")
        return

    m["gap"] = m["ba"] - m["est_ba"]  # positive = overperforming (hot), negative = underperforming (cold)
    m["profit"] = m.apply(lambda r: profit(r["win"], r["odds"]), axis=1)

    split_idx = int(len(m) * 0.6)
    m = m.sort_values("game_date")
    train, test = m.iloc[:split_idx], m.iloc[split_idx:]

    print(f"\ncorr(gap, win) on TRAIN: {np.corrcoef(train['gap'], train['win'].astype(int))[0,1]:+.3f}")

    print("\n-- strategy: bet UNDER when batter is 'hot' (gap in top quartile), bet OVER when 'cold' (bottom quartile) --")
    for label, part in [("TRAIN", train), ("TEST (holdout)", test)]:
        hi_thresh = train["gap"].quantile(0.75)
        lo_thresh = train["gap"].quantile(0.25)
        hot_under = part[(part.gap >= hi_thresh) & (part.side == "under")]
        cold_over = part[(part.gap <= lo_thresh) & (part.side == "over")]
        combined = pd.concat([hot_under, cold_over])
        s = stat(combined)
        passed = gate(s)
        print(f"  {label:<16} n={s['n']:<6} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")

    con.close()


if __name__ == "__main__":
    main()
