"""
Grade the real client-provided 2023-2026 closing-odds warehouse against real
game outcomes, then test the ACTUAL policy already adopted by the project
(per Phase1-MLB-Developer-Handoff.md section 7):

  Side (bet this side by default):  hits->under, total_bases->under,
                                     h2h(game_side)->away, totals(game_total)->under
  Veto (do not bet at all):         home_runs, rbis, runs_scored,
                                     pitcher_strikeouts, pitcher_outs

This tests the policy at full real scale (millions of real closing-odds
rows, not a production sample) using the best available closing price
across all 56 books (the warehouse's "best_over_odds"/"best_under_odds"
convention -- numerically-best American price, i.e. optimistic/line-shopping
best case, disclosed plainly as a modeling assumption, not a claim every
bettor gets this exact price).

Grading source: real boxscore data already in this repo's db (2024-04-01
onward) -- so 2023 cannot be graded yet without a separate real-outcomes
backfill for that season (disclosed, not silently dropped).

Same gate discipline as the rest of this repo: n>=500, ROI 95% CI lower
bound > 0. Also splits by season so a real multi-year holdout check is
possible (train on earlier seasons, confirm on later ones) rather than a
single pooled number.
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
    return {
        "n": n, "wr": round(d["win"].mean() * 100, 1), "roi": round(m, 2),
        "lo": round(m - 1.96 * se, 2) if not np.isnan(se) else None,
        "hi": round(m + 1.96 * se, 2) if not np.isnan(se) else None,
    }


def gate(s):
    return s["n"] >= MIN_GRADED and s["lo"] is not None and s["lo"] > 0


def print_stat(label, s):
    passed = gate(s)
    print(f"  {label:<40} n={s['n']:<8} WR={s['wr']}  ROI={s['roi']}  CI=[{s['lo']},{s['hi']}]  {'PASS' if passed else 'fail'}")


def load_boxscore_lookup(con):
    df = con.execute("""
        SELECT game_pk, player_name, hits, home_runs, rbi, total_bases,
               strikeouts, outs, is_starter, position_type, team_id,
               runs_scored
        FROM boxscore
    """).fetchdf()
    df["name_norm"] = df["player_name"].map(norm_name)
    return df


def grade_batter_market(odds_df, box_df, stat_col, line_col="line"):
    box_batters = box_df[box_df.position_type.isin(["Catcher", "Hitter", "Infielder", "Outfielder"])]
    merged = odds_df.merge(
        box_batters[["game_pk", "name_norm", stat_col]],
        on=["game_pk", "name_norm"], how="inner",
    )
    merged = merged.dropna(subset=[stat_col])
    return merged


def grade_pitcher_market(odds_df, box_df, stat_col):
    box_p = box_df[(box_df.position_type == "Pitcher") & (box_df.is_starter == True)]
    merged = odds_df.merge(
        box_p[["game_pk", "name_norm", stat_col]],
        on=["game_pk", "name_norm"], how="inner",
    )
    merged = merged.dropna(subset=[stat_col])
    return merged


def main():
    con = duckdb.connect(DB, read_only=True)

    odds = con.execute("""
        SELECT co.*, g.game_pk
        FROM client_closing_odds co
        JOIN client_games g ON co.event_id = g.event_id
        WHERE co.season IN (2023, 2024, 2025, 2026)
    """).fetchdf()
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["best_over_odds"] = pd.to_numeric(odds["best_over_odds"], errors="coerce")
    odds["best_under_odds"] = pd.to_numeric(odds["best_under_odds"], errors="coerce")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    odds["season"] = pd.to_numeric(odds["season"], errors="coerce")
    print(f"real closing-odds rows loaded (2023-2026, now that 2023 box scores are backfilled): {len(odds):,}")

    box = load_boxscore_lookup(con)
    print(f"boxscore rows available for grading: {len(box):,}")

    results = {}

    # --- hits -> always under ---
    m = odds[odds.market_key == "batter_hits"]
    g = grade_batter_market(m, box, "hits")
    g["win"] = g["hits"] < g["line"]
    g["profit"] = g.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)
    print("\n=== hits: always UNDER (client policy default) ===")
    print_stat("FULL 2023-2026", stat(g))
    for s in [2023, 2024, 2025, 2026]:
        print_stat(f"season {s}", stat(g[g.season == s]))
    results["hits_under"] = g

    # --- total_bases -> always under ---
    m = odds[odds.market_key == "batter_total_bases"]
    g = grade_batter_market(m, box, "total_bases")
    g["win"] = g["total_bases"] < g["line"]
    g["profit"] = g.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)
    print("\n=== total_bases: always UNDER (client policy default) ===")
    print_stat("FULL 2023-2026", stat(g))
    for s in [2023, 2024, 2025, 2026]:
        print_stat(f"season {s}", stat(g[g.season == s]))
    results["tb_under"] = g

    # --- pitcher_strikeouts -> minus-money only (proxy for veto re-check, no confidence available here) ---
    m = odds[odds.market_key == "pitcher_strikeouts"]
    g = grade_pitcher_market(m, box, "strikeouts")
    over = g.copy(); over["win"] = over["strikeouts"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
    under = g.copy(); under["win"] = under["strikeouts"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
    both = pd.concat([over, under])
    both = both[both.odds < 0]  # minus-money only
    both["profit"] = both.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    print("\n=== pitcher_strikeouts: minus-money only, both sides (currently vetoed by client policy) ===")
    print_stat("FULL 2023-2026", stat(both))
    for s in [2023, 2024, 2025, 2026]:
        print_stat(f"season {s}", stat(both[both.season == s]))

    # --- game markets: h2h always away, spreads away, totals always under ---
    team_runs = con.execute("""
        SELECT game_pk, team_id, sum(runs_scored) AS team_runs
        FROM boxscore WHERE position_type IN ('Catcher','Hitter','Infielder','Outfielder')
        GROUP BY 1,2 HAVING sum(runs_scored) IS NOT NULL
    """).fetchdf()
    game_scores = team_runs.groupby("game_pk")["team_runs"].sum().reset_index(name="total_runs")

    m = odds[(odds.market_key == "totals")].merge(game_scores, on="game_pk", how="inner")
    m["win"] = m["total_runs"] < m["line"]
    m["profit"] = m.apply(lambda r: profit(r["win"], r["best_under_odds"]), axis=1)
    print("\n=== totals: always UNDER (client policy default) ===")
    print_stat("FULL 2023-2026", stat(m))
    for s in [2023, 2024, 2025, 2026]:
        print_stat(f"season {s}", stat(m[m.season == s]))

    # home/away team_id per game_pk, via lineups (event_id, team_side, player_id) -> boxscore (player_id -> team_id)
    side_map = con.execute("""
        SELECT g.game_pk, l.team_side, b.team_id, count(*) n
        FROM lineups l
        JOIN client_games g ON l.event_id = g.event_id
        JOIN boxscore b ON b.player_id = l.player_id AND b.game_pk = g.game_pk
        GROUP BY 1,2,3
    """).fetchdf()
    side_map["game_pk"] = pd.to_numeric(side_map["game_pk"], errors="coerce")
    side_map = side_map.sort_values("n", ascending=False).drop_duplicates(subset=["game_pk", "team_side"])
    home_ids = side_map[side_map.team_side == "home"][["game_pk", "team_id"]].rename(columns={"team_id": "home_team_id"})
    away_ids = side_map[side_map.team_side == "away"][["game_pk", "team_id"]].rename(columns={"team_id": "away_team_id"})
    team_scores = team_runs.merge(home_ids, on="game_pk").merge(away_ids, on="game_pk")

    game_res = team_scores[team_scores.team_id == team_scores.home_team_id][["game_pk", "team_runs"]].rename(columns={"team_runs": "home_runs_"})
    away_res = team_scores[team_scores.team_id == team_scores.away_team_id][["game_pk", "team_runs"]].rename(columns={"team_runs": "away_runs_"})
    game_res = game_res.merge(away_res, on="game_pk")

    m = odds[odds.market_key == "h2h__away"].merge(game_res, on="game_pk", how="inner")
    m["win"] = m["away_runs_"] > m["home_runs_"]
    m["profit"] = m.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)
    print("\n=== h2h: always AWAY moneyline (client policy default) ===")
    print_stat("FULL 2023-2026", stat(m))
    for s in [2023, 2024, 2025, 2026]:
        print_stat(f"season {s}", stat(m[m.season == s]))

    m = odds[odds.market_key == "spreads__away"].merge(game_res, on="game_pk", how="inner")
    m["win"] = (m["away_runs_"] + m["line"]) > m["home_runs_"]
    m["profit"] = m.apply(lambda r: profit(r["win"], r["best_over_odds"]), axis=1)
    print("\n=== spreads: always AWAY (near-miss in Addendum 1, re-checked at full scale) ===")
    print_stat("FULL 2023-2026", stat(m))
    for s in [2023, 2024, 2025, 2026]:
        print_stat(f"season {s}", stat(m[m.season == s]))

    con.close()


if __name__ == "__main__":
    main()
