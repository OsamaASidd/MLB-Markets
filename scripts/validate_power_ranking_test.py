import pathlib
import duckdb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")


def implied_prob(ml):
    return np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT * FROM game_features
        WHERE l10_home_games = 10 AND l10_away_games = 10
          AND home_close_ml IS NOT NULL AND away_close_ml IS NOT NULL
          AND home_open_ml IS NOT NULL AND away_open_ml IS NOT NULL
    """).fetchdf()
    con.close()

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["l10_diff"] = df["l10_home_wins"] - df["l10_away_wins"]
    m_home = implied_prob(df["home_close_ml"]); m_away = implied_prob(df["away_close_ml"])
    df["market_home_prob"] = m_home / (m_home + m_away)
    o_home = implied_prob(df["home_open_ml"]); o_away = implied_prob(df["away_open_ml"])
    df["open_home_prob"] = o_home / (o_home + o_away)
    df["elo_home_prob"] = 1 / (1 + 10 ** (-(df["elo_diff"]) / 400))

    for label, part in [("TRAIN 2014-2017", df[df.season <= 2017]), ("TEST 2018-2019", df[df.season >= 2018])]:
        print(f"\n=== {label}  n={len(part):,} ===")
        for col, name in [("elo_diff", "elo_diff"), ("l10_diff", "l10_diff"),
                           ("market_home_prob", "closing market prob"), ("open_home_prob", "opening market prob")]:
            r = np.corrcoef(part[col], part["home_win"])[0, 1]
            print(f"  corr(home_win, {name:<22}) = {r:+.3f}")

        # elo vs OPENING line (less efficient than close -- best chance for elo to add value)
        part = part.copy()
        part["edge_open"] = part["elo_home_prob"] - part["open_home_prob"]
        for lo, hi in [(0.10, 1.0), (-1.0, -0.10)]:
            sub = part[(part["edge_open"] >= lo) & (part["edge_open"] < hi)] if lo > 0 else part[(part["edge_open"] >= lo) & (part["edge_open"] < hi)]
            if len(sub) < 20:
                continue
            side = "home" if lo > 0 else "away"
            wr = sub["home_win"].mean() if side == "home" else (1 - sub["home_win"]).mean()
            avg_odds_col = "home_close_ml" if side == "home" else "away_close_ml"
            avg_odds = sub[avg_odds_col].mean()
            print(f"  elo edge vs OPEN {lo:+.2f}+ -> bet {side:<5} n={len(sub):<6} actual_WR={wr*100:.1f}%  avg_close_odds={avg_odds:.0f}")


if __name__ == "__main__":
    main()
