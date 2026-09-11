"""
Explore whether Elo power-rankings and L10 recent form carry information the
market (closing moneyline) hasn't already priced in -- on TRAIN (2014-2017)
ONLY. Any pattern found here gets validated against TEST (2018-2019), a
true holdout, before being trusted. This mirrors the discipline used
everywhere else in this repo.
"""
import pathlib
import duckdb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")


def implied_prob(ml):
    # American odds -> implied probability (still vig-inflated, not de-vigged)
    return np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT * FROM game_features
        WHERE l10_home_games = 10 AND l10_away_games = 10
          AND home_close_ml IS NOT NULL AND away_close_ml IS NOT NULL
    """).fetchdf()
    con.close()

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["l10_diff"] = df["l10_home_wins"] - df["l10_away_wins"]
    df["market_home_prob_raw"] = implied_prob(df["home_close_ml"])
    df["market_away_prob_raw"] = implied_prob(df["away_close_ml"])
    df["vig"] = df["market_home_prob_raw"] + df["market_away_prob_raw"] - 1
    df["market_home_prob"] = df["market_home_prob_raw"] / (df["market_home_prob_raw"] + df["market_away_prob_raw"])
    df["elo_home_prob"] = 1 / (1 + 10 ** (-(df["elo_diff"]) / 400))

    train = df[df["season"] <= 2017].copy()
    test = df[df["season"] >= 2018].copy()

    print("=" * 90)
    print(f"TRAIN (2014-2017): n={len(train):,}   TEST (2018-2019): n={len(test):,}")
    print("=" * 90)

    print("\n-- correlation with home_win (TRAIN only) --")
    for col in ["elo_diff", "l10_diff", "market_home_prob"]:
        r = np.corrcoef(train[col], train["home_win"])[0, 1]
        print(f"  {col:<22} r={r:+.3f}")

    print("\n-- does Elo disagree with the market? where, and who's right? (TRAIN) --")
    # the "edge" = elo's implied prob minus market's implied prob for the home side
    train["edge"] = train["elo_home_prob"] - train["market_home_prob"]
    for lo, hi in [(-1, -0.15), (-0.15, -0.08), (-0.08, 0.08), (0.08, 0.15), (0.15, 1)]:
        sub = train[(train["edge"] >= lo) & (train["edge"] < hi)]
        if len(sub) < 20:
            continue
        print(f"  edge in [{lo:+.2f},{hi:+.2f})  n={len(sub):<6} actual_home_WR={sub['home_win'].mean()*100:.1f}%  "
              f"avg_market_implied={sub['market_home_prob'].mean()*100:.1f}%  avg_elo_implied={sub['elo_home_prob'].mean()*100:.1f}%")

    print("\n-- L10 hot/cold split, controlling loosely for market price (TRAIN) --")
    for label, cond in [
        ("home L10>=7, market underdog (home_close_ml>0)", (train.l10_home_wins >= 7) & (train.home_close_ml > 0)),
        ("home L10<=3, market favorite (home_close_ml<0)", (train.l10_home_wins <= 3) & (train.home_close_ml < 0)),
        ("away L10>=7, market underdog (away_close_ml>0)", (train.l10_away_wins >= 7) & (train.away_close_ml > 0)),
        ("away L10<=3, market favorite (away_close_ml<0)", (train.l10_away_wins <= 3) & (train.away_close_ml < 0)),
    ]:
        sub = train[cond]
        if len(sub) < 20:
            print(f"  {label:<48} n={len(sub)} (too few)")
            continue
        wr = sub["home_win"].mean() if "home" in label else (1 - sub["home_win"]).mean()
        print(f"  {label:<48} n={len(sub):<6} actual_WR={wr*100:.1f}%")


if __name__ == "__main__":
    main()
