"""
Does point-in-time starting-pitcher quality (rolling RA/9) predict game
outcomes better than team-level Elo did? Same train/test discipline: explore
on TRAIN (first 60% of dates), confirm on TEST (last 40%, true holdout).
"""
import pathlib
import duckdb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")
WINDOWS = [3, 5, 8]


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("SELECT * FROM pitcher_game_features ORDER BY game_date").fetchdf()
    con.close()

    split_idx = int(len(df) * 0.6)
    df["is_train"] = False
    df.iloc[:split_idx, df.columns.get_loc("is_train")] = True

    print("=" * 90)
    print(f"pitcher_game_features: n={len(df):,}  {df['game_date'].min()} -> {df['game_date'].max()}")
    print("=" * 90)

    for w in WINDOWS:
        hcol, acol = f"home_ra9_l{w}", f"away_ra9_l{w}"
        sub = df.dropna(subset=[hcol, acol])
        sub = sub.copy()
        sub["ra9_diff"] = sub[acol] - sub[hcol]  # positive = home pitcher has lower (better) RA/9
        train = sub[sub.is_train]
        test = sub[~sub.is_train]
        if len(train) < 100 or len(test) < 50:
            print(f"\nwindow={w}: too few rows after split (train={len(train)}, test={len(test)}) -- skipped")
            continue
        r_train = np.corrcoef(train["ra9_diff"], train["home_win"])[0, 1]
        r_test = np.corrcoef(test["ra9_diff"], test["home_win"])[0, 1]
        print(f"\nwindow={w} starts (n_train={len(train):,}, n_test={len(test):,}):")
        print(f"  corr(home_win, ra9_diff) TRAIN={r_train:+.3f}   TEST={r_test:+.3f}")

        # decile check on TRAIN: does home win rate move monotonically with ra9_diff?
        train = train.copy()
        train["decile"] = pd.qcut(train["ra9_diff"], 5, duplicates="drop")
        print("  TRAIN home-win-rate by ra9_diff quintile (should rise left->right if real):")
        for d, grp in train.groupby("decile", observed=True):
            print(f"    {str(d):<22} n={len(grp):<5} home_WR={grp['home_win'].mean()*100:.1f}%")


if __name__ == "__main__":
    import pandas as pd
    main()
