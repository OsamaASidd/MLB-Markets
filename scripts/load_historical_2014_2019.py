"""
Load the free, public 2014-2019 MLB historical Vegas lines dataset
(pwu97/bettingtools on GitHub, originally sourced from
sportsbookreviewsonline.com's public score/odds archives, no signup or
payment required) into db/mlb_markets.duckdb as `historical_game_odds`.

This gives a genuine 6-year real sample for game-level markets (moneyline,
run line, totals) to complement the client's own 2023-2026 warehouse data --
closing the gap toward a multi-year sample without paying for a vendor
whose own historical archive doesn't go back before 2023 anyway (see
milestone report Addendum 3 investigation).
"""
import pathlib
import duckdb
import pandas as pd
import pyreadr

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
DB = str(ROOT / "db" / "mlb_markets.duckdb")

YEARS = [2014, 2015, 2016, 2017, 2018, 2019]


def main():
    frames = []
    for y in YEARS:
        result = pyreadr.read_r(str(RAW / f"mlb_odds_{y}.rda"))
        key = list(result.keys())[0]
        df = result[key]
        df["season"] = y
        frames.append(df)
        print(f"{y}: {len(df):,} rows, columns={list(df.columns)}")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\nTOTAL: {len(all_df):,} rows, {all_df['date'].min()} -> {all_df['date'].max()}")

    con = duckdb.connect(DB)
    con.execute("DROP TABLE IF EXISTS historical_game_odds")
    con.register("all_df", all_df)
    con.execute("CREATE TABLE historical_game_odds AS SELECT * FROM all_df")
    con.unregister("all_df")
    con.execute("CHECKPOINT")
    n = con.execute("SELECT count(*) FROM historical_game_odds").fetchone()[0]
    print(f"loaded into duckdb: {n:,} rows")
    con.close()


if __name__ == "__main__":
    main()
