"""
Test the most commonly cited real MLB weather angle: wind blowing out
correlates with more runs/home runs. Real weather data (temp, wind
speed/direction, dome flag) joined via event_id -> events.game_pk ->
boxscore-derived real final scores. No odds needed for step 1 -- this
first checks whether the underlying baseball fact is even true in this
real 2023-2026 sample, before asking whether the betting market has
already priced it in (research says it typically has, by closing time).
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")


def main():
    con = duckdb.connect(DB, read_only=True)

    df = con.execute("""
        WITH team_totals AS (
            SELECT game_pk, sum(runs_scored) AS total_runs, sum(home_runs) AS total_hrs
            FROM boxscore
            WHERE position_type IN ('Catcher', 'Hitter', 'Infielder', 'Outfielder')
            GROUP BY 1
            HAVING sum(runs_scored) IS NOT NULL
        )
        SELECT w.event_id, e.game_pk, w.is_dome, w.temperature_f, w.wind_speed_mph,
               w.wind_direction_degrees, t.total_runs, t.total_hrs
        FROM weather w
        JOIN events e ON w.event_id = e.event_id
        JOIN team_totals t ON e.game_pk = t.game_pk
        WHERE w.is_dome = False
    """).fetchdf()

    print(f"real non-dome games with weather + real final score: n={len(df):,}")

    df["wind_speed_mph"] = df["wind_speed_mph"].astype(float)
    df["temperature_f"] = df["temperature_f"].astype(float)

    print("\n-- total runs by wind speed bucket --")
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 100)]:
        sub = df[(df.wind_speed_mph >= lo) & (df.wind_speed_mph < hi)]
        if len(sub) < 20:
            continue
        print(f"  wind {lo}-{hi} mph   n={len(sub):<5} avg_total_runs={sub.total_runs.mean():.2f}  avg_HR={sub.total_hrs.mean():.2f}")

    print("\n-- total runs by temperature bucket --")
    for lo, hi in [(0, 50), (50, 65), (65, 80), (80, 120)]:
        sub = df[(df.temperature_f >= lo) & (df.temperature_f < hi)]
        if len(sub) < 20:
            continue
        print(f"  temp {lo}-{hi}F      n={len(sub):<5} avg_total_runs={sub.total_runs.mean():.2f}  avg_HR={sub.total_hrs.mean():.2f}")

    import numpy as np
    print("\n-- correlations --")
    print(f"  corr(wind_speed, total_runs) = {np.corrcoef(df.wind_speed_mph.fillna(0), df.total_runs)[0,1]:+.3f}")
    print(f"  corr(temperature, total_runs) = {np.corrcoef(df.temperature_f.dropna(), df.loc[df.temperature_f.notna(),'total_runs'])[0,1]:+.3f}")
    print(f"  corr(wind_speed, total_HR)    = {np.corrcoef(df.wind_speed_mph.fillna(0), df.total_hrs)[0,1]:+.3f}")
    print(f"  corr(temperature, total_HR)   = {np.corrcoef(df.temperature_f.dropna(), df.loc[df.temperature_f.notna(),'total_hrs'])[0,1]:+.3f}")

    con.close()


if __name__ == "__main__":
    main()
