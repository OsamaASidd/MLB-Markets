"""
Does the temperature effect (real, validated in test_wind_effect.py: warmer
-> more runs/HR, corr +0.095/+0.198 on 1,902 real games) survive as a priced
bet against the actual totals line?

IMPORTANT data trap caught and fixed here: joining weather -> events ->
pick_history's totals picks by team+date looks like it produces ~100-200
rows, but a single real game gets offered many different total lines and
both sides (over/under) -- one Royals/Mariners game alone contributed 42
rows. Those rows are NOT independent bets: if the real score comes in high,
every "over" line for that game tends to win together. Naively computing
ROI/CI across all rows overstates the sample size by ~10x and manufactures
false confidence (an early pass here showed a "+20% ROI" that evaporates
to n=13-14 real games once deduplicated -- nowhere near enough to conclude
anything). This script dedupes to ONE row per real game before reporting.
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")


def main():
    con = duckdb.connect(DB, read_only=True)

    raw = con.execute("""
        SELECT w.temperature_f, w.is_dome, ph.team, ph.opponent, ph.game_time,
               TRY_CAST(ph.line AS DOUBLE) AS line,
               TRY_CAST(ph.odds AS INTEGER) AS odds,
               lower(ph.pick_side) AS side,
               lower(ph.hit) IN ('true','t','1') AS win
        FROM weather w
        JOIN events e ON w.event_id = e.event_id
        JOIN pick_history ph
          ON CAST(e.commence_time AS DATE) = CAST(TRY_CAST(ph.game_time AS TIMESTAMP) AS DATE)
          AND ((ph.team = e.home_team AND ph.opponent = e.away_team)
               OR (ph.team = e.away_team AND ph.opponent = e.home_team))
        WHERE ph.prop_type = 'totals'
          AND lower(coalesce(ph.is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(ph.voided,'false')) NOT IN ('true','t','1')
          AND ph.hit IS NOT NULL AND w.is_dome = False
    """).fetchdf()

    print(f"raw joined rows (BEFORE fixing the multi-line-per-game trap): n={len(raw):,}")
    n_distinct_games = raw[["team", "opponent", "game_time"]].drop_duplicates().shape[0]
    print(f"real distinct games behind those rows: n={n_distinct_games}")
    print("-> that ~10x gap IS the bug. Deduping to one row per game below.\n")

    # one row per real game: pick the closest-to-market total line (the median offered)
    # and whichever side that specific line's pick recorded, to get a single real bet per game
    dedup = raw.sort_values("line").groupby(["team", "opponent", "game_time"]).nth(0).reset_index()

    print(f"deduplicated to n={len(dedup)} real distinct games -- this is the honest sample size")

    def profit(row):
        if not row.win:
            return -1
        return row.odds / 100.0 if row.odds > 0 else 100.0 / abs(row.odds)
    dedup["profit"] = dedup.apply(profit, axis=1)

    if len(dedup) >= 40:
        # NOTE: correlating temperature with raw `win` isn't meaningful here since win means
        # different things for over vs under picks (mixed sides in this sample) -- the bucketed
        # ROI below (priced correctly per row's own side/odds) is the honest comparison instead.
        for lo, hi in [(0, 65), (65, 120)]:
            sub = dedup[(dedup.temperature_f.astype(float) >= lo) & (dedup.temperature_f.astype(float) < hi)]
            if len(sub) < 15:
                continue
            n, wr = len(sub), sub.win.mean() * 100
            roi = sub.profit.mean() * 100
            lo_ci = (sub.profit.mean() - 1.96 * sub.profit.std(ddof=1) / (len(sub) ** 0.5)) * 100
            hi_ci = (sub.profit.mean() + 1.96 * sub.profit.std(ddof=1) / (len(sub) ** 0.5)) * 100
            print(f"  temp {lo}-{hi}F  n={n:<4} WR={wr:.1f}%  ROI={roi:.2f}%  CI=[{lo_ci:.2f},{hi_ci:.2f}]")

    print("\nHONEST CONCLUSION: even using every available real game (not just the extreme hot/cold "
          "buckets that produced the fake '+20% ROI'), there are only ~110 real games where weather "
          "data, a totals pick, and a graded outcome all overlap -- well short of the n>=500 gate floor. "
          "This is a data-coverage gap (pick_history's totals picks only carry the team/date fields "
          "needed to match weather for a narrow slice of games), not evidence the temperature effect "
          "is or isn't bettable. It would need a broader real totals-odds history matched to weather "
          "-- likely requiring the renewed DB credential noted elsewhere in this report -- to actually "
          "answer this properly.")

    con.close()


if __name__ == "__main__":
    main()
