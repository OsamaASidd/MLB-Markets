# Addendum 35: forward odds collection (API-Sports.io) — pre-registered plan

Written before any meaningful volume has accumulated, specifically so the
method can't be adjusted later once results start coming in — same
discipline as every other addendum in this project (32, 34) that locked a
spec before running it.

## What's being collected, and why

`scripts/collect_api_sports_odds.py` (Windows Scheduled Task
`MLB_APISports_OddsCollector`, daily 9:00 AM) pulls real MLB odds from
API-Sports.io — including player props (`Pitcher Outs`, `Pitcher
Strikeouts`, `Player Runs`, `Player RBIs`, `Player Hits`, `Player Total
Bases`, `Player Home Runs`) — into a separate local warehouse
(`db/api_sports_odds.duckdb`, git-ignored).

**Confirmed empirically (not assumed): this source has zero historical
odds before ~2026-09-14.** Every pull from today forward is genuinely new
real data, not a re-slice of something already tested. See the session
transcript / `reports/gate_results.csv` history for the exhaustive checks
(per-game, per-date, whole-season, unfiltered) that established this.

## The pre-registered test — locked now, applied once enough volume exists

When this data is tested against the FAIL markets, it will follow exactly
this spec, decided now:

- **Scope**: the markets this source actually has props for —
  `pitcher_outs`, `pitcher_strikeouts`, `batter_runs_scored`, `rbis`,
  `hits`, `total_bases`, `home_runs` (subject to which props the
  bookmakers actually post live — not every market gets a line every day).
- **Combining old + new data**: outcomes for API-Sports games will be
  joined to the existing real `boxscore` table (this API has no
  player-level box-score endpoint of its own — confirmed, see session
  notes) by team + date, same matching approach already used elsewhere in
  this project (Addendum 7). Only genuinely graded (finished, matched)
  picks count toward `n`.
- **Model**: XGBoost, this project's already-established baseline —
  chosen once, not selected per-market based on which one happens to look
  best on this new data.
- **Split**: chronological, once enough calendar time has passed that a
  real train/test split is meaningful (not before — a same-week
  train/test split on accumulating-in-real-time data isn't a real holdout).
- **Cut**: `edge>0.0`, no threshold search, same as every other addendum.
- **Decision rule**: the official client gate rule (n≥500 → ROI>0; else
  95% CI lower bound > 0), applied once per market, reported honestly —
  pass or fail, whichever it is.

## What "reported honestly" means here specifically

This test will not be run repeatedly, checked, and re-run if the current
answer isn't a pass — it runs once real volume exists, and whatever
verdict comes out is the addendum. If nothing clears the gate, that's
written up the same way Addenda 32-34 wrote up their non-passing results:
explicitly, with the real numbers, not quietly dropped.

## Realistic timeline

n≥500 graded picks for a specific market requires real games with that
prop actually posted and graded. Based on one day's pull (20 games, a
handful of `Pitcher Outs`/`Player Runs` lines among them), reaching n≥500
on the thinner props realistically takes **weeks of daily collection at
minimum**, longer for narrower props. This is not a shortcut to a faster
answer than the rest of this project's live-volume-accumulation findings
(Addendum 1's strikeouts near-miss, Addendum 9's pitcher_outs) — it's the
same kind of real evidence, just from a second live source going forward.
