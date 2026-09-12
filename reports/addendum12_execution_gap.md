# Addendum 12: Why production underperforms the best-price baseline — execution gap vs. selection gap

**Question:** Addendum 9 showed `hits->under` and `total_bases->under` are flat
at the theoretical best available price (best of 56 books). Addendum 1 showed
the same two markets run at −4% to −7% ROI in real production. This addendum
matches individual production picks to the real closing-odds warehouse,
player-by-player, game-by-game, line-by-line, to find out why.

**Verdict up front: this is overwhelmingly an EXECUTION gap, not a selection
gap** — with an honest caveat that the matched sample is small and confined to
an 8-day window, so it should be read as strong directional evidence, not a
final number.

## Match rate

Production's real graded `under`-side picks for `hits`/`total_bases`
(`mlb_market_type IS NOT NULL`, `is_synthetic=False`, `voided=False`, `hit` and
`odds` populated) total **14,165 rows**, spanning **2026-05-17 to 2026-07-27**.
The client's closing-odds warehouse (`client_closing_odds`) only has 2026 data
through **2026-05-24** — an 8-day overlap with production, not the full
10-week production window. `game_pk` is populated on only 29% of production
rows, so the match key used was normalized player name (NFKD, accent-stripped,
lowercased — same pattern as `scripts/grade_client_odds_warehouse.py`) + exact
`game_date` + exact `line` + market.

- Production rows falling on any date the odds warehouse also covers: **1,181
  / 14,165 (8.3%)**.
- Of those, uniquely matched to a specific player/game/line row in the odds
  warehouse: **960 (81.3% of the 1,181 in-window rows)**.
- Overall match rate against the full 14,165: **6.8%**.

**The low overall match rate is a calendar-coverage limitation, not a matching
failure.** Within the days both datasets actually cover, the name/date/line
match succeeds 81% of the time — the normalization approach works. The
ceiling is set by the odds warehouse's 2026 backfill stopping in May while
production kept logging picks through July. Production ROI in the overlap
window itself (n=1,181, ROI −7.74%, CI [−12.63%, −2.85%]) is directionally
consistent with the full-period number (n=14,165, ROI −4.32%, CI [−5.72%,
−2.91%]) — worse, in fact, not better — so the window is not an easy/atypical
slice; if anything it's a harder one.

Of the 960 matched picks, 696 had a real `best_under_odds` quote in the market
for that exact player/game/line (the rest were lines thin enough that no book
in the warehouse posted an under price at all — itself informative: a
meaningful share of production's picks are on lines with little to no real
"under" liquidity).

## Test A: price gap (holding selection constant)

On the 696 cleanly matched picks — same player, same game, same line, same
win/loss outcome, only the price changes:

| | avg odds | avg implied prob | ROI | 95% CI |
|---|---:|---:|---:|---|
| **Production's actual price** | −246 | 58.7% | **−16.69%** | [−23.29%, −10.09%] |
| **Best available price (same spot)** | +24 | 49.1% | **+2.61%** | [−5.34%, +10.55%] |

That's a ~19-point ROI swing from price alone, with win/loss held fixed. Split
by market:

| Market | n | ROI @ production price | ROI @ best available price |
|---|---:|---:|---:|
| hits | 401 | −18.63% | +4.89% |
| total_bases | 295 | −14.04% | −0.50% |

Both markets flip from clearly losing to flat-or-better once the price is
swapped for the best available quote on the identical bet. `total_bases` at
best price (−0.50%, CI crossing zero) lines up with Addendum 9's full-scale
finding (−0.2%). `hits` at best price here (+4.89%, CI [−6.40%, +16.18%])
is noisier and shouldn't be over-read on n=401 from one week, but it is not
the clear loser production's actual price shows.

**How bad is the price production is getting?** 542 of 696 matched picks
(77.9%) got a strictly worse price than the best one available in the market
for that exact bet; only 74 (10.6%) got the identical price, and 80 (11.5%)
got a better price (market moved in production's favor, or the warehouse
snapshot is slightly stale). The gap is not a rounding issue — production's
average odds on these bets is −246 (implied 58.7% probability) against a real
best-available average of +24 (implied 49.1%), a ~10-point implied-probability
gap on average, and some individual picks are off by hundreds of points (e.g.
one matched pick logged at −3000 against a market best of +145 on the same
player/game/line). None of the matched rows carry a populated `bookmaker`
value, so which specific book or feed produced these captured prices can't be
traced from this data — that itself is a gap worth closing operationally.

## Test B: does confidence-based selection make things worse?

On the same 696-row matched set, confidence correlates **positively**, not
adversely, with outcome quality:

- corr(confidence, win) = **+0.11**
- corr(confidence, profit at best-available price) = **+0.05** (weak, but the
  right sign)
- By confidence bucket, ROI-at-best-price rises monotonically: −4.6% (conf
  ≤50, n=388) → +2.4% (50–60, n=186) → +24.5% (60–70, n=109) → +35.5% (70+,
  n=13).
- corr(confidence, the price-shopping gap itself) = **−0.05**, essentially
  zero — meaning higher-confidence picks are not specifically the ones with
  worse execution; the price problem is spread roughly evenly across
  confidence tiers, not concentrated where selection is most active.

If confidence-based selection were actively picking the worst-value spots
(hypothesis B), higher confidence would correlate with *worse* outcomes at
best price, or with *larger* price gaps. Neither shows up. The signal, weak as
it is on this sample, points the normal direction: higher confidence, better
result. The small n at the top bucket (13) means this shouldn't be oversold,
but there is no evidence here of an adverse selection effect.

## Honest verdict

**(A) Execution gap: strongly supported.** Holding the exact same picks
(same player, game, line, win/loss) fixed and swapping only the price flips
ROI from −16.7% to +2.6% — a ~19-point swing that, on its own, is large enough
to explain the entire −4% to −7% gap between production and the best-price
baseline, and then some. 78% of matched picks got a measurably worse price
than what was available in the market at the time.

**(B) Selection gap: not supported.** Confidence correlates positively with
both win rate and best-price ROI on the matched subset, and is uncorrelated
with how bad the execution price is. Nothing here suggests the
confidence-based filter is anti-selecting good bets.

**Caveat, stated plainly:** the matched sample (696–960 rows) is a small
fraction (6.8%) of the full 14,165-row production population, confined to an
8-day window in May 2026 because that's where the odds warehouse and
production history currently overlap. The within-window match rate (81%) and
the window's own ROI (directionally consistent with, in fact worse than, the
full-period number) both suggest this slice is not a cherry-picked easy week,
but a single 8-day window is not a substitute for a full-period audit. The
conclusion — execution, not selection, drives the gap — is well-supported by
the size and direction of the effect, but should be re-confirmed once a
longer-overlap odds backfill (extending the 2026 warehouse past May, or
backfilling further into production's window) is available.

## Recommendation

Extend the `client_closing_odds` 2026 backfill to cover production's full
active window (through at least 2026-07-27), then re-run this same match at
full scale. In parallel, instrument `pick_history` to log which bookmaker/feed
supplied the captured `odds` value for hits/total_bases picks (currently 0%
populated in the matched sample) — that single fix would make the next
version of this audit traceable to a root cause (stale feed, single thin
book, wrong-side mapping) instead of only quantifying the size of the gap.
