# Milestone 1 — MLB Market Gate Audit

**Scope:** Run every MLB market through the gate on the full real sample, give a
straight PASS/FAIL per market with the exact cause and a practical improvement
lever for each failure.

**Data:** real production `pick_history` (the picks the live system actually made
and graded), 74,385 graded MLB picks, **2026-05-17 → 2026-07-27**. Not synthetic,
not a hypothetical backtest over every priced line — this is what the deployed
system actually recommended and how those recommendations actually resolved.

**Gate rule** (client spec): **PASS iff n ≥ 500 graded picks AND the 95% ROI
confidence interval's lower bound is > 0.** Below 500 picks, no result can PASS
outright even with a positive point estimate — there isn't enough evidence yet.

**Where everything lives:** `MLB Markets/db/mlb_markets.duckdb` (38 MB, real data,
committed to this repo), `MLB Markets/scripts/{gate.py,diagnose.py}` (rerunnable),
`MLB Markets/reports/gate_results.csv` (raw numbers behind every row below).

---

## Headline result

**All 10 markets FAIL the gate on the full sample.** Zero clean passes.

This includes **batter_hits** and **total_bases-unders**, the two markets a prior
harness-based backtest (against the raw historical odds warehouse, not live
production picks) had declared PASSED. On real production data they don't hold:

| Market | Prior harness verdict | Real production result |
|---|---|---|
| batter_hits | PASS | **FAIL** — ROI −4.1% to −6.1% across every cut, n=2,508–15,005 |
| total_bases (unders) | PASS | **FAIL** — ROI −2.3% to −4.4%, n=5,255–8,376 |

This is the single most important finding in this audit: **the harness backtest
and live production performance disagree**, and production is what actually pays
or loses money. See "Why the harness said PASS and production says FAIL" below —
this is a lever, not a dead end.

One market — **pitcher_strikeouts** — is a genuine near-miss, not a real fail:
+3.34% ROI on its best cut, positive and walk-forward-stable, just short on
sample size (n=477 vs 500 needed). Full detail below.

---

## Per-market results (full sample, `all` cut — the actual ask for this milestone)

| Market | n | Win% | ROI% | 95% CI | CLV avg | Gate |
|---|---:|---:|---:|---|---:|---|
| pitcher_strikeouts | 1,949 | 50.6 | −6.08 | [−10.31, −1.86] | +1.59 | FAIL |
| hits | 15,005 | 54.2 | −5.36 | [−7.85, −2.87] | +1.38 | FAIL |
| total_bases | 14,351 | 54.4 | −7.08 | [−8.57, −5.59] | +0.94 | FAIL |
| rbis | 11,576 | 56.9 | −6.53 | [−8.25, −4.81] | +1.41 | FAIL |
| home_runs | 14,604 | 76.1 | −11.53 | [−13.08, −9.98] | +0.81 | FAIL |
| runs_scored | 6,859 | 56.5 | −3.79 | [−5.90, −1.69] | +0.68 | FAIL |
| spreads | 3,367 | 50.0 | −5.66 | [−9.08, −2.24] | +0.83 | FAIL |
| h2h (moneyline) | 1,670 | 50.0 | −6.07 | [−11.00, −1.15] | +0.49 | FAIL |
| totals | 4,204 | 50.3 | −5.74 | [−8.62, −2.87] | +0.58 | FAIL |
| pitcher_outs | 798 | 49.5 | −6.37 | [−13.03, +0.30] | +1.28 | FAIL |

Best deployable cut per market (`conf≥60, minus-money` — the actual "recommendable
pick" filter used in production) is in `gate_results.csv`; none clear the gate
either, though several (strikeouts, spreads, h2h, pitcher_outs) get close on point
estimate while failing on sample size or CI width. Full numbers, all three cuts,
all 10 markets: `reports/gate_results.csv`.

---

## Per-market cause and improvement lever

### pitcher_strikeouts — FAIL (near-miss, not a dead market)
- **Cause:** the `conf≥60, minus-money` cut is genuinely positive (+3.34% ROI,
  61% win rate, n=477) and **walk-forward stable** (train +3.6%, test +2.9% on
  unseen dates) — this is a real signal, not overfitting. It fails the gate
  purely on sample size: the 95% CI is [−4.1%, +10.8%], which can't yet separate
  from zero at n=477. Plus-money strikeout picks are a separate, confirmed loser
  (−11.8% ROI) and should be dropped outright regardless of what happens with the
  minus-money side.
- **Lever:** keep shipping this cut as-is (minus-money, conf≥60 only) and let it
  accumulate. At the current +3.3% rate, roughly ~600 graded picks would push the
  CI lower bound above zero — about 5-6 more weeks of live volume at the current
  ~90 picks/week pace, or faster if the pitch-mix feature work already in progress
  tightens the win rate. Do not change the selection rule while waiting; changing
  it now would reset the sample-size clock.

### hits — FAIL (was declared PASS previously — does not hold in production)
- **Cause:** production picks skew toward short-priced favorites (avg odds around
  −55 on the `over` side, which carries 61% of volume) at win rates too close to
  what the price already implies. Plus-money hits picks are the worst slice
  (36.2% win rate, −8.22% ROI) but even the safest minus-money bucket (`≤−250`,
  72.6% win rate) still loses −4.81% — the vig on short prices is eating a
  real-but-thin edge the same way it does for strikeouts. Confidence is also not
  cleanly monotonic against actual win rate in the 30-55 range (e.g. conf 41-50
  band wins less often than the conf 38-47 band below it) — the score is noisy in
  exactly the range most picks fall into.
- **Lever:** (1) drop plus-money hits picks entirely — same fix as strikeouts.
  (2) Investigate why this market passed the harness's historical-odds backtest
  but fails on live picks — check whether the harness proxy is using a wider or
  more favorable set of hypothetical lines than what the live system actually
  gets filled at (entry-price slippage), since that gap is large enough (harness
  PASS vs −5% live) to suggest a real discrepancy between backtest and
  production odds capture, not just noise.

### total_bases — FAIL (unders specifically were declared PASS previously — does not hold)
- **Cause:** the `under` side (63.7% win rate, avg odds −224, 58% of volume) is
  the side that was previously reported passing; on the full production sample it
  is −4.38% ROI (conf≥60 cut: −2.31%, CI [−4.22%, −0.41%] — still clearly
  negative, not just wide). The `over` side is worse (41.4% win rate at plus
  average odds, −10.08% ROI in the plus-money bucket). Same vig-vs-thin-edge
  pattern as hits.
- **Lever:** same as hits — this needs the harness-vs-production discrepancy
  investigated before any further model work, since the model apparently was
  never actually earning what the backtest said it would on this market either.

### rbis — FAIL
- **Cause:** heavily skewed to the `under` side (64% of volume, avg odds −313,
  71.5% win rate) — a short-price "safe" bet that still loses money (−3.85% ROI
  even at the safest odds bucket, ≤−250) because the win rate doesn't clear what
  that price demands. The `over` (plus-money) side is far worse: 30.9% win rate,
  −12.04% ROI, the single worst bucket in this market.
- **Lever:** drop plus-money RBI overs outright (−12% ROI, no ambiguity). On the
  under side, the fix has to be pricing discipline, not projection accuracy — the
  model is picking the right side often enough (71-74% win rate) but at odds too
  short to pay for that accuracy; needs a minimum-odds floor before a minus-money
  RBI under is recommended at all.

### home_runs — FAIL (worst market in the sample)
- **Cause, verified at the row level, and it's a real anomaly worth flagging on
  its own:** the `over` (batter hits a HR) side is priced at a median of
  **+19,900** for 792 of its 2,108 picks — essentially lottery-ticket odds — and
  went **0-for-792** on that bucket. Only the shorter plus-money buckets
  (+200 to +799) show any hit rate at all (14-17%), and even those aren't quite
  profitable once the vig is applied. The `under` (no-HR) side is priced at an
  average of **−3,240** (implied ~97% win probability) but only wins 87.9% of
  the time — a 9-point calibration gap against the market's own price, which is
  why even the "safe" side loses −4.51% to −4.49% ROI on its own.
- **Lever:** this is the clearest, most mechanical fix in the whole audit. (1)
  Hard-cap the odds range this market is allowed to fire in — no `over` picks
  beyond roughly +800 (win rate craters to near-zero past that and there is no
  price that makes a 0%-hit-rate bucket profitable), and no `under` picks shorter
  than roughly −500 (the vig at −3,240 is unbeatable even at a near-perfect win
  rate). (2) Separately, investigate why the projection is generating "recommend"
  verdicts on batters at <1% true home-run probability in the first place — that
  is a projection-calibration bug, not just a pricing-filter gap.

### runs_scored — FAIL, closest to break-even of the non-strikeout markets
- **Cause:** minus-money cut is −1.9% ROI with a CI of [−4.02%, +0.23%] — the
  only market besides strikeouts whose CI upper bound sits close to zero from a
  small negative point estimate. Side mix is similar to RBIs (under-heavy,
  63-72% win rate at short-to-medium prices) without RBIs' extreme plus-money
  blowup. This market also has known data-quality gaps: it runs in
  `pick_history`-only mode because the odds warehouse has no coverage for it, and
  roughly a third of raw candidates get excluded for missing/unresolved event
  matches (per the July harness run notes).
- **Lever:** tighten to minus-money only (drop the −8.89% plus-money bucket) and
  re-check at conf≥55 instead of 60 — the 60 cut actually made the point estimate
  worse here (−3.77% vs −1.9% at no confidence floor), suggesting the confidence
  score is misordered for this market specifically and a different threshold or
  a market-specific recalibration is worth testing before writing this one off.

### spreads — FAIL, worth another look
- **Cause:** full sample −5.66% ROI, but the `conf≥60 minus-money` cut is
  actually **+5.53%** (65.3% win rate, n=513) — it clears the 500-pick floor but
  the CI [−1.33%, +12.39%] still crosses zero, so it's a real fail, not a pass
  by a technicality. It is the single most promising cut of any market outside
  strikeouts.
- **Lever:** this is a "needs more evidence" case exactly like strikeouts, not a
  "structurally broken" case. Keep shipping this exact cut, do not touch the
  selection logic, and let it accumulate; walk-forward already shows TRAIN +5.2%
  / TEST +6.0%, which is directionally consistent and worth continuing to track.

### h2h (moneyline) — FAIL, same shape as spreads
- **Cause:** `conf≥60 minus-money` is +1.94% ROI (62% win rate, n=490 — just
  under the 500 floor) with CI [−5.36%, +9.25%]. Notably this is the only market
  with **negative CLV** on that cut (−2.49%), meaning the closing line moves
  against these picks more often than not, which is a weaker signal than
  strikeouts' or spreads' positive CLV.
- **Lever:** same "let it accumulate" path as spreads, but flag the negative CLV
  for a second look — pair it with the pick's model edge to see whether it's
  concentrated in a specific team/park subset before assuming it's just noise.

### totals — FAIL
- **Cause:** flat across every cut (−1.69% to −5.74%), no cut approaches
  positive with a meaningful CI. No single obvious driver (side mix is close to
  even, odds distribution isn't as skewed as home_runs or rbis) — this reads as
  a market where the model simply doesn't have a real edge yet, rather than a
  pricing-filter problem.
- **Lever:** lowest priority for further tuning; the data doesn't point to an
  obvious lever the way home_runs (odds cap) or hits (harness/production gap)
  do. Worth revisiting only after the higher-signal markets are resolved.

### pitcher_outs — FAIL, data-starved
- **Cause:** n=798 total, only 176 at the `conf≥60 minus-money` cut — even
  though that cut shows +4.4% ROI, the CI is enormous [−8.25%, +17.06%] because
  the sample is too small to say anything with confidence. This market runs in
  `pick_history`-only mode (no odds-warehouse coverage), same provider gap as
  runs_scored.
- **Lever:** this market simply needs more live volume before it can be judged
  at all — no model or pricing change will fix a sample-size problem. Keep it
  running at its current selection rule and revisit at the next milestone once
  n roughly triples.

---

## Why the harness said PASS and production says FAIL

This is the one finding that applies across markets, not just to hits and
total_bases: **a backtest that scores every line the book ever offered is not
the same test as scoring the picks the live system actually made.** The
`STRIKEOUTS_VERDICT.md` writeup already flagged this exact gap for strikeouts
(warehouse test: −9% ROI hard fail; production-picks test: +3.3% near-pass) —
this audit shows the gap can run in *either* direction: strikeouts looks better
in production than in the historical-odds backtest, hits and total_bases look
much worse. That inconsistency, not any single market's number, is the strongest
argument for treating `pick_history` (what was actually picked and graded) as the
canonical gate input going forward, not the historical-odds-warehouse replay.

**Concrete follow-up for Milestone 2:** diff the confidence-scoring code path
that ran live in May-July 2026 against whatever code path the harness's
historical-odds backtest used for the same markets. If they materially disagree
on hits and total_bases, that's the actual bug to fix before any further
projection work on those two markets.

---

## Data-quality notes (not gate failures, but worth flagging)

- **CLV capture is 58-64% across markets**, not the ~92% referenced in prior
  client conversations — that figure appears to be a blended all-sport number,
  not MLB-specific. Every ROI/CLV figure in this report is computed only from
  rows where CLV/odds/hit are all present (see `gate.py`'s `ph` view filter) —
  gaps don't inflate or bias any number above, but they do mean roughly 4 in 10
  picks per market can't be scored for CLV at all.
- **A `strikeouts` prop_type with n=2 exists in `pick_history` alongside
  `pitcher_strikeouts` (n=1,949)** — this is a data-entry artifact (2 stray
  rows, both losses), not a real distinct market. Excluded from this audit;
  worth a one-line fix in whatever writes `prop_type` to stop it recurring.
- **Local odds-warehouse mirror is incomplete** (`cache_mlb_historical_odds` —
  only 3 weeks of May 2023 pulled before an earlier extraction stalled) and was
  intentionally **not** rebuilt for this milestone: the gate only needs
  `pick_history`, which already carries the entry odds for every graded pick.
  Rebuilding the full warehouse mirror (millions of rows) would be needed only
  for the harness-vs-production reconciliation recommended above.
- **The read-only DB credential used to build this mirror
  (`harness_readonly.gzuzuqxvfjszlfclhcfz`, last used 2026-07-29) has since
  stopped authenticating** — confirmed by a direct connection test during this
  audit. This report's sample therefore ends 2026-07-27, about six weeks behind
  today (2026-09-11). A refreshed read-only credential is needed before the next
  milestone to bring the sample current; `scripts/pull_pick_history.py` in this
  repo is ready to run as soon as one is available.

---

## What's in this repo

```
MLB Markets/
  db/mlb_markets.duckdb          real data: pick_history, boxscore, lineups,
                                  weather, opposing_pitcher, player_metadata,
                                  ballpark_factors, algorithm_weights,
                                  statcast xstats/exit-velo (38 MB)
  scripts/pull_pick_history.py   refresh pick_history from the warehouse
                                  (needs a valid DB_PASSWORD in .env)
  scripts/compact_db.py          rebuild the compact db from a raw mirror
  scripts/gate.py                the gate itself — rerun anytime, writes
                                  reports/gate_results.csv
  scripts/diagnose.py            per-market side/odds/confidence diagnostics
                                  behind the causes above
  reports/gate_results.csv       every number in the results table above
  reports/MILESTONE_1_GATE_REPORT.md   this file
```

To rerun: `pip install -r requirements.txt`, then `python scripts/gate.py`.
