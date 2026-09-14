# "Log All Picks" — design note (not a code change)

Design proposal only. No code in this repo or `betgenius/` is changed by
this note. Any implementation needs a written GO per the standing M1 rule
(no changes to `scoring_mlb_v2.ts`, `scoring_mlb_hits_model.ts`,
`algorithm_weights`, or deploys, without one).

## Current behavior (confirmed in code)

`handleLogAllPicks` in `Dashboard.tsx` bulk-inserts every currently-shown
`recommendations_cache` row into `bets`, with a duplicate check and a
Kelly-derived stake (BET → Kelly stake, SKIP_PRICE → discretionary stake,
PASS_NO_EDGE → $0). It has no awareness of this audit's per-market
backtest-gate status — it logs whatever the live board is showing,
uniformly, regardless of market.

## The gap

A user hitting "Log All Picks" today has no way to know that the picks
they're bulk-logging span markets with very different backtest track
records — `batter_hits`/`batter_rbis`/`spreads` pass the official gate on
real, adequately-sized data; `total_bases`/`batter_home_runs`/`h2h` fail;
`pitcher_outs`/`batter_runs_scored`/`batter_strikeouts` fail on newly
backfilled real data (this audit, Addenda 27–31). Right now that
distinction is invisible at the point the user commits real stakes.

## Proposal: informational only, never a filter

**Do not hide or veto any market.** The M1 rule is explicit that
hits/total_bases stay live and show/veto policy is not live — this
proposal does not change that, for any market. The idea is strictly to
surface the audit result as a label, not to act on it.

1. Add a per-market badge on each pick row (Dashboard + the "Log All
   Picks" confirmation) showing the gate status from the latest published
   milestone report: `PASS`, `FAIL`, or `UNTESTED`, sourced from a static
   JSON the harness milestone run already produces (no new live
   computation, no scoring change) — e.g. `{"batter_hits": "PASS",
   "batter_total_bases": "FAIL", ...}`.
2. On "Log All Picks", if the batch includes any `FAIL`-tagged market,
   show a one-line, dismissible note: *"N of these picks are on a market
   that has not beaten the vig on historical backtests — see [report
   link]."* No blocking, no confirmation dialog, no stake change. The
   user proceeds exactly as today if they click through.
3. Log which badge state each bet was logged under
   (`bets.gate_status_at_log_time`, a new nullable column) purely for
   later analysis — does knowing the badge change user behavior, does
   it change realized ROI by cohort. Analysis only; never feeds back into
   scoring or the recommendation itself.

## Explicitly out of scope for this proposal

- Auto-excluding FAIL-market picks from "Log All Picks" — this is a
  veto/hide behavior and the M1 rule forbids treating any unshipped
  filter as the live board.
- Changing `confidence >= 60` or any other production filter.
- Changing stake sizing based on gate status.
- Any change to `scoring_mlb_v2.ts`, `scoring_mlb_hits_model.ts`, or
  `algorithm_weights`.

## Why label rather than filter

The three-market audit result (`pitcher_outs`, `batter_runs_scored`,
`batter_strikeouts`) is based on real, well-powered data and is a genuine
finding — but "the backtest says no edge" is a decision input for the
person risking money, not a decision the platform should make silently on
their behalf by hiding the pick. This mirrors the same principle already
applied to hits/total_bases in the standing M1 rule: show the real number,
let the human decide, and never quietly substitute an unshipped filter for
the live board.

## Rollout, if approved

1. Written GO required (per standing rule) before any of this is built.
2. Ship the static gate-status JSON + badge display first (read-only,
   lowest risk).
3. Ship the "Log All Picks" inline note second.
4. Consider the `gate_status_at_log_time` column only if item 2 shows the
   badge is actually being noticed (basic engagement check) — otherwise
   it's dead weight in the schema.
