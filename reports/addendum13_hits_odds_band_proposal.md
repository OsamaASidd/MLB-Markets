# Addendum 13: Proposal — moderate-odds band filter for `batter_hits` under

**Status: PROPOSAL ONLY. Not deployed. Requires written GO from the project owner
before any production change.**

This document turns the Addendum 10 finding into a concrete, reviewable
change request. It does not modify any file in the production app. Nothing
in this document should be deployed without an explicit yes from the project
owner, consistent with the existing internal rule that scoring/selection
logic for markets already live in production is not to be changed without
sign-off.

## 1. The rule being proposed

**Market:** `batter_hits` &nbsp; **Side:** `under` &nbsp; **Scope:** this rule only
changes *whether the pick fires*, not the scoring math underneath it.

> Show the `batter_hits` under recommendation **only if** the best available
> price across books is between **-249 and -101 inclusive**. Outside that
> band — either a heavier favorite than -250, or any underdog/plus-money
> price — **veto**: do not show the pick.

This is the same shape of rule already used in this codebase for
side/veto decisions (see `betgenius/supabase/functions/_shared/mlb_ev_policy.ts`,
read for reference only, not modified): a small, explicit, auditable
condition that fails safe (no pick shown) outside its stated bounds. In that
same style, the eventual production check would read conceptually as:

```
// ILLUSTRATIVE ONLY — not committed, not deployed.
// Same shape as the existing mlb_ev_policy.ts veto checks.
if (market === "batter_hits" && side === "under") {
  const price = bestAvailableOdds;              // American odds, best across books
  if (price > -101 || price < -249) {
    return false;  // veto: outside the validated moderate-odds band
  }
}
```

The actual implementation location, gating mechanism, and exact price-lookup
field would need to match however `mlbRecommendationShown` and its callers
currently source "best available price" — that's an implementation detail
for the approved-and-scheduled work, not this proposal.

## 2. The evidence

Source: `reports/verify_moderate_odds_hits_output.txt`, `scripts/verify_moderate_odds_hits.py`, client's full 2024-2026 closing-odds warehouse graded against real box scores.

| Period | n (graded) | ROI | 95% CI | Gate verdict |
|---|---:|---:|---|---|
| 2024 | 5,327 | +1.36% | [-0.81%, 3.52%] | fail (CI crosses 0) |
| 2025 | 6,549 | +2.01% | [0.03%, 3.98%] | pass (barely) |
| 2026 (partial, through 2026-05-24) | 1,144 | +4.21% | [-0.59%, 9.01%] | fail (small n) |
| **Pooled 2024-2026** | **13,020** | **+1.93%** | **[0.54%, 3.33%]** | **PASS** |

Gate rule applied: n ≥ 500 and ROI CI lower bound > 0.

**Honest caveat:** only the pooled sample clears the gate with full
confidence. No single season does on its own — 2024 and 2026 have CIs that
cross zero. What supports treating this as a real, promising signal rather
than noise is that all three point estimates are positive and trend upward
(1.36% → 2.01% → 4.21%), not reversing. But that trend is three data points,
not a proof.

**Boundary condition:** this same filter does **not** work on `total_bases`
— tested identically, it stays flat (pooled n=53,086, ROI +0.22%, CI
[-0.47%, 0.91%], fails the gate). This is a `batter_hits`-specific effect,
not a general odds-avoidance rule, and should not be assumed to extend to
other markets without separately testing each one.

## 3. Risk assessment — will this hold up going forward?

**The multiplicity concern, checked directly.** Addendum 7 documented a case
where a tempting result (+20% ROI) turned out to be an illusion: 115 "picks"
were really only 14 independent games, because multiple correlated bets on
the same game were counted as separate independent samples. The natural
question here is whether the same thing is happening to this 13,020-pick
sample — batters in the same game share the same weather, ballpark, and
opposing pitcher, so their outcomes are not fully independent of each other.

I queried the same underlying data to check this directly (not assumed):

- 13,020 picks in the moderate-odds band map to **3,730 distinct games** —
  an average of 3.5 picks per game, max 29 in one game. That's real
  clustering, but far less severe than Addendum 7's case (which was ~8-10x
  inflation from a single game contributing 42 of 115 rows).
- To quantify the effect rather than guess at it, I recomputed the pooled
  95% CI using a cluster-robust standard error (clustering by game, the
  standard correction for exactly this kind of within-game correlation)
  instead of the naive i.i.d. assumption:
  - Naive CI: [0.54%, 3.33%] (as reported)
  - Cluster-robust CI: **[0.38%, 3.48%]**
  - Standard error inflates by only ~1.11x; implied effective sample size
    is roughly ~10,600 rather than the raw 13,020.

**Bottom line: real but modest clustering, and the result survives it.** The
CI lower bound stays above zero even after correcting for game-level
clustering — not the same failure mode as Addendum 7. This check only covers
within-game correlation, though; it does not rule out other non-independence
(e.g. same-player streaks across games, systematic pricing patterns within
the band), which was not separately tested here.

**Other things that could make this not hold up:**
- **Small effective edge.** A ~2% ROI is modest; it can be erased by book
  price movement, a widened band definition drifting from what was tested,
  or normal variance over a shorter forward window.
- **Regime change.** MLB batting environment, book pricing behavior, or
  the client's own book-price sourcing could shift after the 2026-05-24
  data cutoff (see §5).
- **Only tested against best-price-across-56-books**, not necessarily what
  production would actually get filled at (Addendum 1/9 both flag a gap
  between best-available and actual fill price for this market).

## 4. Approval and deployment (if approved)

**This proposal requires explicit written GO from the project owner before
any code change.** `batter_hits` is already live in production, and existing
internal policy treats any change to what fires or vetoes a live market as
requiring sign-off before deployment — this rule qualifies.

If approved, the concrete change would be a small, additive veto condition
in the same file and same style as the existing side/veto checks in
`_shared/mlb_ev_policy.ts` (or wherever the equivalent production check
currently lives for `batter_hits`) — scoped only to `market === "batter_hits"
&& side === "under"`, checked against the best-available price at
recommendation time, failing safe (no pick) outside -249..-101. It would not
touch `scoring_mlb_v2.ts` or `algorithm_weights` — this is a selection/veto
filter layered on top of the existing score, not a change to the scoring
math itself.

## 5. Recommended next step before shipping, even with a GO

Do not go straight from "written GO" to "live for all users." Recommended:

1. **Paper-trade forward for 4-6 weeks** (shadow-log what the filter would
   have picked, without showing it to subscribers) to see if the ~2% edge
   and the win rate (~62%) hold on genuinely new, out-of-sample data.
2. **Re-run this same verification once the odds warehouse extends past its
   current 2026-05-24 cutoff** (the read-only DB credential used to build it
   has since stopped authenticating — a refreshed credential is needed
   first) to add real 2026 volume to the thinnest, most favorable-looking
   season in the current sample.
3. If both check out, ship behind a flag with a rollback plan, matching the
   "one deploy, one change" discipline already used for other production
   changes in this codebase.
