# MLB Markets audit — handoff (2026-09-15)

For a new session, a new engineer, or the client. Read this before touching
anything else in the repo. Goal: give whoever picks this up the real
picture, so nobody re-does 31 addenda of work or re-litigates settled
questions.

## TL;DR

**3 of 11 testable markets pass the client's own official gate rule on
real, adequately-sized data: `hits`, `rbis`, `spreads`. The other 8 are
confirmed FAIL — not "pending," not "untested," actually tested at real
scale and negative.** This is on branch `3-pass-markets`
(`reports/MILESTONE_1_GATE_REPORT.md` has the full 31-addendum trail).

Branch `8-pooled-ml` holds a separate, deprioritized research line
(pooled cross-market XGBoost models) — it is **not** a competing result
and does not change the count above. See `POOLED_ML_STATUS.md` on that
branch for why.

## Official gate rule (verified against the client's own code)

From `betgenius/harness/lib/metrics.ts`, `evaluateEvGate`:
- n ≥ 500 **and** ROI > 0 → PASS
- else, 95% ROI CI lower bound > 0 → PASS
- else → FAIL

This project's own scripts used a **stricter, wrong** rule for most of its
life (`n≥500 AND CI>0`). Fixed in Addendum 28. Flipped `spreads` from FAIL
to PASS on re-check. If anyone hands you a different rule, verify it
against the actual `metrics.ts` file, don't take it on faith.

## The 11-market scoreboard

| Market | Status | n | ROI | Addendum |
|---|---|---|---|---|
| `hits` | **PASS** | large | positive, CI>0 | 27/28 |
| `rbis` | **PASS** | large | positive, CI>0 | 27/28 |
| `spreads` | **PASS** | large | positive, CI>0 (flipped by the gate-rule fix) | 28 |
| `total_bases` | FAIL | large | negative | 27-29 (statcast features tried, still negative) |
| `batter_home_runs` | FAIL | large | -2.20% (88.4% accuracy — accuracy ≠ ROI) | 24 |
| `pitcher_strikeouts` | FAIL | large | negative | 9, 17 |
| `h2h` | FAIL | large | negative | 9, 17 |
| `totals` | FAIL | large | negative | 9, 17 |
| `pitcher_outs` | FAIL | n=5,451 (TEST 3,334) | -1.21%, CI[-7.2%,4.78%] | 30 |
| `batter_runs_scored` | FAIL | n=79,131 (TEST 52,981) | -2.54%, CI[-4.05%,-1.03%] | 30 |
| `batter_strikeouts` | FAIL | n=60,862 (TEST 41,489) | -0.87%, CI[-3.1%,1.35%] | 30 |

The last three were previously "untestable"/underpowered — resolved via a
real historical-odds backfill from The Odds API (paid credential), matched
to real box-score outcomes. Not fixture data, not synthetic.

## What's been tried and genuinely ruled out (don't re-do these)

- **Pooled cross-market ML models** (branch `8-pooled-ml`): Model B showed
  an implausible AUC 0.857 first pass — traced to alt-line ladders being
  counted as independent bets (fixed, corrected to 0.764). Later found to
  give **different PASS/FAIL verdicts run to run** — genuine training
  instability, not a data bug. This is why it's deprioritized, not because
  no one tried.
- **Individual per-market XGBoost models** with real point-in-time features
  (Elo, L10, bullpen fatigue, park factors): 3 solid passes, 2 markets
  that only "pass" under a best-of-5 threshold search (flagged explicitly
  as multiple-comparisons noise, not a real finding, per Addendum 27/28).
- **Real Statcast quality features** (exit velo, xStats) folded into
  `total_bases`: still FAIL, n=858, ROI -6.87% (Addendum 29).
- **Pre-registered 5-model comparison on all 8 FAIL markets** (XGBoost,
  LightGBM, HistGBM, RandomForest, LogisticRegression — same features/
  split as each market's published result, one pre-specified cut, no
  threshold search): 5 of 40 combinations technically passed, but every
  one disagrees with the other 4 models on the same data and has a CI
  spanning deep negative — a real, concrete demonstration of the
  multiple-comparisons risk, not 5 new passes (Addendum 32,
  `scripts/multi_model_comparison.py`). **Do not re-run this hoping for a
  different combination to "win" — that's the exact behavior this
  experiment was designed to catch.**
- **Extending the three newly-backfilled markets back to 2020**: The Odds
  API does not carry historical player-prop odds before **2023-05-03** —
  confirmed empirically against 12,822 real requests, all rejected with
  the same `HISTORICAL_MARKETS_UNAVAILABLE_AT_DATE` error. Not a
  credential or code issue (Addendum 31).
- **ESPN's public API as an alternate data source**: confirmed directly
  (both a 2021 historical game and a live 2026 game) that it never carries
  player-prop odds at all, any era — a different, more permanent kind of
  "no" than the Odds API's date limit.
- **The `batter_strikeouts` resolver bug** (`resolve-picks/index.ts`,
  `prop_type === "strikeouts"` matches pitcher-K before the
  `batter_strikeouts` check): real, confirmed directly in code. Does
  **not** affect any backtest verdict above — those come from an
  independently-backfilled dataset, not from live grading. Documented
  only; do not patch without the client's written GO (see below).

## Standing constraints (from the client, still in force unless told otherwise)

- No changes to `scoring_mlb_v2.ts`, `scoring_mlb_hits_model.ts`, or
  `algorithm_weights` without a **written GO**.
- `hits` and `total_bases` stay live on the dashboard regardless of gate
  status — no hide/veto recommendation for either.
- No deploying `process-games-mlb` from a review/phase1 branch (strips the
  live GLM scorer).
- No production DB writes; read-only access only.
- `HITS_CONFIRMED_LINEUP_EXPERIMENT.active` must stay `false` until GO'd.
- Do not re-buy 2020-2026 player-prop history — confirmed not to exist
  pre-2023-05-03, would just burn real API credits for nothing.

## What's genuinely still open (real, not a rescue for the 8 FAILs)

From the client's own engineer (Saqib), in response to four research
questions — each requires a **written GO** before any live wiring, and
none of them currently produces usable data for the three newly-backfilled
FAIL markets specifically:

1. **CLV / line-movement audit** — real, doable now without a new GO
   (read-only analysis on already-existing 2025 warehouse snapshots, no
   new API pull). Only applies to `batter_hits`/`h2h`/`spreads`/`totals`
   (the markets with warehouse snapshot data) — **not** to `pitcher_outs`/
   `batter_runs_scored`/`batter_strikeouts`, which have zero warehouse
   rows, so there's nothing to compute CLV on for them. 2025 measured so
   far: avg CLV +0.07% — real but tiny, not a rescue for `hits` (which
   already passes) either.
2. **Pitch-level / at-bat Statcast** (vs. today's specific pitcher-hand +
   pitch-type mix, not seasonal aggregates) — real idea, no data pulled
   yet, wiring it is a scoring change, written GO + branch-only per the
   client.
3. **Confirmed lineup/batting-order timing** — already measured on `hits`:
   halves the bleed on projected-lineup rows but hits still fails that
   *particular in-sample slice* even after excluding them. Needs a
   same-day re-score after lineups lock, which isn't built. Doesn't apply
   to the three FAIL markets.
4. **Umpire/park K-zone for `batter_strikeouts`** — factor already exists
   in the live scorer, but the client's own note is explicit: "impossible
   on 0 clean graded batter-K rows... do not run a fake gate." No usable
   data here without first fixing the resolver bug under a written GO.

## Explicit guardrails (do not do these — asked for, repeatedly, and refused)

- Do not search across models/thresholds until one "happens to pass" and
  report it as a finding. That's noise, not an edge, no matter how many
  models are tried.
- Do not loosen or reinterpret the official gate rule to manufacture a
  different verdict.
- Do not activate any experiment flagged `active: false` or patch the
  resolver bug without the client's written GO.
- Do not treat a best-of-N threshold search as a pre-specified result —
  always report both, flagged separately, the way Addenda 27/30 do.
- Do not claim a live-grading bug fix retroactively changes a backtest
  verdict computed from separately-sourced data — it doesn't, mechanically.

## If picking this up to run one more legitimate ML attempt

Pre-registering the method **before** running it, specifically so the
result can't be cherry-picked after the fact:

- **Scope**: the 8 FAIL markets, same real point-in-time features already
  built (Elo, L10, bullpen fatigue, park factors, real Statcast where
  available) — no new, untested features.
- **Models**: XGBoost (already the primary model used throughout this
  audit) plus 2-3 other standard tabular learners (e.g. LightGBM/CatBoost
  if available, else scikit-learn's HistGradientBoostingClassifier,
  RandomForest, and a plain logistic regression as a baseline) — not real
  Azure AutoML, since this project has no Azure subscription/credentials;
  substitute a small, disciplined local comparison instead and say so.
- **Split**: one 75/25 chronological split per market, decided once,
  reused for every model — no re-splitting to find a friendlier cut.
- **Decision rule**: official gate rule (n≥500→ROI>0; else CI>0), applied
  once per model per market, on the pre-specified `edge>0.0` cut. No
  threshold search folded into this pass — if a threshold search is run
  separately, it gets reported separately and flagged as such, same as
  every prior addendum.
- **Reporting**: every model's result for every market gets reported —
  passes AND fails — not just whichever comes out best. If nothing
  passes, that's the answer, not a reason to keep searching.

## Client-conversation angle, if the pressure is about expectations

If the ask driving this is "the client expects more than 3 passes" —
that's a conversation about expectations, not a data problem to keep
attacking:
- 3 real passes out of 11 is a legitimate, usable result — it tells the
  client exactly where to keep money live and where not to.
- The alternative (reporting a manufactured pass) costs more than an
  honest FAIL does, if/when it loses live at the rate the real data
  predicted — and it traces back to this audit specifically.
- The genuinely open research (CLV, pitch-level features, lineup timing)
  is real forward motion to offer the client — it's just gated on their
  own written GO, which is their call to make, not a stall on this side.

## Repo / branch map

- `3-pass-markets` — the actual M1 deliverable. Clean git history (the
  real client-data exposure from Addendum 28 was purged via
  `git filter-repo` and force-pushed 2026-09-15).
- `8-pooled-ml` — deprioritized pooled-ML research, kept separate,
  `POOLED_ML_STATUS.md` explains why it's not a competing result.
- `deliverables/` — E2E checklist, scheduler monitor design, All Picks
  gate-status design note (separate from the gate audit itself).
- `scripts/scheduler_monitor.py` — independent read-only cron-health
  check, fixture-tested.
