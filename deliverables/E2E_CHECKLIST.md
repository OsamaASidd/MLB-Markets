# End-to-end pipeline checklist (M1 deliverable)

Read-only verification checklist for the live pick pipeline, for anyone with
the `harness_readonly` role (or equivalent read-only access) — no writes, no
scoring changes, no deploys. Grounded in the real data flow documented in
`betgenius/CLAUDE.md` ("Key data flow") and the actual table/function names
found in this codebase, not assumed ones.

Each stage: what should be true, and a read-only SQL snippet to check it.
Run against a fresh clone's `HARNESS_DATABASE_URL` (or ask Osama's client
contact to run these directly against production, since this contractor's
credential is not always live).

## 1. Odds ingest → warehouse

`fetch-odds-mlb` (cron) writes into `cache_mlb_historical_odds`. Confirm a
recent slate actually landed:

```sql
SELECT market_key, COUNT(*) AS rows, MAX(snapshot_timestamp) AS latest_snap
FROM cache_mlb_historical_odds
WHERE commence_time >= now() - interval '2 days'
GROUP BY market_key
ORDER BY market_key;
```

Expect a row per active market (`batter_hits`, `batter_home_runs`,
`batter_rbis`, `batter_total_bases`, `pitcher_strikeouts`, `h2h`, `spreads`,
`totals`) with `latest_snap` within the last ~30 min of the check (matches
the `snapshot-odds-writer-30min` cron). **Do not expect rows** for
`batter_runs_scored`, `batter_strikeouts`, or `pitcher_outs` — confirmed
zero in the warehouse (Addendum 30/31 of the gate report); that's a known,
documented gap, not a check failure.

## 2. Scoring → recommendations_cache

`process-games-mlb` scores candidates via `scoring_mlb_v2.ts` +
`algorithm_weights` and writes `recommendations_cache`.

```sql
SELECT market_type, COUNT(*) AS picks, MAX(created_at) AS latest
FROM recommendations_cache
WHERE created_at >= now() - interval '1 day'
GROUP BY market_type
ORDER BY market_type;
```

Confirm `batter_hits` and `batter_total_bases` are present (the two markets
that must stay live per the M1 rule) and `latest` is recent. If either is
missing on a day with real games, that's a live-board regression — flag it,
don't fix scoring without a written GO.

## 3. Dashboard reads the cache

No DB check possible here (client-side render) — manual check: open
Dashboard, confirm picks shown match `recommendations_cache` rows from step
2 for the same slate (same player/market/line/confidence). Confirm the
`confidence >= 60` production filter (per Saqib's note) is what's actually
gating what's shown, not a warehouse-Poisson or unshipped filter.

## 4. Log All Picks → bets table

`handleLogAllPicks` (Dashboard.tsx) inserts into `bets` with a
duplicate-check and Kelly-derived stake. Confirm no duplicate rows are
possible for the same slate:

```sql
SELECT user_id, player_name, prop_type, line, pick_side, COUNT(*)
FROM bets
WHERE created_at >= now() - interval '1 day'
GROUP BY 1,2,3,4,5
HAVING COUNT(*) > 1;
```

Expect zero rows. A duplicate here means the `checkDuplicateBet` /
`loggedPicks` short-circuit in Dashboard.tsx has a real bug — reproduce
before reporting, since the short-circuit is client-state-based and a
page refresh mid-slate could plausibly produce a false positive here that
isn't actually a bug.

## 5. Resolution → outcomes

`resolve-picks` settles `bets`/`pick_history` against
`fetch-mlb-boxscores` results.

```sql
SELECT status, COUNT(*)
FROM bets
WHERE created_at <= now() - interval '1 day'  -- games should be final by now
GROUP BY status;
```

A large `pending` count for games that finished over 24h ago means
`resolve-picks` is silently failing — check `cron_heartbeat` (see the
scheduler monitor, `scripts/scheduler_monitor.py`) for that job's status
before assuming it's a resolution-logic bug.

**Known, documented resolver issue** (do not "fix" without a written GO —
see `03_BATTER_STRIKEOUTS.md`): `resolve-picks` checks
`prop_type === "strikeouts"` as pitcher-K before checking
`mlb_market_type === "batter_strikeouts"`, so batter-strikeout picks
resolve against the wrong stat and void. Confirmed directly in
`resolve-picks/index.ts` (~line 1161 vs 1174). Expect most/all
`batter_strikeouts` rows in `bets`/`pick_history` to show `void` status —
that is the known bug, not a new finding to re-investigate.

## 6. CLV capture (informational, not gating)

```sql
SELECT COUNT(*) FILTER (WHERE clv_pct IS NOT NULL) AS with_clv,
       COUNT(*) AS total
FROM pick_history
WHERE created_at >= now() - interval '7 days';
```

Confirms `capture_closing_odds_mlb` is actually writing `clv_pct` on
recent picks. Per Saqib's note this path does not depend on warehouse
snapshot depth, so it should be populated even during the current
low-snapshot-depth 2026 window.

## What this checklist is not

Not a substitute for the client's own `health-monitor` edge function
(already live, real alerting). Not a scoring or gate re-run — that's the
gate report's job (`scripts/gate.py`, `MILESTONE_1_GATE_REPORT.md`). This
is a manual, read-only spot-check anyone can run periodically without
service-role access, to catch a silent pipeline break between the
already-covered automated checks.
