# MLB Markets

Milestone 1 deliverable: gate audit of every MLB market against the real
production pick history, plus the local database and every script behind it.

**Honest headline result, 31 addenda in: 3 of 11 testable markets pass the
client's own official gate rule on real, adequately-sized data — `hits`,
`rbis`, `spreads`. The other 8 (`total_bases`, `batter_home_runs`,
`pitcher_strikeouts`, `h2h`, `totals`, `pitcher_outs`, `batter_runs_scored`,
`batter_strikeouts`) are confirmed FAIL, not "pending" or "untested."**

**Start here:**
- [`reports/MILESTONE_1_GATE_REPORT.md`](reports/MILESTONE_1_GATE_REPORT.md) — full PASS/FAIL
  results, root causes, improvement levers, and 31 addenda of follow-up research.
- [`reports/market_verdicts_onepager.html`](reports/market_verdicts_onepager.html) — the
  lean, one-table client handoff view (status, latest number, official gate, lever, decision
  needed, per market).
- [`reports/pass_fail_verdicts.html`](reports/pass_fail_verdicts.html) — full shareable verdict
  summary across every market, section-by-section.
- [`deliverables/`](deliverables/) — the non-model M1 deliverables (E2E checklist, scheduler
  monitor, All Picks design note) requested separately from the gate audit itself.

## Layout

- `db/mlb_markets.duckdb` — the local database everything reads from. **Not committed** —
  see [Data access](#data-access-before-you-run-anything) below before trying to run anything.
- `scripts/` — every script behind every number in the report, one script (or small group)
  per finding. Full index in the [testing guide](#testing--reproduction-guide) below.
- `reports/` — the report itself, both HTML deliverables, and the saved raw output of every
  script that produced a reported result (`*_output.txt` / `gate_results.csv`).
- `deliverables/` — the M1 work-order items outside the gate audit itself: the end-to-end
  pipeline checklist, and the "Log All Picks" gate-status design note.
- `data_raw/` — the one external dataset small enough to commit: a free, public 2014-2019
  MLB odds archive (six `.rda` files, ~35KB each), used in Addenda 4-6. Real historical-odds
  caches from The Odds API (Addenda 30-31) are gitignored — see the report for how to
  regenerate them with a live API key, and Addendum 31 for why the 2020-2022 extension
  specifically cannot be regenerated (the provider doesn't carry player-prop odds that far
  back, confirmed empirically, not a missing-credential problem).

## Setup

```
pip install -r requirements.txt
```

`scripts/pull_pick_history.py` additionally needs a `DB_PASSWORD` in a local `.env`
(see `.env.example`) — not needed for anything else.

## Data access before you run anything

Almost every script connects to `db/mlb_markets.duckdb` — most of them `read_only`. That
file is **not currently committed** and going forward is `.gitignore`'d: at various points
it has held the client's real, proprietary production data (pick history, box scores, and
a full 2.42M-row real odds warehouse).

**⚠ Resolved 2026-09-15, previously a real exposure — history now clean.** This file
was committed and pushed to this repo's public GitHub remote across 6 commits (38MB →
42MB) before a later commit removed it from the working tree — which does **not** remove
it from git history. That history has since been rewritten with `git filter-repo` to
strip the file from every commit on both `master` and `milestone-1-mlb-gate-audit`, and
both branches have been force-pushed. Verified after the rewrite: zero commits reference
the file anymore (`git log --all -- db/mlb_markets.duckdb` returns nothing), repo size
dropped from 80MB to 667KB, and the repo has 0 forks and 0 PRs, so there's no known
external copy of the old history. One residual, low-likelihood caveat: GitHub can retain
now-dangling old commits by direct SHA for a grace period before internal garbage
collection — anyone who already had the exact old commit SHA before the rewrite could
still fetch it during that window. Nobody outside this engagement is known to have had
that. See `reports/MILESTONE_1_GATE_REPORT.md`, Addendum 28, for the original disclosure.

This means **a fresh clone of this repo cannot run most of these scripts
out of the box** — that's expected, not a bug. Three ways the db gets populated:

| Path | Script(s) | Needs | Unlocks |
|---|---|---|---|
| A. Free public data only | `load_historical_2014_2019.py` | Nothing — reads `data_raw/*.rda`, already committed | Addenda 4-6 (external 2014-2019 game-odds sample) |
| B. Original small snapshot | `pull_pick_history.py` + `compact_db.py` | A valid `DB_PASSWORD` — **the credential used for the current snapshot has since expired** (see report's Data-quality notes); the client would need to issue a fresh one | Addenda 1-3 (`pick_history`, `boxscore`, etc.) |
| C. Full real odds warehouse | `load_client_odds_warehouse.py` | The client's raw xlsx export (~112MB, real production data) — not included in this repo | Addenda 7-24 (everything built on `client_closing_odds`/`client_games`) |

If you're the client (or anyone else with direct access to the actual `db/mlb_markets.duckdb`
file used to produce this report), copy it into `db/` and every script below runs as
documented. If you don't have it, the practical way to verify a result is to read the
matching `reports/*_output.txt` file — every number quoted in the report was captured
straight from one of these runs and is committed alongside it.

## Testing / reproduction guide

Scripts are grouped below in the order they were actually run, matching the report's
addenda. Each row: what it needs beyond the base db, what it writes, which addendum cites it.

### Tier 1 — runs with zero client data (free public data only)

```
python scripts/load_historical_2014_2019.py       # creates db/mlb_markets.duckdb if absent,
                                                    # loads historical_game_odds (2014-2019)
python scripts/gate_historical_game_markets.py     # 12 generic mechanical strategies, 6-year sample
python scripts/build_power_rankings.py             # point-in-time Elo + L10 -> game_features
python scripts/explore_power_ranking_patterns.py   # pattern search, TRAIN split only
python scripts/validate_power_ranking_test.py      # confirms/rejects on TEST holdout
python scripts/test_fade_streak_strategy.py        # prices the L10 mean-reversion pattern as a real bet
python scripts/build_pitcher_quality.py            # point-in-time starter RA/9 -> pitcher_game_features
python scripts/explore_pitcher_quality_patterns.py # tests the pitcher-quality signal
```
| Script | Output | Addendum |
|---|---|---|
| `load_historical_2014_2019.py` | (loads table) | 4 |
| `gate_historical_game_markets.py` | `reports/gate_historical_game_markets_output.txt` | 4 |
| `build_power_rankings.py` → `validate_power_ranking_test.py` | (quoted directly in report text) | 5 |
| `test_fade_streak_strategy.py` | `reports/fade_streak_output.txt` | 5 |
| `build_pitcher_quality.py` / `explore_pitcher_quality_patterns.py` | (quoted directly in report text) | 6 |

### Tier 2 — needs Path B (`pull_pick_history.py` + `compact_db.py`)

| Script | Needs | Output | Addendum |
|---|---|---|---|
| `gate.py` | `pick_history` | `reports/gate_results.csv` | 1 |
| `diagnose.py` | `pick_history` | (printed diagnostics) | 1 |
| `factor_reweight.py` | `pick_history` (`score_*` factors) | `reports/factor_reweight_output.txt` | 2 |
| `load_warehouse_picks.py` | `harness/out/*.csv` (harness backtest exports, not included) | loads `warehouse_picks` (79,310 rows) | 7 |
| `sweep_warehouse.py` | `warehouse_picks` | `reports/sweep_warehouse_output.txt` | 7 |
| `validate_against_production.py` | `warehouse_picks` + `pick_history` | (printed cross-check) | 7 |

### Tier 3 — needs Path C (`load_client_odds_warehouse.py`, the full real warehouse)

Weather/strategy research and the weight cross-reference:

| Script | Output | Addendum |
|---|---|---|
| `test_wind_effect.py` | (printed — confirms the real temp/wind-runs correlation) | 7 |
| `test_temperature_totals_strategy.py` | `reports/temperature_totals_output.txt` | 7 |
| `cross_reference_weights_vs_correlation.py` | `reports/weight_vs_correlation_output.txt` | 8, 11 |

The client's own real odds warehouse, grading, and levers:

| Script | Output | Addendum |
|---|---|---|
| `grade_client_odds_warehouse.py` | `reports/client_warehouse_grading_output.txt` | 9 |
| `find_roi_improvement_levers.py` | `reports/roi_improvement_levers_output.txt` | 10 |
| `verify_moderate_odds_hits.py` | `reports/verify_moderate_odds_hits_output.txt` | 10 |
| `backfill_2023_boxscores.py` | (fills in real 2023 box scores via the free MLB Stats API) | 14 |
| `overall_portfolio_roi.py` | `reports/overall_portfolio_roi_output.txt` | 15 |
| `test_bullpen_fatigue.py` | `reports/bullpen_fatigue_output.txt` | 16 |
| `sweep_remaining_markets.py` | `reports/sweep_remaining_markets_output.txt` | 17 |
| `finegrain_plus_odds_sweep.py` | `reports/finegrain_plus_odds_output.txt` | 18 |
| `test_statcast_regression.py` | `reports/statcast_regression_output.txt` | 19 |
| `test_ballpark_factor_strategy.py` | `reports/ballpark_factor_strategy_output.txt` | 19 |

The ML models — pooled, cross-market, properly holdout-validated:

| Script | Output | Addendum |
|---|---|---|
| `xgboost_model.py` | `reports/xgboost_model_output.txt` | 20 |
| `xgboost_calibrated.py` | `reports/xgboost_calibrated_output.txt` | 21 |
| `xgboost_pooled_A_factors.py` | `reports/xgboost_pooled_A_output.txt` | 22 |
| `xgboost_pooled_B_multiyear.py` | `reports/xgboost_pooled_B_output.txt` | 23 |
| `check_favorite_accuracy_vs_roi.py` | `reports/favorite_accuracy_vs_roi_output.txt` | 24 |

Run any Tier 3 script exactly as committed and its printed output should reproduce the
corresponding `reports/*_output.txt` file (modulo trivial floating-point noise from
XGBoost's own non-determinism, already visible run-to-run in Addendum 23's numbers).

### Tier 4 — needs a live Odds API key (`ODDS_API_KEY` in `.env`), real paid credits

Individual per-market models, the gate-rule correction, and the real backfills that
resolved the three previously-untestable/underpowered markets:

| Script | Output | Addendum |
|---|---|---|
| `xgboost_individual_markets.py` | (printed per-market table) | 27, 28 |
| `backfill_pitcher_outs_odds.py` / `backfill_runs_scored_odds.py` / `backfill_batter_strikeouts_odds.py` | `data_raw/*_odds_cache.jsonl` (gitignored) | 30 |
| `test_backfilled_markets.py` | `reports/backfilled_markets_test_output.txt` | 30 |
| `backfill_2020_2022_boxscores.py` → `build_event_id_mapping_2020_2022.py` → `backfill_2020_2022_market_odds.py` | confirms 2020-2022 player-prop odds don't exist at this provider (not a code/credential issue) | 31 |

### Outside the gate audit: `scripts/scheduler_monitor.py`

Independent, read-only cron-health second opinion — reads `betgenius`'s real
`cron_heartbeat` table with the same staleness rule as its production
`detect_silent_crons()` view, for anyone with only `harness_readonly` access
(no service-role key, no Twilio access). `python scripts/scheduler_monitor.py --test`
runs a fixture-only self-test with no DB needed; without `--test` it needs
`betgenius/harness/.env`'s `HARNESS_DATABASE_URL`. See `deliverables/E2E_CHECKLIST.md`
for how this fits into the rest of the pipeline checks.

### Verifying without rerunning anything

If you don't have `db/mlb_markets.duckdb`, every number in the report traces to one of the
`reports/*_output.txt` / `reports/gate_results.csv` files above — they're the actual,
uncut console output of the script that produced them, committed as-is. Cross-reference
the addendum number against the tables above to find the right file.
