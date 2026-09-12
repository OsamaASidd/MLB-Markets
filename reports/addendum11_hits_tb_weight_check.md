# Addendum 11: hits / total_bases weight-vs-correlation check, done properly

Follow-up to Addendum 8, which flagged `pitcher_strikeouts`'s two
heaviest-weighted factors as real-world noise by matching `w_mlb_*` weight
columns to `score_*` columns via name-stripping — an approach that matched
only one `w_mlb_hits_*` column out of ~20 in the schema. This redo reads the
actual scoring code (`scoring_mlb_v2.ts`) to find the true mapping, and it
overturns the premise: **the `w_mlb_hits_*`/`w_mlb_tb_*` columns are dead
schema, not live weights.**

## The structural finding: hits and total_bases don't have their own weights

`scoreBatterMarket()` (scoring_mlb_v2.ts:4272) is one shared function for
`hits`/`totalBases`/`rbi`/`homeRuns`. Every factor inside it reads from a single
module-scope `W_BATTER` object (`DEFAULT_W_BATTER`, line 2872) — there is no
per-market weight object in the scoring code itself.

The loader that populates `W_BATTER` from the database
(`_shared/mlb_weights.ts`, `COL_TO_PATH` lines 37-121 and
`loadMlbWeightsFromDB()` lines 191-369) reads only the **generic**,
non-market-prefixed columns — `w_mlb_batter_hit_rate`, `w_mlb_batter_babip`,
`w_mlb_bullpen_quality`, etc. A repo-wide search found **zero references
anywhere in the betgenius codebase** (scoring, loader,
`process-games-mlb/index.ts`, harness, or any tracked migration) to
`w_mlb_hits_*` or `w_mlb_tb_*`. Those ~20-per-market columns exist in the live
`algorithm_weights` table but nothing reads them.

Per-market differentiation *is* architected — via D-534
(scoring_mlb_v2.ts:5888-5953: `setMlbWeightsWithPerMarket`/`setActiveMarket`,
sourced from a `mlb_market_weight_overrides` JSONB column, keyed by the same
generic column names) — but the live row's overrides are `{}` (confirmed by
direct query). So **hits and total_bases are scored today with the identical
global weight set as home_runs and rbis**, differentiated only by the
`isPowerMarket` branch inside `scoreBatterMarket` (hits ≠ power market;
total_bases = power market), not by the `w_mlb_hits_*`/`w_mlb_tb_*` columns.
The D-534 migration's own stated motivation ("R4 doubled helps batter_hits but
hurts batter_total_bases") is exactly the problem here — built, never
populated.

## True weight → factor mapping (read from the code, cited by line)

Every row is `W_BATTER.<key>` multiplying that factor's bucketed magnitude,
gated by `marketStat`. Weight = live `algorithm_weights` row (id=1); corr =
TRAIN-only (first 60% of dates, `pick_history`,
`prop_type='hits'`/`'total_bases'`, non-synthetic, non-voided, `hit IS NOT
NULL`) — same split discipline as `factor_reweight.py`. n(train) = 9,003
(hits) / 8,610 (total_bases) of 15,005 / 14,351 total.

**FLAG-A** = weight ≥1.0, |corr| <0.03 (heavily weighted, no signal).
**FLAG-B** = weight ≤0.25, |corr| ≥0.08 (underweighted, real signal).

### hits (23 factors; sorted by |weight|)

| weight column | factor (line) | weight | TRAIN corr | flag |
|---|---|---:|---:|---|
| w_mlb_batter_line_hit_rate | score_batter_line_hit_rate (5121) | 2.0 | n/a (0 in train)† | — |
| w_mlb_batter_pitcher_quality | score_opposing_pitcher_quality (4437) | 1.5 | −0.014 | **A** |
| w_mlb_batter_vs_pitcher_hand_split | score_batter_vs_pitcher_hand_split (4834) | 1.5 | −0.001 (n=138) | **A** |
| w_mlb_batter_contact_rate | score_batter_contact_rate (5050) | 1.5 | n/a (0 in train)† | — |
| w_mlb_batter_xwoba | score_batter_xwoba (5150) | 1.5 | +0.015 | **A** |
| w_mlb_batter_weather_temp | score_weather_temp (4518) | 1.25 | +0.019 | **A** |
| w_mlb_batter_recent_ab | score_recent_at_bats (4465) | −1.0 | −0.015 | **A** |
| w_mlb_lineup_spot | score_lineup_spot (4563) | 1.0 | −0.005 (n=138) | **A** |
| w_mlb_batter_opp_pitcher_pitchtype_quality | score_opp_pitcher_pitchtype_quality (4696) | 1.0 | +0.011 (n=138) | **A** |
| w_mlb_batter_xba | score_batter_xba (4740) | 1.0 | −0.001 (n=138) | **A** |
| w_mlb_bullpen_quality | score_bullpen_quality (4857) | 1.0 | +0.007 (n=138) | **A** |
| w_mlb_pitcher_hard_contact_allowed | score_pitcher_hard_contact_allowed (5000) | 1.0 | n/a (n=31)† | — |
| w_mlb_batter_hard_hit | score_batter_hard_hit (5218) | 1.0 | +0.039 | — |
| w_mlb_batter_handedness_matchup | score_handedness_matchup (4484) | −0.75 | −0.021 | — |
| w_mlb_pitcher_baa_vs_hand | score_pitcher_baa_vs_hand (4646) | 0.75 | n/a (0 in train)† | — |
| w_mlb_batter_ballpark_hits_factor | score_ballpark_factor (4500) | 0.5 | −0.006 | — |
| w_mlb_batter_lineup_consistency | score_lineup_consistency (4542) | 0.5 | +0.006 | — |
| w_mlb_day_after_night_fatigue | score_day_after_night_fatigue (4587) | 0.5 | n/a (0 in train)† | — |
| w_mlb_travel_getaway | score_travel_getaway (4604) | 0.5 | n/a (0 in train)† | — |
| w_mlb_hitter_streak_fatigue | score_hitter_streak_fatigue (4666) | 0.5 | n/a (0 in train)† | — |
| w_mlb_batter_form | score_batter_form (4409) | 0.25 | +0.016 | — |
| w_mlb_batter_hit_rate | score_batter_hit_rate (4368) | **0.125** | **+0.049** | close to B, doesn't clear 0.08 |
| w_mlb_batter_babip | score_batter_babip (4781) | −0.125 | n/a (n=71)† | — |

### total_bases (28 factors; sorted by |weight|)

| weight column | factor (line) | weight | TRAIN corr | flag |
|---|---|---:|---:|---|
| w_mlb_batter_line_hit_rate | score_batter_line_hit_rate (5121) | 2.0 | n/a (0 in train)† | — |
| w_mlb_batter_power_rate | score_batter_power_rate (4388) | 1.5 | +0.037 | — |
| w_mlb_batter_pitcher_quality | score_opposing_pitcher_quality (4437) | 1.5 | −0.030 | — |
| w_mlb_batter_pitcher_hr_rate | score_pitcher_hr_rate (4451) | 1.5 | +0.038 | — |
| w_mlb_batter_vs_pitcher_hand_split | score_batter_vs_pitcher_hand_split (4834) | 1.5 | +0.000 | **A** |
| w_mlb_batter_xwoba | score_batter_xwoba (5150) | 1.5 | −0.006 | **A** |
| w_mlb_batter_weather_temp | score_weather_temp (4518) | 1.25 | −0.023 | **A** |
| w_mlb_batter_recent_ab | score_recent_at_bats (4465) | −1.0 | −0.000 | **A** |
| w_mlb_lineup_spot | score_lineup_spot (4563) | 1.0 | **+0.057** | — |
| w_mlb_batter_opp_pitcher_pitchtype_quality | score_opp_pitcher_pitchtype_quality (4696) | 1.0 | −0.003 | **A** |
| w_mlb_batter_xba | score_batter_xba (4740) | 1.0 | +0.006 | **A** |
| w_mlb_batter_exit_velo_trend | score_batter_exit_velo_trend (4752) | 1.0 | **+0.086** | — |
| w_mlb_batter_barrel_rate | score_batter_barrel_rate (4765) | 1.0 | +0.051 | — |
| w_mlb_batter_xslg_regression | score_batter_xslg_regression (4776) | 1.0 | −0.020 | **A** |
| w_mlb_bullpen_quality | score_bullpen_quality (4857) | 1.0 | −0.006 | **A** |
| w_mlb_pitcher_hard_contact_allowed | score_pitcher_hard_contact_allowed (5000) | 1.0 | n/a (n=36)† | — |
| w_mlb_batter_hard_hit | score_batter_hard_hit (5218) | 1.0 | +0.025 | **A** |
| w_mlb_batter_handedness_matchup | score_handedness_matchup (4484) | −0.75 | −0.008 | — |
| w_mlb_pitcher_baa_vs_hand | score_pitcher_baa_vs_hand (4646) | 0.75 | n/a (0 in train)† | — |
| w_mlb_batter_launch_angle | score_batter_launch_angle (5176) | 0.75 | **−0.157** | ⚠ see below |
| w_mlb_batter_sweet_spot | score_batter_sweet_spot (5195) | 0.75 | −0.002 | — |
| w_mlb_batter_form_power | score_batter_form_power (4410) | 0.5 | +0.015 | — |
| w_mlb_batter_ballpark_hits_factor | score_ballpark_factor (4500) | 0.5 | −0.027 | — |
| w_mlb_batter_weather_wind | score_weather_wind (4536) | −0.5 | +0.022 | — |
| w_mlb_batter_lineup_consistency | score_lineup_consistency (4542) | 0.5 | −0.049 | — |
| w_mlb_day_after_night_fatigue | score_day_after_night_fatigue (4587) | 0.5 | n/a (0 in train)† | — |
| w_mlb_travel_getaway | score_travel_getaway (4604) | 0.5 | n/a (0 in train)† | — |
| w_mlb_hitter_streak_fatigue | score_hitter_streak_fatigue (4666) | 0.5 | n/a (0 in train)† | — |
| w_mlb_batter_babip | score_batter_babip (4781) | −0.125 | −0.007 (n=163) | — |

† Wired into scoring after the 60/40 split point (0 rows in TRAIN, ~5,300-6,000
in TEST; full range 2026-05-17 to 2026-07-27) — a rollout-timing gap, not a
data-quality bug. Test-only (out-of-methodology, informational) correlations:
`line_hit_rate` +0.051 (hits) / **+0.163** (total_bases); `contact_rate` +0.019
(hits); all others |r| ≤ 0.04. `line_hit_rate` carries the single heaviest
weight (2.0) in the whole batter set, and its only available reading for
total_bases is the strongest correlation found anywhere in this check —
plausibly well-calibrated, but unconfirmable under the required split until
more TRAIN-period rows accumulate.

## Flagged: 18 miscalibrations (9 hits, 9 total_bases)

All 18 are FLAG-A (heavily weighted ≥1.0, |corr| <0.03). **No FLAG-B cases**
in either market — nothing is both underweighted (≤0.25) and correlated
(≥0.08); `score_batter_hit_rate` (weight 0.125, corr +0.049) is closest but
doesn't clear 0.08. The 9 flagged factors per market are listed in the tables
above; five (`vs_pitcher_hand_split`, `xwoba`, `weather_temp`, `recent_ab`,
`opp_pitcher_pitchtype_quality`) are flagged identically in *both* markets —
the same weight column doing the same non-work twice, per the shared-weight
finding above.

**Additional flag beyond the stated thresholds:** `score_batter_launch_angle`
in total_bases (weight 0.75, corr **−0.157**) is the strongest correlation in
this whole check — and it runs backwards. The factor logic (line 5162-5173)
assigns its most positive score to launch angles it calls the "line-drive
sweet spot," yet real outcomes correlate strongly negatively with that score.
Not "no signal" — a candidate sign/threshold bug, separate from the
overweighting pattern below.

## Verdict

**Yes — hits and total_bases show the same overweight-the-noise pattern as
pitcher_strikeouts, and it's structural, not incidental.** In both markets the
top 3-4 heaviest weights (1.25-1.5, second only to the still-unverifiable
`line_hit_rate`) attach to factors with |corr| < 0.02 on TRAIN —
`pitcher_quality`, `vs_pitcher_hand_split`, `xwoba`, `weather_temp` chief among
them. Meanwhile the few factors that do show real TRAIN signal
(`exit_velo_trend` +0.086, `lineup_spot` +0.057, `barrel_rate` +0.051, all
total_bases) sit at a middling weight of 1.0, no more favored than the noise
factors at the same or higher weight. Less dramatic than strikeouts (no
1.5-weight factor here is as flatly zero as `opposing_lineup_k`'s +0.006), but
the shape is the same: weight does not track real correlation for these
markets either.

The bigger practical finding is architectural: fixing this by editing
`w_mlb_hits_*`/`w_mlb_tb_*` — as the original framing assumed — would do
**nothing**, since production never reads those columns. A real fix has to go
through the generic `w_mlb_batter_*` columns (also touching home_runs/rbis) or
through populating `algorithm_weights.mlb_market_weight_overrides` for
`batter_hits`/`batter_total_bases` — the D-534 machinery for exactly this
already exists, and currently sits unused (`{}`).
