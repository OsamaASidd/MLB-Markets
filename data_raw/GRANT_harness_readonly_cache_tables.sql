-- harness_readonly SELECT grants — named cache tables only
-- Apply by: whoever manages the database. Developers do not self-apply.
-- Scope: SELECT only. No INSERT/UPDATE/DELETE/TRUNCATE.
-- Do NOT run: GRANT SELECT ON ALL TABLES IN SCHEMA public
-- Do NOT run: ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES
-- Those two would expose bets / auth / billing tables. Not requested.

-- Dead end (do not grant): public.cache_mlb_historical_odds_snapshot
-- Dropped in 20260528000002_d360pre_drop_unused_d358_snapshot_table.sql

-- Already granted earlier (skip if already OK):
--   cache_statcast_pitcher_arsenal, mlb_pitcher_season_stats (D-892)
--   cache_mlb_historical_outcomes, cache_mlb_historical_bullpen (D-894)
--   cache_mlb_historical_pitcher_statcast (D-895)
--   cron.job, cron.job_run_details, cron_progress, recommendations_cache (D-896)

-- Rollback:
--   REVOKE SELECT ON TABLE
--     public.cache_statcast_batters_pull_rate,
--     public.cache_mlb_park_dimensions,
--     public.cache_mlb_ballpark_orientation,
--     public.cache_statcast_batters_contact_rate,
--     public.cache_statcast_batters_sprint_speed,
--     public.cache_statcast_framing,
--     public.cache_mlb_batter_splits,
--     public.cache_mlb_pitcher_splits,
--     public.cache_mlb_bullpen_stats,
--     public.cache_mlb_pitcher_last3,
--     public.cache_mlb_pen_rest,
--     public.cache_mlb_bullpen_high_leverage,
--     public.cache_mlb_pitcher_inn1,
--     public.cache_savant_team_chase,
--     public.cache_mlb_team_manager_hook,
--     public.cache_mlb_team_oaa,
--     public.cache_umpire_stats,
--     public.cache_team_batting_stats,
--     public.cache_odds_snapshots,
--     public.props_cache
--   FROM harness_readonly;

DO $$
DECLARE
  t text;
  names text[] := ARRAY[
    'cache_statcast_batters_pull_rate',
    'cache_mlb_park_dimensions',
    'cache_mlb_ballpark_orientation',
    'cache_statcast_batters_contact_rate',
    'cache_statcast_batters_sprint_speed',
    'cache_statcast_framing',
    'cache_mlb_batter_splits',
    'cache_mlb_pitcher_splits',
    'cache_mlb_bullpen_stats',
    'cache_mlb_pitcher_last3',
    'cache_mlb_pen_rest',
    'cache_mlb_bullpen_high_leverage',
    'cache_mlb_pitcher_inn1',
    'cache_savant_team_chase',
    'cache_mlb_team_manager_hook',
    'cache_mlb_team_oaa',
    'cache_umpire_stats',
    'cache_team_batting_stats',
    'cache_odds_snapshots',
    'props_cache'
  ];
BEGIN
  FOREACH t IN ARRAY names LOOP
    BEGIN
      EXECUTE format('GRANT SELECT ON TABLE public.%I TO harness_readonly', t);
      RAISE NOTICE 'GRANT SELECT public.% : OK', t;
    EXCEPTION WHEN undefined_table THEN
      RAISE NOTICE 'GRANT SELECT public.% : SKIP (no such table)', t;
    WHEN OTHERS THEN
      RAISE NOTICE 'GRANT SELECT public.% : FAIL %', t, SQLERRM;
    END;
  END LOOP;
END $$;

-- Verify:
-- SELECT table_name, privilege_type
-- FROM information_schema.role_table_grants
-- WHERE grantee = 'harness_readonly'
--   AND table_name IN (
--     'cache_odds_snapshots','props_cache','cache_mlb_team_oaa',
--     'cache_mlb_batter_splits','cache_team_batting_stats'
--   )
-- ORDER BY 1;
