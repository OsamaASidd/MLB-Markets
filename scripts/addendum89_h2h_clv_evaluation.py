"""
Addendum 89: CLV (closing-line-value) evaluation for h2h -- the
highest-leverage untried check per the external review. Rationale:
confirming a ~3% ROI edge needs roughly (1.96/0.03)^2 ~ 4,300 bets (per-
bet profit variance is close to 1 at near-even odds). CLV is a
PROPORTION (did the line move in our favor by closing, yes/no) with far
lower variance -- detecting a genuine ~55-60% CLV-positive rate needs
one to two orders of magnitude fewer bets. If the closing line really is
well-calibrated (confirmed: Addendum 67 EDA, decile corr 0.957-0.973),
then a model that reliably beats it (CLV-positive more than half the
time) has real predictive information, even before enough bets have
settled to prove it out in ROI terms.

Design (kept clean, no circularity): "our bet" uses ONLY the OPENING
snapshot's de-vigged price as market_prob (not closing -- using closing
would make the bet decision see information from after the bet was
supposedly placed). The model is the SAME already-proven clean point-in-
time feature set (Addendum 68: elo/l10/fatigue/bullpen/starter-rolling-
ERA/park), CV-AUC-tuned XGBoost, walk-forward. CLV-positive = the
closing price subsequently moved further toward our side than the
opening price we bet at (movement = close_prob - open_prob > 0 for the
side we bet) -- i.e. line-shopping in TIME, not just across books.

Statistical test: one-sample proportion test (binomial), H0: CLV-
positive rate = 50%, against the pooled walk-forward held-out bet set.
Single pre-specified bet-selection rule (edge>0 using OPENING price),
gate/test touched exactly once.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
from xgboost_individual_markets import DB, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, TEAM_NAME_ALIAS, add_bullpen_features, add_park_features  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_enriched_pool, FEATURES, build_starter_rolling_era  # noqa: E402

N_FOLDS = 5
GLM_FEATURES = [f for f in FEATURES if f not in ("side_code", "market_prob")]


def build_opening_snapshot_pool(con2):
    """Earliest pre-game snapshot per (event_id, pick_side) -- the price
    we'd actually bet at, before any line movement has happened."""
    snaps = con2.execute("""
        SELECT event_id, pick_side, odds, snapshot_time, game_time
        FROM cache_odds_snapshots_game_side
        WHERE snapshot_time <= game_time
    """).fetchdf()
    snaps["implied"] = implied_prob(snaps["odds"].values)
    key = ["event_id", "pick_side"]
    first_t = snaps.groupby(key)["snapshot_time"].transform("min")
    last_t = snaps.groupby(key)["snapshot_time"].transform("max")
    opening = snaps[snaps.snapshot_time == first_t].groupby(key)["implied"].median().rename("open_prob")
    closing = snaps[snaps.snapshot_time == last_t].groupby(key)["implied"].median().rename("close_prob")
    n_snaps = snaps.groupby(key)["snapshot_time"].nunique().rename("n_snapshots")
    mv = pd.concat([opening, closing, n_snaps], axis=1).reset_index()
    mv["movement"] = mv["close_prob"] - mv["open_prob"]
    return mv


def build_extended_features(con, con2, games):
    """Extend the game-level feature table beyond the 2023-2025 real-odds
    window (through the present) using cache_mlb_historical_outcomes
    directly -- same pattern as Addendum 70/74/76. No odds needed here,
    only real outcomes + the already-established point-in-time features."""
    n0 = len(games)
    existing_pks = set(games.game_pk)
    raw = con.execute("""
        SELECT game_pk, home_team, away_team, home_score, away_score,
               TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND TRY_CAST(commence_time AS DATE) > DATE '2026-06-25'
    """).fetchdf()
    raw = raw[~raw.game_pk.isin(existing_pks)]
    raw = raw[~raw.home_team.isin(EXCLUDE_TEAMS) & ~raw.away_team.isin(EXCLUDE_TEAMS)]
    print(f"  {len(raw)} additional real games found beyond the original 2023-2025 window (through {raw.game_date.max() if len(raw) else 'n/a'})")
    return raw


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log + clean point-in-time features (Addendum 48/68)...")
    games, id_to_name = build_fuller_games(con, con2)
    train_pool = build_enriched_pool(con, con2, games, id_to_name, "h2h")
    print(f"  TRAIN pool (real 2023-2025 h2h odds): n={len(train_pool)}  "
          f"date range {pd.to_datetime(train_pool.game_date).min().date()} -> {pd.to_datetime(train_pool.game_date).max().date()}")

    print("\nfitting ONE model on the full 2023-2025 pool (CV-AUC-tuned, ROI never touched)...")
    X_train, y_train = train_pool[GLM_FEATURES], train_pool["win"].astype(int)
    model, cv_score, params = cv_select(X_train, y_train)
    print(f"  CV_AUC={cv_score:.4f}  params={params}")

    print("\nbuilding opening-snapshot pool from live cache_odds_snapshots_game_side (recent window)...")
    mv = build_opening_snapshot_pool(con2)
    id_map = pd.read_csv(ROOT / "data_raw" / "event_id_mapping_h2h_totals_gap.csv")[["event_id", "game_pk"]].drop_duplicates(subset=["event_id"])
    mv_away = mv[mv.pick_side == "away"].merge(id_map, on="event_id", how="inner")
    print(f"  away-side opening/closing snapshot rows mapped to real game_pk: n={len(mv_away)}")

    print("\nbuilding EXTENDED point-in-time features covering the recent window (genuinely out-of-sample vs training)...")
    ext_raw = build_extended_features(con, con2, games)
    name_to_id = {v: k for k, v in id_to_name.items()}
    ext_raw["home_team"] = ext_raw["home_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    ext_raw["away_team"] = ext_raw["away_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    ext_raw["home_team_id"] = ext_raw["home_team"].map(name_to_id)
    ext_raw["away_team_id"] = ext_raw["away_team"].map(name_to_id)
    ext_raw = ext_raw.dropna(subset=["home_team_id", "away_team_id", "game_date"]).drop_duplicates(subset=["game_pk"])
    ext_raw["home_team_id"] = ext_raw["home_team_id"].astype("int64")
    ext_raw["away_team_id"] = ext_raw["away_team_id"].astype("int64")
    ext_raw["home_runs_"] = ext_raw["home_score"].astype(float)
    ext_raw["away_runs_"] = ext_raw["away_score"].astype(float)
    ext_raw["event_id"] = None

    games_ext = pd.concat([
        games[["game_pk", "game_date", "home_team_id", "away_team_id", "home_runs_", "away_runs_", "event_id"]],
        ext_raw[["game_pk", "game_date", "home_team_id", "away_team_id", "home_runs_", "away_runs_", "event_id"]],
    ], ignore_index=True)
    games_ext = games_ext.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    assert games_ext.game_pk.nunique() == len(games_ext), "duplicate game_pk after extension -- STOP"

    from xgboost_individual_markets import build_elo_l10, build_bullpen_fatigue
    elo_l10 = build_elo_l10(games_ext[["game_pk", "home_team_id", "away_team_id", "home_runs_", "away_runs_", "game_date"]])
    games_ext = games_ext.merge(elo_l10, on="game_pk", how="inner")
    fatigue = build_bullpen_fatigue(con)
    games_ext["home_fatigue"] = games_ext.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    games_ext["away_fatigue"] = games_ext.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)
    games_ext = add_bullpen_features(con2, games_ext, id_to_name)
    games_ext = add_park_features(con, con2, games_ext)

    parks = con.execute("SELECT park_name, runs_factor, hr_factor, k_factor, hits_factor FROM ballpark_factors").fetchdf()
    for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
        parks[c] = pd.to_numeric(parks[c], errors="coerce")
    venue_by_event = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    venue_by_event = venue_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    event_to_gamepk = con.execute("SELECT event_id, game_pk FROM client_games").fetchdf()
    event_to_gamepk["game_pk"] = pd.to_numeric(event_to_gamepk["game_pk"], errors="coerce")
    event_to_gamepk = event_to_gamepk.dropna(subset=["game_pk"])
    event_to_gamepk["game_pk"] = event_to_gamepk["game_pk"].astype("int64")
    park_by_gamepk = venue_by_event.merge(event_to_gamepk, on="event_id", how="inner")[
        ["game_pk", "runs_factor", "hr_factor", "k_factor", "hits_factor"]].drop_duplicates(subset=["game_pk"])
    games_ext = games_ext.merge(park_by_gamepk, on="game_pk", how="left")

    op = con.execute("SELECT game_pk, home_starter_id, away_starter_id FROM opposing_pitcher").fetchdf()
    op["game_pk"] = pd.to_numeric(op["game_pk"], errors="coerce")
    op = op.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    op["game_pk"] = op["game_pk"].astype("int64")
    op["home_starter_id"] = op["home_starter_id"].round().astype("Int64")
    op["away_starter_id"] = op["away_starter_id"].round().astype("Int64")
    games_ext = games_ext.merge(op, on="game_pk", how="left")
    roll = build_starter_rolling_era(con)
    roll_h = roll.rename(columns={"player_id": "home_starter_id", "sp_roll_era": "home_sp_roll_era", "sp_roll_ip": "home_sp_roll_ip"})
    roll_h["home_starter_id"] = roll_h["home_starter_id"].astype("Int64")
    games_ext = games_ext.merge(roll_h[["home_starter_id", "game_pk", "home_sp_roll_era", "home_sp_roll_ip"]], on=["home_starter_id", "game_pk"], how="left")
    roll_a = roll.rename(columns={"player_id": "away_starter_id", "sp_roll_era": "away_sp_roll_era", "sp_roll_ip": "away_sp_roll_ip"})
    roll_a["away_starter_id"] = roll_a["away_starter_id"].astype("Int64")
    games_ext = games_ext.merge(roll_a[["away_starter_id", "game_pk", "away_sp_roll_era", "away_sp_roll_ip"]], on=["away_starter_id", "game_pk"], how="left")
    games_ext["sp_roll_era_diff"] = games_ext["away_sp_roll_era"] - games_ext["home_sp_roll_era"]
    games_ext["sp_roll_ip_diff"] = games_ext["home_sp_roll_ip"] - games_ext["away_sp_roll_ip"]
    games_ext["elo_diff"] = (games_ext["home_elo"] + 24) - games_ext["away_elo"]
    games_ext["l10_diff"] = games_ext["home_l10"] - games_ext["away_l10"]
    games_ext["fatigue_diff"] = games_ext["home_fatigue"] - games_ext["away_fatigue"]
    games_ext["win"] = games_ext["away_runs_"] > games_ext["home_runs_"]

    n0 = len(mv_away)
    combo = mv_away.merge(games_ext[["game_pk"] + GLM_FEATURES + ["game_date", "win"]].drop_duplicates(subset=["game_pk"]),
                           on="game_pk", how="inner")
    print(f"  row-count check (merge with extended feature pool): before={n0} after={len(combo)} "
          f"{'OK' if len(combo) <= n0 else '*** FAN-OUT, STOP ***'}")
    assert len(combo) <= n0
    combo["game_date"] = pd.to_datetime(combo["game_date"])
    combo = combo.dropna(subset=["open_prob", "movement"]).reset_index(drop=True)
    print(f"  final CLV test pool: n={len(combo)}  date range "
          f"{combo.game_date.min() if len(combo) else 'n/a'} -> {combo.game_date.max() if len(combo) else 'n/a'}")

    if len(combo) < 20:
        print(f"\n  INSUFFICIENT DATA (n={len(combo)}) -- UNDERPOWERED even for a CLV test")
        return

    test_pred = model.predict_proba(combo[GLM_FEATURES])[:, 1]
    combo["model_prob"] = test_pred
    combo["edge"] = combo["model_prob"] - combo["open_prob"]

    sub = combo[combo.edge > 0.0]
    print(f"\n=== bets at OPEN price with edge>0 (single pre-specified cut): n={len(sub)} ===")
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return

    clv_positive = int((sub["movement"] > 0).sum())
    n = len(sub)
    rate = clv_positive / n
    btest = binomtest(clv_positive, n, p=0.5, alternative="greater")
    print(f"\n=== CLV EVALUATION (report-only per the approved gate revision, NOT a PASS/FAIL by itself) ===")
    print(f"  n={n}  CLV-positive={clv_positive} ({round(100*rate,1)}%)  "
          f"one-sided binomial test vs 50%: p-value={btest.pvalue:.4f}")
    print(f"  {'STATISTICALLY SIGNIFICANT CLV edge (p<0.05)' if btest.pvalue < 0.05 else 'not significant at p<0.05'}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
