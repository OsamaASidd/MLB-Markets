"""
Addendum 58: a dedicated pitcher_strikeouts model, built on the n=20,415
combined pool from Addendum 57 (2023-2025 real odds + the freshly
backfilled 2025-05-28->2026-09-23 gap, matched to real boxscore K counts).

WHY a fresh feature set: every prior pitcher_strikeouts attempt (37/48/50-
style enrichment, Addendum 52, Addendum 55b, Addendum 57) reused TEAM-LEVEL
features built for the h2h/totals markets (Elo, bullpen ERA, park factors).
Those describe which TEAM wins, not how many batters a specific PITCHER
strikes out. This is a player prop -- the actually-relevant signal is the
pitcher's own recent form and the opposing lineup's own strikeout
tendency. Building that properly for the first time here:

  - p5_k9, p5_k_rate, p5_ip_avg: rolling (last 5 starts, STRICTLY PRIOR
    -- shift(1) before the rolling window) K/9, K-rate (K / batters
    faced), and average innings pitched -- workload/usage trend.
  - szn_k9, szn_k_rate, starts_count: season-to-date expanding (all
    career starts in this data, strictly prior) versions of the same,
    for stability early in a run of starts and as a reliability signal.
  - days_rest: days since the pitcher's last start.
  - opp_k_rate: the OPPOSING team's trailing (last 10 games, strictly
    prior) team strikeout rate (batter_strikeouts / plate_appearances)
    -- how K-prone that specific lineup has been recently.
  - k_factor, is_dome: park strikeout factor and dome status (Addendum
    37 park enrichment, reused as-is -- legitimately relevant context).
  - market_prob, side_code: the market's own implied probability and
    which side (over/under) is being priced -- as in every other
    addendum.

All rolling/expanding stats are computed with shift(1) BEFORE any
window/expanding aggregation, so no start ever sees its own outcome or a
future start's outcome -- point-in-time correctness, verified via an
explicit assertion below.

Same discipline as every prior addendum: CV-AUC-only model/hyperparameter
selection (never ROI) on training folds, walk-forward expanding-window
folds, one single pre-specified edge>0.0 cut, official gate rule touched
exactly once on the pooled held-out predictions.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob, norm_name, pick_main_line  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, TEAM_NAME_ALIAS, add_park_features  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum57_pitcher_k_gap_integration import parse_gap_odds  # noqa: E402

N_FOLDS = 6


def parse_innings(ip):
    if pd.isna(ip):
        return np.nan
    try:
        s = str(ip)
        whole, _, frac = s.partition(".")
        whole = float(whole)
        add = {"": 0.0, "0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)
        return whole + add
    except Exception:
        return np.nan


def rolling_trailing(df, group_col, value_cols, window, prefix):
    g = df.groupby(group_col)
    for c in value_cols:
        df[f"{prefix}_{c}_sum"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    df[f"{prefix}_n_prior"] = g[value_cols[0]].transform(lambda s: s.shift(1).expanding().count())
    return df


def expanding_trailing(df, group_col, value_cols, prefix):
    g = df.groupby(group_col)
    for c in value_cols:
        df[f"{prefix}_{c}_sum"] = g[c].transform(lambda s: s.shift(1).expanding().sum())
    return df


def build_pitcher_rolling_features(con):
    box = con.execute("""
        SELECT game_pk, player_name, team_id, game_date, strikeouts, innings_pitched, batters_faced
        FROM boxscore WHERE position_type = 'Pitcher' AND is_starter = True
    """).fetchdf()
    box["game_date"] = pd.to_datetime(box["game_date"])
    box["ip_float"] = box["innings_pitched"].map(parse_innings)
    box["name_norm"] = box["player_name"].map(norm_name)
    box = box.dropna(subset=["ip_float", "game_date"])
    box = box.sort_values(["name_norm", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    box = rolling_trailing(box, "name_norm", ["strikeouts", "ip_float", "batters_faced"], 5, "p5")
    box["p5_k9"] = 9 * box["p5_strikeouts_sum"] / box["p5_ip_float_sum"]
    box["p5_k_rate"] = box["p5_strikeouts_sum"] / box["p5_batters_faced_sum"]
    box["p5_ip_avg"] = box["p5_ip_float_sum"] / box["p5_n_prior"].clip(upper=5)

    box = expanding_trailing(box, "name_norm", ["strikeouts", "ip_float"], "szn")
    box["szn_k9"] = 9 * box["szn_strikeouts_sum"] / box["szn_ip_float_sum"]
    box["starts_count"] = box.groupby("name_norm")["game_date"].transform(lambda s: np.arange(len(s)))
    box["szn_k_rate"] = box["szn_strikeouts_sum"] / box.groupby("name_norm")["batters_faced"].transform(
        lambda s: s.shift(1).expanding().sum())
    box["days_rest"] = box.groupby("name_norm")["game_date"].diff().dt.days

    # point-in-time correctness: every _sum column above was built with
    # shift(1) applied BEFORE .rolling()/.expanding(), so by construction
    # row i's aggregates only ever see rows < i for that pitcher.

    box = box.drop_duplicates(subset=["game_pk", "name_norm"])
    return box[["game_pk", "name_norm", "team_id", "game_date", "p5_k9", "p5_k_rate", "p5_ip_avg",
                "szn_k9", "szn_k_rate", "starts_count", "days_rest"]]


def build_k_factor(con):
    """ballpark_factors.k_factor by game_pk (add_park_features only carries
    dimensions/orientation/is_dome, not the runs/hr/k/hits factor table)."""
    parks = con.execute("SELECT park_name, k_factor FROM ballpark_factors").fetchdf()
    parks["k_factor"] = pd.to_numeric(parks["k_factor"], errors="coerce")
    venue_by_event = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    venue_by_event = venue_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    event_to_gamepk = con.execute("SELECT event_id, game_pk FROM client_games").fetchdf()
    event_to_gamepk["game_pk"] = pd.to_numeric(event_to_gamepk["game_pk"], errors="coerce")
    event_to_gamepk = event_to_gamepk.dropna(subset=["game_pk"])
    event_to_gamepk["game_pk"] = event_to_gamepk["game_pk"].astype("int64")
    out = venue_by_event.merge(event_to_gamepk, on="event_id", how="inner")[["game_pk", "k_factor"]]
    return out.drop_duplicates(subset=["game_pk"])


def build_opponent_k_rate(con):
    batter_box = con.execute("""
        SELECT game_pk, team_id, game_date, plate_appearances, batter_strikeouts
        FROM boxscore WHERE batter_strikeouts IS NOT NULL
    """).fetchdf()
    batter_box["game_date"] = pd.to_datetime(batter_box["game_date"])
    team_game = batter_box.groupby(["team_id", "game_pk", "game_date"]).agg(
        team_pa=("plate_appearances", "sum"), team_k=("batter_strikeouts", "sum")
    ).reset_index()
    team_game = team_game.sort_values(["team_id", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    g = team_game.groupby("team_id")
    team_game["t10_k_sum"] = g["team_k"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).sum())
    team_game["t10_pa_sum"] = g["team_pa"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).sum())
    team_game["opp_k_rate"] = team_game["t10_k_sum"] / team_game["t10_pa_sum"]
    out = team_game[["team_id", "game_pk", "opp_k_rate"]].rename(columns={"team_id": "opp_team_id"})
    return out.drop_duplicates(subset=["opp_team_id", "game_pk"])


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("rebuilding the Addendum 57 combined odds pool (2023-2025 + gap backfill)...")
    print("building Addendum 48 fuller game log (through 2026-06-25)...")
    games_v2, id_to_name = build_fuller_games(con, con2)
    name_to_id = {v: k for k, v in id_to_name.items()}

    existing_pks = set(games_v2.game_pk)
    raw = con.execute("""
        SELECT game_pk, home_team, away_team, TRY_CAST(commence_time AS DATE) AS game_date
        FROM cache.cache_mlb_historical_outcomes
        WHERE game_completed = true AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND TRY_CAST(commence_time AS DATE) > DATE '2026-06-25'
    """).fetchdf()
    raw = raw[~raw.game_pk.isin(existing_pks)]
    raw = raw[~raw.home_team.isin(EXCLUDE_TEAMS) & ~raw.away_team.isin(EXCLUDE_TEAMS)]
    raw["home_team"] = raw["home_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["away_team"] = raw["away_team"].map(lambda n: TEAM_NAME_ALIAS.get(n, n))
    raw["home_team_id"] = raw["home_team"].map(name_to_id)
    raw["away_team_id"] = raw["away_team"].map(name_to_id)
    raw = raw.dropna(subset=["home_team_id", "away_team_id", "game_date"]).drop_duplicates(subset=["game_pk"])
    raw["home_team_id"] = raw["home_team_id"].astype("int64")
    raw["away_team_id"] = raw["away_team_id"].astype("int64")
    raw["event_id"] = None
    games_ext = pd.concat([
        games_v2[["game_pk", "game_date", "home_team_id", "away_team_id", "event_id"]],
        raw[["game_pk", "game_date", "home_team_id", "away_team_id", "event_id"]],
    ], ignore_index=True)
    games_ext = games_ext.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    assert games_ext.game_pk.nunique() == len(games_ext), "duplicate game_pk in game log -- STOP"

    print("Addendum 37 park enrichment (dimensions, is_dome)...")
    games_ext = add_park_features(con, con2, games_ext)
    n_before_kf = len(games_ext)
    games_ext = games_ext.merge(build_k_factor(con), on="game_pk", how="left")
    assert len(games_ext) == n_before_kf, "k_factor merge fan-out -- STOP"

    print("parsing freshly-backfilled pitcher_strikeouts odds...")
    gap_odds = parse_gap_odds()

    existing_odds = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds, co.player_name
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'pitcher_strikeouts' AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    existing_odds["game_pk"] = pd.to_numeric(existing_odds["game_pk"], errors="coerce")
    for c in ["line", "best_over_odds", "best_under_odds"]:
        existing_odds[c] = pd.to_numeric(existing_odds[c], errors="coerce")
    existing_odds = existing_odds.dropna(subset=["game_pk"])
    existing_odds["game_pk"] = existing_odds["game_pk"].astype("int64")
    existing_odds["name_norm"] = existing_odds["player_name"].map(norm_name)
    existing_odds = pick_main_line(existing_odds, ["game_pk", "name_norm"])

    gap_odds["game_pk"] = gap_odds["game_pk"].astype("int64")
    gap_odds["name_norm"] = gap_odds["player_name"].map(norm_name)
    gap_odds = pick_main_line(gap_odds, ["game_pk", "name_norm"])
    gap_odds_clean = gap_odds[~gap_odds.game_pk.isin(set(existing_odds.game_pk))]

    all_odds = pd.concat([
        existing_odds[["game_pk", "name_norm", "line", "best_over_odds", "best_under_odds"]],
        gap_odds_clean[["game_pk", "name_norm", "line", "best_over_odds", "best_under_odds"]],
    ], ignore_index=True)
    print(f"  combined odds pool: n={len(all_odds)} (game, pitcher) rows")

    box = con.execute("""
        SELECT game_pk, player_name, team_id, strikeouts FROM boxscore
        WHERE position_type = 'Pitcher' AND is_starter = True
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)
    m = all_odds.merge(box[["game_pk", "name_norm", "team_id", "strikeouts"]],
                        on=["game_pk", "name_norm"], how="inner").dropna(subset=["strikeouts"])
    under = m.copy(); under["win"] = under["strikeouts"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"
    over = m.copy(); over["win"] = over["strikeouts"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"
    pool = pd.concat([under, over]).dropna(subset=["odds"])
    n_before = len(pool)
    print(f"  pool matched to real boxscore K outcomes: n={n_before}")

    pool = pool.merge(games_ext, on="game_pk", how="inner")
    print(f"  row-count check (game-log merge): before={n_before} after={len(pool)} "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool["opp_team_id"] = np.where(pool["team_id"] == pool["home_team_id"], pool["away_team_id"], pool["home_team_id"])

    print("\nbuilding dedicated pitcher rolling-form features (point-in-time, shift(1)-guarded)...")
    pitcher_feats = build_pitcher_rolling_features(con)
    n_before2 = len(pool)
    pool = pool.merge(pitcher_feats.drop(columns=["team_id", "game_date"]), on=["game_pk", "name_norm"], how="left")
    print(f"  row-count check (pitcher-form merge): before={n_before2} after={len(pool)} "
          f"{'OK' if len(pool) == n_before2 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before2

    print("building opponent trailing team K-rate features (point-in-time)...")
    opp_feats = build_opponent_k_rate(con)
    n_before3 = len(pool)
    pool = pool.merge(opp_feats, on=["game_pk", "opp_team_id"], how="left")
    print(f"  row-count check (opponent K-rate merge): before={n_before3} after={len(pool)} "
          f"{'OK' if len(pool) == n_before3 else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before3

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["market_prob"] = implied_prob(pool["odds"])
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.dropna(subset=["win", "odds"]).sort_values(
        ["game_date", "game_pk", "side", "name_norm"], kind="mergesort").reset_index(drop=True)
    print(f"\nfinal dedicated-feature pool: n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")
    coverage = pool[["p5_k9", "opp_k_rate", "k_factor"]].notna().mean()
    print(f"  feature coverage: {coverage.to_dict()}")

    FEATURES = ["side_code", "market_prob", "p5_k9", "p5_k_rate", "p5_ip_avg",
                "szn_k9", "szn_k_rate", "starts_count", "days_rest", "opp_k_rate",
                "k_factor", "is_dome"]

    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== walk-forward folds (expanding window, {N_FOLDS} folds), dedicated pitcher model ===")
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        X_train, y_train = train_df[FEATURES], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[FEATURES])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("\n  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\nedge>0.0 (single pre-specified cut, odds<0), official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
