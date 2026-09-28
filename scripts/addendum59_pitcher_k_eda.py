"""
Addendum 59: real EDA on the pitcher_strikeouts pool (n=20,415) BEFORE
building any more models. Every prior attempt (52, 55b, 57, 58) jumped
straight to XGBoost/logistic regression; none of them actually looked at
where the signal is (or isn't) first. This script:

  1. Market calibration: bin market_prob into deciles, compare to actual
     win rate. If the market is already well-calibrated across the whole
     range, that's direct evidence there's no systematic mispricing left
     to find with a black-box model on the SAME inputs.
  2. Per-feature point-biserial correlation with the actual win/loss
     outcome (not CV-AUC on a black-box model -- raw, interpretable
     correlation, feature by feature).
  3. Segment breakdowns: over vs under side, low-vs-high starts_count
     (small-sample/rookie pitchers vs established ones), by line level,
     by market_prob extremity (favorites vs dogs), by odds side.
  4. Residual analysis: for rows the market prices confidently (extreme
     market_prob), does actual win rate diverge from market_prob more
     than for coin-flip-priced rows? That WOULD be exploitable structure
     a model could learn -- absence of it explains the ~0.56 AUC ceiling
     mechanically instead of just re-confirming it.

Reuses the exact same pool-construction code as Addendum 58 (imported,
not re-copied) so this analysis is on the identical, already-sanity-
checked dataset.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob, norm_name, pick_main_line  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, TEAM_NAME_ALIAS, add_park_features  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games, EXCLUDE_TEAMS  # noqa: E402
from addendum57_pitcher_k_gap_integration import parse_gap_odds  # noqa: E402
from addendum58_pitcher_k_dedicated_model import (  # noqa: E402
    build_pitcher_rolling_features, build_opponent_k_rate, build_k_factor,
)

pd.set_option("display.width", 140)


def build_pool(con, con2):
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
    ], ignore_index=True).sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    assert games_ext.game_pk.nunique() == len(games_ext)

    games_ext = add_park_features(con, con2, games_ext)
    games_ext = games_ext.merge(build_k_factor(con), on="game_pk", how="left")

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
    n0 = len(pool)
    pool = pool.merge(games_ext, on="game_pk", how="inner")
    assert len(pool) == n0

    pool["opp_team_id"] = np.where(pool["team_id"] == pool["home_team_id"], pool["away_team_id"], pool["home_team_id"])
    n1 = len(pool)
    pool = pool.merge(build_pitcher_rolling_features(con).drop(columns=["team_id", "game_date"]),
                       on=["game_pk", "name_norm"], how="left")
    assert len(pool) == n1
    n2 = len(pool)
    pool = pool.merge(build_opponent_k_rate(con), on=["game_pk", "opp_team_id"], how="left")
    assert len(pool) == n2

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["market_prob"] = implied_prob(pool["odds"])
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool["push_margin"] = (pool["strikeouts"] - pool["line"]).abs()  # distance from the line -- how "close" the K count was
    return pool.dropna(subset=["win", "odds"]).reset_index(drop=True)


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building pool (same construction as Addendum 58)...")
    pool = build_pool(con, con2)
    print(f"  n={len(pool)}")

    print("\n=== 1. MARKET CALIBRATION (decile of market_prob vs actual win rate) ===")
    pool["prob_decile"] = pd.qcut(pool["market_prob"], 10, labels=False, duplicates="drop")
    calib = pool.groupby("prob_decile").agg(
        n=("win", "size"), mean_market_prob=("market_prob", "mean"), actual_win_rate=("win", "mean")
    )
    calib["gap"] = calib["actual_win_rate"] - calib["mean_market_prob"]
    print(calib.round(4).to_string())
    corr_calib = np.corrcoef(calib["mean_market_prob"], calib["actual_win_rate"])[0, 1]
    print(f"  decile-level correlation(market_prob, actual_win_rate) = {corr_calib:.4f}  "
          f"(near 1.0 == well-calibrated == little room for a model to add value on top of the price)")

    print("\n=== 2. PER-FEATURE POINT-BISERIAL CORRELATION WITH WIN ===")
    FEATURES = ["market_prob", "p5_k9", "p5_k_rate", "p5_ip_avg", "szn_k9", "szn_k_rate",
                "starts_count", "days_rest", "opp_k_rate", "k_factor", "is_dome", "side_code"]
    rows = []
    for f in FEATURES:
        sub = pool.dropna(subset=[f, "win"])
        if sub[f].nunique() < 2:
            continue
        r, p = stats.pointbiserialr(sub["win"].astype(int), sub[f])
        rows.append({"feature": f, "n": len(sub), "corr_with_win": r, "p_value": p})
    corr_df = pd.DataFrame(rows).sort_values("corr_with_win", key=lambda s: s.abs(), ascending=False)
    print(corr_df.round(4).to_string(index=False))

    print("\n=== 3. SEGMENT BREAKDOWNS ===")
    print("\n-- by side (over vs under) --")
    print(pool.groupby("side").agg(n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n-- by starts_count bucket (pitcher experience in this dataset) --")
    pool["starts_bucket"] = pd.cut(pool["starts_count"], [-1, 2, 5, 10, 20, 1000],
                                    labels=["0-2", "3-5", "6-10", "11-20", "20+"])
    print(pool.groupby("starts_bucket", observed=True).agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n-- by line level (low K lines vs high K lines) --")
    pool["line_bucket"] = pd.cut(pool["line"], [0, 4.5, 5.5, 6.5, 7.5, 20],
                                 labels=["<=4.5", "5.5", "6.5", "7.5", "8.5+"])
    print(pool.groupby("line_bucket", observed=True).agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n-- by market_prob extremity (favorite vs coinflip vs dog, this side) --")
    pool["fav_bucket"] = pd.cut(pool["market_prob"], [0, 0.40, 0.47, 0.53, 0.60, 1.0],
                                 labels=["dog<40%", "40-47%", "47-53% (coinflip)", "53-60%", "fav>60%"])
    print(pool.groupby("fav_bucket", observed=True).agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n-- by days_rest bucket --")
    pool["rest_bucket"] = pd.cut(pool["days_rest"], [-1, 3, 4, 5, 6, 100], labels=["<=3", "4", "5", "6", "7+"])
    print(pool.groupby("rest_bucket", observed=True).agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_market_prob=("market_prob", "mean")).round(4))

    print("\n=== 4. RESIDUAL / MISPRICING CHECK ===")
    print("does the market's calibration GAP (actual - implied) vary systematically by feature buckets?")
    print("(if gap is ~flat and near 0 everywhere, there's no exploitable residual for ANY model to find)")
    for seg_col in ["starts_bucket", "line_bucket", "fav_bucket", "rest_bucket"]:
        seg = pool.groupby(seg_col, observed=True).apply(
            lambda g: pd.Series({"n": len(g), "gap": g["win"].mean() - g["market_prob"].mean()}),
            include_groups=False)
        print(f"\n  gap by {seg_col}:")
        print(seg.round(4).to_string())

    print("\n=== 5. OUTCOME VARIANCE CHECK (is strikeouts inherently noisy relative to the line?) ===")
    print(f"  mean |strikeouts - line| (push_margin): {pool['push_margin'].mean():.2f}")
    print(f"  std of strikeouts within same (line, side_code=over) rows: "
          f"{pool[pool.side=='over'].groupby('line')['strikeouts'].std().mean():.2f}")
    close_calls = (pool["push_margin"] <= 0.5).mean()
    print(f"  fraction of rows decided by <=0.5 K margin (near-coinflip outcomes by design): {close_calls:.3f}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
