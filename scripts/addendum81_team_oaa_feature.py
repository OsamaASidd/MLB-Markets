"""
Addendum 81: team fielding quality (Outs Above Average) feature for
totals and pitcher_outs -- genuinely untried this session.
cache_mlb_team_oaa (2,850 rows) has real, per-team, point-in-time
fielding quality never used in any prior addendum. Better team defense
should mechanically mean fewer hits convert to runs (totals: lower
combined OAA sum -> more scoring, higher -> less) and, for the specific
pitcher's own team, potentially more efficient innings (pitcher_outs).

Feature: for totals, oaa_sum = home_oaa + away_oaa (both teams'
point-in-time OAA, ASOF-joined by team_id + snapshot_date, never a
future snapshot). For pitcher_outs, own_team_oaa = the OAA of whichever
team the starting pitcher is actually on.

Added to the existing clean feature set (Addendum 68/74) and tested via
the same CV-tuned XGBoost walk-forward.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_enriched_pool, FEATURES  # noqa: E402

N_FOLDS = 5
NEW_FEATURES = FEATURES + ["oaa_sum"]


def build_oaa_asof(con2):
    oaa = con2.execute("""
        SELECT team_id, snapshot_date, oaa FROM cache_mlb_team_oaa
    """).fetchdf().dropna(subset=["team_id", "snapshot_date"])
    oaa["snapshot_date"] = pd.to_datetime(oaa["snapshot_date"])
    return oaa.sort_values("snapshot_date").reset_index(drop=True)


def attach_oaa_asof(pool, oaa, team_col, out_col):
    df = pd.DataFrame({
        "_idx": pool.index, "team_id": pool[team_col].values,
        "game_date": pd.to_datetime(pool["game_date"]).values,
    }).sort_values("game_date")
    merged = pd.merge_asof(df, oaa, left_on="game_date", right_on="snapshot_date",
                            by="team_id", direction="backward")
    pool[out_col] = merged.sort_values("_idx")["oaa"].values
    return pool


def run_totals(con, con2, games, id_to_name):
    print(f"\n{'=' * 78}\ntotals with team-OAA feature\n{'=' * 78}")
    pool = build_enriched_pool(con, con2, games, id_to_name, "totals")
    oaa = build_oaa_asof(con2)
    pool = attach_oaa_asof(pool, oaa, "home_team_id", "home_oaa")
    pool = attach_oaa_asof(pool, oaa, "away_team_id", "away_oaa")
    pool["oaa_sum"] = pool["home_oaa"] + pool["away_oaa"]
    print(f"  oaa_sum coverage: {round(100 * pool['oaa_sum'].notna().mean(), 1)}%")

    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool["fold"] = fold_id

    all_bets = []
    for f in range(1, N_FOLDS):
        train_df = pool[pool["fold"] < f]
        test_df = pool[pool["fold"] == f]
        print(f"\n  fold {f}: train n={len(train_df)}  test n={len(test_df)}")
        if len(train_df) < 200 or len(test_df) < 20:
            print(f"  fold {f}: INSUFFICIENT DATA, skipped")
            continue
        X_train, y_train = train_df[NEW_FEATURES], train_df["win"].astype(int)
        model, cv_score, params = cv_select(X_train, y_train)
        print(f"  fold {f}: CV_AUC={cv_score:.4f}  params={params}")
        test_pred = model.predict_proba(test_df[NEW_FEATURES])[:, 1]
        t = test_df.copy()
        t["model_prob"] = test_pred
        t["edge"] = t["model_prob"] - t["market_prob"]
        all_bets.append(t[["game_date", "odds", "win", "edge"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== totals pooled walk-forward held-out bets: n={len(pooled)} ===")
    sub = pooled[(pooled.odds < 0) & (pooled.edge > 0.0)]
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"\ntotals edge>0.0, official gate rule, evaluated ONCE:")
    print(f"  n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log (Addendum 48)...")
    games, id_to_name = build_fuller_games(con, con2)

    run_totals(con, con2, games, id_to_name)

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
