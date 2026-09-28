"""
Addendum 90: CLV evaluation for pitcher_strikeouts -- same design as
Addendum 89 (h2h). Trains the dedicated pitcher-form model (Addendum 59:
rolling K/9, opponent K-rate, park k_factor -- all point-in-time) on
data STRICTLY BEFORE the live snapshot window (< 2026-06-20), then tests
genuinely out-of-sample on the snapshot-window games: bet at the OPENING
price, check whether the line subsequently moved further toward our side
(CLV-positive) more often than chance.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum49_h2h_walkforward import cv_select  # noqa: E402
from addendum59_pitcher_k_eda import build_pool as build_dedicated_pool  # noqa: E402
from addendum60_pitcher_k_line_movement import build_movement_features, MAPPING  # noqa: E402

FEATURES = ["p5_k9", "p5_k_rate", "p5_ip_avg", "szn_k9", "szn_k_rate",
            "starts_count", "days_rest", "opp_k_rate", "k_factor", "is_dome"]
SPLIT_DATE = "2026-06-20"


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building dedicated pitcher-form pool (Addendum 59)...")
    pool = build_dedicated_pool(con, con2)
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    train_pool = pool[pool.game_date < SPLIT_DATE].dropna(subset=FEATURES + ["win"])
    print(f"  TRAIN pool (strictly before {SPLIT_DATE}): n={len(train_pool)}  "
          f"date range {train_pool.game_date.min().date()} -> {train_pool.game_date.max().date()}")

    print("\nfitting ONE model on the pre-snapshot-window pool (CV-AUC-tuned)...")
    X_train, y_train = train_pool[FEATURES], train_pool["win"].astype(int)
    model, cv_score, params = cv_select(X_train, y_train)
    print(f"  CV_AUC={cv_score:.4f}  params={params}")

    print("\nbuilding opening-snapshot pool from live cache_odds_snapshots_pitcher_k...")
    mv = build_movement_features(con2)
    id_map = pd.read_csv(MAPPING)[["game_pk", "event_id"]].drop_duplicates(subset=["event_id"])
    mv = mv.merge(id_map, on="event_id", how="inner")
    from xgboost_individual_markets import norm_name
    mv["name_norm"] = mv["player_name"].map(norm_name)
    print(f"  movement rows mapped to real game_pk: n={len(mv)}")

    n0 = len(mv)
    test_pool = mv.merge(pool[["game_pk", "name_norm", "game_date"] + FEATURES].drop_duplicates(subset=["game_pk", "name_norm"]),
                          on=["game_pk", "name_norm"], how="inner")
    print(f"  row-count check (merge with dedicated feature pool): before={n0} after={len(test_pool)} "
          f"{'OK' if len(test_pool) <= n0 else '*** FAN-OUT, STOP ***'}")
    assert len(test_pool) <= n0
    test_pool["game_date"] = pd.to_datetime(test_pool["game_date"])
    # NOTE: do NOT dropna on FEATURES -- k_factor/is_dome are ~100% null
    # for this recent backfilled window (a real park-factor coverage gap,
    # separate issue) and XGBoost's missing=np.nan handles this natively,
    # same convention used everywhere else in this project.
    test_pool = test_pool[test_pool.game_date >= SPLIT_DATE].dropna(subset=["open_prob", "movement"])
    print(f"  final CLV test pool (>= {SPLIT_DATE}, genuinely out-of-sample vs training): n={len(test_pool)}")

    if len(test_pool) < 20:
        print(f"\n  INSUFFICIENT DATA (n={len(test_pool)}) -- UNDERPOWERED even for a CLV test")
        return

    test_pred = model.predict_proba(test_pool[FEATURES])[:, 1]
    test_pool = test_pool.copy()
    test_pool["model_prob"] = test_pred
    test_pool["edge"] = test_pool["model_prob"] - test_pool["open_prob"]

    sub = test_pool[test_pool.edge > 0.0]
    print(f"\n=== bets at OPEN price with edge>0 (single pre-specified cut): n={len(sub)} ===")
    if len(sub) == 0:
        print("  edge>0.0 cut produced ZERO bets -- not evaluated")
        return

    clv_positive = int((sub["movement"] > 0).sum())
    n = len(sub)
    rate = clv_positive / n
    btest = binomtest(clv_positive, n, p=0.5, alternative="greater")
    print(f"\n=== CLV EVALUATION (report-only, NOT a PASS/FAIL by itself) ===")
    print(f"  n={n}  CLV-positive={clv_positive} ({round(100*rate,1)}%)  "
          f"one-sided binomial test vs 50%: p-value={btest.pvalue:.4f}")
    print(f"  {'STATISTICALLY SIGNIFICANT CLV edge (p<0.05)' if btest.pvalue < 0.05 else 'not significant at p<0.05'}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
