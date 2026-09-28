"""
Addendum 76: market-offset ridge-logistic GLM for batter_runs_scored.
Reuses Addendum 62's corrected pool-construction (both known fan-out bugs
already fixed there) unmodified, swapping the raw implied_prob(odds)
market feature for a real de-vig using best_over_odds/best_under_odds
(both already present via build_market_dataset_both_sides inside
Addendum 62's build_pool).
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob  # noqa: E402
from addendum62_batter_runs_scored_eda import build_pool  # noqa: E402
from glm_market_offset_walkforward import run_walkforward  # noqa: E402

GLM_FEATURES = ["elo_diff", "l10_diff", "fatigue_diff", "runs_factor", "hr_factor", "k_factor", "hits_factor"]


def main():
    con = duckdb.connect(DB, read_only=True)

    print("rebuilding batter_runs_scored pool (both fan-out bugs fixed, Addendum 62)...")
    pool = build_pool(con)

    over_p = implied_prob(pool["best_over_odds"].values)
    under_p = implied_prob(pool["best_under_odds"].values)
    total = over_p + under_p
    fair_over = over_p / total
    fair_under = under_p / total
    pool["market_prob"] = np.where(pool["side"].values == "over", fair_over, fair_under)
    print(f"  de-vig applied: mean overround={round(100 * (total.mean() - 1), 2)}%")
    pool = pool.dropna(subset=["win", "odds", "market_prob"])
    print(f"  pool ready: n={len(pool):,}")

    run_walkforward(pool, GLM_FEATURES, "batter_runs_scored", n_folds=6)

    con.close()


if __name__ == "__main__":
    main()
