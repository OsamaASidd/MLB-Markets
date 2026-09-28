"""
Addendum 73: market-offset ridge-logistic GLM for totals, same
architecture as Addendum 72 (h2h). Reuses Addendum 68's already-built
enriched pool (fuller game log + bullpen/park/corrected point-in-time
starter features) unmodified -- totals' pool already carries BOTH
best_over_odds/best_under_odds on every row (survives Addendum 48's
build_pool untouched), so de-vig needs no extra query.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import build_fuller_games  # noqa: E402
from addendum68_h2h_totals_fixed_starter_feature import build_enriched_pool, FEATURES  # noqa: E402
from glm_market_offset_walkforward import run_walkforward  # noqa: E402

GLM_FEATURES = [f for f in FEATURES if f not in ("side_code", "market_prob")]


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building fuller game log (Addendum 48)...")
    games, id_to_name = build_fuller_games(con, con2)

    print("\nbuilding totals pool with corrected point-in-time starter feature...")
    pool = build_enriched_pool(con, con2, games, id_to_name, "totals")
    print(f"  pool n={len(pool)}")

    over_p = implied_prob(pool["best_over_odds"].values)
    under_p = implied_prob(pool["best_under_odds"].values)
    total = over_p + under_p
    fair_over = over_p / total
    fair_under = under_p / total
    pool["market_prob"] = np.where(pool["side"].values == "over", fair_over, fair_under)
    pool["hold"] = total - 1.0
    print(f"  de-vig applied: mean overround={round(100 * (total.mean() - 1), 2)}%")

    run_walkforward(pool, GLM_FEATURES, "totals", hold_col="hold")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
