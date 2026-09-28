"""
Addendum 77: market-offset ridge-logistic GLM for pitcher_strikeouts.
Reuses Addendum 59's already-built pool (dedicated pitcher-form features:
rolling K/9, opponent K-rate, park k_factor, is_dome -- all point-in-time
by construction) unmodified, swapping raw implied_prob(odds) for a real
de-vig using best_over_odds/best_under_odds (both survive on every row
from the original odds-parsing merge).
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, implied_prob  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB  # noqa: E402
from addendum59_pitcher_k_eda import build_pool  # noqa: E402
from glm_market_offset_walkforward import run_walkforward  # noqa: E402

GLM_FEATURES = ["p5_k9", "p5_k_rate", "p5_ip_avg", "szn_k9", "szn_k_rate",
                "starts_count", "days_rest", "opp_k_rate", "k_factor", "is_dome"]


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("building pitcher_strikeouts pool (Addendum 59 dedicated features)...")
    pool = build_pool(con, con2)

    over_p = implied_prob(pool["best_over_odds"].values)
    under_p = implied_prob(pool["best_under_odds"].values)
    total = over_p + under_p
    fair_over = over_p / total
    fair_under = under_p / total
    pool["market_prob"] = np.where(pool["side"].values == "over", fair_over, fair_under)
    print(f"  de-vig applied: mean overround={round(100 * (total.mean() - 1), 2)}%")
    pool = pool.dropna(subset=["win", "odds", "market_prob"])
    print(f"  pool ready: n={len(pool):,}")

    run_walkforward(pool, GLM_FEATURES, "pitcher_strikeouts", n_folds=6)

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
