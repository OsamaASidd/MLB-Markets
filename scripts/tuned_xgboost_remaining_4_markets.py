"""
Addendum 36: disciplined hyperparameter tuning on the 4 FAIL markets that
showed ZERO passes across all 6 models in Addendum 33 (batter_total_bases,
h2h, totals, batter_runs_scored) -- completing the tuning pass Addendum 34
only ran on the 4 markets that had shown at least one pass.

Same pre-registered method as Addendum 34, no changes:
  - ONE model family (XGBoost), chosen once, not per-market.
  - ONE pre-specified hyperparameter grid, identical to Addendum 34.
  - Selection metric: CV-averaged ROC AUC on the TRAINING split only
    (5-fold StratifiedKFold) -- ROI/gate is never touched during selection.
  - Refit once on the full training split, scored ONCE on the untouched
    held-out test split, at the same edge>0.0 cut and official gate rule.
  - Every result reported, pass or fail.

Reuses the same pool/feature-building code already published in
multi_model_comparison.py -- does not modify it.
"""
from tuned_xgboost_4_markets import ROOT, DB, tune_and_evaluate  # reuse Addendum 34's grid/eval code
import duckdb  # noqa: E402
from multi_model_comparison import (  # noqa: E402
    build_games_and_box, build_warehouse_pool, build_game_pool,
)
from test_backfilled_markets import build_market_dataset  # noqa: E402
from xgboost_individual_markets import norm_name  # noqa: E402


def main():
    con = duckdb.connect(DB, read_only=True)
    print("building real team game log, Elo, L10, bullpen fatigue, park factors...")
    games, box = build_games_and_box(con)

    print("\n=== Addendum 36: tuned XGBoost on the 4 markets that showed ZERO passes in Addendum 33 ===")

    # batter_total_bases -- warehouse pool, per-year split (same as Addendum 32/33/34)
    pool = build_warehouse_pool(con, box, "batter_total_bases", "total_bases", "batter")
    tune_and_evaluate("batter_total_bases", pool, games, "per_year")

    # h2h, totals -- game pool, per-year split
    for market_key in ["h2h", "totals"]:
        pool = build_game_pool(con, games, market_key)
        tune_and_evaluate(market_key, pool, games, "per_year")

    # batter_runs_scored -- real backfilled odds, chronological split
    cache_path = ROOT / "data_raw" / "runs_scored_odds_cache.jsonl"
    if cache_path.exists():
        box_bk = con.execute("""
            SELECT game_pk, player_name, runs_scored, is_starter, position_type FROM boxscore
        """).fetchdf()
        box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
        pool = build_market_dataset(
            cache_path, "batter_runs_scored", "runs_scored", box_bk,
            ["Catcher", "Hitter", "Infielder", "Outfielder"], False,
        )
        tune_and_evaluate("batter_runs_scored", pool, games, "chronological")
    else:
        print(f"  batter_runs_scored: cache file missing ({cache_path.name}), skipping")

    con.close()


if __name__ == "__main__":
    main()
