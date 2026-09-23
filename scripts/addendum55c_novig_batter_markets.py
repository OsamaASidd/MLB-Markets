"""
Addendum 55c: no-vig market_prob correction, applied ONCE each to three
already-published individual-market pipelines -- batter_total_bases
(Addendum 39's genuine 2025 holdout), batter_runs_scored (Addendum 30's
real-odds baseline), batter_strikeouts (Addendum 53's enriched result).

WHY this run exists: every edge calculation in this project's history
(edge = model_prob - implied_prob(odds)) has compared the model's
predicted probability against the RAW, single-side implied probability of
the offered odds -- which is inflated by the bookmaker's overround (vig).
Standard practice in betting analytics is to de-vig: for a two-sided prop
with BOTH sides' real odds available for the same (player, game, line),

    fair_prob_side = implied_prob(this_side_odds)
                     / (implied_prob(over_odds) + implied_prob(under_odds))

using implied_prob() completely unmodified from xgboost_individual_markets.py.

GROUND RULES (this is a one-shot, pre-specified correction, not a new
feature search): do NOT retrain, re-tune, or re-select any model. Every
model below is fit EXACTLY as its source script already fits it -- same
features (including market_prob as a training feature, computed the
ORIGINAL raw way, untouched), same CV-AUC-only hyperparameter selection
where applicable, same chronological split. The ONLY change is: after the
model produces its (unchanged) model_prob predictions on the (unchanged)
test split, market_prob is recomputed once, de-vigged, and used ONLY to
recompute edge = model_prob - market_prob and re-evaluate the gate. This
is why every "RAW" number printed below is a reproduction check (it must
match the market's already-published verdict) and every "DEVIG" number is
the new, one-shot result. The gate is evaluated exactly once per market
per market_prob definition (raw reproduction, then devig) -- no threshold
search, no re-splitting, no model swap.

Data-plumbing finding (verified empirically before writing this script,
not assumed): none of the three markets required re-parsing the raw
JSONL/warehouse caches to recover the "other side"'s odds -- all three
already compute both best_over_odds/best_under_odds internally before
picking one side:
  - batter_total_bases: multi_model_comparison.build_warehouse_pool()
    queries `client_closing_odds.best_over_odds`/`best_under_odds`
    directly and never trims them from its returned pool -- both survive,
    untouched, all the way through build_features_v2().
  - batter_runs_scored / batter_strikeouts: test_backfilled_markets.
    parse_cache() already parses both best_over_dec/best_under_dec per
    (event, player); build_market_dataset() computes both American-odds
    columns too, but its FINAL column-select trims them away. This script
    adds build_market_dataset_both_sides(), a byte-identical reproduction
    of build_market_dataset() (same parse_cache/decimal_to_american calls,
    same merge, same row count -- asserted below) that simply keeps the
    two extra columns instead of dropping them. No new join, no new data
    source, no re-parse of anything -- just not throwing the columns away.

Reused UNMODIFIED (imported, never copy-edited): gate/stat/profit/
implied_prob/norm_name/build_team_game_log/build_elo_l10/
build_bullpen_fatigue (xgboost_individual_markets.py); build_games_and_box/
build_warehouse_pool (multi_model_comparison.py); build_games_v2/
build_box_ids/build_features_v2/add_opp_pitcher_features (addendum37_
historical_enrichment.py); parse_cache/decimal_to_american/
build_market_dataset (test_backfilled_markets.py); build_games_v3/
split_chronological/BASE_FEATURES/ENRICHED_FEATURES/GAME_LEVEL_COLS
(addendum53_batter_strikeouts_enrichment.py); BASE_FEATURES/
MARKET_PARK_FACTOR (addendum39_explainable_holdout.py); PARAM_GRID/
FIXED_PARAMS (tuned_xgboost_4_markets.py).
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
import joblib  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from xgboost_individual_markets import (  # noqa: E402
    DB, gate, stat, profit, implied_prob, norm_name,
    build_team_game_log, build_elo_l10, build_bullpen_fatigue,
)
from multi_model_comparison import build_games_and_box, build_warehouse_pool  # noqa: E402
from addendum37_historical_enrichment import (  # noqa: E402
    CACHE_DB, build_games_v2, build_box_ids, build_features_v2, add_opp_pitcher_features,
)
from test_backfilled_markets import parse_cache, decimal_to_american, build_market_dataset  # noqa: E402
from addendum39_explainable_holdout import (  # noqa: E402
    BASE_FEATURES as TB_BASE_FEATURES, MARKET_PARK_FACTOR,
)
from addendum53_batter_strikeouts_enrichment import (  # noqa: E402
    BASE_FEATURES as SK_BASE_FEATURES, ENRICHED_FEATURES as SK_ENRICHED_FEATURES,
    GAME_LEVEL_COLS as SK_GAME_LEVEL_COLS, build_games_v3, split_chronological as sk_split_chronological,
)
from tuned_xgboost_4_markets import PARAM_GRID, FIXED_PARAMS  # noqa: E402

# Already-published raw-baseline numbers this run reproduces and then
# corrects (cited, not re-derived -- see each market's own addendum for the
# full published number). Used only for an honest side-by-side print.
PUBLISHED_BASELINE = {
    "batter_total_bases": "Addendum 39/46: edge>0.0 cut produced ZERO bets on the 2025 holdout "
                           "(raw market_prob) -- not evaluated as PASS/FAIL",
    "batter_runs_scored": "Addendum 30: n=52,981  ROI=-2.54%  CI=[-4.05%,-1.03%]  FAIL",
    "batter_strikeouts": "Addendum 53 Model B (enriched, tuned): n=1,211  ROI=1.24%  FAIL "
                         "(flagged SUSPECT -- near-zero marginal CV-AUC vs Model A)",
}


# --------------------------------------------------------------------------
# De-vig helper -- the ONE new piece of logic in this whole script.
# --------------------------------------------------------------------------
def devig_market_prob(df, tag):
    """Standard two-sided no-vig fair probability, using implied_prob()
    completely unmodified. Falls back to the untouched raw implied_prob(odds)
    for the (small) remainder of rows where only one side is quoted at the
    picked line/consensus -- there is no vig to remove if we don't know the
    other side's price. Prints the mandatory vig-magnitude/coverage check."""
    over_odds = pd.to_numeric(df["best_over_odds"], errors="coerce")
    under_odds = pd.to_numeric(df["best_under_odds"], errors="coerce")
    odds = pd.to_numeric(df["odds"], errors="coerce")
    over_prob = implied_prob(over_odds)
    under_prob = implied_prob(under_odds)
    total = over_prob + under_prob
    both = over_odds.notna().values & under_odds.notna().values

    with np.errstate(invalid="ignore", divide="ignore"):
        fair_over = over_prob / total
        fair_under = under_prob / total
    fair_side = np.where(df["side"].values == "over", fair_over, fair_under)
    raw = implied_prob(odds)
    market_prob = np.where(both, fair_side, raw)

    overround_pct = (total[both] - 1.0) * 100
    n_both = int(both.sum())
    print(f"  [{tag}] no-vig check: n={len(df):,}  both-sides-quoted={n_both:,} "
          f"({round(100 * n_both / len(df), 1) if len(df) else 0}%)  "
          f"mean overround={round(float(overround_pct.mean()), 2) if n_both else 'n/a'}%  "
          f"median overround={round(float(np.median(overround_pct)), 2) if n_both else 'n/a'}%  "
          f"(remaining {len(df) - n_both:,} rows: single side quoted, raw implied_prob kept, no vig removable)")
    return market_prob


def report_cut(test, edge, label):
    sub = test[(test.odds < 0) & (edge > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    if s["n"] is None or s["n"] == 0:
        print(f"    [{label}] edge>0.0 cut produced ZERO bets -- not evaluated")
        return {"label": label, "verdict": "no_bets", **s}
    passed = gate(s)
    underpowered = (s["n"] < 500) and (s["lo"] is not None) and (s["hi"] is not None) and (s["hi"] - s["lo"] > 40)
    verdict = "PASS" if passed else ("UNDERPOWERED" if underpowered else "FAIL")
    print(f"    [{label}] n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return {"label": label, "verdict": verdict, **s}


def bet_flip_count(test, edge_raw, edge_devig):
    raw_bet = ((test.odds < 0) & (edge_raw > 0)).values
    devig_bet = ((test.odds < 0) & (edge_devig > 0)).values
    flips = int((raw_bet != devig_bet).sum())
    into_bets = int((~raw_bet & devig_bet).sum())
    out_of_bets = int((raw_bet & ~devig_bet).sum())
    print(f"    bet-selection flips (raw -> devig) on this test split: {flips} of {len(test):,} rows  "
          f"(newly IN: {into_bets}, newly OUT: {out_of_bets})")
    return flips, into_bets, out_of_bets


# --------------------------------------------------------------------------
# Market 1: batter_total_bases -- Addendum 39's explainable holdout,
# logistic regression, 6 pre-registered features, chronological 70/30 split.
# --------------------------------------------------------------------------
def run_total_bases(con, con2):
    print("\n" + "=" * 78)
    print("MARKET 1: batter_total_bases  (Addendum 39 pipeline, unmodified training)")
    print("=" * 78)

    games, box = build_games_and_box(con)
    games_v2 = build_games_v2(con, con2, games)
    box_ids = build_box_ids(con)

    market_name, stat_col = "batter_total_bases", "total_bases"
    park_col = MARKET_PARK_FACTOR[market_name]
    features = TB_BASE_FEATURES + [park_col]

    pool = build_warehouse_pool(con, box, market_name, stat_col, "batter")
    pool = build_features_v2(pool, games_v2, "batter", box_ids, con2)
    assert {"best_over_odds", "best_under_odds"}.issubset(pool.columns), \
        "both-sides odds columns did not survive the unmodified merge chain -- STOP"

    pool = pool.dropna(subset=features + ["win", "odds", "game_date"]).copy()
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    cut = int(len(pool) * 0.70)
    train, test = pool.iloc[:cut].copy(), pool.iloc[cut:].copy()
    print(f"  features (unchanged): {features}")
    print(f"  full range: {pool['game_date'].min().date()} -> {pool['game_date'].max().date()}")
    print(f"  train n={len(train):,}  test n={len(test):,}")

    if len(train) < 100 or len(test) < 20:
        print("  INSUFFICIENT DATA -- not evaluated")
        return

    X_train, y_train = train[features], train["win"].astype(int)
    X_test = test[features]
    # No hyperparameter search -- unchanged from Addendum 39 (default C=1.0).
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    test = test.copy()
    test["model_prob"] = model.predict_proba(X_test)[:, 1]

    market_prob_raw = test["market_prob"].values  # unchanged training feature: implied_prob(odds)
    edge_raw = test["model_prob"].values - market_prob_raw
    market_prob_devig = devig_market_prob(test, market_name)
    edge_devig = test["model_prob"].values - market_prob_devig

    print("  --- reproduction check (raw market_prob, should match Addendum 39/46) ---")
    r_raw = report_cut(test, edge_raw, "RAW reproduction")
    print("  --- Addendum 55c: de-vigged market_prob, one-shot correction ---")
    r_devig = report_cut(test, edge_devig, "DEVIG (Addendum 55c)")
    bet_flip_count(test, edge_raw, edge_devig)
    print(f"  published baseline: {PUBLISHED_BASELINE[market_name]}")
    return {"market": market_name, "raw": r_raw, "devig": r_devig}


# --------------------------------------------------------------------------
# Market 2 & 3 shared plumbing: real backfilled odds cache, both sides kept.
# --------------------------------------------------------------------------
def build_market_dataset_both_sides(cache_path, market_key, stat_col, box, position_filter, starter_only):
    """Byte-identical reproduction of test_backfilled_markets.build_market_dataset
    (same parse_cache/decimal_to_american calls, same merge, same dropna) --
    the ONLY difference is the final column-select keeps best_over_odds/
    best_under_odds instead of discarding them. Row-count parity against the
    original, unmodified function is asserted by the caller."""
    odds = parse_cache(cache_path, market_key)
    print(f"  parsed {len(odds):,} real (event,player) rows from cache")
    odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce")
    odds["name_norm"] = odds["player_name"].map(norm_name)
    odds["best_over_odds"] = odds["best_over_dec"].map(decimal_to_american)
    odds["best_under_odds"] = odds["best_under_dec"].map(decimal_to_american)
    odds = odds.dropna(subset=["game_pk"])

    box_f = box[box.position_type.isin(position_filter)]
    if starter_only:
        box_f = box_f[box_f.is_starter == True]
    m = odds.merge(box_f[["game_pk", "name_norm", stat_col]], on=["game_pk", "name_norm"], how="inner")
    m = m.dropna(subset=[stat_col])
    print(f"  {len(m):,} rows matched to a real box-score outcome")

    under = m.copy(); under["win"] = under[stat_col] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = under["name_norm"]
    over = m.copy(); over["win"] = over[stat_col] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = over["name_norm"]
    both = pd.concat([under, over]).dropna(subset=["odds"]).assign(market_key=market_key)
    cols = ["market_key", "game_pk", "commence_time", "odds", "win", "side", "tiebreak",
            "best_over_odds", "best_under_odds"]
    return both[cols]


def assert_parity(cache_path, market_key, stat_col, box, position_filter, starter_only, mine):
    """Mandatory row-count sanity check: confirm the both-sides-retaining
    reproduction above is not silently a different dataset from the
    published, unmodified build_market_dataset()."""
    orig = build_market_dataset(cache_path, market_key, stat_col, box, position_filter, starter_only)
    assert len(orig) == len(mine), (
        f"row-count MISMATCH vs unmodified build_market_dataset for {market_key}: "
        f"orig={len(orig)} mine={len(mine)} -- STOP")
    assert orig["win"].sum() == mine["win"].sum()
    print(f"  parity check OK: both-sides reproduction matches unmodified build_market_dataset "
          f"exactly (n={len(mine):,}, same win count)")


# --------------------------------------------------------------------------
# Market 2: batter_runs_scored -- Addendum 30's exact real-odds pipeline.
# --------------------------------------------------------------------------
def run_runs_scored(con):
    print("\n" + "=" * 78)
    print("MARKET 2: batter_runs_scored  (Addendum 30 / test_backfilled_markets.py pipeline, "
          "unmodified training)")
    print("=" * 78)

    games = build_team_game_log(con)
    elo_l10 = build_elo_l10(games)
    fatigue = build_bullpen_fatigue(con)
    games = games.merge(elo_l10, on="game_pk")
    games["home_fatigue"] = games.apply(lambda r: fatigue.get((r.home_team_id, r.game_date), 0), axis=1)
    games["away_fatigue"] = games.apply(lambda r: fatigue.get((r.away_team_id, r.game_date), 0), axis=1)

    parks = con.execute("SELECT * FROM ballpark_factors").fetchdf()
    for c in ["runs_factor", "hr_factor", "k_factor", "hits_factor"]:
        parks[c] = pd.to_numeric(parks[c], errors="coerce")
    park_by_event = con.execute("SELECT DISTINCT event_id, venue_name FROM weather").fetchdf()
    park_by_event = park_by_event.merge(parks, left_on="venue_name", right_on="park_name", how="inner")
    event_to_gamepk = con.execute("SELECT event_id, game_pk FROM client_games").fetchdf()
    event_to_gamepk["game_pk"] = pd.to_numeric(event_to_gamepk["game_pk"], errors="coerce")
    park_by_gamepk = park_by_event.merge(event_to_gamepk, on="event_id", how="inner")[
        ["game_pk", "runs_factor", "hr_factor", "k_factor", "hits_factor"]]
    games = games.merge(park_by_gamepk, on="game_pk", how="left")

    box = con.execute("""
        SELECT game_pk, player_name, runs_scored, outs AS pitcher_outs_stat, is_starter, position_type
        FROM boxscore
    """).fetchdf()
    box["name_norm"] = box["player_name"].map(norm_name)

    market_key, stat_col = "batter_runs_scored", "runs_scored"
    cache_path = ROOT / "data_raw" / "runs_scored_odds_cache.jsonl"
    position_filter, starter_only = ["Catcher", "Hitter", "Infielder", "Outfielder"], False

    pool = build_market_dataset_both_sides(cache_path, market_key, stat_col, box, position_filter, starter_only)
    assert_parity(cache_path, market_key, stat_col, box, position_filter, starter_only, pool)

    pool = pool.merge(games[["game_pk", "game_date", "home_elo", "away_elo", "home_l10", "away_l10",
                              "home_fatigue", "away_fatigue", "runs_factor", "hr_factor", "k_factor", "hits_factor"]],
                       on="game_pk", how="inner")
    if len(pool) < 500:
        print(f"  INSUFFICIENT DATA: n={len(pool)}")
        return

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])  # unchanged training feature
    features = ["side_code", "market_prob", "elo_diff", "l10_diff", "fatigue_diff",
                "runs_factor", "hr_factor", "k_factor", "hits_factor"]

    pool = pool.sort_values(["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    split = int(len(pool) * 0.75)
    train, test = pool.iloc[:split].copy(), pool.iloc[split:].copy()
    print(f"  TRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
        reg_lambda=3.0, eval_metric="logloss", missing=np.nan, random_state=0, n_jobs=1,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred >= 0.5)
    print(f"  AUC TEST={auc:.3f}  accuracy TEST={acc:.3f}  (reproduces Addendum 30, reported only)")

    test = test.copy()
    test["model_prob"] = test_pred
    market_prob_raw = test["market_prob"].values
    edge_raw = test["model_prob"].values - market_prob_raw
    market_prob_devig = devig_market_prob(test, market_key)
    edge_devig = test["model_prob"].values - market_prob_devig

    print("  --- reproduction check (raw market_prob, should match Addendum 30) ---")
    r_raw = report_cut(test, edge_raw, "RAW reproduction")
    print("  --- Addendum 55c: de-vigged market_prob, one-shot correction ---")
    r_devig = report_cut(test, edge_devig, "DEVIG (Addendum 55c)")
    bet_flip_count(test, edge_raw, edge_devig)
    print(f"  published baseline: {PUBLISHED_BASELINE[market_key]}")
    return {"market": market_key, "raw": r_raw, "devig": r_devig}


# --------------------------------------------------------------------------
# Market 3: batter_strikeouts -- Addendum 53's enriched, CV-AUC-tuned pipeline.
# --------------------------------------------------------------------------
def attach_features_both_sides(pool, games_v3, box_ids, con2):
    """Byte-identical reproduction of addendum53's attach_features(), except
    market_prob (the training feature) is computed the SAME raw way -- the
    only addition is that best_over_odds/best_under_odds (already present in
    `pool` via build_market_dataset_both_sides) simply survive these merges
    untouched, same as every other passenger column."""
    n0 = len(pool)
    pool = pool.merge(games_v3[SK_GAME_LEVEL_COLS], on="game_pk", how="inner")
    print(f"  pool + game-level features, INNER on game_pk: n={n0:,} -> {len(pool):,}")

    pool["name_norm"] = pool["tiebreak"]
    n1 = len(pool)
    pool = add_opp_pitcher_features(pool, games_v3, box_ids, con2)
    assert len(pool) == n1, "*** opposing-pitcher merge changed row count -- FAN-OUT, STOP ***"
    print(f"  pool + opposing-starting-pitcher quality, LEFT join: n={n1:,} -> {len(pool):,}  OK")

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])  # unchanged training feature
    pool["game_date"] = pd.to_datetime(pool["game_date"])
    return pool


def tune_and_evaluate_rowlevel(label, train, test, features):
    """Same GridSearchCV fit as addendum53.tune_and_evaluate (same PARAM_GRID/
    FIXED_PARAMS, same 5-fold StratifiedKFold, same threading backend) --
    exposes the row-level test predictions instead of computing edge/gate
    internally, so this script can recompute edge with both a raw and a
    de-vigged market_prob on the SAME, unmodified model predictions."""
    X_train, y_train = train[features], train["win"].astype(int)
    X_test, y_test = test[features], test["win"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        xgb.XGBClassifier(**FIXED_PARAMS),
        PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=os.cpu_count(), refit=True,
    )
    with joblib.parallel_backend("threading"):
        search.fit(X_train, y_train)  # CV AUC only -- no ROI/gate touched here

    best_model = search.best_estimator_
    test_pred = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    test_acc = accuracy_score(y_test, test_pred >= 0.5)

    t = test.copy()
    t["model_prob"] = test_pred
    print(f"\n  [{label}]")
    print(f"    features ({len(features)}): {features}")
    print(f"    best CV AUC (train only, 5-fold)={search.best_score_:.4f}  params={search.best_params_}")
    print(f"    test AUC={test_auc:.4f}  test acc={test_acc:.4f}  (reported only, not used for selection)")
    return t, {"cv_auc": round(search.best_score_, 4), "best_params": search.best_params_,
               "test_auc": round(test_auc, 4), "test_acc": round(test_acc, 4)}


def run_strikeouts(con, con2):
    print("\n" + "=" * 78)
    print("MARKET 3: batter_strikeouts  (Addendum 53 enriched pipeline, unmodified CV-AUC tuning)")
    print("=" * 78)

    box_bk = con.execute("""
        SELECT game_pk, player_name, batter_strikeouts, is_starter, position_type FROM boxscore
    """).fetchdf()
    box_bk["name_norm"] = box_bk["player_name"].map(norm_name)
    market_key, stat_col = "batter_strikeouts", "batter_strikeouts"
    cache_path = ROOT / "data_raw" / "batter_strikeouts_odds_cache.jsonl"
    position_filter, starter_only = ["Catcher", "Hitter", "Infielder", "Outfielder"], False

    pool = build_market_dataset_both_sides(cache_path, market_key, stat_col, box_bk, position_filter, starter_only)
    print(f"  pool (pre game-log join): n={len(pool):,}  unique game_pk={pool.game_pk.nunique():,}")
    assert_parity(cache_path, market_key, stat_col, box_bk, position_filter, starter_only, pool)

    games_v3 = build_games_v3(con, con2)
    box_ids = build_box_ids(con)

    pool = attach_features_both_sides(pool, games_v3, box_ids, con2)
    train, test = sk_split_chronological(pool)
    print(f"  TRAIN n={len(train):,} ({train.game_date.min().date()}->{train.game_date.max().date()})  "
          f"TEST n={len(test):,} ({test.game_date.min().date()}->{test.game_date.max().date()})")

    if len(pool) < 500 or len(train) < 50 or len(test) < 50:
        print("  INSUFFICIENT DATA -- stopping")
        return

    results = {}
    for label, features in [("Model A: baseline features (no enrichment), tuned", SK_BASE_FEATURES),
                             ("Model B: enriched features (Addendum 37 groups), tuned", SK_ENRICHED_FEATURES)]:
        t, meta = tune_and_evaluate_rowlevel(label, train, test, features)

        market_prob_raw = t["market_prob"].values
        edge_raw = t["model_prob"].values - market_prob_raw
        market_prob_devig = devig_market_prob(t, f"{market_key} / {label.split(':')[0]}")
        edge_devig = t["model_prob"].values - market_prob_devig

        print("    --- reproduction check (raw market_prob, should match Addendum 53) ---")
        r_raw = report_cut(t, edge_raw, "RAW reproduction")
        print("    --- Addendum 55c: de-vigged market_prob, one-shot correction ---")
        r_devig = report_cut(t, edge_devig, "DEVIG (Addendum 55c)")
        bet_flip_count(t, edge_raw, edge_devig)
        results[label] = {"meta": meta, "raw": r_raw, "devig": r_devig}

    print(f"  published baseline: {PUBLISHED_BASELINE[market_key]}")
    return {"market": market_key, **results}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    print("=== Addendum 55c: de-vigged market_prob, one-shot correction, three markets ===")
    print("De-vig formula (standard, two-sided): fair_prob_side = implied_prob(this_side_odds) / "
          "(implied_prob(over_odds) + implied_prob(under_odds)); implied_prob() unmodified from "
          "xgboost_individual_markets.py. Fallback to raw implied_prob(odds) only for the rows "
          "where the other side's price isn't quoted at the picked line (no vig removable there).")

    all_results = {}
    r1 = run_total_bases(con, con2)
    if r1:
        all_results[r1["market"]] = r1
    r2 = run_runs_scored(con)
    if r2:
        all_results[r2["market"]] = r2
    r3 = run_strikeouts(con, con2)
    if r3:
        all_results[r3["market"]] = r3

    con.close()
    con2.close()

    print("\n" + "=" * 78)
    print("SUMMARY -- Addendum 55c")
    print("=" * 78)
    for market, r in all_results.items():
        print(f"\n{market}:")
        print(f"  published baseline: {PUBLISHED_BASELINE[market]}")
        if "raw" in r:
            print(f"  RAW reproduction : {r['raw']}")
            print(f"  DEVIG (this run) : {r['devig']}")
        else:
            for label, sub in r.items():
                if label == "market":
                    continue
                print(f"  [{label}]")
                print(f"    RAW reproduction : {sub['raw']}")
                print(f"    DEVIG (this run) : {sub['devig']}")


if __name__ == "__main__":
    main()
