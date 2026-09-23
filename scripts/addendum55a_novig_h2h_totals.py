"""
Addendum 55a: recompute h2h and totals' best-known walk-forward result with
a de-vigged (no-vig) market_prob instead of the raw single-side implied
probability every prior addendum has used.

BACKGROUND: every edge calculation in this project's history has been
`edge = model_prob - implied_prob(this_side_odds)` -- the RAW, single-side
implied probability, which still contains the bookmaker's overround (vig).
Standard practice in betting analytics is to remove the vig first: for a
two-sided market, convert BOTH sides' odds to raw implied probability, then
normalize:  fair_prob_side = implied_prob(side_odds) / (implied_prob(side_A)
+ implied_prob(side_B)).  This is a one-shot, pre-specified correction to
the edge-calculation baseline -- NOT a new feature, NOT a re-tune.

GROUND RULES (see prompt): do not retrain/re-tune/re-select the model.
Reuse the EXACT SAME model-training process already established for each
market's best-known walk-forward result (same CV-AUC-only hyperparameter
selection via cv_select/PARAM_GRID_WIDE, same features -- including
market_prob as a *feature*, computed the same raw way it always has been,
so the trained model itself is byte-for-byte the same pipeline as
Addendum 49 (h2h) / Addendum 50 (totals)). The ONLY change: at the point
where each fold's held-out predictions are used to decide which bets clear
the edge>0.0 gate, ALSO compute a de-vigged market_prob and use THAT for
edge = model_prob - market_prob instead of the raw one. Gate evaluated
EXACTLY ONCE per market on the pooled de-vigged result.

h2h: build_pool() only ever pulls the away side's odds (market_key=
'h2h__away'). The other side lives at market_key='h2h__home' (verified
directly against client_closing_odds this session -- 7,440 rows each,
exactly one row per event_id, best_over_odds carries the moneyline price
for that market_key's side, best_under_odds always null for h2h). Paired
by event_id (a true 1:1 join at the odds-table level, confirmed: every
(event_id, market_key) combination is unique). That event-level pair table
is then matched onto build_pool()'s own away-side rows by (game_pk,
away_odds) -- NOT by game_pk alone -- because a small number of game_pks
carry more than one event_id (double-header data-quality artifact already
present in build_pool()'s existing behavior, left unmodified); matching on
the odds value too pins down the correct home-side price for each duplicate
without fanning out the pool. 18 of 7,088 (game_pk, away_odds) combinations
are still ambiguous after that (kept first row deterministically, disclosed
below) -- immaterial at this n.

totals: build_pool()'s totals branch already carries best_over_odds AND
best_under_odds on every row (both under-rows and over-rows are built from
the same merged frame before the under/over split) -- direct, no extra
join needed, exactly as the task predicted.
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import duckdb  # noqa: E402
from xgboost_individual_markets import DB, gate, stat, profit, implied_prob, pick_main_line  # noqa: E402
from addendum37_historical_enrichment import CACHE_DB, TEAM_NAME_ALIAS  # noqa: E402
from addendum48_h2h_totals_fuller_gamelog import (  # noqa: E402
    build_fuller_games, build_pool, build_features, FEATURES, EXCLUDE_TEAMS,
)
from addendum49_h2h_walkforward import attach_enrichment, cv_select, PARAM_GRID_WIDE  # noqa: E402
from addendum50_totals_gap_integration import (  # noqa: E402
    decimal_to_american, parse_gap_odds, build_extended_games, GAP_CACHE,
)

N_FOLDS = 6


def run_walkforward(pool, label):
    """Identical walk-forward structure to Addendum 49/50: expanding-window
    folds, CV-AUC hyperparameter selection on each fold's TRAINING data only
    (cv_select/PARAM_GRID_WIDE, unmodified), held-out predictions pooled.
    Tracks BOTH the raw edge (market_prob, the existing feature/baseline)
    and the de-vigged edge (market_prob_novig, computed post-hoc, used only
    at bet-selection time) side by side on the exact same folds/models."""
    fold_id = pd.qcut(np.arange(len(pool)), N_FOLDS, labels=False)
    pool = pool.copy()
    pool["fold"] = fold_id

    all_bets = []
    print(f"\n=== {label}: walk-forward folds (expanding window, {N_FOLDS} folds) ===")
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
        t["edge_raw"] = t["model_prob"] - t["market_prob"]
        t["edge_novig"] = t["model_prob"] - t["market_prob_novig"]
        all_bets.append(t[["game_date", "odds", "win", "edge_raw", "edge_novig", "overround"]])

    pooled = pd.concat(all_bets, ignore_index=True)
    print(f"\n=== {label}: pooled walk-forward held-out bets across folds 1-{N_FOLDS - 1}: n={len(pooled)} ===")
    return pooled


def report_gate(pooled, edge_col, label):
    sub = pooled[(pooled.odds < 0) & (pooled[edge_col] > 0.0)]
    profits = sub.apply(lambda r: profit(r["win"], r["odds"]), axis=1)
    s = stat(profits, sub["win"])
    verdict = "PASS" if gate(s) else "FAIL"
    print(f"  [{label}] edge>0.0 (odds<0), official gate rule: "
          f"n={s['n']}  WR={s['wr']}%  ROI={s['roi']}%  CI=[{s['lo']},{s['hi']}]  {verdict}")
    return s, verdict


def sign_flip_report(pooled):
    cand = pooled[pooled.odds < 0].copy()
    raw_pos = cand["edge_raw"] > 0.0
    novig_pos = cand["edge_novig"] > 0.0
    flipped_to_pos = (~raw_pos & novig_pos).sum()
    flipped_to_neg = (raw_pos & ~novig_pos).sum()
    print(f"  sign flips among odds<0 candidates (n={len(cand)}): "
          f"{flipped_to_pos} flipped negative->positive (newly enter the bet set), "
          f"{flipped_to_neg} flipped positive->negative (drop out of the bet set), "
          f"total changed = {flipped_to_pos + flipped_to_neg}")
    return flipped_to_pos, flipped_to_neg


# ---------------------------------------------------------------- h2h -----

def build_h2h_home_side_map(con):
    """Event-level away/home odds pair table (true 1:1 per event_id), used
    to attach the missing 'other side' price onto build_pool()'s away-only
    h2h rows, matched by (game_pk, away_odds) to avoid fanning out the
    small number of game_pks that carry more than one event_id."""
    pair = con.execute("""
        SELECT co.event_id, co.game_pk, co.market_key, co.best_over_odds AS odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key IN ('h2h__away', 'h2h__home')
          AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    pair["game_pk"] = pd.to_numeric(pair["game_pk"], errors="coerce")
    pair["odds"] = pd.to_numeric(pair["odds"], errors="coerce")
    pair = pair.dropna(subset=["game_pk", "odds"])
    pair["game_pk"] = pair["game_pk"].astype("int64")

    wide = pair.pivot_table(index=["event_id", "game_pk"], columns="market_key",
                             values="odds", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"h2h__away": "away_odds", "h2h__home": "home_odds"})
    wide = wide.dropna(subset=["away_odds", "home_odds"])
    n_wide = len(wide)
    dup = wide.duplicated(subset=["game_pk", "away_odds"]).sum()
    wide = wide.drop_duplicates(subset=["game_pk", "away_odds"], keep="first")
    print(f"  h2h other-side pair table: {n_wide} event-level (away,home) pairs built; "
          f"{dup} ambiguous (game_pk,away_odds) combos collapsed (kept first, immaterial at this n)")
    return wide[["game_pk", "away_odds", "home_odds"]]


def run_h2h(con, con2):
    print("\n" + "=" * 78)
    print("h2h -- de-vigged edge, same pipeline as Addendum 49 (unmodified)")
    print("=" * 78)
    games, id_to_name = build_fuller_games(con, con2)
    pool = build_pool(con, games, "h2h")
    n_before = len(pool)
    print(f"  h2h pool from build_pool() (unmodified): n={n_before}")

    home_map = build_h2h_home_side_map(con)
    n_pre_join = len(pool)
    pool = pool.merge(home_map, left_on=["game_pk", "odds"], right_on=["game_pk", "away_odds"], how="left")
    n_matched = pool["home_odds"].notna().sum()
    print(f"  row-count sanity check (other-side join): before={n_pre_join}  after={len(pool)}  "
          f"{'OK, no fan-out' if len(pool) == n_pre_join else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_pre_join, "other-side join changed row count -- STOP"
    print(f"  matched home-side odds for {n_matched}/{len(pool)} rows "
          f"({100 * n_matched / len(pool):.1f}%)")

    pool = attach_enrichment(con, con2, pool, id_to_name)
    print(f"  row-count sanity check (enrichment): before={n_before}  after={len(pool)}  "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool = build_features(pool)  # unmodified -- sets market_prob = implied_prob(odds), the RAW feature, same as always
    pool["game_date"] = pd.to_datetime(pool["game_date"])

    # de-vigged probability, computed post-hoc, NOT fed back as a training feature
    pool["p_away_raw"] = implied_prob(pool["odds"])
    pool["p_home_raw"] = implied_prob(pool["home_odds"])
    pool["overround"] = pool["p_away_raw"] + pool["p_home_raw"]
    pool["market_prob_novig"] = pool["p_away_raw"] / pool["overround"]

    n_pre_drop = len(pool)
    pool = pool.dropna(subset=["win", "odds", "market_prob_novig"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"  final h2h pool: n={len(pool)} (dropped {n_pre_drop - len(pool)} rows missing "
          f"win/odds/other-side match)  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")
    print(f"  MEASURED h2h vig: mean overround={pool['overround'].mean():.4f} "
          f"(i.e. avg {100 * (pool['overround'].mean() - 1):.2f}% vig)  "
          f"median={pool['overround'].median():.4f}  min={pool['overround'].min():.4f}  max={pool['overround'].max():.4f}")

    pooled = run_walkforward(pool, "h2h")
    print("\nh2h RESULTS:")
    s_raw, v_raw = report_gate(pooled, "edge_raw", "RAW market_prob (baseline, this run)")
    s_novig, v_novig = report_gate(pooled, "edge_novig", "DE-VIGGED market_prob (Addendum 55a)")
    sign_flip_report(pooled)
    print("\n  For reference, Addendum 49's originally-reported baseline: n=1,114  ROI=-0.37%  CI=[-5.55,4.81]  FAIL")
    return {"market": "h2h", "raw": s_raw, "raw_verdict": v_raw, "novig": s_novig, "novig_verdict": v_novig}


# -------------------------------------------------------------- totals ----

def run_totals(con, con2):
    print("\n" + "=" * 78)
    print("totals -- de-vigged edge, same pipeline as Addendum 50 (unmodified)")
    print("=" * 78)
    print("building Addendum 48 fuller game log (through 2026-06-25)...")
    games_v2, id_to_name = build_fuller_games(con, con2)
    print("extending game log with cache_mlb_historical_outcomes through 2026-09-23...")
    games_ext = build_extended_games(con, con2, games_v2, id_to_name)
    print(f"  extended game log: n={len(games_ext)}  date range {games_ext.game_date.min()} -> {games_ext.game_date.max()}")

    print("parsing freshly-backfilled totals odds...")
    gap_odds = parse_gap_odds()

    existing_odds = con.execute("""
        SELECT co.game_pk, co.line, co.best_over_odds, co.best_under_odds
        FROM (SELECT co.*, g.game_pk FROM client_closing_odds co
              JOIN client_games g ON co.event_id = g.event_id) co
        WHERE co.market_key = 'totals' AND co.game_pk IS NOT NULL AND co.game_pk != ''
    """).fetchdf()
    existing_odds["game_pk"] = pd.to_numeric(existing_odds["game_pk"], errors="coerce")
    existing_odds["line"] = pd.to_numeric(existing_odds["line"], errors="coerce")
    existing_odds["best_over_odds"] = pd.to_numeric(existing_odds["best_over_odds"], errors="coerce")
    existing_odds["best_under_odds"] = pd.to_numeric(existing_odds["best_under_odds"], errors="coerce")
    existing_odds = existing_odds.dropna(subset=["game_pk"])
    existing_odds["game_pk"] = existing_odds["game_pk"].astype("int64")
    existing_odds = pick_main_line(existing_odds, ["game_pk"], over_odds_col="best_over_odds")

    overlap = set(existing_odds.game_pk) & set(gap_odds.game_pk)
    print(f"  existing odds: {len(existing_odds)} games | gap-backfilled odds: {len(gap_odds)} games | overlap: {len(overlap)}")
    gap_odds_clean = gap_odds[~gap_odds.game_pk.isin(set(existing_odds.game_pk))]
    all_odds = pd.concat([existing_odds[["game_pk", "line", "best_over_odds", "best_under_odds"]],
                           gap_odds_clean], ignore_index=True)
    print(f"  combined totals odds pool: n={len(all_odds)} games")

    m = all_odds.merge(games_ext, on="game_pk", how="inner")
    m["total_runs"] = m["home_runs_"] + m["away_runs_"]
    under = m.copy(); under["win"] = under["total_runs"] < under["line"]; under["odds"] = under["best_under_odds"]; under["side"] = "under"; under["tiebreak"] = ""
    over = m.copy(); over["win"] = over["total_runs"] > over["line"]; over["odds"] = over["best_over_odds"]; over["side"] = "over"; over["tiebreak"] = ""
    pool = pd.concat([under, over]).dropna(subset=["odds"])
    n_before = len(pool)
    print(f"  totals pool (pre-enrichment): n={n_before}")

    pool = attach_enrichment(con, con2, pool, id_to_name)
    print(f"  row-count sanity check (enrichment): before={n_before}  after={len(pool)}  "
          f"{'OK' if len(pool) == n_before else '*** FAN-OUT, STOP ***'}")
    assert len(pool) == n_before

    pool["side_code"] = pool["side"].astype("category").cat.codes
    pool["elo_diff"] = (pool["home_elo"] + 24) - pool["away_elo"]
    pool["l10_diff"] = pool["home_l10"] - pool["away_l10"]
    pool["fatigue_diff"] = pool["home_fatigue"] - pool["away_fatigue"]
    pool["market_prob"] = implied_prob(pool["odds"])  # RAW feature, unmodified, same as Addendum 50
    pool["game_date"] = pd.to_datetime(pool["game_date"])

    # de-vigged probability -- both sides' odds already present on every row (no extra join needed)
    pool["p_over_raw"] = implied_prob(pool["best_over_odds"])
    pool["p_under_raw"] = implied_prob(pool["best_under_odds"])
    pool["overround"] = pool["p_over_raw"] + pool["p_under_raw"]
    pool["market_prob_novig"] = np.where(pool["side"] == "over",
                                          pool["p_over_raw"] / pool["overround"],
                                          pool["p_under_raw"] / pool["overround"])

    pool = pool.dropna(subset=["win", "odds", "market_prob_novig"]).sort_values(
        ["game_date", "game_pk", "side", "tiebreak"], kind="mergesort").reset_index(drop=True)
    print(f"  final totals pool: n={len(pool)}  date range {pool.game_date.min().date()} -> {pool.game_date.max().date()}")
    print(f"  MEASURED totals vig: mean overround={pool['overround'].mean():.4f} "
          f"(i.e. avg {100 * (pool['overround'].mean() - 1):.2f}% vig)  "
          f"median={pool['overround'].median():.4f}  min={pool['overround'].min():.4f}  max={pool['overround'].max():.4f}")

    pooled = run_walkforward(pool, "totals")
    print("\ntotals RESULTS:")
    s_raw, v_raw = report_gate(pooled, "edge_raw", "RAW market_prob (baseline, this run)")
    s_novig, v_novig = report_gate(pooled, "edge_novig", "DE-VIGGED market_prob (Addendum 55a)")
    sign_flip_report(pooled)
    print("\n  For reference, Addendum 50's baseline (single-split, no gap backfill): n=234  ROI=-9.38%  CI=[-23.35,4.59]  FAIL")
    print("  For reference, per HANDOFF/prompt: totals AUC~0.50 throughout, essentially no edge regardless of market_prob treatment.")
    return {"market": "totals", "raw": s_raw, "raw_verdict": v_raw, "novig": s_novig, "novig_verdict": v_novig}


def main():
    con = duckdb.connect(DB, read_only=True)
    con2 = duckdb.connect(CACHE_DB, read_only=True)
    con.execute(f"ATTACH '{CACHE_DB}' AS cache (READ_ONLY)")

    results = []
    results.append(run_h2h(con, con2))
    results.append(run_totals(con, con2))

    print("\n" + "=" * 78)
    print("SUMMARY -- Addendum 55a (de-vigged market_prob, one-shot correction)")
    print("=" * 78)
    for r in results:
        print(f"\n{r['market']}:")
        print(f"  RAW    (this run):   n={r['raw']['n']}  ROI={r['raw']['roi']}%  "
              f"CI=[{r['raw']['lo']},{r['raw']['hi']}]  {r['raw_verdict']}")
        print(f"  NO-VIG (Addendum 55a): n={r['novig']['n']}  ROI={r['novig']['roi']}%  "
              f"CI=[{r['novig']['lo']},{r['novig']['hi']}]  {r['novig_verdict']}")

    con.close()
    con2.close()


if __name__ == "__main__":
    main()
