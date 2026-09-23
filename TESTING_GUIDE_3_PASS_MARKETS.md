# Testing guide — `3-pass-markets` (the Milestone 1 deliverable)

How to reproduce every verdict in this branch yourself, from a clean
checkout, in the order that makes the most sense to run them.

Read `HANDOFF.md` first if you haven't — this guide is "how to run the
numbers," not "what the numbers mean." That context lives there.

---

## 1. Setup

```
cd "MLB Markets"
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install flaml lightgbm      # needed for scripts/multi_model_comparison.py (Addendum 32/33)
```

**Data**: `db/mlb_markets.duckdb` (real data, committed to this repo — no
credentials or network access needed for anything in this guide). All
scripts below open it **read-only**; nothing here writes to it.

---

## 2. The headline gate result — `hits`, `rbis`, `spreads` PASS; 8 FAIL

```
python scripts/gate.py
```

Runs all 10 core markets (`pitcher_strikeouts`, `hits`, `total_bases`,
`rbis`, `home_runs`, `runs_scored`, `spreads`, `h2h`, `totals`,
`pitcher_outs`) against three cuts (`all`, `minus-money`,
`conf>=60 minus-money`), applies the official gate rule
(`n>=500 → ROI>0`; `n<500 → 95% CI lower bound>0`, verified against the
client's `betgenius/harness/lib/metrics.ts:evaluateEvGate`), and writes
`reports/gate_results.csv`.

**What to expect**: `spreads` clears the gate on its best cut once the
corrected rule is applied (Addendum 28). `hits` and `rbis` pass via the
individual-market XGBoost work below, not this script alone — `gate.py`
tests the *existing* production selection logic, not a trained model.
Cross-check any single row against `reports/MILESTONE_1_GATE_REPORT.md`'s
per-market tables if a number looks different — the report states which
addendum superseded which run.

---

## 3. The 3 backfilled real-data markets — `pitcher_outs`, `batter_runs_scored`, `batter_strikeouts`

```
python scripts/test_backfilled_markets.py
```

These 3 were previously "untestable" (too little data) until a real
historical-odds backfill from The Odds API was matched to real box-score
outcomes (Addendum 30). **Expected: all 3 FAIL** —
`pitcher_outs` n=5,451 (test 3,334) ROI −1.21%; `batter_runs_scored`
n=79,131 (test 52,981) ROI −2.54%; `batter_strikeouts` n=60,862 (test
41,489) ROI −0.87%. If your run disagrees, check
`data_raw/*_odds_cache.jsonl` exist first — the backfill caches are what
this script reads; it does not re-hit the API.

---

## 4. Individual per-market XGBoost — the real signal check

```
python scripts/xgboost_individual_markets.py
```

Trains one XGBoost model per market, real point-in-time features (Elo,
L10, bullpen fatigue, park factors), reports both a no-threshold-search
result (`edge>0.0`) and a best-of-5-thresholds result on the *same* test
set — explicitly labeled as multiple-comparisons exposure, not a second
finding. This is the source of the `hits`/`rbis` passes.

---

## 5. Multi-model comparison — Addenda 32 & 33 (does a different model change anything?)

```
python scripts/multi_model_comparison.py
```

Runs 6 models (XGBoost, LightGBM, HistGradientBoostingClassifier,
RandomForest, LogisticRegression, and `flaml` AutoML — the local
substitute for Azure AutoML, since this project has no Azure
subscription/credentials) across all 8 FAIL markets, one pre-specified
`edge>0.0` cut, no threshold search. Takes several minutes (FLAML's
default per-market time budget is 60s × 8 markets).

**Expected: 6 of 48 market/model combinations technically clear the gate,
spread across 4 of the 8 markets** — but no two models agree on the same
market, and every passing CI still spans into negative territory. This is
the demonstrated case for why a single model clearing the bar isn't
treated as a real pass in this project. Full output saved at
`reports/multi_model_comparison_output.txt`.

---

## 6. Disciplined hyperparameter tuning — Addendum 34 (does tuning recover any of those 4?)

```
python scripts/tuned_xgboost_4_markets.py
```

Takes the 4 markets that showed *any* pass in step 5
(`batter_home_runs`, `pitcher_strikeouts`, `pitcher_outs`,
`batter_strikeouts`) and re-tests with **one** model family (XGBoost,
chosen once, not per-market), **one** pre-specified hyperparameter grid,
selected by cross-validated AUC on the training split only — ROI is never
touched during model selection, only on the final single held-out
evaluation.

**Expected: 3 of 4 reverse to FAIL** (`pitcher_strikeouts` and
`batter_strikeouts` decisively so — their CIs no longer even touch zero).
`batter_home_runs` clears the gate again (ROI +0.68%, CI still crosses
−1.07%) — flagged in the report as an unresolved near-miss, **not**
counted as a confirmed pass. Full output at
`reports/tuned_xgboost_4_markets_output.txt`.

> This script uses `joblib.parallel_backend("threading")` rather than the
> default process-based backend — `n_jobs=-1`/loky fails on this sandboxed
> Windows Python with `ModuleNotFoundError: No module named
> '_posixsubprocess'`. If you hit that error elsewhere, apply the same fix.

---

## 7. What a "PASS" printed by any of the above does *and doesn't* mean

- **n≥500 only requires a positive point estimate**, not a CI that clears
  zero — that's the official rule, not a bug. A lot of the fragile passes
  in steps 5-6 exist *because* of this permissive arm.
- **A pass from one model/search, unreplicated, is not promoted to the
  confirmed scoreboard.** Only `hits`, `rbis`, `spreads` currently qualify
  — each holds without needing a search across models or thresholds to
  find them.
- If you want to try to convert a flagged near-miss (currently just
  `batter_home_runs`) into a confirmed pass, the legitimate next step is
  an **independent holdout** — a different time split, decided once, not
  re-cut until it looks good — not another model or another tuning pass
  on the same split.

---

## 8. Full narrative

`reports/MILESTONE_1_GATE_REPORT.md` — 34 addenda, in order, each one
disclosing what was tried, what worked, what didn't, and why. Addenda
32-34 (the multi-model sweep, FLAML addition, and disciplined tuning) are
at the end.
