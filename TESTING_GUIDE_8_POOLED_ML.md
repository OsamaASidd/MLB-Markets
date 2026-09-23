# Testing guide — `8-pooled-ml` (deprioritized research branch)

How to reproduce the pooled cross-market ML models yourself. Read
`POOLED_ML_STATUS.md` on that branch first — **this branch is not an
alternative result to `3-pass-markets`**; it's exploratory work the client
asked to deprioritize, kept here so it isn't lost. Nothing below changes
any PASS/FAIL verdict in the main report.

---

## 1. Setup

```
git checkout 8-pooled-ml
cd "MLB Markets"        # if not already there
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Same `db/mlb_markets.duckdb` as `3-pass-markets` (real data, committed,
read-only). If you already have a venv from testing that branch, it's
reusable — the dependency list is unchanged.

---

## 2. Model A — pooled, one XGBoost model across all markets, real production factors

```
python scripts/xgboost_pooled_A_factors.py
```

One unified model with market type as a categorical feature, trained on
the ~150-factor production feature set — which only exists for the window
it was actually logged in (2026-05-17 to 2026-07-27; confirmed by direct
query, this feature set doesn't exist before that). 75/25 chronological
split within that single window.

**Expected**: no threshold clears the gate cleanly.
`ALL MARKETS edge>0.0` → n=3,550, ROI −0.05%, CI [−2.55, 2.44] — fail, and
it stays negative-to-flat across every edge threshold tried
(`reports/xgboost_pooled_A_output.txt` has the full per-market
breakdown). `pitcher_strikeouts` and `pitcher_outs` show n=0 at this cut —
real data-coverage gaps in the factor table for those two, not a bug.

---

## 3. Model B — pooled, all 4 real years (2023-2026), self-built features

```
python scripts/xgboost_pooled_B_multiyear.py
```

Same pooling idea, but built to span the full real 2023-2026 window using
features constructed independently of the production factor table (team
Elo rebuilt from real box scores, L10 form, real ballpark factors, real
bullpen fatigue) — 8 markets included (`batter_runs_scored` and
`pitcher_outs` excluded; both have a real, confirmed coverage gap in the
odds warehouse for this specific model, documented in the script's
docstring). Split: 75/25 chronological **within each calendar year**,
then pooled.

**Expected**: `edge>0.03, minus-money` is the one cut that clears the gate
(n=1,264, ROI +5.65%, CI [1.56%, 9.75%] — PASS) out of 4 thresholds tried;
`edge>0.0`, `edge>0.05`, and `edge>0.08` don't. **This is exactly the
result flagged as unreliable in `POOLED_ML_STATUS.md` (Addendum 26): the
same model, re-run, produces a different PASS/FAIL verdict from run to
run** — genuine training instability, not something this guide can make
deterministic. Don't treat one run's PASS here as confirmation of
anything; if you want to see the instability yourself, re-run the script
a few times and compare `edge>0.03`'s verdict across runs.

---

## 4. Accuracy vs. ROI — why "raise accuracy" and "raise ROI" aren't the same ask

```
python scripts/check_favorite_accuracy_vs_roi.py
```

Zero modeling — just bets whichever side the market's own price already
favors, and reports accuracy next to ROI. **Expected**:
`batter_home_runs` shows ~88.9% accuracy while still losing money — the
concrete illustration for why a high-accuracy target doesn't imply a
profitable one (a market can be "usually right" and still lose to the
vig on the rare miss). This is cited directly in the main report's
Addendum 24.

---

## 5. What NOT to conclude from any of this

- **Model B's one PASS is not a 4th confirmed market**, on top of the 3
  in `3-pass-markets` — it's the same branch's own documented reason this
  whole line of work is deprioritized, not a competing result to report
  alongside the audit.
- **Don't re-run either pooled model hoping for a more favorable verdict
  on a specific run.** The instability finding means the verdict itself
  is the noise; picking whichever run looks best and reporting that is
  the exact multiple-comparisons trap the main report's guardrails exist
  to catch, just via re-running instead of re-modeling.
- If this line of work is picked back up, the prerequisite (per
  `POOLED_ML_STATUS.md`) is resolving the run-to-run instability first —
  fixed seeds, deterministic data ordering, and repeated runs reported in
  full — before any single result from it is treated as real.

---

## 6. Full narrative

`reports/MILESTONE_1_GATE_REPORT.md` Addenda 20-26 (this report file is
shared between both branches) — includes the AUC 0.857→0.764
row-multiplicity bug fix (Addendum 23) and the training-instability
finding itself (Addendum 26), with exact numbers from the runs that
surfaced them.
