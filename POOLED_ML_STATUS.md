# This branch: pooled ML research (out of scope for M1, kept for reference)

This branch exists to preserve the pooled cross-market ML research
(Addenda 20-26 of `reports/MILESTONE_1_GATE_REPORT.md`) as tracked,
reviewable files, separate from `3-pass-markets` — the actual Milestone 1
gate-audit deliverable.

**This is not an alternative "solution" to the gate audit and does not
change any PASS/FAIL verdict.** It's a different, exploratory line of work
the client asked to deprioritize ("no new research/models for now"), kept
here so it isn't lost, not because it's a competing result.

## What's actually in it, honestly

- **Model A** (`xgboost_pooled_A_factors.py`) and **Model B**
  (`xgboost_pooled_B_multiyear.py`): pooled, cross-market XGBoost models,
  properly holdout-validated. Model B's first run showed an implausible
  AUC 0.857 / ROI +46.63% — traced to a row-multiplicity bug (alt-line
  ladders counted as independent bets) and corrected to AUC 0.764. A
  **later, separate finding (Addendum 26): re-running Model B gives
  different PASS/FAIL verdicts run to run** — a genuine training-
  instability problem, not a data or code bug, which is exactly why this
  line of work was deprioritized rather than shipped.
- **Accuracy-vs-ROI check** (`check_favorite_accuracy_vs_roi.py`):
  `batter_home_runs` hits 88.4% accuracy with -2.20% ROI — a real
  illustration of why accuracy and ROI are not interchangeable metrics,
  cited in the main report to explain a recurring source of confusion.
- **`reports/ml_model_walkthrough.html`**: a plain-language walkthrough of
  both models for a non-technical reader, including the instability
  finding above — not a claim that either model is ready to ship.

## The actual, current, official result

Still `3-pass-markets`: `hits`, `rbis`, `spreads` pass the client's
official gate rule on real, adequately-sized data; 8 other markets are
confirmed FAIL. Nothing on this branch changes that. If pooled ML is
picked back up later, it needs to resolve the training-instability
finding before any result from it is treated as real — a model whose
verdict flips between runs isn't reporting a stable finding yet.
