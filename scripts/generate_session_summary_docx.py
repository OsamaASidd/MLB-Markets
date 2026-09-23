"""One-off generator for a plain, unstyled .docx recap of this session's work."""
import pathlib
from docx import Document

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "Session_Summary_2026-09-21.docx"

doc = Document()

def p(text=""):
    doc.add_paragraph(text)

p("MLB Markets Audit -- Session Summary (2026-09-21)")
p()
p("Official scoreboard: 3 of 11 markets pass the client's gate rule on real, adequately-sized data.")
p("Passing: hits, rbis, spreads.")
p("Failing (confirmed): total_bases, batter_home_runs, pitcher_strikeouts, h2h, totals, pitcher_outs, batter_runs_scored, batter_strikeouts.")
p("This count did not change during this session. Every attempt below either confirmed an existing FAIL or produced a result flagged as unresolved/unreplicated, not a new confirmed pass.")
p()

p("1. Multi-model comparison (Addendum 32, extended in this session as Addendum 33)")
p("Existing pre-registered comparison (5 models: XGBoost, LightGBM, HistGBM, RandomForest, LogisticRegression) was extended with a 6th model, flaml AutoML -- the local substitute for Azure AutoML, since this project has no Azure subscription/credentials.")
p("Result: 6 of 48 market/model combinations technically passed, spread across 4 of the 8 FAIL markets (batter_home_runs, pitcher_strikeouts, pitcher_outs, batter_strikeouts). No two models agreed on the same market; every passing CI still spanned into negative territory. Consistent with Addendum 32's original finding: this is the signature of multiple-comparisons noise, not real edges.")
p("Script: scripts/multi_model_comparison.py. Output: reports/multi_model_comparison_output.txt.")
p()

p("2. Disciplined hyperparameter tuning (Addendum 34)")
p("Pre-registered before running: one model (XGBoost, chosen once, not per-market), one hyperparameter grid, selected by cross-validated AUC on training data only -- never on ROI. Applied to the 4 markets that showed any pass in step 1.")
p("Result: 3 of 4 reversed to FAIL under honest tuning -- pitcher_strikeouts and batter_strikeouts decisively so (CIs no longer touch zero). batter_home_runs passed again (+0.68% ROI, CI still crosses -1.07%) -- flagged as an unresolved near-miss, not promoted to the confirmed scoreboard, since it has never passed independently of a search process.")
p("Script: scripts/tuned_xgboost_4_markets.py. Output: reports/tuned_xgboost_4_markets_output.txt.")
p()

p("3. Disciplined tuning on the remaining 4 FAIL markets (Addendum 36)")
p("Same method applied to the 4 markets that had shown zero passes in step 1: batter_total_bases, h2h, totals, batter_runs_scored.")
p("Result: batter_total_bases FAIL (-1.13%), h2h FAIL decisively (-31.59%), batter_runs_scored FAIL decisively (-2.62%, CI excludes zero). totals passed once (n=601, ROI +1.62%, CI [-5.98%, 9.22%]) -- flagged as unresolved/unreplicated for the same reason as batter_home_runs: it fails on every other test in this project (gate.py, individual XGBoost, the 6-model sweep) and its CI is too wide to be informative on its own.")
p("Script: scripts/tuned_xgboost_remaining_4_markets.py.")
p()

p("4. MLB2 delivery pack review (separate vendor pack, same client system)")
p("Reviewed a separate delivery pack claiming 7 of 10 markets pass in a backtest. Findings: 2 of the 7 came from fixing real data join gaps (batter_runs_scored and pitcher_outs had been silently under-measured); the rest came from a box-score-derived signal plus a recalibrated no-vig price filter, chosen once via a total-units spec-selection protocol.")
p("Critical caveats stated in the pack itself: this is a backtest only -- on the live slice, every market in that pack still fails the gate. Their own strictest test (holding out the last third of history) confirms only 4 of the 7 cleanly; batter_runs_scored reverses to FAIL on that holdout, and 2 more become underpowered rather than proven.")
p()

p("5. API-Sports.io MLB data source investigation")
p("A new API-Sports.io MLB Pro plan key was provided and tested directly against the live API (not assumed from documentation).")
p("Finding: the account and plan are real and active. The bet-type catalog includes the exact player props needed (Pitcher Outs, Pitcher Strikeouts, Player Runs, RBIs, etc.), but the /odds endpoint has zero historical coverage before approximately 2026-09-14 -- confirmed across per-game, per-date, whole-month, and fully unfiltered whole-season queries (2021 through 2025, all zero rows). There is no player-level box-score endpoint on this API at all.")
p("Conclusion: this source cannot backfill any FAIL market's history. It can only be used as a forward-collection source starting from the day it is queried.")
p()

p("6. Forward odds collector -- built, tested, and scheduled")
p("scripts/collect_api_sports_odds.py pulls real MLB odds (including player props) daily and stores them in a new local warehouse, db/api_sports_odds.duckdb (git-ignored, separate from the committed main warehouse so it doesn't bloat the repo with a growing binary).")
p("First manual run captured 20 real games and 14,062 odds rows in one pull.")
p("A Windows Scheduled Task (MLB_APISports_OddsCollector) now runs this daily at 9:00 AM automatically.")
p("A pre-registered test plan for this data was written before any real volume accumulated: reports/addendum35_forward_collection_plan.md -- one model, one join method to real outcomes, one cut, the official gate rule, applied once enough volume exists, reported honestly either way. Realistic timeline: weeks of daily collection at minimum before any market reaches n>=500.")
p()

p("7. Testing guides")
p("TESTING_GUIDE_3_PASS_MARKETS.md -- how to reproduce every verdict on the 3-pass-markets branch, script by script, with expected results and how to interpret a PASS without over-reading it.")
p("TESTING_GUIDE_8_POOLED_ML.md -- how to reproduce the pooled ML models on the 8-pooled-ml branch, including how to reproduce the documented run-to-run training instability finding, and an explicit section on what not to conclude from a single favorable run.")
p()

p("8. Local repository cleanup")
p("Removed approximately 230MB of verified byte-identical duplicate files: a stray Desktop folder containing duplicate copies of the real client odds export, plus duplicate copies of the same export sitting at the project root. The canonical copy in harness_out/ (the path cited in the reports) was left untouched. Also cleared scripts/__pycache__.")
p("Flagged but not touched, pending a decision: betgenius/ (383MB, mostly a regenerable node_modules folder) contains a number of loose scratch scripts unrelated to MLB (naming suggests an NBA project) mixed into the client's proprietary, NDA-protected, non-version-controlled source clone.")
p()

p("9. Open items for next session")
p("- Decide whether to commit the Addendum 33-36 work (currently uncommitted on 3-pass-markets) and under what branch name.")
p("- Let the API-Sports forward collector run for several weeks before attempting any test against it.")
p("- Decide on the betgenius/ scratch-script cleanup.")
p("- batter_home_runs and totals remain flagged as unresolved near-misses, not confirmed passes -- an independent holdout (a different time split, decided once) is the legitimate next step if either is worth pursuing further.")

doc.save(str(OUT))
print(f"saved: {OUT}")
