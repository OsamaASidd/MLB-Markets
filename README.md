# MLB Markets

Milestone 1 deliverable: gate audit of every MLB market against the real
production pick history, plus the local database and scripts behind it.

Start here: [`reports/MILESTONE_1_GATE_REPORT.md`](reports/MILESTONE_1_GATE_REPORT.md)
for the full PASS/FAIL results, per-market root causes, and improvement levers.

## Layout

- `db/mlb_markets.duckdb` — local copy of the real MLB pick history, boxscores,
  lineups, weather, and other point-in-time context, pulled from the production
  warehouse (odds warehouse table intentionally excluded — see report).
- `scripts/gate.py` — the gate itself (`n>=500 AND ROI 95% CI lower bound > 0`).
  Run it any time with `python scripts/gate.py`.
- `scripts/diagnose.py` — per-market side mix / odds bucket / confidence
  calibration breakdown, the basis for the root-cause section of the report.
- `scripts/pull_pick_history.py` — refreshes `pick_history` from the warehouse.
  Needs a valid `DB_PASSWORD` in `.env` (see `.env.example`) — the credential
  used to build the current snapshot has since expired; see the report's
  "Data-quality notes" section.
- `scripts/compact_db.py` — rebuilds the shipped db from a raw warehouse mirror,
  dropping large free-text columns not needed for grading (keeps the file small).
- `reports/gate_results.csv` — raw numbers behind every row in the report.

## Setup

```
pip install -r requirements.txt
python scripts/gate.py
```
