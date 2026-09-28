"""
gate_old: the ORIGINAL, unmodified gate rule -- exactly matching the
client's own betgenius/harness/lib/metrics.ts (evaluateEvGate), verified
against the source at the start of this audit. Kept as its own module,
separate from gate_new.py, so every future result can be reported against
BOTH the old and new (client-approved 2026-09-26) rule side by side.

Rule: n >= 500 -> PASS iff ROI > 0.
      n <  500 -> PASS iff 95% CI lower bound on ROI > 0.

Works on the same `stat` dict shape used throughout this project's
addenda (xgboost_individual_markets.stat(): keys n, wr, roi, lo, hi).
"""

MIN_GRADED = 500


def gate_old(s):
    """s: dict with keys n, roi, lo, hi. Returns 'PASS' or 'FAIL'."""
    n = s.get("n")
    if n is None or n == 0:
        return "FAIL"
    if n >= MIN_GRADED:
        return "PASS" if (s.get("roi") is not None and s["roi"] > 0) else "FAIL"
    lo = s.get("lo")
    return "PASS" if (lo is not None and lo > 0) else "FAIL"
