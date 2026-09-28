"""
gate_new: client-approved (2026-09-26, Saqib Jawed -- "Approve all 4")
revised gate logic. Four changes were proposed and approved:

  1. MONITORING status for promising-but-underpowered results: n < min_n,
     ROI > 0, CI crosses zero, but the CI is narrow enough to plausibly
     resolve with more data (default width threshold: 40 percentage
     points -- the same number disclosed in the original proposal to the
     client, not tuned after the fact). Re-check cadence (how often a
     MONITORING market gets re-evaluated, and the max number of re-checks
     before a final verdict is locked in) is an OPERATIONAL scheduling
     decision the client is still finalizing (Saqib: "working on it",
     2026-09-26) -- NOT encoded in this function. gate_new() only
     classifies a single snapshot of results; it doesn't decide when to
     re-run.
  2. Point-in-time / lookahead-leakage feature audit requirement -- a
     process checklist item for any NEW feature before it feeds a gate-
     relevant model. Not a numeric rule, not encoded here. (Three real
     violations of this were found and fixed this session -- see
     addendum68_h2h_totals_fixed_starter_feature.py's docstring.)
  3. CLV (closing-line-value) tracked as a supplementary, REPORT-ONLY
     signal -- passed through in the result detail for visibility, NEVER
     used to change the PASS/MONITORING/FAIL verdict.
  4. Minimum sample size derived from a power calculation instead of a
     flat n=500. derive_min_n() below implements the calculation.
     gate_new() accepts an explicit min_n override; defaults to 500 (the
     old flat threshold) because no market-specific derived thresholds
     have been computed/approved yet -- this is a real, still-open
     follow-up, not assumed done.

Works on the same `stat` dict shape used throughout this project's
addenda (xgboost_individual_markets.stat(): keys n, wr, roi, lo, hi).
"""
import numpy as np
from scipy.stats import norm

MIN_GRADED_DEFAULT = 500
MONITORING_CI_WIDTH_THRESHOLD = 40  # percentage points; disclosed to the client in the original proposal


def derive_min_n(observed_win_rate, min_detectable_effect_roi_pct, confidence=0.95, power=0.80):
    """Standard two-sided power calculation for a proportion, sized to
    detect a given minimum ROI effect (in percentage points) at the
    requested confidence/power, given a market's own observed bet-level
    win-rate variance. NOT yet run/approved for any specific market --
    a ready-to-use helper once real observed variance + a chosen minimum
    detectable effect (MDE) are supplied and signed off on."""
    p = observed_win_rate
    variance = p * (1 - p)
    z_alpha = norm.ppf(1 - (1 - confidence) / 2)
    z_beta = norm.ppf(power)
    mde = min_detectable_effect_roi_pct / 100
    n = ((z_alpha + z_beta) ** 2 * variance) / (mde ** 2)
    return int(np.ceil(n))


def gate_new(s, min_n=MIN_GRADED_DEFAULT, monitoring_ci_width_threshold=MONITORING_CI_WIDTH_THRESHOLD,
             clv_positive_rate=None):
    """s: dict with keys n, roi, lo, hi.
    Returns (verdict, detail) where verdict in {"PASS","MONITORING","FAIL"}
    and detail is a dict carrying the reason + any report-only CLV info."""
    n = s.get("n")
    detail = {"clv_positive_rate": clv_positive_rate}
    if n is None or n == 0:
        detail["reason"] = "no data"
        return "FAIL", detail

    if n >= min_n:
        verdict = "PASS" if (s.get("roi") is not None and s["roi"] > 0) else "FAIL"
        detail["reason"] = f"n>={min_n}, ROI-only rule"
        return verdict, detail

    lo, hi, roi = s.get("lo"), s.get("hi"), s.get("roi")
    if lo is not None and lo > 0:
        detail["reason"] = "n<min_n, CI lower bound > 0"
        return "PASS", detail

    if (roi is not None and roi > 0 and lo is not None and hi is not None
            and (hi - lo) < monitoring_ci_width_threshold):
        detail["reason"] = ("positive ROI point estimate, CI crosses zero, "
                             f"but CI width {round(hi - lo, 2)} < {monitoring_ci_width_threshold} -- "
                             "plausibly resolves with more data")
        return "MONITORING", detail

    detail["reason"] = "n<min_n, CI does not clear zero and is not narrow enough for MONITORING"
    return "FAIL", detail


def evaluate_both(s, min_n=MIN_GRADED_DEFAULT, clv_positive_rate=None):
    """Convenience: run both gates on the same stat dict, for the
    'old' / 'new' column pair now required in every report."""
    from gate_old import gate_old
    old_verdict = gate_old(s)
    new_verdict, detail = gate_new(s, min_n=min_n, clv_positive_rate=clv_positive_rate)
    return {"n": s.get("n"), "roi": s.get("roi"), "lo": s.get("lo"), "hi": s.get("hi"),
            "old_gate": old_verdict, "new_gate": new_verdict, "new_gate_reason": detail.get("reason"),
            "clv_positive_rate": clv_positive_rate}
