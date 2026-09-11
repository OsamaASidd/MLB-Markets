"""
Honest attempt at using the score_* factor columns already logged in
pick_history to build a better-than-`confidence` selection rule for live
production, per market.

Context: the live `algorithm_weights` row (updated 2026-07-06) records
backtest_win_pct=65.6, backtest_roi=55.1 on backtest_picks=224 -- a tiny
sample claiming an implausibly large ROI, the same shape as the "under 0.5"
false positive documented in the milestone report. That is a strong signal
the current production weights are themselves overfit, which would help
explain why every market's live ROI is negative despite the weights having
looked great in whatever backtest produced that row.

Method (train/test split BY DATE, same discipline as the rest of this repo):
  1. For each market, split graded picks into TRAIN (first 60% of dates) and
     TEST (last 40%, genuinely unseen).
  2. On TRAIN only, compute each score_* column's correlation with `win`.
  3. Take the top-K |correlation| columns (min |r| and min non-null n to
     avoid noise from sparse/near-constant columns), sign-align them, and
     build a combined z-score.
  4. Fit ONLY a threshold (top quartile of the combined score) on TRAIN.
  5. Apply that exact threshold to TEST and report the out-of-sample ROI.
     No further tuning after seeing TEST -- if it's not accurate here,
     it's not accurate.

This will not always find something -- and won't pretend to.
"""
import pathlib
import warnings
import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # constant-column corrcoef -> NaN, filtered below anyway

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

TOP_K = 8
MIN_ABS_CORR = 0.03
MIN_NONNULL = 300

MARKETS = [
    "pitcher_strikeouts", "hits", "total_bases", "rbis", "home_runs",
    "runs_scored", "spreads", "h2h", "totals", "pitcher_outs",
]


def load_market(con, market):
    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    select_cols = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in score_cols)
    df = con.execute(f"""
        SELECT
          TRY_CAST(odds AS INTEGER) AS odds,
          TRY_CAST(confidence AS INTEGER) AS confidence,
          lower(hit) IN ('true','t','1') AS win,
          TRY_CAST(game_date AS DATE) AS game_date,
          CASE WHEN lower(hit) IN ('true','t','1')
               THEN CASE WHEN TRY_CAST(odds AS INTEGER) > 0
                         THEN TRY_CAST(odds AS INTEGER) / 100.0
                         ELSE 100.0 / abs(TRY_CAST(odds AS INTEGER)) END
               ELSE -1 END AS profit,
          {select_cols}
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL
          AND prop_type = '{market}'
          AND lower(coalesce(is_synthetic, 'false')) NOT IN ('true', 't', '1')
          AND lower(coalesce(voided, 'false')) NOT IN ('true', 't', '1')
          AND hit IS NOT NULL
          AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchdf()
    return df, score_cols


def gate(n, roi_lo):
    return n is not None and n >= 500 and roi_lo is not None and roi_lo > 0


def roi_ci(profits):
    n = len(profits)
    if n == 0:
        return n, None, None, None
    m = profits.mean() * 100
    se = profits.std(ddof=1) / np.sqrt(n) * 100
    return n, round(m, 2), round(m - 1.96 * se, 2), round(m + 1.96 * se, 2)


def main():
    con = duckdb.connect(DB, read_only=True)
    results = []
    for market in MARKETS:
        df, score_cols = load_market(con, market)
        df = df.dropna(subset=["game_date"]).sort_values("game_date")
        n_total = len(df)
        print(f"\n{'='*95}\n{market}  (n={n_total:,})\n{'='*95}")
        if n_total < 400:
            print("  too few rows to split train/test meaningfully -- skipped")
            continue

        split_idx = int(n_total * 0.6)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]

        # baseline: current `confidence` column's own performance, both halves
        for label, part in [("TRAIN baseline (confidence>=60, minus)", train), ("TEST baseline (confidence>=60, minus)", test)]:
            sub = part[(part.confidence >= 60) & (part.odds < 0)]
            n, roi, lo, hi = roi_ci(sub.profit)
            print(f"  {label:<38} n={n:<6} ROI={roi} CI=[{lo},{hi}]")

        # correlate each score_* column with win, on TRAIN only
        corrs = []
        for c in score_cols:
            s = train[c]
            nn = s.notna().sum()
            if nn < MIN_NONNULL:
                continue
            r = np.corrcoef(s.fillna(s.median()), train["win"].astype(int))[0, 1]
            if np.isnan(r) or abs(r) < MIN_ABS_CORR:
                continue
            corrs.append((c, r, nn))
        corrs.sort(key=lambda x: -abs(x[1]))
        top = corrs[:TOP_K]
        if not top:
            print("  no score_* factor cleared the min-correlation bar on TRAIN -- no reweight attempted")
            continue

        print("  top factors (TRAIN-only correlation with win):")
        for c, r, nn in top:
            print(f"    {c:<45} r={r:+.3f}  (n_nonnull={nn})")

        def combined_score(part):
            zs = []
            for c, r, _ in top:
                col = part[c]
                mu, sd = train[c].mean(), train[c].std(ddof=0)
                if not sd or np.isnan(sd):
                    continue
                z = (col.fillna(mu) - mu) / sd
                zs.append(np.sign(r) * z)
            return pd.concat(zs, axis=1).sum(axis=1) if zs else pd.Series(0, index=part.index)

        train = train.copy()
        test = test.copy()
        train["combo"] = combined_score(train)
        test["combo"] = combined_score(test)

        threshold = train["combo"].quantile(0.75)  # fit ONLY on train

        for label, part in [("TRAIN (top quartile combo, minus-money)", train), ("TEST  (top quartile combo, minus-money) -- true holdout", test)]:
            sub = part[(part.combo >= threshold) & (part.odds < 0)]
            n, roi, lo, hi = roi_ci(sub.profit)
            passed = gate(n, lo)
            print(f"  {label:<52} n={n:<6} ROI={roi} CI=[{lo},{hi}]  {'PASS' if passed else 'fail'}")
            results.append({"market": market, "split": label, "n": n, "roi": roi, "ci_lo": lo, "ci_hi": hi, "pass": passed})

    print(f"\n{'='*95}\nSUMMARY -- any market where the TEST (holdout) split actually passed\n{'='*95}")
    wins = [r for r in results if r["pass"] and "TEST" in r["split"]]
    if not wins:
        print("  none. No factor-reweighted rule beat the gate on genuine held-out data.")
    for r in wins:
        print(f"  {r['market']:<20} n={r['n']} ROI={r['roi']}% CI=[{r['ci_lo']},{r['ci_hi']}]")
    con.close()


if __name__ == "__main__":
    main()
