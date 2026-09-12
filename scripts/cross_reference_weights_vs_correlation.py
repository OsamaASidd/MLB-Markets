"""
Cross-reference the LIVE production algorithm_weights values (w_mlb_*) against
this repo's own independently-computed factor-correlation findings
(factor_reweight.py, Addendum 2) to find specific, named miscalibrations:
factors the live system weights heavily that show near-zero real correlation
with outcomes, or factors it barely uses that show real signal.

This is only possible because of the betgenius architecture deep-dive: we now
know exactly which w_mlb_* column controls which score_* factor's contribution
to the live confidence score (scoring_mlb_v2.ts multiplies each factor's
bucketed magnitude by its w_mlb_* weight -- see the betgenius briefing).
"""
import pathlib
import re
import warnings
import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = str(ROOT / "db" / "mlb_markets.duckdb")

# market -> the w_mlb_* prefix used for that market's factor weights in
# algorithm_weights (per the schema dumped earlier: w_mlb_hits_*, w_mlb_tb_*,
# w_mlb_rbi_*, w_mlb_hr_* for the four batter markets that share factor
# names; pitcher_strikeouts factors mostly have no market prefix (w_mlb_pitcher_*)
MARKETS = ["pitcher_strikeouts", "hits", "total_bases", "rbis", "home_runs", "runs_scored"]

MARKET_WEIGHT_PREFIX = {
    "hits": "w_mlb_hits_", "total_bases": "w_mlb_tb_", "rbis": "w_mlb_rbi_",
    "home_runs": "w_mlb_hr_", "runs_scored": "w_mlb_runs_",
}
# other markets' reserved prefixes -- when a market has no dedicated prefix
# (pitcher_strikeouts), weight columns starting with any of these belong to
# a DIFFERENT market and must be excluded, not treated as generic/shared
OTHER_MARKET_PREFIXES = ("w_mlb_hits_", "w_mlb_tb_", "w_mlb_rbi_", "w_mlb_hr_",
                          "w_mlb_runs_", "w_mlb_outs_", "w_mlb_bk_", "w_mlb_pk_",
                          "w_mlb_side_", "w_mlb_total_")


def get_live_weights(con):
    cols = [c[0] for c in con.execute("DESCRIBE algorithm_weights").fetchall()]
    row = con.execute("SELECT * FROM algorithm_weights WHERE id=1").fetchone()
    return dict(zip(cols, row))


def factor_key_from_weight_col(wcol):
    # w_mlb_pitcher_k_rate -> pitcher_k_rate ; w_mlb_hits_lineup_spot -> lineup_spot
    m = re.match(r"^w_mlb_(hits_|tb_|rbi_|hr_|runs_|outs_|bk_|pk_|side_|total_|batter_|pitcher_|umpire_|game_)?(.+)$", wcol)
    return m.group(2) if m else None


def load_market_correlations(con, market):
    """Recompute TRAIN-only score_* vs win correlations for one market (same
    logic as factor_reweight.py, standalone here so this script is self-contained)."""
    cols = [c[0] for c in con.execute("DESCRIBE pick_history").fetchall()]
    score_cols = [c for c in cols if c.startswith("score_")]
    select_cols = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in score_cols)
    df = con.execute(f"""
        SELECT TRY_CAST(game_date AS DATE) AS game_date,
               lower(hit) IN ('true','t','1') AS win,
               {select_cols}
        FROM pick_history
        WHERE mlb_market_type IS NOT NULL AND prop_type = '{market}'
          AND lower(coalesce(is_synthetic,'false')) NOT IN ('true','t','1')
          AND lower(coalesce(voided,'false')) NOT IN ('true','t','1')
          AND hit IS NOT NULL AND TRY_CAST(odds AS INTEGER) IS NOT NULL
    """).fetchdf()
    df = df.dropna(subset=["game_date"]).sort_values("game_date")
    train = df.iloc[: int(len(df) * 0.6)]
    out = {}
    for c in score_cols:
        s = train[c]
        if s.notna().sum() < 200:
            continue
        r = np.corrcoef(s.fillna(s.median()), train["win"].astype(int))[0, 1]
        if not np.isnan(r):
            out[c.replace("score_", "")] = round(r, 3)
    return out


def main():
    con = duckdb.connect(DB, read_only=True)
    weights = get_live_weights(con)

    print("=" * 100)
    print("LIVE WEIGHT vs REAL-WORLD CORRELATION cross-reference")
    print("(weight = current production multiplier; corr = this repo's TRAIN-only correlation with actual wins)")
    print("=" * 100)

    flags = []
    for market in MARKETS:
        corr = load_market_correlations(con, market)
        prefix = MARKET_WEIGHT_PREFIX.get(market, "w_mlb_")
        print(f"\n--- {market} ---")
        rows = []
        for wcol, wval in weights.items():
            if not wcol.startswith("w_mlb_"):
                continue
            fkey = factor_key_from_weight_col(wcol)
            if fkey is None or fkey not in corr:
                continue
            # only consider weight columns plausibly tied to this market: its own
            # dedicated prefix, or -- for markets with no dedicated prefix
            # (pitcher_strikeouts) -- a generic column that isn't reserved by
            # one of the OTHER markets' explicit prefixes
            if market in MARKET_WEIGHT_PREFIX:
                if not wcol.startswith(prefix):
                    continue
            else:
                if wcol.startswith(OTHER_MARKET_PREFIXES):
                    continue
            try:
                wnum = float(wval)
            except (TypeError, ValueError):
                continue
            rows.append((wcol, wnum, corr[fkey]))
        rows.sort(key=lambda r: -abs(r[1]))
        for wcol, wnum, r in rows[:12]:
            flag = ""
            if abs(wnum) >= 1.0 and abs(r) < 0.03:
                flag = "  <-- FLAG: heavily weighted, ~zero real correlation"
                flags.append((market, wcol, wnum, r))
            elif abs(wnum) <= 0.25 and abs(r) >= 0.08:
                flag = "  <-- FLAG: barely weighted, real correlation found"
                flags.append((market, wcol, wnum, r))
            print(f"  {wcol:<40} weight={wnum:<7} corr={r:+.3f}{flag}")

    print("\n" + "=" * 100)
    print(f"NAMED MISCALIBRATIONS FOUND: {len(flags)}")
    print("=" * 100)
    for market, wcol, wnum, r in flags:
        print(f"  [{market}] {wcol}: weight={wnum}, real corr={r:+.3f}")

    con.close()


if __name__ == "__main__":
    main()
