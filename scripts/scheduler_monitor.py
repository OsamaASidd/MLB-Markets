"""
Independent, read-only scheduler/cron health check for the M1 work order.

This does NOT replace the client's own production health-monitor edge
function (supabase/functions/health-monitor/index.ts, already live, SMS
alerts via Twilio, runs every 30min via pg_cron) -- that's real,
already-shipped monitoring with a much richer check list (error_log
accumulation, run_log staleness, algorithm_weights staleness). Duplicating
it would be redundant and this contractor has no service-role access
anyway.

What this IS: an outside, harness_readonly-scoped second opinion, usable
by anyone with only the read-only DB role (no service-role key, no
Twilio/SMS access) -- reads public.cron_heartbeat directly (D-272-INF-2,
confirmed present in betgenius/supabase/migrations/20260520000002_...sql)
and applies the exact same staleness rule as the production
detect_silent_crons() view (elapsed > 2x expected_interval_seconds),
without depending on that view being granted to a non-standard role.

Usage:
    python3 scripts/scheduler_monitor.py            # live check (needs HARNESS_DATABASE_URL)
    python3 scripts/scheduler_monitor.py --test      # fixture-only self-test, no DB needed

Exit code 0 = all crons healthy or DB unreachable-but-reported-clearly.
Exit code 1 = at least one cron is silent past 2x its expected interval.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / "betgenius" / "harness" / ".env")
except ImportError:
    pass


def evaluate(rows):
    """rows: list of dicts with job_name, last_fired_at (datetime, tz-aware),
    last_status, consecutive_failures, expected_interval_seconds (or None).
    Returns (silent, healthy) lists. Pure function -- fixture-testable
    without a DB."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    silent, healthy = [], []
    for r in rows:
        interval = r.get("expected_interval_seconds")
        if interval is None:
            healthy.append(r)  # matches detect_silent_crons: NULL interval -> not silent
            continue
        elapsed = (now - r["last_fired_at"]).total_seconds()
        if elapsed > interval * 2:
            silent.append({**r, "elapsed_seconds": elapsed})
        else:
            healthy.append(r)
    return silent, healthy


def run_live():
    import re
    env_path = ROOT / "betgenius" / "harness" / ".env"
    if not env_path.exists():
        print(f"[scheduler_monitor] no {env_path} -- can't run live, run --test instead")
        return 2
    url = re.search(r"HARNESS_DATABASE_URL=(.+)", env_path.read_text()).group(1).strip()

    try:
        import psycopg2
    except ImportError:
        print("[scheduler_monitor] psycopg2 not installed -- pip install psycopg2-binary")
        return 2

    try:
        conn = psycopg2.connect(url)
    except Exception as e:
        print(f"[scheduler_monitor] DB connection failed: {e}")
        print("[scheduler_monitor] this is a read-only check -- reporting the failure "
              "rather than guessing at cron health.")
        return 2

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT job_name, last_fired_at, last_status, consecutive_failures,
                   expected_interval_seconds
            FROM public.cron_heartbeat
            ORDER BY job_name
        """)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[scheduler_monitor] query failed (role may lack SELECT on "
              f"cron_heartbeat): {e}")
        return 2
    finally:
        conn.close()

    silent, healthy = evaluate(rows)
    print(f"[scheduler_monitor] {len(healthy)} healthy, {len(silent)} silent "
          f"(of {len(rows)} known cron jobs)")
    for r in silent:
        print(f"  SILENT: {r['job_name']} -- last fired {r['elapsed_seconds']/60:.0f}min ago, "
              f"expected every {r['expected_interval_seconds']/60:.0f}min, "
              f"last_status={r['last_status']}, consecutive_failures={r['consecutive_failures']}")
    return 1 if silent else 0


def run_test():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    fixture = [
        {"job_name": "process-games-mlb", "last_fired_at": now - timedelta(minutes=10),
         "last_status": "success", "consecutive_failures": 0, "expected_interval_seconds": 1800},
        {"job_name": "fetch-odds-mlb", "last_fired_at": now - timedelta(hours=5),
         "last_status": "error", "consecutive_failures": 6, "expected_interval_seconds": 1800},
        {"job_name": "algorithm_weights_refresh", "last_fired_at": now - timedelta(days=3),
         "last_status": "success", "consecutive_failures": 0, "expected_interval_seconds": None},
    ]
    silent, healthy = evaluate(fixture)
    assert len(silent) == 1 and silent[0]["job_name"] == "fetch-odds-mlb", \
        f"expected exactly fetch-odds-mlb silent, got {[s['job_name'] for s in silent]}"
    assert len(healthy) == 2, f"expected 2 healthy, got {len(healthy)}"
    print("[scheduler_monitor] --test PASSED: staleness rule matches "
          "detect_silent_crons() semantics (elapsed > 2x expected_interval_seconds, "
          "NULL interval never flags silent)")
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_test())
    sys.exit(run_live())
