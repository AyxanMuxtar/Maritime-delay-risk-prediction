"""
src/quality_checks.py
─────────────────────
Automated data quality gates that run after each pipeline stage.

Severity levels
---------------
  ABORT : check failed → pipeline should exit non-zero
  WARN  : check failed → log, continue
  FLAG  : check flagged individual rows → log, continue, write flags to DB
  PASS  : check passed

Each function returns a QualityCheckResult dict:
    {
        "check_name":  str,
        "status":      "PASS" | "WARN" | "FAIL" | "FLAG",
        "severity":    "ABORT" | "WARN" | "FLAG",
        "stage":       "raw" | "staging" | "analytics",
        "details":     dict (check-specific metrics),
        "message":     str (human-readable summary),
    }

Public API
----------
    check_row_count(conn, table)                    → dict
    check_null_ratio(conn, table, max_ratio)        → dict
    check_date_continuity(conn, table, max_gap)     → dict
    check_value_ranges(conn, table)                 → dict
    check_feature_completeness(conn, table)         → dict
    check_freshness(conn, table, max_days)          → dict
    check_freshness_monthly(conn, table)            → dict
    run_all_checks(conn, stage)                     → list[dict]
    format_check_report(results)                    → str (pretty table)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Severity constants ────────────────────────────────────────────────────────
ABORT = "ABORT"
WARN  = "WARN"
FLAG  = "FLAG"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Row count (ABORT on failure)
# ══════════════════════════════════════════════════════════════════════════════

def check_row_count(conn, table: str = "raw.weather_daily") -> dict:
    """After raw load: table must have > 0 rows, else abort."""
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    passed = n > 0
    return {
        "check_name": "row_count",
        "status":     "PASS" if passed else "FAIL",
        "severity":   ABORT,
        "stage":      "raw",
        "details":    {"rows": n, "table": table},
        "message":    f"{table}: {n:,} rows" if passed else
                      f"{table} is empty — pipeline cannot continue",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Null ratio (WARN on failure)
# ══════════════════════════════════════════════════════════════════════════════

def check_null_ratio(
    conn,
    table:     str   = "staging.weather_daily",
    max_ratio: float = 0.05,
) -> dict:
    """
    After staging: each column should have < max_ratio (default 5%) nulls.

    Returns the list of offending columns with their null percentages.
    """
    df = conn.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf()
    columns = [c for c in df.columns if c not in ("city", "date")]

    n_total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if n_total == 0:
        return {
            "check_name": "null_ratio",
            "status":     "FAIL",
            "severity":   WARN,
            "stage":      "staging",
            "details":    {},
            "message":    f"{table} is empty",
        }

    offenders = {}
    for col in columns:
        n_nulls = conn.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL'
        ).fetchone()[0]
        ratio = n_nulls / n_total
        if ratio > max_ratio:
            offenders[col] = round(ratio * 100, 2)

    passed = len(offenders) == 0
    return {
        "check_name": "null_ratio",
        "status":     "PASS" if passed else "WARN",
        "severity":   WARN,
        "stage":      "staging",
        "details":    {"threshold_pct": max_ratio * 100,
                       "rows_checked":  n_total,
                       "offenders":     offenders},
        "message":    f"All columns within {max_ratio*100:.0f}% null threshold" if passed else
                      f"{len(offenders)} column(s) exceed {max_ratio*100:.0f}% nulls: {offenders}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Date continuity (WARN on failure)
# ══════════════════════════════════════════════════════════════════════════════

def check_date_continuity(
    conn,
    table:   str = "staging.weather_daily",
    max_gap: int = 3,
) -> dict:
    """After staging: no gaps > max_gap days per city."""
    gap_rows = conn.execute(f"""
        SELECT city,
               date,
               LEAD(date) OVER (PARTITION BY city ORDER BY date) AS next_date,
               DATEDIFF('day', date,
                 LEAD(date) OVER (PARTITION BY city ORDER BY date)) AS gap_days
        FROM {table}
    """).fetchdf()
    gap_rows = gap_rows[gap_rows["gap_days"].notna()
                        & (gap_rows["gap_days"] > max_gap)]

    passed = len(gap_rows) == 0
    worst = int(gap_rows["gap_days"].max()) if not passed else 0
    by_city = gap_rows.groupby("city").size().to_dict() if not passed else {}

    return {
        "check_name": "date_continuity",
        "status":     "PASS" if passed else "WARN",
        "severity":   WARN,
        "stage":      "staging",
        "details":    {"max_gap_allowed": max_gap,
                       "gaps_found":      int(len(gap_rows)),
                       "worst_gap_days":  worst,
                       "by_city":         by_city},
        "message":    f"No gaps > {max_gap} days" if passed else
                      f"{len(gap_rows)} gaps > {max_gap} days "
                      f"(worst: {worst} days)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Value ranges (FLAG on failure — writes to raw.quality_flags)
# ══════════════════════════════════════════════════════════════════════════════

def check_value_ranges(
    conn,
    table: str = "staging.weather_daily",
) -> dict:
    """
    After staging: check physical plausibility of core variables.
    Violating rows are inserted into meta.quality_flags (created if absent).
    """
    # Per-variable physical bounds (below, above)
    bounds = {
        "temperature_2m_max": (-50, 60),
        "temperature_2m_min": (-60, 55),
        "wind_speed_10m_max": (0, 300),
        "relative_humidity_2m_mean": (0, 100),
        "precipitation_sum":  (0, 500),
    }

    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.quality_flags (
            flagged_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            city        VARCHAR,
            date        DATE,
            column_name VARCHAR,
            value       DOUBLE,
            reason      VARCHAR
        )
    """)

    total_flagged = 0
    by_column = {}
    for col, (lo, hi) in bounds.items():
        # Skip if column doesn't exist
        exists = conn.execute(f"""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema || '.' || table_name = '{table}'
              AND column_name = '{col}'
        """).fetchone()[0]
        if not exists:
            continue

        flagged = conn.execute(f"""
            INSERT INTO meta.quality_flags (city, date, column_name, value, reason)
            SELECT city, date, '{col}', "{col}",
                   CASE
                     WHEN "{col}" < {lo} THEN 'below_{lo}'
                     WHEN "{col}" > {hi} THEN 'above_{hi}'
                   END
            FROM {table}
            WHERE "{col}" < {lo} OR "{col}" > {hi}
            RETURNING 1
        """).fetchall()
        n = len(flagged)
        if n > 0:
            by_column[col] = n
            total_flagged += n

    passed = total_flagged == 0
    return {
        "check_name": "value_ranges",
        "status":     "PASS" if passed else "FLAG",
        "severity":   FLAG,
        "stage":      "staging",
        "details":    {"bounds": bounds,
                       "flagged_total":    total_flagged,
                       "flagged_by_column": by_column},
        "message":    "All values within physical bounds" if passed else
                      f"{total_flagged} out-of-range values flagged "
                      f"to meta.quality_flags",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature completeness (WARN on failure)
# ══════════════════════════════════════════════════════════════════════════════

def check_feature_completeness(
    conn,
    table: str = "analytics.daily_enriched",
) -> dict:
    """
    After analytics: critical engineered features must be present and non-null.
    Lag columns are allowed to be NaN on the first few rows per city.
    """
    required = [
        "temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum",
        "wave_height", "visibility_mean", "visibility_is_known",
        "hdd", "cdd", "temp_range_c",
        "month_sin", "month_cos", "doy_sin", "doy_cos",
        "is_risk_day",
        "wind_change_1d",
        "precip_change_1d",
        "wind_3d_max",
        "precip_3d_sum",
        "strong_wind_recent",
        "wind_precip_combo",
        "low_visibility_recent",
    ]

    # Check presence
    cols = conn.execute(f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema || '.' || table_name = '{table}'
    """).fetchdf()["column_name"].tolist()

    missing = [c for c in required if c not in cols]
    if missing:
        return {
            "check_name": "feature_completeness",
            "status":     "FAIL",
            "severity":   WARN,
            "stage":      "analytics",
            "details":    {"missing_columns": missing},
            "message":    f"{len(missing)} required features missing: {missing}",
        }

    # Check nullness (excluding lag columns which have expected leading NaN)
    nullable_offenders = {}
    for col in required:
        n_null = conn.execute(
            f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL'
        ).fetchone()[0]
        if n_null > 0:
            nullable_offenders[col] = n_null

    passed = len(nullable_offenders) == 0
    return {
        "check_name": "feature_completeness",
        "status":     "PASS" if passed else "WARN",
        "severity":   WARN,
        "stage":      "analytics",
        "details":    {"required": required,
                       "null_offenders": nullable_offenders},
        "message":    f"All {len(required)} required features present and non-null"
                      if passed else
                      f"Nulls in required features: {nullable_offenders}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Freshness (two variants — strict 2-day, monthly 35-day)
# ══════════════════════════════════════════════════════════════════════════════

def check_freshness(
    conn,
    table:    str = "raw.weather_daily",
    max_days: int = 2,
) -> dict:
    """
    After raw load: latest date should be within max_days of today.
    Strict version from spec — fires WARN on monthly-schedule setups.
    """
    max_date = conn.execute(
        f"SELECT MAX(date) FROM {table}"
    ).fetchone()[0]
    if max_date is None:
        return {
            "check_name": "freshness",
            "status":     "FAIL",
            "severity":   WARN,
            "stage":      "raw",
            "details":    {"max_days_allowed": max_days},
            "message":    f"{table} has no data",
        }

    days_behind = (date.today() - max_date).days
    passed = days_behind <= max_days
    return {
        "check_name": "freshness",
        "status":     "PASS" if passed else "WARN",
        "severity":   WARN,
        "stage":      "raw",
        "details":    {"latest_date":       str(max_date),
                       "days_behind_today": days_behind,
                       "max_days_allowed":  max_days},
        "message":    f"Latest data is {days_behind} days old "
                      f"(threshold: {max_days})" if not passed else
                      f"Data is {days_behind} days old — within threshold",
    }


def check_freshness_monthly(
    conn,
    table: str = "raw.weather_daily",
) -> dict:
    """
    Monthly-schedule-aware freshness: latest date should be within 35 days.
    Use this instead of check_freshness when the pipeline runs monthly.
    """
    r = check_freshness(conn, table, max_days=35)
    r["check_name"] = "freshness_monthly"
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 7. Predictions completeness (WARN — runs after predict stage)
# ══════════════════════════════════════════════════════════════════════════════

def check_predictions_completeness(daily_df, expected_cities: list[str]) -> dict:
    """
    Verify the predictions DataFrame:
      - Covers every expected city
      - Covers every day of its target month (no gaps)
      - Has valid source flags (only 'short_horizon' or 'climatology')
      - Has probabilities in [0, 1]
    """
    import pandas as pd

    issues = []
    if len(daily_df) == 0:
        return {
            "check_name": "predictions_completeness",
            "status":     "FAIL",
            "severity":   WARN,
            "stage":      "predict",
            "details":    {},
            "message":    "Predictions DataFrame is empty",
        }

    # City coverage
    cities_in = set(daily_df["city"].unique())
    cities_expected = set(expected_cities)
    if cities_in != cities_expected:
        missing = cities_expected - cities_in
        extra   = cities_in - cities_expected
        if missing:
            issues.append(f"missing cities: {sorted(missing)}")
        if extra:
            issues.append(f"unexpected cities: {sorted(extra)}")

    # Day coverage per city
    dates = pd.to_datetime(daily_df["date"])
    target_month = dates.dt.strftime("%Y-%m").iloc[0]
    expected_days = pd.Period(target_month).days_in_month
    by_city = daily_df.groupby("city").size()
    bad_cities = by_city[by_city != expected_days].to_dict()
    if bad_cities:
        issues.append(
            f"day count mismatch (expected {expected_days}): {bad_cities}"
        )

    # Source flag validity
    valid_sources = {"short_horizon", "climatology"}
    bad_sources = set(daily_df["source"].unique()) - valid_sources
    if bad_sources:
        issues.append(f"invalid source flags: {bad_sources}")

    # Probability range
    bad_probs = daily_df[
        (daily_df["probability"] < 0) | (daily_df["probability"] > 1)
    ]
    if len(bad_probs) > 0:
        issues.append(f"{len(bad_probs)} probabilities outside [0, 1]")

    passed = len(issues) == 0
    return {
        "check_name": "predictions_completeness",
        "status":     "PASS" if passed else "WARN",
        "severity":   WARN,
        "stage":      "predict",
        "details":    {
            "rows":           len(daily_df),
            "target_month":   target_month,
            "expected_days":  expected_days,
            "cities":         sorted(cities_in),
            "by_source":      daily_df["source"].value_counts().to_dict(),
        },
        "message":    "All predictions cover their target month cleanly"
                      if passed else
                      f"Issues: {'; '.join(issues)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Batch runner & pretty-print
# ══════════════════════════════════════════════════════════════════════════════

def run_all_checks(
    conn,
    stage:           str,
    monthly:         bool = True,
    predictions_df       = None,
    expected_cities: list[str] | None = None,
) -> list[dict]:
    """
    Run all checks relevant to a given pipeline stage.

    Parameters
    ----------
    stage           : 'raw' | 'staging' | 'analytics' | 'predict'
    monthly         : Use the 35-day freshness check instead of the 2-day one
    predictions_df  : Daily predictions DataFrame (required for stage='predict')
    expected_cities : List of city names (required for stage='predict')
    """
    results = []

    if stage == "raw":
        results.append(check_row_count(conn, "raw.weather_daily"))
        if monthly:
            results.append(check_freshness_monthly(conn))
        else:
            results.append(check_freshness(conn))

    elif stage == "staging":
        results.append(check_null_ratio(conn))
        results.append(check_date_continuity(conn))
        results.append(check_value_ranges(conn))

    elif stage == "analytics":
        results.append(check_feature_completeness(conn))

    elif stage == "predict":
        if predictions_df is None or expected_cities is None:
            raise ValueError(
                "stage='predict' requires predictions_df and expected_cities"
            )
        results.append(
            check_predictions_completeness(predictions_df, expected_cities)
        )

    else:
        raise ValueError(f"Unknown stage: {stage!r}")

    return results


def format_check_report(results: list[dict]) -> str:
    """Produce a readable text report for a list of check results."""
    if not results:
        return "(no checks run)"

    rows = []
    header = f"{'STATUS':<8} {'SEVERITY':<8} {'STAGE':<12} {'CHECK':<26} MESSAGE"
    rows.append(header)
    rows.append("─" * len(header))

    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "FLAG": "🚩"}
    for r in results:
        status = icon.get(r["status"], r["status"])
        rows.append(
            f"{status:<8} {r['severity']:<8} {r['stage']:<12} "
            f"{r['check_name']:<26} {r['message']}"
        )
    return "\n".join(rows)


def any_aborting(results: list[dict]) -> bool:
    """True if any result has severity=ABORT and status≠PASS."""
    return any(r["severity"] == ABORT and r["status"] != "PASS" for r in results)
