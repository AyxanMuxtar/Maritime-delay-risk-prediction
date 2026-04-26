"""
src/database.py
───────────────
DuckDB analytical database for the Caspian Maritime Delay-Risk project.

Three-layer schema design
-------------------------
  raw        → Direct copies of ingested CSVs. No transformations.
  staging    → Cleaned data: nulls handled, types enforced, duplicates removed.
  analytics  → Feature-enriched tables: derived columns, rolling stats, risk flags.

Public API
----------
    get_connection(db_path)        → duckdb.DuckDBPyConnection
    create_schemas(conn)           → None
    create_raw_tables(conn)        → None
    load_raw_data(conn, data_dir)  → dict[str, int]   (table → row count)
    build_staging(conn)            → None
    build_analytics(conn)          → None
    validate_database(conn, ...)   → pd.DataFrame      (validation report)
    run_query(conn, sql)           → pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Connection
# ══════════════════════════════════════════════════════════════════════════════

def get_connection(db_path: str | Path = "data/caspian_weather.duckdb"):
    """
    Return a DuckDB connection, creating the database file if needed.

    Parameters
    ----------
    db_path : Path to the .duckdb file. Use ':memory:' for in-memory.

    Returns
    -------
    duckdb.DuckDBPyConnection
    """
    import duckdb

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    logger.info("Connected to DuckDB: %s", db_path)
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# 2. Schema creation
# ══════════════════════════════════════════════════════════════════════════════

def create_schemas(conn) -> None:
    """Create raw, staging, and analytics schemas if they don't exist."""
    for schema in ("raw", "staging", "analytics"):
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        logger.info("  Schema '%s' ready.", schema)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Raw tables
# ══════════════════════════════════════════════════════════════════════════════

# Column definitions for the raw weather table.
# Matches the output of ingestion.fetch_historical() → _payload_to_dataframe().
_RAW_WEATHER_COLUMNS = """
    city                          VARCHAR NOT NULL,
    date                          DATE    NOT NULL,
    temperature_2m_max            DOUBLE,
    temperature_2m_min            DOUBLE,
    temperature_2m_mean           DOUBLE,
    apparent_temperature_mean     DOUBLE,
    wind_speed_10m_max            DOUBLE,
    wind_gusts_10m_max            DOUBLE,
    wind_direction_10m_dominant   DOUBLE,
    precipitation_sum             DOUBLE,
    rain_sum                      DOUBLE,
    snowfall_sum                  DOUBLE,
    weather_code                  DOUBLE,
    relative_humidity_2m_mean     DOUBLE,
    dew_point_2m_mean             DOUBLE,
    surface_pressure_mean         DOUBLE,
    shortwave_radiation_sum       DOUBLE
"""

_RAW_VISIBILITY_COLUMNS = """
    city                          VARCHAR NOT NULL,
    date                          DATE    NOT NULL,
    visibility_mean               DOUBLE,
    visibility_min                DOUBLE,
    visibility_hours_below_1km    DOUBLE
"""

_RAW_FORECAST_COLUMNS = """
    city                          VARCHAR NOT NULL,
    date                          DATE    NOT NULL,
    temperature_2m_max            DOUBLE,
    temperature_2m_min            DOUBLE,
    wind_speed_10m_max            DOUBLE,
    wind_gusts_10m_max            DOUBLE,
    precipitation_sum             DOUBLE,
    snowfall_sum                  DOUBLE,
    weather_code                  DOUBLE,
    relative_humidity_2m_mean     DOUBLE,
    visibility_mean               DOUBLE
"""


def create_raw_tables_if_absent(conn) -> None:
    """
    Like create_raw_tables() but uses CREATE TABLE IF NOT EXISTS so it
    does NOT wipe existing data. Use this for incremental pipeline runs.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS raw.weather_daily (
            {_RAW_WEATHER_COLUMNS},
            PRIMARY KEY (city, date)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS raw.visibility_daily (
            {_RAW_VISIBILITY_COLUMNS},
            PRIMARY KEY (city, date)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS raw.forecast_7day (
            {_RAW_FORECAST_COLUMNS}
        )
    """)
    logger.info("Raw tables ready (preserving any existing rows)")


def create_raw_tables(conn) -> None:
    """
    Create raw-layer tables. Drops and recreates to ensure clean state.
    """
    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.weather_daily (
            {_RAW_WEATHER_COLUMNS},
            PRIMARY KEY (city, date)
        )
    """)
    logger.info("  Created raw.weather_daily")

    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.visibility_daily (
            {_RAW_VISIBILITY_COLUMNS},
            PRIMARY KEY (city, date)
        )
    """)
    logger.info("  Created raw.visibility_daily")

    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.forecast_7day (
            {_RAW_FORECAST_COLUMNS}
        )
    """)
    logger.info("  Created raw.forecast_7day")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Load raw data
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data(
    conn,
    data_dir: str | Path = "data/raw",
) -> dict[str, int]:
    """
    Load CSV files from data/raw/ into raw-layer tables.

    Detects file naming convention:
      {city}_historical_*.csv    → raw.weather_daily
      {city}_hourly_visibility_* → aggregated then → raw.visibility_daily
      {city}_forecast_7day.csv   → raw.forecast_7day

    Returns dict mapping table name → total rows loaded.
    """
    data_dir = Path(data_dir)
    loaded: dict[str, int] = {}

    # ── Weather historical ────────────────────────────────────────────────────
    weather_files = sorted(data_dir.glob("*_historical_*.csv"))

    # If city-specific files exist, skip combined all_cities files to avoid duplicates.
    city_specific_weather = [
        f for f in weather_files
        if not f.name.startswith("all_cities_")
    ]
    if city_specific_weather:
        weather_files = city_specific_weather
    if weather_files:
        for f in weather_files:
            # Read CSV header to determine which columns are present
            df_sample = pd.read_csv(f, nrows=0)
            # Select only columns that exist in the raw table
            raw_cols = [
                "city", "date",
                "temperature_2m_max", "temperature_2m_min",
                "temperature_2m_mean", "apparent_temperature_mean",
                "wind_speed_10m_max", "wind_gusts_10m_max",
                "wind_direction_10m_dominant",
                "precipitation_sum", "rain_sum", "snowfall_sum",
                "weather_code", "relative_humidity_2m_mean",
                "dew_point_2m_mean", "surface_pressure_mean",
                "shortwave_radiation_sum",
            ]
            present = [c for c in raw_cols if c in df_sample.columns]
            select_expr = ", ".join(present)

            conn.execute(f"""
                INSERT OR REPLACE INTO raw.weather_daily ({select_expr})
                SELECT {select_expr}
                FROM read_csv_auto('{f}', header=true, dateformat='%Y-%m-%d')
            """)
            logger.info("  Loaded %s → raw.weather_daily", f.name)

            # If file also has visibility columns, load those too
            vis_cols = ["visibility_mean", "visibility_min", "visibility_hours_below_1km"]
            vis_present = [c for c in vis_cols if c in df_sample.columns]
            if vis_present:
                vis_select = ", ".join(["city", "date"] + vis_present)
                # Only insert rows where visibility data exists (non-null)
                where_clause = " OR ".join(f"{c} IS NOT NULL" for c in vis_present)
                conn.execute(f"""
                    INSERT OR REPLACE INTO raw.visibility_daily ({vis_select})
                    SELECT {vis_select}
                    FROM read_csv_auto('{f}', header=true, dateformat='%Y-%m-%d')
                    WHERE {where_clause}
                """)
                logger.info("  Loaded visibility from %s → raw.visibility_daily", f.name)

        count = conn.execute("SELECT COUNT(*) FROM raw.weather_daily").fetchone()[0]
        loaded["raw.weather_daily"] = count

    # ── Hourly visibility CSVs (aggregate to daily, then load) ────────────────
    vis_hourly_files = sorted(data_dir.glob("*_hourly_visibility_*.csv"))
    if vis_hourly_files:
        for f in vis_hourly_files:
            conn.execute(f"""
                INSERT OR IGNORE INTO raw.visibility_daily
                SELECT
                    city,
                    CAST(datetime AS DATE) AS date,
                    AVG(visibility)        AS visibility_mean,
                    MIN(visibility)        AS visibility_min,
                    SUM(CASE WHEN visibility < 1000 THEN 1 ELSE 0 END)
                                           AS visibility_hours_below_1km
                FROM read_csv_auto('{f}', header=true)
                GROUP BY city, CAST(datetime AS DATE)
            """)
            logger.info("  Loaded %s → raw.visibility_daily (hourly aggregated)", f.name)

        count = conn.execute("SELECT COUNT(*) FROM raw.visibility_daily").fetchone()[0]
        loaded["raw.visibility_daily"] = count

    # ── Forecasts ─────────────────────────────────────────────────────────────
    forecast_files = sorted(data_dir.glob("*_forecast_7day.csv"))
    if forecast_files:
        for f in forecast_files:
            df_sample = pd.read_csv(f, nrows=0)
            fcast_cols = [
                "city", "date",
                "temperature_2m_max", "temperature_2m_min",
                "wind_speed_10m_max", "wind_gusts_10m_max",
                "precipitation_sum", "snowfall_sum",
                "weather_code", "relative_humidity_2m_mean",
                "visibility_mean",
            ]
            present = [c for c in fcast_cols if c in df_sample.columns]
            select_expr = ", ".join(present)

            conn.execute(f"""
                INSERT INTO raw.forecast_7day ({select_expr})
                SELECT {select_expr}
                FROM read_csv_auto('{f}', header=true, dateformat='%Y-%m-%d')
            """)
            logger.info("  Loaded %s → raw.forecast_7day", f.name)

        count = conn.execute("SELECT COUNT(*) FROM raw.forecast_7day").fetchone()[0]
        loaded["raw.forecast_7day"] = count

    logger.info("Raw data load complete: %s", loaded)
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# 5. Staging layer
# ══════════════════════════════════════════════════════════════════════════════

def build_staging(conn) -> None:
    """
    Build staging-layer tables from raw data.

    Transformations applied:
      - Remove duplicate (city, date) rows (keep first)
      - Enforce NOT NULL on critical columns
      - Left-join visibility data onto weather
      - Cast types explicitly

    Note: visibility imputation + visibility_is_known flag are handled
    by src/cleaning.py:clean_raw_to_staging() which REPLACES the output
    of this function when called from the Day 4 pipeline. This function
    is kept as a minimal reference / fallback.
    """
    conn.execute("""
        CREATE OR REPLACE TABLE staging.weather_daily AS
        WITH deduped_weather AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY date) AS rn
            FROM raw.weather_daily
        ),
        clean_weather AS (
            SELECT * EXCLUDE (rn)
            FROM deduped_weather
            WHERE rn = 1
              AND city IS NOT NULL
              AND date IS NOT NULL
        ),
        deduped_vis AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY date) AS rn
            FROM raw.visibility_daily
        ),
        clean_vis AS (
            SELECT * EXCLUDE (rn)
            FROM deduped_vis
            WHERE rn = 1
        )
        SELECT
            w.city,
            w.date,

            -- Temperature
            ROUND(w.temperature_2m_max, 2)          AS temperature_2m_max,
            ROUND(w.temperature_2m_min, 2)          AS temperature_2m_min,
            ROUND(w.temperature_2m_mean, 2)         AS temperature_2m_mean,
            ROUND(w.apparent_temperature_mean, 2)    AS apparent_temperature_mean,

            -- Wind
            ROUND(w.wind_speed_10m_max, 2)          AS wind_speed_10m_max,
            ROUND(w.wind_gusts_10m_max, 2)          AS wind_gusts_10m_max,
            ROUND(w.wind_direction_10m_dominant, 1)  AS wind_direction_10m_dominant,

            -- Precipitation
            ROUND(w.precipitation_sum, 2)            AS precipitation_sum,
            ROUND(w.rain_sum, 2)                     AS rain_sum,
            ROUND(w.snowfall_sum, 2)                 AS snowfall_sum,

            -- Atmosphere
            CAST(w.weather_code AS INTEGER)          AS weather_code,
            ROUND(w.relative_humidity_2m_mean, 2)    AS relative_humidity_2m_mean,
            ROUND(w.dew_point_2m_mean, 2)            AS dew_point_2m_mean,
            ROUND(w.surface_pressure_mean, 2)        AS surface_pressure_mean,
            ROUND(w.shortwave_radiation_sum, 2)      AS shortwave_radiation_sum,

            -- Visibility (NULL for pre-2022 dates — expected)
            ROUND(v.visibility_mean, 1)              AS visibility_mean,
            ROUND(v.visibility_min, 1)               AS visibility_min,
            CAST(v.visibility_hours_below_1km AS INTEGER)
                                                     AS visibility_hours_below_1km,

        FROM clean_weather w
        LEFT JOIN clean_vis v
            ON w.city = v.city AND w.date = v.date
        ORDER BY w.city, w.date
    """)

    count = conn.execute("SELECT COUNT(*) FROM staging.weather_daily").fetchone()[0]
    cities = conn.execute(
        "SELECT COUNT(DISTINCT city) FROM staging.weather_daily"
    ).fetchone()[0]
    logger.info(
        "  Built staging.weather_daily: %d rows, %d cities", count, cities
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Analytics layer
# ══════════════════════════════════════════════════════════════════════════════

def build_analytics(conn) -> None:
    """
    Build analytics-layer tables from staging data.

    Creates:
      - analytics.daily_enriched    → daily data + risk flags, rolling stats
      - analytics.monthly_summary   → monthly aggregates + target label
    """

    # ── Daily enriched ────────────────────────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE analytics.daily_enriched AS
        SELECT
            *,

            -- Calendar features
            EXTRACT(YEAR FROM date)    AS year,
            EXTRACT(MONTH FROM date)   AS month,
            EXTRACT(DOW FROM date)     AS day_of_week,
            EXTRACT(DOY FROM date)     AS day_of_year,

            -- Season label (meteorological)
            CASE
                WHEN EXTRACT(MONTH FROM date) IN (12, 1, 2) THEN 'Winter'
                WHEN EXTRACT(MONTH FROM date) IN (3, 4, 5)  THEN 'Spring'
                WHEN EXTRACT(MONTH FROM date) IN (6, 7, 8)  THEN 'Summer'
                ELSE 'Autumn'
            END AS season,

            -- Individual risk flags (1 = threshold breached)
            CASE WHEN wind_speed_10m_max > 50 THEN 1 ELSE 0 END
                AS risk_wind,
            CASE WHEN wind_gusts_10m_max > 75 THEN 1 ELSE 0 END
                AS risk_gust,
            CASE WHEN precipitation_sum > 15 THEN 1 ELSE 0 END
                AS risk_precip,
            CASE WHEN snowfall_sum > 5 THEN 1 ELSE 0 END
                AS risk_snow,
            CASE WHEN visibility_mean IS NOT NULL AND visibility_mean < 1000
                 THEN 1 ELSE 0 END
                AS risk_visibility,
            -- Combined: any risk flag triggered
            CASE WHEN
                wind_speed_10m_max > 50
                OR wind_gusts_10m_max > 75
                OR precipitation_sum > 15
                OR snowfall_sum > 5
                OR (visibility_mean IS NOT NULL AND visibility_mean < 1000)
            THEN 1 ELSE 0 END
                AS is_risk_day,

            -- Rolling stats (7-day window)
            AVG(wind_speed_10m_max) OVER w7     AS wind_speed_7d_avg,
            MAX(wind_speed_10m_max) OVER w7     AS wind_speed_7d_max,
            AVG(precipitation_sum)  OVER w7     AS precip_7d_avg,
            SUM(precipitation_sum)  OVER w7     AS precip_7d_sum,
            AVG(temperature_2m_mean) OVER w7    AS temp_7d_avg,

            -- Lag features (previous day)
            LAG(wind_speed_10m_max, 1) OVER w_city    AS wind_speed_lag1,
            LAG(precipitation_sum, 1)  OVER w_city    AS precip_lag1,
            LAG(temperature_2m_mean, 1) OVER w_city   AS temp_lag1

        FROM staging.weather_daily
        WINDOW
            w7     AS (PARTITION BY city ORDER BY date
                       ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
            w_city AS (PARTITION BY city ORDER BY date)
        ORDER BY city, date
    """)

    count = conn.execute("SELECT COUNT(*) FROM analytics.daily_enriched").fetchone()[0]
    logger.info("  Built analytics.daily_enriched: %d rows", count)

    # ── Monthly summary + target label ────────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE TABLE analytics.monthly_summary AS
        SELECT
            city,
            EXTRACT(YEAR FROM date)  AS year,
            EXTRACT(MONTH FROM date) AS month,
            DATE_TRUNC('month', date) AS month_start,

            -- Row counts
            COUNT(*)                        AS total_days,
            SUM(is_risk_day)                AS risk_days,
            ROUND(SUM(is_risk_day) * 100.0 / COUNT(*), 1)
                                            AS risk_day_pct,

            -- TARGET LABEL
            CASE WHEN SUM(is_risk_day) >= 5 THEN 1 ELSE 0 END
                                            AS high_risk_month,

            -- Risk breakdown
            SUM(risk_wind)                  AS wind_risk_days,
            SUM(risk_gust)                  AS gust_risk_days,
            SUM(risk_precip)                AS precip_risk_days,
            SUM(risk_snow)                  AS snow_risk_days,
            SUM(risk_visibility)            AS visibility_risk_days,

            -- Temperature stats
            ROUND(AVG(temperature_2m_mean), 2)  AS temp_mean,
            ROUND(MAX(temperature_2m_max), 2)   AS temp_max,
            ROUND(MIN(temperature_2m_min), 2)   AS temp_min,
            ROUND(STDDEV(temperature_2m_mean), 2) AS temp_std,

            -- Wind stats
            ROUND(AVG(wind_speed_10m_max), 2)   AS wind_mean,
            ROUND(MAX(wind_speed_10m_max), 2)   AS wind_max,
            ROUND(STDDEV(wind_speed_10m_max), 2) AS wind_std,
            ROUND(AVG(wind_gusts_10m_max), 2)   AS gust_mean,
            ROUND(MAX(wind_gusts_10m_max), 2)   AS gust_max,

            -- Precipitation stats
            ROUND(SUM(precipitation_sum), 2)    AS precip_total,
            ROUND(AVG(precipitation_sum), 2)    AS precip_daily_avg,
            ROUND(MAX(precipitation_sum), 2)    AS precip_max,
            SUM(CASE WHEN precipitation_sum = 0 THEN 1 ELSE 0 END)
                                                AS dry_days,

            -- Snowfall stats
            ROUND(SUM(snowfall_sum), 2)         AS snow_total,
            ROUND(MAX(snowfall_sum), 2)         AS snow_max,

            -- Pressure stats
            ROUND(AVG(surface_pressure_mean), 2)    AS pressure_mean,
            ROUND(STDDEV(surface_pressure_mean), 2) AS pressure_std,
            ROUND(MIN(surface_pressure_mean), 2)    AS pressure_min,

            -- Humidity stats
            ROUND(AVG(relative_humidity_2m_mean), 2) AS humidity_mean,
            ROUND(MAX(relative_humidity_2m_mean), 2) AS humidity_max,

            -- Visibility stats (NULL if no visibility data in this month)
            ROUND(AVG(visibility_mean), 1)          AS vis_mean_avg,
            ROUND(MIN(visibility_min), 1)           AS vis_min_worst,
            SUM(CASE WHEN visibility_hours_below_1km >= 4 THEN 1 ELSE 0 END)
                                                    AS sustained_fog_days

        FROM analytics.daily_enriched
        GROUP BY city,
                 EXTRACT(YEAR FROM date),
                 EXTRACT(MONTH FROM date),
                 DATE_TRUNC('month', date)
        ORDER BY city,
                 EXTRACT(YEAR FROM date),
                 EXTRACT(MONTH FROM date)
    """)

    count = conn.execute("SELECT COUNT(*) FROM analytics.monthly_summary").fetchone()[0]
    logger.info("  Built analytics.monthly_summary: %d rows", count)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_database(
    conn,
    expected_cities: list[str] | None = None,
    expected_start:  str | None       = None,
    expected_end:    str | None       = None,
) -> pd.DataFrame:
    """
    Run a battery of validation checks on the database.

    Parameters
    ----------
    expected_cities : List of city names (defaults to src.config.CITIES keys)
    expected_start  : ISO date string (defaults to src.config.DATE_RANGE['start'])
    expected_end    : ISO date string (defaults to src.config.DATE_RANGE['end'])

    Defaults are read from src.config so a config change propagates
    automatically — no stale hardcoded dates can cause validation to
    pass when it shouldn't.

    Returns a DataFrame with columns: check, status, detail
    """
    # Resolve defaults from src.config so validation always matches the
    # project-wide source of truth. Explicit arguments still win.
    if expected_cities is None or expected_start is None or expected_end is None:
        from src.config import CITIES as _CFG_CITIES, DATE_RANGE as _CFG_DATE_RANGE
        if expected_cities is None:
            expected_cities = list(_CFG_CITIES.keys())
        if expected_start is None:
            expected_start = _CFG_DATE_RANGE["start"]
        if expected_end is None:
            expected_end = _CFG_DATE_RANGE["end"]

    results: list[dict] = []

    def _check(name: str, sql: str, expect_fn, fail_msg_fn):
        try:
            val = conn.execute(sql).fetchone()[0]
            passed = expect_fn(val)
            results.append({
                "check":  name,
                "status": "✅ PASS" if passed else "⚠️ FAIL",
                "detail": str(val) if passed else fail_msg_fn(val),
            })
        except Exception as exc:
            results.append({
                "check": name, "status": "❌ ERROR", "detail": str(exc)
            })

    # Check 1: tables exist
    for tbl in ["raw.weather_daily", "staging.weather_daily",
                "analytics.daily_enriched", "analytics.monthly_summary"]:
        _check(
            f"Table exists: {tbl}",
            f"SELECT COUNT(*) FROM {tbl}",
            lambda v: v > 0,
            lambda v: f"Table is empty ({v} rows)",
        )

    # Check 2: city count
    _check(
        "City count (raw)",
        "SELECT COUNT(DISTINCT city) FROM raw.weather_daily",
        lambda v: v == len(expected_cities),
        lambda v: f"Expected {len(expected_cities)}, got {v}",
    )

    # Check 3: row counts per city
    per_city = conn.execute("""
        SELECT city, COUNT(*) AS n
        FROM staging.weather_daily
        GROUP BY city ORDER BY city
    """).fetchdf()
    for _, row in per_city.iterrows():
        city, n = row["city"], row["n"]
        expected_min = 1800  # ~5 years minimum
        results.append({
            "check":  f"Row count: {city}",
            "status": "✅ PASS" if n >= expected_min else "⚠️ FAIL",
            "detail": f"{n} rows",
        })

    # Check 4: date range
    _check(
        "Date range start",
        "SELECT MIN(date) FROM staging.weather_daily",
        lambda v: str(v) <= expected_start,
        lambda v: f"Starts at {v}, expected <= {expected_start}",
    )
    _check(
        "Date range end",
        "SELECT MAX(date) FROM staging.weather_daily",
        lambda v: str(v) >= expected_end,
        lambda v: f"Ends at {v}, expected >= {expected_end}",
    )

    # Check 5: no duplicate (city, date)
    _check(
        "No duplicates (staging)",
        """SELECT COUNT(*) FROM (
            SELECT city, date, COUNT(*) AS n
            FROM staging.weather_daily
            GROUP BY city, date HAVING n > 1
        )""",
        lambda v: v == 0,
        lambda v: f"{v} duplicate city-date pairs found",
    )

    # Check 6: date gaps
    gap_sql = """
        SELECT COUNT(*) FROM (
            SELECT city, date,
                   LEAD(date) OVER (PARTITION BY city ORDER BY date) AS next_date,
                   DATEDIFF('day', date,
                     LEAD(date) OVER (PARTITION BY city ORDER BY date)) AS gap
            FROM staging.weather_daily
        ) WHERE gap > 1
    """
    _check(
        "No date gaps (staging)",
        gap_sql,
        lambda v: v == 0,
        lambda v: f"{v} gaps detected",
    )

    # Check 7: analytics monthly summary has target label
    _check(
        "Target label exists",
        """SELECT COUNT(DISTINCT high_risk_month)
           FROM analytics.monthly_summary""",
        lambda v: v == 2,   # should have both 0 and 1
        lambda v: f"Only {v} distinct label value(s) — class imbalance issue",
    )

    # Check 8: risk day counts are reasonable
    _check(
        "Risk days present",
        "SELECT SUM(is_risk_day) FROM analytics.daily_enriched",
        lambda v: v > 0,
        lambda v: "No risk days flagged — thresholds may be too strict",
    )

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Query helper
# ══════════════════════════════════════════════════════════════════════════════

def run_query(conn, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
    return conn.execute(sql).fetchdf()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def build_database(
    db_path: str | Path = "data/caspian_weather.duckdb",
    data_dir: str | Path = "data/raw",
    expected_start: str | None = None,
    expected_end:   str | None = None,
) -> tuple:
    """
    End-to-end: create DB, load raw data, build staging & analytics, validate.

    When expected_start / expected_end are None, validate_database() reads
    them from src.config.DATE_RANGE.

    Returns (connection, load_counts, validation_df).
    """
    conn = get_connection(db_path)
    create_schemas(conn)
    create_raw_tables(conn)
    counts = load_raw_data(conn, data_dir)
    build_staging(conn)
    build_analytics(conn)
    validation = validate_database(
        conn,
        expected_start=expected_start,
        expected_end=expected_end,
    )
    return conn, counts, validation
