"""
src/cleaning.py
───────────────
Data cleaning pipeline for the Caspian Maritime Delay-Risk project.

Responsibilities
----------------
- Handle missing values per column using tailored strategies
- Flag outliers (IQR method) WITHOUT removing them — removal is a
  modelling-time decision, not a cleaning-time one
- Validate temporal continuity (no missing dates)
- Winsorize heavy-tailed precipitation/snowfall by city to prevent
  one outlier station (e.g. Anzali) from skewing cross-city features
- Orchestrate raw → staging transition in DuckDB

Public API
----------
    handle_missing_values(df, strategy)          → pd.DataFrame
    flag_outliers(df, columns, method, threshold) → pd.DataFrame
    validate_date_continuity(df, city)            → dict
    winsorize_by_city(df, caps)                   → pd.DataFrame
    clean_raw_to_staging(conn)                    → dict (summary stats)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Missing value handling
# ══════════════════════════════════════════════════════════════════════════════

# Default per-column strategies. Rationale per variable:
#   - Temperature/humidity/pressure → forward-fill (smooth meteorological vars)
#   - Precipitation/snowfall → zero (if API returned null, it means no event)
#   - Wind direction → drop (can't meaningfully interpolate circular data)
#   - Weather code → mode (categorical, fill with most common)
#   - Visibility → leave as-is (nulls are expected pre-2022)
_DEFAULT_STRATEGIES: dict[str, str] = {
    "temperature_2m_max":            "ffill",
    "temperature_2m_min":            "ffill",
    "temperature_2m_mean":           "ffill",
    "apparent_temperature_mean":     "ffill",
    "wind_speed_10m_max":            "ffill",
    "wind_gusts_10m_max":            "ffill",
    "wind_direction_10m_dominant":   "keep",
    "precipitation_sum":             "zero",
    "rain_sum":                      "zero",
    "snowfall_sum":                  "zero",
    "weather_code":                  "mode",
    "relative_humidity_2m_mean":     "ffill",
    "dew_point_2m_mean":             "ffill",
    "surface_pressure_mean":         "ffill",
    "shortwave_radiation_sum":       "ffill",
    # Visibility: imputed with median AFTER the visibility_is_known flag
    # is added (see features.add_visibility_missingness_flag). The model
    # learns to ignore imputed values via the flag.
    "visibility_mean":               "median",
    "visibility_min":                "median",
    "visibility_hours_below_1km":    "median",
}


def handle_missing_values(
    df:       pd.DataFrame,
    strategy: Optional[dict[str, str]] = None,
    group_by: str = "city",
) -> pd.DataFrame:
    """
    Impute or drop missing values per column using the given strategy map.

    Strategies supported:
      'ffill'   forward-fill within each group, then backfill to catch leading NaNs
      'bfill'   backfill only
      'zero'    replace NaN with 0
      'mean'    per-group mean
      'median'  per-group median
      'mode'    per-group mode (most common value)
      'drop'    drop rows where this column is NaN
      'keep'    leave NaN in place (for columns where null is meaningful)

    Parameters
    ----------
    df       : DataFrame with 'city' column (or another group_by key)
    strategy : {column: strategy} mapping. Defaults to _DEFAULT_STRATEGIES.
    group_by : Column name to group on for per-group imputation (usually 'city')

    Returns
    -------
    pd.DataFrame — new object, input not mutated
    """
    strategy = strategy or _DEFAULT_STRATEGIES
    out = df.copy()
    drop_mask = pd.Series(False, index=out.index)

    for col, strat in strategy.items():
        if col not in out.columns:
            continue

        n_missing_before = out[col].isna().sum()
        if n_missing_before == 0:
            continue

        if strat == "keep":
            continue

        elif strat == "zero":
            out[col] = out[col].fillna(0)

        elif strat == "ffill":
            # Forward-fill within each city, then backfill leading NaNs
            out[col] = (
                out.groupby(group_by, group_keys=False)[col]
                   .transform(lambda s: s.ffill().bfill())
            )

        elif strat == "bfill":
            out[col] = (
                out.groupby(group_by, group_keys=False)[col]
                   .transform(lambda s: s.bfill())
            )

        elif strat == "mean":
            out[col] = out.groupby(group_by)[col].transform(
                lambda s: s.fillna(s.mean())
            )

        elif strat == "median":
            out[col] = out.groupby(group_by)[col].transform(
                lambda s: s.fillna(s.median())
            )

        elif strat == "mode":
            def _fill_mode(s):
                mode = s.mode()
                if len(mode) == 0:
                    return s
                return s.fillna(mode.iloc[0])
            out[col] = out.groupby(group_by)[col].transform(_fill_mode)

        elif strat == "drop":
            drop_mask |= out[col].isna()

        else:
            logger.warning("Unknown strategy '%s' for column '%s' — skipping", strat, col)
            continue

        n_missing_after = out[col].isna().sum()
        n_filled = n_missing_before - n_missing_after
        if n_filled > 0:
            logger.info("  %s: filled %d/%d nulls with '%s'",
                        col, n_filled, n_missing_before, strat)

    if drop_mask.any():
        n_drop = int(drop_mask.sum())
        out = out.loc[~drop_mask].reset_index(drop=True)
        logger.info("  Dropped %d rows due to 'drop' strategy", n_drop)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. Outlier flagging (IQR or z-score)
# ══════════════════════════════════════════════════════════════════════════════

def flag_outliers(
    df:        pd.DataFrame,
    columns:   list[str],
    method:    str = "iqr",
    threshold: float = 1.5,
    group_by:  str = "city",
) -> pd.DataFrame:
    """
    Add a boolean flag column for each target column indicating outlier rows.
    Does NOT remove outliers — that's a modelling decision.

    Parameters
    ----------
    df        : Input DataFrame with 'city' column
    columns   : Which numeric columns to check
    method    : 'iqr' (Q1-1.5·IQR, Q3+1.5·IQR) or 'zscore' (|z| > threshold)
    threshold : IQR multiplier (default 1.5) or z-score threshold (default 1.5)
    group_by  : Calculate bounds per this column (so thresholds are city-specific)

    Returns
    -------
    pd.DataFrame with added columns `<col>_is_outlier` (bool) for each input column.
    """
    out = df.copy()
    n_flagged_total = 0

    for col in columns:
        if col not in out.columns:
            logger.warning("  Column '%s' not in DataFrame — skipping", col)
            continue

        flag_col = f"{col}_is_outlier"

        if method == "iqr":
            def _iqr_flag(s):
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or pd.isna(iqr):
                    return pd.Series(False, index=s.index)
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                return (s < lower) | (s > upper)

            out[flag_col] = (
                out.groupby(group_by, group_keys=False)[col]
                   .transform(_iqr_flag)
                   .fillna(False)
                   .astype(bool)
            )

        elif method == "zscore":
            def _z_flag(s):
                mu = s.mean()
                sd = s.std()
                if sd == 0 or pd.isna(sd):
                    return pd.Series(False, index=s.index)
                return ((s - mu) / sd).abs() > threshold

            out[flag_col] = (
                out.groupby(group_by, group_keys=False)[col]
                   .transform(_z_flag)
                   .fillna(False)
                   .astype(bool)
            )

        else:
            raise ValueError(f"method must be 'iqr' or 'zscore', got '{method}'")

        n_flagged = int(out[flag_col].sum())
        n_flagged_total += n_flagged
        logger.info("  %s: flagged %d outliers (%.1f%%) using %s method",
                    col, n_flagged, n_flagged / len(out) * 100, method)

    logger.info("Total outlier flags set: %d across %d columns",
                n_flagged_total, len(columns))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Temporal continuity validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_date_continuity(
    df:   pd.DataFrame,
    city: Optional[str] = None,
) -> dict:
    """
    Check for missing dates in a per-city daily sequence.

    Parameters
    ----------
    df   : DataFrame with 'date' and 'city' columns
    city : Optional — if provided, check only that city

    Returns
    -------
    dict with keys:
      - checked_cities : list of cities examined
      - total_expected : total days expected across all cities
      - total_actual   : total days actually present
      - gaps           : {city: [missing_date, ...]}
      - gap_count      : {city: int}
      - status         : '✅ OK' if no gaps, '⚠️ GAPS' otherwise
    """
    cities = [city] if city else sorted(df["city"].unique())
    gaps: dict[str, list] = {}
    gap_count: dict[str, int] = {}
    total_expected = 0
    total_actual = 0

    for c in cities:
        sub = df[df["city"] == c]
        if len(sub) == 0:
            gaps[c] = []
            gap_count[c] = 0
            continue

        sub_dates = pd.to_datetime(sub["date"])
        start, end = sub_dates.min(), sub_dates.max()
        expected = pd.date_range(start, end, freq="D")
        missing = expected.difference(pd.DatetimeIndex(sub_dates.values))

        gaps[c] = [d.strftime("%Y-%m-%d") for d in missing]
        gap_count[c] = len(missing)
        total_expected += len(expected)
        total_actual += len(sub)

    any_gaps = any(v > 0 for v in gap_count.values())
    return {
        "checked_cities": cities,
        "total_expected": total_expected,
        "total_actual":   total_actual,
        "gaps":           gaps,
        "gap_count":      gap_count,
        "status":         "⚠️ GAPS" if any_gaps else "✅ OK",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Winsorizing (per-city caps on heavy-tailed variables)
# ══════════════════════════════════════════════════════════════════════════════

def winsorize_by_city(
    df:   pd.DataFrame,
    caps: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Cap heavy-tailed variables at per-city thresholds.

    The Caspian cities have very different precipitation climates — Anzali
    routinely gets 60+ mm rainstorms while Aktau's 99th percentile is ~25 mm.
    Without winsorizing, cross-city statistics (std, variance) are dominated
    by Anzali's tail, and downstream risk models over-weight those events.

    Parameters
    ----------
    df   : DataFrame with 'city' column
    caps : {column: {city: cap_value}} — see config.WINSORIZE_CAPS

    Returns
    -------
    pd.DataFrame with values clipped to city-specific caps.
    """
    out = df.copy()
    total_capped = 0

    for col, per_city_caps in caps.items():
        if col not in out.columns:
            continue
        for city, cap_val in per_city_caps.items():
            mask = (out["city"] == city) & (out[col] > cap_val)
            n_capped = int(mask.sum())
            if n_capped > 0:
                out.loc[mask, col] = cap_val
                total_capped += n_capped
                logger.info(
                    "  Winsorized %s for %s: %d values > %.1f capped",
                    col, city, n_capped, cap_val
                )

    if total_capped == 0:
        logger.info("  No values required winsorizing")
    else:
        logger.info("Total values winsorized: %d", total_capped)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. Raw → staging orchestration (DuckDB)
# ══════════════════════════════════════════════════════════════════════════════

def add_visibility_missingness_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary column `visibility_is_known` before visibility imputation.

    For the 2015-2024 window, visibility data is only available from 2022
    onwards (historical-forecast-api coverage). Pre-2022 rows have NaN
    visibility. This function marks which rows have REAL visibility data
    vs. which will be imputed — the model uses this flag to decide how
    much to trust the visibility features.

    MUST be called BEFORE handle_missing_values() or the flag will be 1
    for all rows (because NaN will already be filled).

    Parameters
    ----------
    df : DataFrame with 'visibility_mean' column

    Returns
    -------
    pd.DataFrame with added 'visibility_is_known' column (int 0/1)
    """
    out = df.copy()
    if "visibility_mean" in out.columns:
        out["visibility_is_known"] = out["visibility_mean"].notna().astype(int)
    else:
        out["visibility_is_known"] = 0
    n_known = int(out["visibility_is_known"].sum())
    n_total = len(out)
    logger.info(
        "  visibility_is_known: %d/%d rows have real visibility (%.1f%%)",
        n_known, n_total, n_known / n_total * 100 if n_total else 0,
    )
    return out


def clean_raw_to_staging(
    conn,
    outlier_columns:  Optional[list[str]] = None,
    winsorize_caps:   Optional[dict]      = None,
    missing_strategy: Optional[dict]      = None,
) -> dict:
    """
    Read raw tables, apply cleaning, write to staging.weather_daily.

    Pipeline:
      1. Pull raw.weather_daily + raw.visibility_daily (LEFT JOINed) into pandas
      2. Handle missing values per strategy
      3. Winsorize per-city caps on heavy-tailed variables
      4. Flag outliers (boolean columns added)
      5. Validate date continuity
      6. Overwrite staging.weather_daily with the cleaned result

    Parameters
    ----------
    conn            : DuckDB connection
    outlier_columns : Columns to flag outliers in (default: config list)
    winsorize_caps  : Per-city caps dict (default: config.WINSORIZE_CAPS)
    missing_strategy: Missing-value strategy dict (default: _DEFAULT_STRATEGIES)

    Returns
    -------
    dict with cleaning statistics: rows_in, rows_out, outliers_flagged,
    values_winsorized, missing_filled, gaps.
    """
    # Lazy imports so this file works without the config module at import-time
    try:
        from src.config import OUTLIER_FLAG_COLUMNS, WINSORIZE_CAPS
    except ImportError:
        OUTLIER_FLAG_COLUMNS = []
        WINSORIZE_CAPS = {}

    outlier_columns = outlier_columns or OUTLIER_FLAG_COLUMNS
    winsorize_caps  = winsorize_caps  or WINSORIZE_CAPS

    logger.info("=== Raw → Staging cleaning pipeline ===")

    # ── Step 1: pull joined raw data into pandas ─────────────────────────────
    df = conn.execute("""
        SELECT
            w.*,
            v.visibility_mean,
            v.visibility_min,
            CAST(v.visibility_hours_below_1km AS INTEGER)
                AS visibility_hours_below_1km
        FROM raw.weather_daily w
        LEFT JOIN raw.visibility_daily v
            ON w.city = v.city AND w.date = v.date
        ORDER BY w.city, w.date
    """).fetchdf()

    rows_in = len(df)
    nulls_before = int(df.select_dtypes("number").isna().sum().sum())
    logger.info("Loaded %d rows (%d total nulls)", rows_in, nulls_before)

    # ── Step 2a: add visibility missingness flag BEFORE imputation ───────────
    # The flag captures which rows had real visibility data (2022+) vs imputed.
    # This MUST happen before handle_missing_values fills NaN visibility values.
    logger.info("Adding visibility_is_known flag...")
    df = add_visibility_missingness_flag(df)

    # ── Step 2b: handle missing values (including median-impute visibility) ──
    logger.info("Handling missing values...")
    df = handle_missing_values(df, strategy=missing_strategy)
    nulls_after = int(df.select_dtypes("number").isna().sum().sum())

    # ── Step 3: winsorize heavy-tailed variables ─────────────────────────────
    logger.info("Winsorizing heavy-tailed variables per-city...")
    df = winsorize_by_city(df, caps=winsorize_caps)

    # ── Step 4: flag outliers (no removal) ───────────────────────────────────
    logger.info("Flagging outliers (IQR method)...")
    df = flag_outliers(df, columns=outlier_columns, method="iqr", threshold=1.5)
    outlier_flags_set = int(
        df[[c for c in df.columns if c.endswith("_is_outlier")]].sum().sum()
    )

    # ── Step 5: date continuity ──────────────────────────────────────────────
    logger.info("Validating date continuity...")
    continuity = validate_date_continuity(df)

    # ── Step 6: write staging table ──────────────────────────────────────────
    # DuckDB reads the pandas DataFrame directly via variable binding
    conn.register("_clean_staging", df)
    conn.execute("""
        CREATE OR REPLACE TABLE staging.weather_daily AS
        SELECT * FROM _clean_staging
    """)
    conn.unregister("_clean_staging")

    rows_out = conn.execute(
        "SELECT COUNT(*) FROM staging.weather_daily"
    ).fetchone()[0]
    logger.info("Wrote %d rows to staging.weather_daily", rows_out)

    # Compute winsorize count summary (vs original raw)
    raw_df = conn.execute("""
        SELECT city, date, precipitation_sum, snowfall_sum
        FROM raw.weather_daily
    """).fetchdf()
    values_winsorized = 0
    for col, per_city in winsorize_caps.items():
        if col not in raw_df.columns:
            continue
        for city, cap in per_city.items():
            mask = (raw_df["city"] == city) & (raw_df[col] > cap)
            values_winsorized += int(mask.sum())

    summary = {
        "rows_in":            rows_in,
        "rows_out":           rows_out,
        "missing_filled":     nulls_before - nulls_after,
        "missing_remaining":  nulls_after,
        "values_winsorized":  values_winsorized,
        "outlier_flags_set":  outlier_flags_set,
        "date_gaps":          sum(continuity["gap_count"].values()),
        "continuity_status":  continuity["status"],
    }

    logger.info("=== Pipeline complete ===")
    for k, v in summary.items():
        logger.info("  %-20s: %s", k, v)

    return summary
