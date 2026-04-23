"""
src/features.py
───────────────
Feature engineering for the Caspian Maritime Delay-Risk project.

Responsibilities
----------------
- Rolling averages (7-day, 30-day) for temperature, precipitation, wind
- Calendar / seasonal indicators (month, quarter, season, day-of-year)
- Temperature range and volatility indicators
- Heating and Cooling Degree Days (HDD / CDD)
- Anomaly scores (deviation from climatological norm for each calendar day)
- Lag features for prediction (yesterday, day-before-yesterday)
- Build analytics.daily_enriched and analytics.monthly_summary

Public API
----------
    add_rolling_features(df, windows, columns)    → pd.DataFrame
    add_seasonal_features(df)                      → pd.DataFrame
    add_temperature_range(df)                      → pd.DataFrame
    add_degree_days(df, base_temp=18)              → pd.DataFrame
    add_anomaly_scores(df, columns)                → pd.DataFrame
    add_lag_features(df, columns, lags)            → pd.DataFrame
    add_wave_proxy(df)                             → pd.DataFrame
    engineer_all_features(df, city_col='city')     → pd.DataFrame
    build_analytics_layer(conn)                    → dict (row counts)
"""

from __future__ import annotations

import logging
from typing import Optional, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Rolling averages
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_features(
    df:       pd.DataFrame,
    windows:  Iterable[int]           = (7, 30),
    columns:  Optional[list[str]]     = None,
    group_by: str                     = "city",
    date_col: str                     = "date",
) -> pd.DataFrame:
    """
    Add rolling-mean and rolling-max features per-city.

    For each (column × window), adds two new columns:
        <column>_<window>d_mean
        <column>_<window>d_max

    Ensures rows are sorted by (group_by, date_col) before computing.
    Uses min_periods=1 so early rows are not NaN.
    """
    default_cols = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    columns = columns or default_cols
    out = df.sort_values([group_by, date_col]).copy()

    for col in columns:
        if col not in out.columns:
            continue
        for w in windows:
            mean_col = f"{col}_{w}d_mean"
            max_col  = f"{col}_{w}d_max"
            out[mean_col] = (
                out.groupby(group_by, group_keys=False)[col]
                   .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
            out[max_col] = (
                out.groupby(group_by, group_keys=False)[col]
                   .transform(lambda s: s.rolling(w, min_periods=1).max())
            )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. Seasonal / calendar indicators
# ══════════════════════════════════════════════════════════════════════════════

def add_seasonal_features(
    df:       pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add calendar features: year, month, quarter, season, day_of_year,
    week_of_year, plus cyclical (sin/cos) encodings of month and day-of-year.

    Cyclical encodings are important because month 12 is adjacent to month 1,
    which tree-based models won't see without the sin/cos transform.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])

    out["year"]         = dt.dt.year.astype(int)
    out["month"]        = dt.dt.month.astype(int)
    out["quarter"]      = dt.dt.quarter.astype(int)
    out["day_of_year"]  = dt.dt.dayofyear.astype(int)
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["day_of_week"]  = dt.dt.dayofweek.astype(int)

    # Meteorological season (Northern Hemisphere)
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    out["season"] = out["month"].map(season_map)

    # Cyclical encodings
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"]   = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Temperature range (volatility indicator)
# ══════════════════════════════════════════════════════════════════════════════

def add_temperature_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature range features:
      - temp_range_c     : daily max - min (°C)
      - temp_range_7d    : 7-day mean of temp_range
    """
    out = df.copy()

    if "temperature_2m_max" in out.columns and "temperature_2m_min" in out.columns:
        out["temp_range_c"] = (
            out["temperature_2m_max"] - out["temperature_2m_min"]
        ).round(2)

        if "city" in out.columns:
            out["temp_range_7d"] = (
                out.sort_values(["city", "date"])
                   .groupby("city", group_keys=False)["temp_range_c"]
                   .transform(lambda s: s.rolling(7, min_periods=1).mean())
                   .round(2)
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. Heating and Cooling Degree Days
# ══════════════════════════════════════════════════════════════════════════════

def add_degree_days(
    df:        pd.DataFrame,
    base_temp: float = 18.0,
    temp_col:  str   = "temperature_2m_mean",
) -> pd.DataFrame:
    """
    Add heating and cooling degree days.

    HDD = max(0, base_temp - T_mean)   — how much heating is needed
    CDD = max(0, T_mean - base_temp)   — how much cooling is needed

    Base temperature defaults to 18°C (US convention).
    """
    out = df.copy()
    if temp_col not in out.columns:
        logger.warning("'%s' not in DataFrame — skipping degree days", temp_col)
        return out

    t = out[temp_col]
    out["hdd"] = (base_temp - t).clip(lower=0).round(2)
    out["cdd"] = (t - base_temp).clip(lower=0).round(2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. Anomaly scores (deviation from climatological norm per calendar day)
# ══════════════════════════════════════════════════════════════════════════════

def add_anomaly_scores(
    df:        pd.DataFrame,
    columns:   Optional[list[str]] = None,
    group_by:  str                 = "city",
    date_col:  str                 = "date",
) -> pd.DataFrame:
    """
    For each (city × day-of-year), compute the multi-year mean & std, then
    express each observation as a z-score: (value - mean) / std.

    A z-score of 0 means 'perfectly typical for this city on this date'.
    A z-score of +3 means 'much hotter/windier/etc. than normal for this date'.

    Adds columns `<column>_anom` to the DataFrame.
    """
    default_cols = [
        "temperature_2m_mean",
        "wind_speed_10m_max",
        "precipitation_sum",
    ]
    columns = columns or default_cols
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["_doy"] = out[date_col].dt.dayofyear

    for col in columns:
        if col not in out.columns:
            continue
        anom_col = f"{col}_anom"

        # Per-city, per-day-of-year mean and std
        grp = out.groupby([group_by, "_doy"])[col]
        mean_map = grp.transform("mean")
        std_map  = grp.transform("std").replace(0, np.nan)

        out[anom_col] = ((out[col] - mean_map) / std_map).round(3).fillna(0)

    out = out.drop(columns=["_doy"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. Lag features (for prediction)
# ══════════════════════════════════════════════════════════════════════════════

def add_lag_features(
    df:        pd.DataFrame,
    columns:   Optional[list[str]]  = None,
    lags:      Iterable[int]        = (1, 2),
    group_by:  str                  = "city",
    date_col:  str                  = "date",
) -> pd.DataFrame:
    """
    Add lagged versions of key features.

    For each (column × lag), adds column `<column>_lag<N>`.
    Uses per-city groupby so we don't leak data across cities.
    """
    default_cols = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    columns = columns or default_cols
    out = df.sort_values([group_by, date_col]).copy()

    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            out[lag_col] = out.groupby(group_by, group_keys=False)[col].shift(lag)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. Wave height proxy (SMB formula — wraps era5_client)
# ══════════════════════════════════════════════════════════════════════════════

def add_wave_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add estimated `wave_height` column using the SMB fetch-limited formula
    from src.era5_client. Applied per-city using the CASPIAN_FETCH_KM lookup.
    """
    try:
        from src.era5_client import add_wave_proxy_to_dataframe
    except ImportError:
        logger.warning("src.era5_client not importable — skipping wave proxy")
        return df

    out = df.copy()
    pieces = []
    for city, sub in out.groupby("city", group_keys=False):
        pieces.append(add_wave_proxy_to_dataframe(sub, city=city))
    return pd.concat(pieces, ignore_index=True).sort_values(["city", "date"])


# ══════════════════════════════════════════════════════════════════════════════
# 8. Orchestration — build the full feature set
# ══════════════════════════════════════════════════════════════════════════════

def engineer_all_features(
    df:       pd.DataFrame,
    city_col: str = "city",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.

    Order matters:
      1. seasonal    → adds month, day_of_year needed later
      2. temp_range  → uses max & min
      3. degree_days → uses temp_mean
      4. wave_proxy  → uses wind_speed & direction
      5. rolling     → uses base vars
      6. anomaly     → uses base vars + day_of_year
      7. lag         → last so lagged columns don't include other engineered features
    """
    logger.info("Running full feature pipeline...")

    df = add_seasonal_features(df, date_col=date_col)
    logger.info("  ✓ seasonal features")

    df = add_temperature_range(df)
    logger.info("  ✓ temperature range")

    df = add_degree_days(df)
    logger.info("  ✓ heating/cooling degree days")

    df = add_wave_proxy(df)
    logger.info("  ✓ wave height proxy (SMB)")

    df = add_rolling_features(df, group_by=city_col, date_col=date_col)
    logger.info("  ✓ rolling features (7d, 30d)")

    df = add_anomaly_scores(df, group_by=city_col, date_col=date_col)
    logger.info("  ✓ anomaly scores")

    df = add_lag_features(df, group_by=city_col, date_col=date_col)
    logger.info("  ✓ lag features")

    logger.info("Feature pipeline complete — %d columns total", len(df.columns))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. Build analytics layer (DuckDB)
# ══════════════════════════════════════════════════════════════════════════════

def build_analytics_layer(conn) -> dict:
    """
    Read staging.weather_daily, engineer all features, apply risk labels,
    and write to analytics.daily_enriched + analytics.monthly_summary.

    Uses the risk threshold logic from src.risk_labeler so the source of
    truth for "what is a risk day" lives in one place.

    Returns dict with row counts.
    """
    logger.info("=== Building analytics layer ===")

    # Lazy imports
    try:
        from src.risk_labeler import label_risk_days, _DEFAULT_THRESHOLDS
    except ImportError:
        logger.warning("Could not import src.risk_labeler — using local thresholds")
        label_risk_days = None
        _DEFAULT_THRESHOLDS = {
            "wind_speed_10m_max": 50.0,
            "wind_gusts_10m_max": 75.0,
            "precipitation_sum":  15.0,
            "snowfall_sum":        5.0,
            "wave_height":         2.5,
            "visibility_mean":  1000.0,
            "visibility_min":    500.0,
            "visibility_hours_below_1km": 4.0,
        }

    try:
        from src.config import HIGH_RISK_MONTH_THRESHOLD
    except ImportError:
        HIGH_RISK_MONTH_THRESHOLD = 5

    # ── Pull staging data ────────────────────────────────────────────────────
    df = conn.execute("""
        SELECT * FROM staging.weather_daily ORDER BY city, date
    """).fetchdf()
    logger.info("Loaded %d staging rows", len(df))

    # ── Feature engineering ──────────────────────────────────────────────────
    df = engineer_all_features(df)

    # ── Risk labelling ───────────────────────────────────────────────────────
    _BELOW_VARS = {"visibility_mean", "visibility_min"}

    is_risk = pd.Series(False, index=df.index)
    for var, thresh in _DEFAULT_THRESHOLDS.items():
        if var not in df.columns:
            continue
        col = df[var]
        if var in _BELOW_VARS:
            # For BELOW-threshold variables (visibility), NaN means "unknown" —
            # fill with LARGE value so NaN never counts as below-threshold.
            # The visibility_is_known flag separately tells the model which
            # rows had real data.
            is_risk |= (col.fillna(1e9) < thresh)
        else:
            # For ABOVE-threshold variables, NaN → 0 is safe: 0 is never > threshold
            is_risk |= (col.fillna(0) > thresh)
    df["is_risk_day"] = is_risk.astype(int)

    # Individual risk flags for interpretability
    # Same NaN convention: fillna(0) for above-threshold, fillna(1e9) for below
    def _flag_above(c, t): return (df[c].fillna(0)   > t).astype(int) if c in df.columns else 0
    def _flag_below(c, t): return (df[c].fillna(1e9) < t).astype(int) if c in df.columns else 0

    df["risk_wind"]       = _flag_above("wind_speed_10m_max", 50)
    df["risk_gust"]       = _flag_above("wind_gusts_10m_max", 75)
    df["risk_precip"]     = _flag_above("precipitation_sum",  15)
    df["risk_snow"]       = _flag_above("snowfall_sum",        5)
    df["risk_wave"]       = _flag_above("wave_height",        2.5)
    df["risk_visibility"] = _flag_below("visibility_mean",  1000)
    df["risk_fog_min"]    = _flag_below("visibility_min",    500)

    # ── Write daily_enriched ─────────────────────────────────────────────────
    conn.register("_analytics_daily", df)
    conn.execute("""
        CREATE OR REPLACE TABLE analytics.daily_enriched AS
        SELECT * FROM _analytics_daily
    """)
    conn.unregister("_analytics_daily")
    daily_n = conn.execute(
        "SELECT COUNT(*) FROM analytics.daily_enriched"
    ).fetchone()[0]
    logger.info("  Wrote analytics.daily_enriched: %d rows", daily_n)

    # ── Monthly summary ──────────────────────────────────────────────────────
    conn.execute(f"""
        CREATE OR REPLACE TABLE analytics.monthly_summary AS
        SELECT
            city,
            EXTRACT(YEAR  FROM date) AS year,
            EXTRACT(MONTH FROM date) AS month,
            DATE_TRUNC('month', date) AS month_start,

            COUNT(*)                    AS total_days,
            SUM(is_risk_day)            AS risk_days,
            ROUND(SUM(is_risk_day) * 100.0 / COUNT(*), 1) AS risk_day_pct,

            CASE WHEN SUM(is_risk_day) >= {HIGH_RISK_MONTH_THRESHOLD}
                 THEN 1 ELSE 0 END AS high_risk_month,

            SUM(risk_wind)       AS wind_risk_days,
            SUM(risk_gust)       AS gust_risk_days,
            SUM(risk_precip)     AS precip_risk_days,
            SUM(risk_snow)       AS snow_risk_days,
            SUM(risk_wave)       AS wave_risk_days,
            SUM(risk_visibility) AS visibility_risk_days,
            SUM(risk_fog_min)    AS severe_fog_days,

            ROUND(AVG(temperature_2m_mean), 2)  AS temp_mean,
            ROUND(MAX(temperature_2m_max),  2)  AS temp_max,
            ROUND(MIN(temperature_2m_min),  2)  AS temp_min,
            ROUND(STDDEV(temperature_2m_mean), 2) AS temp_std,

            ROUND(AVG(wind_speed_10m_max), 2)   AS wind_mean,
            ROUND(MAX(wind_speed_10m_max), 2)   AS wind_max,
            ROUND(STDDEV(wind_speed_10m_max),2) AS wind_std,

            ROUND(SUM(precipitation_sum), 2)    AS precip_total,
            ROUND(AVG(precipitation_sum), 2)    AS precip_daily_avg,
            ROUND(MAX(precipitation_sum), 2)    AS precip_max,

            ROUND(SUM(hdd), 1) AS hdd_total,
            ROUND(SUM(cdd), 1) AS cdd_total,

            ROUND(AVG(visibility_mean), 1)   AS vis_mean_avg,
            ROUND(MIN(visibility_min),  1)   AS vis_min_worst,

            ROUND(AVG(wave_height), 2) AS wave_mean,
            ROUND(MAX(wave_height), 2) AS wave_max

        FROM analytics.daily_enriched
        GROUP BY city,
                 EXTRACT(YEAR  FROM date),
                 EXTRACT(MONTH FROM date),
                 DATE_TRUNC('month', date)
        ORDER BY city,
                 EXTRACT(YEAR  FROM date),
                 EXTRACT(MONTH FROM date)
    """)
    monthly_n = conn.execute(
        "SELECT COUNT(*) FROM analytics.monthly_summary"
    ).fetchone()[0]
    logger.info("  Wrote analytics.monthly_summary: %d rows", monthly_n)

    return {
        "daily_enriched_rows":  daily_n,
        "monthly_summary_rows": monthly_n,
        "feature_columns":      len(df.columns),
    }
