"""
src/risk_labeler.py
───────────────────
Delay-risk day flagging and monthly label generation.

No imports from other src modules — all thresholds are either passed
as arguments (default: None → uses module-level defaults below) or
imported explicitly by the caller from src.api_client.

Usage
-----
    from src.risk_labeler import label_risk_days, monthly_features
    # optionally override thresholds:
    from src.api_client import RISK_THRESHOLDS
    daily_risk = label_risk_days(df, thresholds=RISK_THRESHOLDS)
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

# ── Default thresholds (mirrors src/config.py — keep in sync) ────────────────
# Missing columns in a given DataFrame are silently skipped.
# This is intentional: visibility_* columns only exist for dates >= 2022-01-01
# (see fetch_historical_forecast_hourly + merge_visibility_into_daily in
# src/ingestion.py). For earlier dates, fog_proxy_flag (Day 4 feature) is used.
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "wind_speed_10m_max":         50.0,    # km/h  — Beaufort 10 / storm force
    "wind_gusts_10m_max":         75.0,    # km/h
    "precipitation_sum":          15.0,    # mm/day
    "snowfall_sum":                5.0,    # cm/day
    "wave_height":                 2.5,    # metres  (ERA5 marine)
    "visibility_mean":          1000.0,    # metres — BELOW = fog risk
    "visibility_min":            500.0,    # metres — BELOW = severe fog
    "visibility_hours_below_1km":   4.0,   # count  — ABOVE = sustained fog
}

_DEFAULT_HIGH_RISK_DAYS: int = 5   # days/month → label = 1

# Variables where LOWER value = higher risk (all others: ABOVE = risk)
_BELOW_THRESHOLD_VARS: set[str] = {
    "visibility_mean",
    "visibility_min",
}


def label_risk_days(
    df: pd.DataFrame,
    thresholds: Optional[dict[str, float]] = None,
) -> pd.Series:
    """
    Return a binary int Series (0 / 1) flagging each day as a delay-risk day.

    A day is flagged if ANY single threshold is breached. Columns not present
    in ``df`` are silently skipped.

    Parameters
    ----------
    df         : Daily weather DataFrame (DatetimeIndex)
    thresholds : {variable_name: threshold_value} mapping.
                 Defaults to _DEFAULT_THRESHOLDS.

    Returns
    -------
    pd.Series[int]  named 'is_risk_day', same index as df
    """
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    risk = pd.Series(False, index=df.index)

    for var, threshold in thresholds.items():
        if var not in df.columns:
            continue
        col = df[var].fillna(0)   # treat NaN as no-risk for now
        if var in _BELOW_THRESHOLD_VARS:
            risk |= col < threshold
        else:
            risk |= col > threshold

    return risk.astype(int).rename("is_risk_day")


def monthly_features(
    df: pd.DataFrame,
    thresholds: Optional[dict[str, float]] = None,
    min_risk_days: int = _DEFAULT_HIGH_RISK_DAYS,
) -> pd.DataFrame:
    """
    Aggregate daily weather into monthly features + binary target label.

    Parameters
    ----------
    df            : Daily weather DataFrame (DatetimeIndex)
    thresholds    : Risk thresholds (defaults to _DEFAULT_THRESHOLDS)
    min_risk_days : Months with >= this many risk days are labelled 1

    Returns
    -------
    pd.DataFrame indexed by month-end (ME), columns:
        risk_days, total_days, risk_day_pct,
        <var>_mean, <var>_max, <var>_min, <var>_std  (for numeric cols),
        high_risk_month  (0 / 1)  ← TARGET LABEL
    """
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    daily_risk = label_risk_days(df, thresholds)

    # Monthly aggregates for all numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_dict = {col: ["mean", "max", "min", "std"] for col in numeric_cols}
    monthly = df[numeric_cols].resample("ME").agg(agg_dict)
    monthly.columns = ["_".join(c) for c in monthly.columns]

    # Risk statistics
    monthly["risk_days"]       = daily_risk.resample("ME").sum()
    monthly["total_days"]      = daily_risk.resample("ME").count()
    monthly["risk_day_pct"]    = (
        monthly["risk_days"] / monthly["total_days"] * 100
    ).round(2)

    # Binary target
    monthly["high_risk_month"] = (monthly["risk_days"] >= min_risk_days).astype(int)

    return monthly


def risk_day_breakdown(
    df: pd.DataFrame,
    thresholds: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Return a per-day DataFrame showing which variable(s) triggered the risk flag.
    Useful for exploratory analysis and threshold calibration.

    Returns
    -------
    pd.DataFrame with one boolean column per threshold variable,
    plus an 'any_risk' column.
    """
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    result = pd.DataFrame(index=df.index)

    for var, threshold in thresholds.items():
        if var not in df.columns:
            result[var + "_breach"] = False
            continue
        col = df[var].fillna(0)
        if var in _BELOW_THRESHOLD_VARS:
            result[var + "_breach"] = col < threshold
        else:
            result[var + "_breach"] = col > threshold

    result["any_risk"] = result.any(axis=1)
    return result.astype(int)
