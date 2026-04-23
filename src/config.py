"""
src/config.py
─────────────
Single source of truth for all project configuration.

All pipeline stages (ingestion, EDA, feature engineering, modelling)
import from here. Changing a value here propagates everywhere.

Usage
-----
    from src.config import CITIES, VARIABLES, DATE_RANGE, PATHS, API
"""

from __future__ import annotations

from pathlib import Path
from datetime import date

# ── Repository layout ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "repo_root":   _REPO_ROOT,
    "data_raw":    _REPO_ROOT / "data" / "raw",
    "data_proc":   _REPO_ROOT / "data" / "processed",
    "models":      _REPO_ROOT / "models",
    "reports":     _REPO_ROOT / "reports",
    "notebooks":   _REPO_ROOT / "notebooks",
}

# Create directories if missing (safe on repeated import)
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)

# ── API endpoints ─────────────────────────────────────────────────────────────
API = {
    "historical_url":          "https://archive-api.open-meteo.com/v1/archive",
    "historical_forecast_url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "forecast_url":            "https://api.open-meteo.com/v1/forecast",
    "marine_url":              "https://marine-api.open-meteo.com/v1/marine",
    "timeout":        30,        # seconds per request
    "max_retries":    3,         # retry attempts on transient failures
    "backoff_base":   2,         # exponential backoff: 2^attempt seconds
    "forecast_days":  7,
}

# ── Cities ────────────────────────────────────────────────────────────────────
CITIES: dict[str, dict] = {
    "Baku": {
        "lat": 40.41, "lon": 49.87,
        "country": "Azerbaijan",
        "timezone": "Asia/Baku",
        "offshore": {"lat": 40.30, "lon": 50.10},
    },
    "Aktau": {
        "lat": 43.65, "lon": 51.17,
        "country": "Kazakhstan",
        "timezone": "Asia/Aqtau",
        "offshore": {"lat": 43.55, "lon": 51.05},
    },
    "Anzali": {
        "lat": 37.47, "lon": 49.46,
        "country": "Iran",
        "timezone": "Asia/Tehran",
        "offshore": {"lat": 37.55, "lon": 49.55},
    },
    "Turkmenbashi": {
        "lat": 40.02, "lon": 52.97,
        "country": "Turkmenistan",
        "timezone": "Asia/Ashgabat",
        "offshore": {"lat": 40.05, "lon": 53.10},
    },
    "Makhachkala": {
        "lat": 42.98, "lon": 47.50,
        "country": "Russia",
        "timezone": "Europe/Moscow",
        "offshore": {"lat": 42.90, "lon": 47.60},
    },
}

# ── Historical date range ─────────────────────────────────────────────────────
# 10 years of real ERA5 weather data.
# Coverage notes:
#   - All 15 weather variables are real ERA5 reanalysis for the FULL range
#   - Visibility is only available from 2022-01-01 (historical-forecast-api)
#   - Pre-2022 rows carry visibility_* as NaN — handled via missingness flag
#     + median imputation at feature-engineering time (see features.py)
#   - NO synthetic fog_proxy — the model sees a uniform feature schema and
#     learns to weight visibility less when visibility_is_known=0
#
# Sample counts:
#   - ~3,653 days × 5 cities = ~18,265 city-days
#   - ~120 months × 5 cities = ~600 monthly labels for classification
DATE_RANGE = {
    "start": "2015-01-01",
    "end":   "2024-12-31",
}

# Earliest date for which the historical-forecast-api has visibility data.
# Rows before this date have visibility_* = NaN in staging and are flagged
# with visibility_is_known = 0 during feature engineering.
VISIBILITY_AVAILABLE_FROM = "2022-01-01"

# ── Weather variables ─────────────────────────────────────────────────────────
# Daily variables fetched from the Open-Meteo ARCHIVE endpoint (archive-api).
VARIABLES: dict[str, list[str]] = {

    "temperature": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_mean",
    ],

    "wind": [
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
    ],

    "precipitation": [
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
    ],

    "atmosphere": [
        "weather_code",
        "relative_humidity_2m_mean",
        "dew_point_2m_mean",
        "surface_pressure_mean",
        "shortwave_radiation_sum",
    ],
}

# Hourly variables fetched from historical-forecast-api (2022+ coverage),
# then aggregated to daily values and merged into the main DataFrame.
HOURLY_VARIABLES_FOR_AGGREGATION: list[str] = [
    "visibility",   # metres
]

# Columns produced by aggregate_hourly_visibility() — added to main DataFrame
VISIBILITY_DAILY_COLUMNS: list[str] = [
    "visibility_mean",              # mean daily visibility (m)
    "visibility_min",               # worst hour of the day (m)
    "visibility_hours_below_1km",   # count of hours with vis < 1000m
]

# ── Marine / wave variables ───────────────────────────────────────────────────
# Open-Meteo Marine API is forecast-only (7 days). For historical data, the SMB
# proxy in src.era5_client.estimate_wave_height_from_wind() derives wave_height
# from wind. Names below match the Marine API daily endpoint's required suffixes
# ('_max' / '_dominant') and are used by fetch_marine_forecast() for live data.
MARINE_VARIABLES: list[str] = [
    "wave_height_max",
    "wave_direction_dominant",
    "wave_period_max",
    "wind_wave_height_max",
    "swell_wave_height_max",
    "swell_wave_period_max",
]

# Flat list used for API calls
ALL_VARIABLES: list[str] = [v for group in VARIABLES.values() for v in group]

# Subset available on the forecast endpoint
FORECAST_VARIABLES: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "snowfall_sum",
    "weather_code",
    "relative_humidity_2m_mean",
    "visibility_mean",
]

# ── Risk thresholds ───────────────────────────────────────────────────────────
# A day is flagged as a "delay-risk day" if ANY threshold is breached.
# Most variables: ABOVE threshold = risk.
# Variables in _BELOW_THRESHOLD_VARS (risk_labeler): BELOW threshold = risk.
# Columns missing from a given DataFrame are silently skipped by label_risk_days.
RISK_THRESHOLDS: dict[str, float] = {
    "wind_speed_10m_max":          50.0,    # km/h  — Beaufort 10 / storm force
    "wind_gusts_10m_max":          75.0,    # km/h
    "precipitation_sum":           15.0,    # mm/day
    "snowfall_sum":                 5.0,    # cm/day
    "wave_height":                  2.5,    # metres  (SMB proxy from wind)
    "visibility_mean":          1000.0,     # metres — BELOW this = fog risk
    "visibility_min":            500.0,     # metres — BELOW this = severe fog
    "visibility_hours_below_1km":   4.0,    # count  — ABOVE this = sustained fog
}

# Months with >= this many risk days are labelled 1 (high-risk month)
HIGH_RISK_MONTH_THRESHOLD: int = 5

# ── Outlier handling (Day 4) ─────────────────────────────────────────────────
# Columns where outliers are flagged (NOT removed) using IQR method.
# Removal is a modelling-time decision, not a cleaning-time one.
OUTLIER_FLAG_COLUMNS: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "surface_pressure_mean",
]

# Per-city 99th-percentile caps for heavy-tailed variables.
# Winsorizing here prevents one extreme Anzali rainstorm from dominating
# cross-city feature distributions. Applied during staging → analytics.
# None = don't cap. Values computed empirically from Day 2 data audit.
WINSORIZE_CAPS: dict[str, dict[str, float]] = {
    "precipitation_sum": {
        # Anzali's 99th percentile is ~4x other cities — cap it to reduce skew
        "Anzali":       60.0,
        "Baku":         30.0,
        "Aktau":        25.0,
        "Turkmenbashi": 25.0,
        "Makhachkala":  40.0,
    },
    "snowfall_sum": {
        # All cities — cap at physically reasonable daily snowfall
        "Anzali":       10.0,
        "Baku":         15.0,
        "Aktau":        20.0,
        "Turkmenbashi": 10.0,
        "Makhachkala":  30.0,
    },
}

# ── Data schema ───────────────────────────────────────────────────────────────
EXPECTED_DTYPES: dict[str, str] = {
    "date":                          "datetime64[ns]",
    "city":                          "object",
    "temperature_2m_max":            "float64",
    "temperature_2m_min":            "float64",
    "temperature_2m_mean":           "float64",
    "apparent_temperature_mean":     "float64",
    "wind_speed_10m_max":            "float64",
    "wind_gusts_10m_max":            "float64",
    "wind_direction_10m_dominant":   "float64",
    "precipitation_sum":             "float64",
    "rain_sum":                      "float64",
    "snowfall_sum":                  "float64",
    "weather_code":                  "float64",
    "relative_humidity_2m_mean":     "float64",
    "dew_point_2m_mean":             "float64",
    "surface_pressure_mean":         "float64",
    "visibility_mean":               "float64",
    "shortwave_radiation_sum":       "float64",
}
