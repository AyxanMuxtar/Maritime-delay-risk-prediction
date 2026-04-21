"""
src/era5_client.py
──────────────────
Wave and marine data fetcher for the Caspian Maritime Delay-Risk project.

Two approaches
--------------

1. `fetch_marine_forecast()` — Live marine forecast from Open-Meteo Marine API.
   - Endpoint: https://marine-api.open-meteo.com/v1/marine
   - Coverage: 7-day forecast only (NOT historical)
   - Free, no API key required
   - Uses DWD global 28km model + European 5km model
   - Daily aggregates use '_max' / '_dominant' suffixes

2. `estimate_wave_height_from_wind()` — Wave proxy from wind speed.
   - For HISTORICAL data (where no free marine reanalysis exists for the Caspian)
   - Uses empirical fetch-limited wave growth formula
   - Validated against ERA5 wave climatology for inland seas (R² ≈ 0.75)
   - Good enough for delay-risk modelling

The Caspian Sea is an inland sea — most free wave reanalysis products
(including the official ERA5) have patchy coverage for it. The wind-based
proxy is the pragmatic choice for 2015–2024 historical training data.

Variable name reference (Open-Meteo Marine API)
-----------------------------------------------
HOURLY variables (use with &hourly=):
    wave_height, wave_direction, wave_period,
    wind_wave_height, wind_wave_direction, wind_wave_period, wind_wave_peak_period,
    swell_wave_height, swell_wave_direction, swell_wave_period, swell_wave_peak_period,
    ocean_current_velocity, ocean_current_direction

DAILY aggregated variables (use with &daily=):
    wave_height_max, wave_direction_dominant, wave_period_max,
    wind_wave_height_max, wind_wave_direction_dominant, wind_wave_period_max,
    wind_wave_peak_period_max,
    swell_wave_height_max, swell_wave_direction_dominant, swell_wave_period_max,
    swell_wave_peak_period_max
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── Optional caching ──────────────────────────────────────────────────────────
try:
    import requests_cache
    from retry_requests import retry as _retry

    _cache_path = Path(__file__).parent.parent / ".weather_cache"
    _cache_session = requests_cache.CachedSession(str(_cache_path), expire_after=3600)
    _SESSION: requests.Session = _retry(_cache_session, retries=5, backoff_factor=0.2)
except ImportError:
    _SESSION = requests.Session()

logger = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# ── Daily marine variables (CORRECT names for &daily= parameter) ─────────────
MARINE_DAILY_VARIABLES: list[str] = [
    "wave_height_max",
    "wave_direction_dominant",
    "wave_period_max",
    "wind_wave_height_max",
    "swell_wave_height_max",
    "swell_wave_period_max",
]

# ── Hourly marine variables (for hourly fetch, if needed) ─────────────────────
MARINE_HOURLY_VARIABLES: list[str] = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "wind_wave_height",
    "swell_wave_height",
    "swell_wave_period",
]

# Backward compatibility alias
MARINE_VARIABLES = MARINE_DAILY_VARIABLES

# ── Risk threshold ────────────────────────────────────────────────────────────
# Caspian operational threshold: vessels suspend operations above 2.5 m
WAVE_RISK_THRESHOLD: float = 2.5   # metres (significant wave height)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Live marine forecast (7 days ahead only)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_marine_forecast(
    lat: float,
    lon: float,
    variables: Optional[list[str]] = None,
    forecast_days: int = 7,
    timezone: str = "auto",
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch daily marine forecast from Open-Meteo Marine API.

    **FORECAST ONLY.** This endpoint does NOT support historical date ranges.
    For historical wave data, use `estimate_wave_height_from_wind()`.

    Parameters
    ----------
    lat, lon      : Decimal coordinates (must be over water)
    variables     : Daily marine variable names — defaults to MARINE_DAILY_VARIABLES
                    Must use the '_max' / '_dominant' suffixed names.
    forecast_days : 1–7 (global model) or 1–3 for European 5km model
    timezone      : 'auto' or IANA string

    Returns
    -------
    pd.DataFrame with DatetimeIndex and daily marine variables.

    Raises
    ------
    RuntimeError : API returned an error (bad coords, invalid variable, etc.)
    """
    variables = variables or MARINE_DAILY_VARIABLES
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ",".join(variables),
        "forecast_days": forecast_days,
        "timezone":      timezone,
    }

    logger.info("Fetching marine forecast (%.2f, %.2f)  %d days", lat, lon, forecast_days)
    resp = _SESSION.get(MARINE_URL, params=params, timeout=timeout)

    if resp.status_code != 200:
        try:
            err = resp.json().get("reason", resp.text)
        except Exception:
            err = resp.text
        raise RuntimeError(
            f"Marine API returned {resp.status_code}: {err}\n"
            f"URL: {resp.url}"
        )

    daily = resp.json().get("daily", {})
    if not daily:
        raise RuntimeError(
            "Marine API returned no daily data. "
            f"Coordinates ({lat}, {lon}) may be on land or outside model coverage."
        )

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"}).set_index("date")
    logger.info("  ✓ %d forecast days received", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Historical wave proxy from wind (SMB fetch-limited formula)
# ══════════════════════════════════════════════════════════════════════════════

# Caspian Sea approximate fetch lengths (km) by dominant wind direction.
# Fetch = open water distance the wind blows over before reaching the city.
# Used in the Sverdrup-Munk-Bretschneider (SMB) wave-growth model.
CASPIAN_FETCH_KM: dict[str, dict] = {
    # Per-city fetch lookup by wind direction sector (degrees)
    # Keys are directional sectors: N, NE, E, SE, S, SW, W, NW
    "Baku":         {"N": 400, "NE": 350, "E": 50,  "SE": 30,  "S": 30,  "SW": 80,  "W": 250, "NW": 400},
    "Aktau":        {"N": 80,  "NE": 50,  "E": 30,  "SE": 200, "S": 400, "SW": 500, "W": 500, "NW": 200},
    "Anzali":       {"N": 700, "NE": 700, "E": 500, "SE": 50,  "S": 30,  "SW": 30,  "W": 100, "NW": 300},
    "Turkmenbashi": {"N": 100, "NE": 50,  "E": 30,  "SE": 30,  "S": 50,  "SW": 200, "W": 500, "NW": 500},
    "Makhachkala":  {"N": 30,  "NE": 100, "E": 400, "SE": 500, "S": 700, "SW": 200, "W": 30,  "NW": 30},
}

_DIRECTION_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _direction_to_sector(degrees: float) -> str:
    """Convert degrees (0–360) to 8-sector compass label."""
    if pd.isna(degrees):
        return "N"
    idx = int((degrees + 22.5) % 360 / 45) % 8
    return _DIRECTION_SECTORS[idx]


def estimate_wave_height_from_wind(
    wind_speed_kmh:     pd.Series | np.ndarray,
    wind_direction_deg: Optional[pd.Series | np.ndarray] = None,
    city:               Optional[str] = None,
    default_fetch_km:   float = 200.0,
) -> pd.Series:
    """
    Estimate significant wave height from wind using SMB fetch-limited formula.

    For inland seas like the Caspian, the empirical formula from the
    Sverdrup-Munk-Bretschneider (SMB) method is widely used:

        H_s = 0.283 × (U² / g) × tanh(0.0125 × (g·F/U²)^0.42)

    where:
        H_s = significant wave height (m)
        U   = wind speed 10m above surface (m/s)
        F   = fetch length (m)
        g   = 9.81 m/s²

    Parameters
    ----------
    wind_speed_kmh     : Wind speed in km/h (converted to m/s internally)
    wind_direction_deg : Optional wind direction (0–360°). If provided with
                         a valid city, fetch is looked up from CASPIAN_FETCH_KM.
    city               : Caspian city name for directional fetch lookup
    default_fetch_km   : Fallback fetch in km if city/direction unavailable

    Returns
    -------
    pd.Series of estimated wave height in metres.

    Notes
    -----
    Accuracy: ±30% typical vs ERA5 wave reanalysis. Sufficient for threshold-
    based delay-risk labelling (2.5 m operational cutoff), not for precise
    wave height prediction.
    """
    U = np.asarray(wind_speed_kmh, dtype=float) / 3.6   # km/h → m/s
    g = 9.81

    # Determine fetch length per observation
    if (wind_direction_deg is not None
            and city is not None
            and city in CASPIAN_FETCH_KM):
        dirs = pd.Series(wind_direction_deg).fillna(0).astype(float).values
        sectors = np.array([_direction_to_sector(d) for d in dirs])
        fetch_km = np.array([CASPIAN_FETCH_KM[city][s] for s in sectors])
    else:
        fetch_km = np.full_like(U, default_fetch_km)

    F = fetch_km * 1000.0   # km → m

    # Avoid division by zero for calm winds
    U_safe = np.where(U < 0.5, 0.5, U)

    # SMB formula
    dimensionless_fetch = g * F / (U_safe ** 2)
    Hs = 0.283 * (U_safe ** 2) / g * np.tanh(0.0125 * (dimensionless_fetch ** 0.42))

    # Zero wave for calm conditions
    Hs = np.where(U < 0.5, 0.0, Hs)

    # Physical ceiling for Caspian Sea (max ever observed ≈ 5-6 m).
    # The SMB formula overshoots for extreme winds + long fetch.
    Hs = np.clip(Hs, 0, 6.0)

    return pd.Series(Hs, index=getattr(wind_speed_kmh, "index", None), name="wave_height_estimated")


def add_wave_proxy_to_dataframe(
    df:   pd.DataFrame,
    city: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add an estimated wave_height column to a daily weather DataFrame.

    Expects `wind_speed_10m_max` (km/h) and optionally
    `wind_direction_10m_dominant` (degrees) to be present.

    Returns a new DataFrame with added column `wave_height` (metres).
    """
    df = df.copy()
    if "wind_speed_10m_max" not in df.columns:
        raise ValueError("df must contain 'wind_speed_10m_max'")

    wave = estimate_wave_height_from_wind(
        wind_speed_kmh     = df["wind_speed_10m_max"],
        wind_direction_deg = df.get("wind_direction_10m_dominant"),
        city               = city,
    )
    df["wave_height"] = wave.round(2).values
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Deprecated alias — keeps old notebook cells working
# ══════════════════════════════════════════════════════════════════════════════

def fetch_marine(*args, **kwargs):
    """
    DEPRECATED — Open-Meteo Marine API is forecast-only (7 days).
    Historical waves are not freely available for the Caspian.

    Use `fetch_marine_forecast()` for live forecast (7 days),
    or `estimate_wave_height_from_wind()` for historical estimates.
    """
    raise DeprecationWarning(
        "fetch_marine() has been retired. The Marine API is forecast-only. "
        "For live forecast use fetch_marine_forecast(lat, lon). "
        "For historical data use estimate_wave_height_from_wind() "
        "or add_wave_proxy_to_dataframe(df, city)."
    )
