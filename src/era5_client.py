"""
src/era5_client.py
──────────────────
ERA5 reanalysis data fetcher via the Open-Meteo Marine API.

Why this works without a CDS account
--------------------------------------
Open-Meteo's Marine API (https://marine-api.open-meteo.com) serves ERA5
ocean/wave reanalysis variables for free — no API key required. This covers
wave height, wave period, wind wave height, and swell. For full ERA5
atmospheric reanalysis over land (pressure levels, soil moisture, etc.),
the official Copernicus CDS API is needed (see OPTIONAL section below).

Variables available via Open-Meteo Marine API (Aggregated to Daily)
---------------------------------------------
  wave_height                : Significant wave height (m) [Daily Max]
  wave_direction             : Mean wave direction (°) [Daily Median]
  wave_period                : Mean wave period (s) [Daily Max]
  wind_wave_height           : Wind-generated wave component (m) [Daily Max]
  wind_wave_direction        : Wind wave direction (°) [Daily Median]
  wind_wave_period           : Wind wave period (s) [Daily Max]
  swell_wave_height          : Swell component height (m) [Daily Max]
  swell_wave_direction       : Swell direction (°) [Daily Median]
  swell_wave_period          : Swell period (s) [Daily Max]
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

# ── Optional caching (reuses same pattern as api_client.py) ──────────────────
try:
    import requests_cache
    from retry_requests import retry as _retry
    from pathlib import Path

    _cache_path = Path(__file__).parent.parent / ".weather_cache"
    _cache_session = requests_cache.CachedSession(str(_cache_path), expire_after=3600)
    _SESSION: requests.Session = _retry(_cache_session, retries=5, backoff_factor=0.2)
except ImportError:
    _SESSION = requests.Session()

logger = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────
MARINE_HISTORICAL_URL = "https://marine-api.open-meteo.com/v1/marine"

# ── Variable lists ────────────────────────────────────────────────────────────
MARINE_VARIABLES: list[str] = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "wind_wave_height",
    "wind_wave_direction",
    "wind_wave_period",
    "swell_wave_height",
    "swell_wave_direction",
    "swell_wave_period",
]

# ── Risk threshold ────────────────────────────────────────────────────────────
# Caspian operational threshold: vessels suspend operations above 2.5 m
WAVE_RISK_THRESHOLD: float = 2.5   # metres (significant wave height)


def fetch_marine(
    lat: float,
    lon: float,
    start: str,
    end: str,
    variables: Optional[list[str]] = None,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Fetch hourly marine reanalysis from Open-Meteo Marine API and aggregate to daily.

    Parameters
    ----------
    lat, lon   : Decimal coordinates (must be over water)
    start, end : ISO date strings 'YYYY-MM-DD'
    variables  : Marine variable names (defaults to MARINE_VARIABLES)
    timezone   : 'auto' or IANA string

    Returns
    -------
    pd.DataFrame with DatetimeIndex and daily aggregated marine variables.
    """
    variables = variables or MARINE_VARIABLES
    
    # FIX: Open-Meteo Marine API requires these variables to be queried as 'hourly'
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     ",".join(variables), 
        "timezone":   timezone,
    }

    logger.info(
        "Fetching ERA5 marine hourly and aggregating to daily %s → %s  (%.2f, %.2f)",
        start, end, lat, lon
    )
    
    resp = _SESSION.get(MARINE_HISTORICAL_URL, params=params, timeout=30)
    resp.raise_for_status()

    # FIX: Parse 'hourly' payload instead of 'daily'
    hourly = resp.json().get("hourly", {})
    if not hourly:
        raise ValueError(
            f"Marine API returned no hourly data. "
            f"Check that coordinates (lat={lat}, lon={lon}) are over water."
        )

    # Load into DataFrame and set Datetime Index
    df_hourly = pd.DataFrame(hourly)
    df_hourly["time"] = pd.to_datetime(df_hourly["time"])
    df_hourly.set_index("time", inplace=True)

    # FIX: Aggregate hourly data into daily metrics for your project
    # Heights and Periods use 'max' (worst-case scenario for risk), Directions use 'median'
    agg_rules = {
        "wave_height": "max",
        "wave_direction": "median",
        "wave_period": "max",
        "wind_wave_height": "max",
        "wind_wave_direction": "median",
        "wind_wave_period": "max",
        "swell_wave_height": "max",
        "swell_wave_direction": "median",
        "swell_wave_period": "max",
    }
    
    # Only apply rules for the variables actually requested
    current_agg_rules = {k: v for k, v in agg_rules.items() if k in variables}
    
    # Resample to Daily ('D')
    df_daily = df_hourly.resample('D').agg(current_agg_rules)
    
    return df_daily


def fetch_marine_all_cities(
    start: str,
    end: str,
    offshore_points: Optional[dict[str, dict]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch marine data for offshore proxy points near each Caspian port.

    The offshore_points dict maps city names to {'lat': ..., 'lon': ...}
    for points ~5–10 km offshore (wave models need open-water coordinates).
    Defaults to pre-defined offshore proxies for the 5 project cities.
    """
    # Offshore proxy points (manually shifted ~5 km into the Caspian)
    default_offshore = {
        "Baku":         {"lat": 40.30, "lon": 50.10},  # SE of Baku bay
        "Aktau":        {"lat": 43.55, "lon": 51.05},  # W of Aktau port
        "Anzali":       {"lat": 37.55, "lon": 49.55},  # N of Anzali lagoon
        "Turkmenbashi": {"lat": 40.05, "lon": 53.10},  # W of port
        "Makhachkala":  {"lat": 42.90, "lon": 47.60},  # W of port
    }
    points = offshore_points or default_offshore
    results: dict[str, pd.DataFrame] = {}

    for name, coords in points.items():
        logger.info("── Marine fetch: %s (%.2f, %.2f)", name, coords["lat"], coords["lon"])
        try:
            df = fetch_marine(coords["lat"], coords["lon"], start, end)
            df.insert(0, "city", name)
            results[name] = df
        except Exception as exc:
            logger.warning("Marine fetch failed for %s: %s", name, exc)

    return results