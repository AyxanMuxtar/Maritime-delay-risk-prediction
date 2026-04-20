"""
src/api_client.py
─────────────────
Open-Meteo API client for the Caspian Maritime Delay-Risk project.

Design decisions
----------------
- Only stdlib + requests + pandas are hard dependencies.
- requests_cache / retry_requests are used if installed, silently skipped otherwise.
- All config constants live here so other modules import from this file,
  never from each other in a circular way.

Usage
-----
    from src.api_client import fetch_historical, fetch_forecast, CITIES
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Optional: caching + retry ─────────────────────────────────────────────────
# Install with:  pip install requests-cache retry-requests
try:
    import requests_cache
    from retry_requests import retry as _retry

    _cache_path = Path(__file__).parent.parent / ".weather_cache"
    _cache_session = requests_cache.CachedSession(str(_cache_path), expire_after=3600)
    SESSION: requests.Session = _retry(_cache_session, retries=5, backoff_factor=0.2)
    _CACHE_ACTIVE = True
except ImportError:
    SESSION = requests.Session()
    _CACHE_ACTIVE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────────────────────
HISTORICAL_URL = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL   = 'https://api.open-meteo.com/v1/forecast'

# ── Cached session ─────────────────────────────────────────────────────────────
_cache_dir = Path(__file__).parent.parent / '.weather_cache'
_cache_session = requests_cache.CachedSession(str(_cache_dir), expire_after=3600)
SESSION = _retry(_cache_session, retries=5, backoff_factor=0.2)

# ── City registry ─────────────────────────────────────────────────────────────
CITIES: dict[str, dict] = {
    'Baku':          {'lat': 40.41, 'lon': 49.87, 'country': 'Azerbaijan'},
    'Aktau':         {'lat': 43.65, 'lon': 51.17, 'country': 'Kazakhstan'},
    'Anzali':        {'lat': 37.47, 'lon': 49.46, 'country': 'Iran'},
    'Turkmenbashi':  {'lat': 40.02, 'lon': 52.97, 'country': 'Turkmenistan'},
    'Makhachkala':   {'lat': 42.98, 'lon': 47.50, 'country': 'Russia'},
}

# ── Default variable list ─────────────────────────────────────────────────────
DAILY_VARIABLES: list[str] = [
    'temperature_2m_max',
    'temperature_2m_min',
    'temperature_2m_mean',
    'wind_speed_10m_max',
    'wind_gusts_10m_max',
    'wind_direction_10m_dominant',
    'precipitation_sum',
    'rain_sum',
    'snowfall_sum',
    'weather_code',
    'relative_humidity_2m_mean',
    'dew_point_2m_mean',
    'apparent_temperature_mean',
    'surface_pressure_mean',
    'visibility_mean',
    'shortwave_radiation_sum',
]

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS: dict[str, float] = {
    'wind_speed_10m_max':  50.0,   # km/h
    'wind_gusts_10m_max':  75.0,   # km/h
    'precipitation_sum':   15.0,   # mm
    'snowfall_sum':         5.0,   # cm
    'visibility_mean':   1000.0,   # metres (below = risk)
}

HIGH_RISK_MONTH_THRESHOLD = 5     # days/month


def fetch_historical(
    lat: float,
    lon: float,
    start: str,
    end: str,
    variables: list[str] | None = None,
    timezone: str = 'auto',
) -> pd.DataFrame:
    """
    Fetch daily historical weather from Open-Meteo archive.

    Parameters
    ----------
    lat, lon   : float — coordinates
    start, end : str   — ISO date strings 'YYYY-MM-DD'
    variables  : list  — daily variables (defaults to DAILY_VARIABLES)
    timezone   : str   — 'auto' or IANA tz string

    Returns
    -------
    pd.DataFrame with DatetimeIndex
    """
    variables = variables or DAILY_VARIABLES
    params = {
        'latitude':   lat,
        'longitude':  lon,
        'start_date': start,
        'end_date':   end,
        'daily':      ','.join(variables),
        'timezone':   timezone,
    }
    logger.info('Fetching historical %s → %s (lat=%.2f, lon=%.2f)', start, end, lat, lon)
    resp = SESSION.get(HISTORICAL_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    df = pd.DataFrame(payload['daily'])
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')


def fetch_forecast(
    lat: float,
    lon: float,
    variables: list[str] | None = None,
    forecast_days: int = 7,
    timezone: str = 'auto',
) -> pd.DataFrame:
    """
    Fetch daily forecast from Open-Meteo forecast API (up to 16 days).
    """
    variables = variables or DAILY_VARIABLES[:10]   # forecast has fewer vars
    params = {
        'latitude':      lat,
        'longitude':     lon,
        'daily':         ','.join(variables),
        'forecast_days': forecast_days,
        'timezone':      timezone,
    }
    logger.info('Fetching %d-day forecast (lat=%.2f, lon=%.2f)', forecast_days, lat, lon)
    resp = SESSION.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    df = pd.DataFrame(payload['daily'])
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')


def fetch_all_cities(
    start: str,
    end: str,
    variables: list[str] | None = None,
    cities: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for all cities. Returns dict of DataFrames.
    """
    cities = cities or CITIES
    results = {}
    for name, meta in cities.items():
        logger.info('Fetching %s ...', name)
        df = fetch_historical(meta['lat'], meta['lon'], start, end, variables)
        df['city'] = name
        results[name] = df
    logger.info('All %d cities fetched.', len(results))
    return results
