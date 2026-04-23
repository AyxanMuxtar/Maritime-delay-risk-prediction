"""
src/ingestion.py
────────────────
Production-grade ingestion layer for the Caspian Maritime Delay-Risk project.

Responsibilities
----------------
- Fetch daily historical weather data from Open-Meteo archive API
- Fetch daily forecast data from Open-Meteo forecast API
- Fetch marine / wave data from Open-Meteo Marine API (ERA5-backed)
- Retry transient HTTP failures with exponential backoff
- Validate inputs and raise descriptive errors on bad responses
- Persist raw data to CSV (no external dependencies beyond stdlib + pandas)
- Provide a lightweight SQLite-based cache to avoid redundant API calls

Public API
----------
    fetch_historical(city, lat, lon, start, end, variables)  → pd.DataFrame
    fetch_forecast(city, lat, lon, variables)                 → pd.DataFrame
    fetch_marine(city, lat, lon, start, end, variables)       → pd.DataFrame
    fetch_all_cities(cities, start, end, variables)           → dict[str, pd.DataFrame]
    save_raw(df, name, directory)                             → Path
    load_raw(name, directory)                                 → pd.DataFrame
    audit_dataframe(df, city, expected_start, expected_end)   → dict
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Default config (overridden by src/config.py values when passed in) ────────
_HISTORICAL_URL          = "https://archive-api.open-meteo.com/v1/archive"
_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_FORECAST_URL            = "https://api.open-meteo.com/v1/forecast"
_MARINE_URL              = "https://marine-api.open-meteo.com/v1/marine"

# Visibility is only in the historical-forecast-api, from this date onwards.
# Earlier dates must use the fog_proxy feature.
_VISIBILITY_AVAILABLE_FROM = "2022-01-01"

_DEFAULT_TIMEOUT    = 30
_DEFAULT_MAX_RETRY  = 3
_DEFAULT_BACKOFF    = 2      # seconds; actual wait = backoff ** attempt


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _validate_date_range(start: str, end: str) -> None:
    """Raise ValueError for obviously wrong date inputs."""
    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError(
            f"Dates must be ISO format 'YYYY-MM-DD'. Got start='{start}', end='{end}'."
        ) from exc

    if s >= e:
        raise ValueError(f"start_date ({start}) must be before end_date ({end}).")

    if e > date.today():
        raise ValueError(
            f"end_date ({end}) is in the future. "
            "Use fetch_forecast() for future dates."
        )

    earliest_supported = date(2015, 1, 1)
    if s < earliest_supported:
        logger.warning(
            "start_date %s is before %s. "
            "ERA5 reanalysis coverage may be incomplete that far back.",
            start, earliest_supported,
        )


def _http_get(
    url: str,
    params: dict,
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
    backoff: int = _DEFAULT_BACKOFF,
) -> dict:
    """
    HTTP GET with exponential backoff retry.

    Retries on:
      - HTTP 429 (rate limit) — waits for Retry-After header if present,
        otherwise uses a rate-limit-friendly schedule (30s, 60s, 120s)
      - HTTP 5xx (server errors) — normal exponential backoff
      - URLError (network transient failures)

    Raises immediately on:
      - HTTP 4xx (except 429) — bad request, wrong params, etc.
    """
    query = urlencode({k: v for k, v in params.items() if v is not None})
    full_url = f"{url}?{query}"
    last_exc: Exception = RuntimeError("No attempts made.")
    rate_limit_wait: float | None = None   # set when server returns 429

    for attempt in range(max_retries + 1):
        if attempt > 0:
            if rate_limit_wait is not None:
                # Rate-limit schedule: 30s, 60s, 120s, 240s...
                wait = max(rate_limit_wait, 30 * (2 ** (attempt - 1)))
                logger.warning(
                    "Retry %d/%d after %.0fs (rate-limited — Open-Meteo free tier)",
                    attempt, max_retries, wait,
                )
                rate_limit_wait = None   # reset; next iteration uses new header if any
            else:
                wait = backoff ** attempt
                logger.warning(
                    "Retry %d/%d after %.0fs  (%s)",
                    attempt, max_retries, wait, type(last_exc).__name__,
                )
            time.sleep(wait)

        try:
            with urlopen(full_url, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)

        except HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                # Honour Retry-After header if present (seconds or HTTP-date)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        rate_limit_wait = float(retry_after)
                    except ValueError:
                        rate_limit_wait = 60.0   # fallback on HTTP-date format
                else:
                    rate_limit_wait = 30.0       # no header → use sane default
                continue
            if exc.code >= 500:
                continue          # retry on server errors
            raise RuntimeError(
                f"HTTP {exc.code} from Open-Meteo: {exc.reason}\n"
                f"URL: {full_url}"
            ) from exc

        except (URLError, TimeoutError, ConnectionResetError) as exc:
            last_exc = exc
            continue              # retry

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Open-Meteo returned non-JSON response.\nURL: {full_url}"
            ) from exc

    raise RuntimeError(
        f"All {max_retries + 1} attempts failed for {url}.\n"
        f"Last error: {last_exc}\n"
        f"TIP: Open-Meteo's free tier limits ~600 requests/minute, "
        f"~10,000/day. If you hit this repeatedly, increase "
        f"delay_between_cities in fetch_all_cities() or split the fetch "
        f"across multiple days."
    )


def _payload_to_dataframe(payload: dict, city: str) -> pd.DataFrame:
    """
    Convert the 'daily' section of an Open-Meteo JSON response to a DataFrame.

    Columns: date (datetime), city (str), + one column per weather variable.
    """
    daily = payload.get("daily")
    if not daily:
        raise RuntimeError(
            f"Open-Meteo response for '{city}' has no 'daily' section. "
            f"Response keys: {list(payload.keys())}"
        )

    df = pd.DataFrame(daily)

    if "time" not in df.columns:
        raise RuntimeError(
            f"'daily' section for '{city}' has no 'time' column. "
            f"Columns: {list(df.columns)}"
        )

    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "city", city)      # city as the first column

    # Coerce all weather columns to float (API sometimes returns int)
    for col in df.columns:
        if col not in ("date", "city"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Public fetch functions
# ══════════════════════════════════════════════════════════════════════════════

def fetch_historical(
    city:      str,
    lat:       float,
    lon:       float,
    start:     str,
    end:       str,
    variables: list[str],
    timezone:  str = "auto",
    timeout:   int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
) -> pd.DataFrame:
    """
    Fetch daily historical weather from the Open-Meteo archive endpoint.

    Parameters
    ----------
    city       : Human-readable name (used as a column value and in logs)
    lat, lon   : Decimal coordinates
    start, end : ISO date strings 'YYYY-MM-DD'
    variables  : Open-Meteo daily variable names
    timezone   : 'auto' or IANA string, e.g. 'Asia/Baku'
    timeout    : Per-request timeout in seconds
    max_retries: Number of retry attempts on transient failures

    Returns
    -------
    pd.DataFrame  columns: [date, city, <variable>, ...]

    Raises
    ------
    ValueError  : Bad date range or invalid inputs
    RuntimeError: API error or malformed response after all retries
    """
    _validate_date_range(start, end)

    if not variables:
        raise ValueError("variables list must not be empty.")

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(variables),
        "timezone":   timezone,
    }

    logger.info("→ Fetching historical  %-15s  %s → %s  (%d vars)",
                city, start, end, len(variables))

    payload = _http_get(_HISTORICAL_URL, params, timeout, max_retries)
    df = _payload_to_dataframe(payload, city)

    logger.info(
        "  ✓ %-15s  %d rows  %s → %s",
        city, len(df), df["date"].min().date(), df["date"].max().date(),
    )
    return df


def fetch_forecast(
    city:      str,
    lat:       float,
    lon:       float,
    variables: list[str],
    forecast_days: int = 7,
    timezone:  str = "auto",
    timeout:   int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
) -> pd.DataFrame:
    """
    Fetch daily forecast from the Open-Meteo forecast endpoint.

    Parameters
    ----------
    city          : City name
    lat, lon      : Coordinates
    variables     : Forecast variable names (subset of historical vars)
    forecast_days : Number of days ahead (1–16)
    timezone      : 'auto' or IANA string

    Returns
    -------
    pd.DataFrame  columns: [date, city, <variable>, ...]
    """
    if not 1 <= forecast_days <= 16:
        raise ValueError(f"forecast_days must be 1–16, got {forecast_days}.")

    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ",".join(variables),
        "forecast_days": forecast_days,
        "timezone":      timezone,
    }

    logger.info("→ Fetching forecast    %-15s  %d days", city, forecast_days)

    payload = _http_get(_FORECAST_URL, params, timeout, max_retries)
    df = _payload_to_dataframe(payload, city)

    logger.info("  ✓ %-15s  %d rows  %s → %s",
                city, len(df),
                df["date"].min().date(), df["date"].max().date())
    return df


def fetch_marine(
    city:      str,
    lat:       float,
    lon:       float,
    start:     str,
    end:       str,
    variables: list[str],
    timezone:  str = "auto",
    timeout:   int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
) -> pd.DataFrame:
    """
    Fetch daily marine / wave data from Open-Meteo Marine API (ERA5-backed).
    Coordinates must be over water — use offshore proxy points.

    Returns
    -------
    pd.DataFrame  columns: [date, city, wave_height, wave_period, ...]
    """
    _validate_date_range(start, end)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(variables),
        "timezone":   timezone,
    }

    logger.info("→ Fetching marine      %-15s  %s → %s", city, start, end)

    try:
        payload = _http_get(_MARINE_URL, params, timeout, max_retries)
    except RuntimeError as exc:
        # Marine API returns an error for some inland-sea coordinates —
        # log a warning and return an empty DataFrame rather than crashing.
        logger.warning(
            "Marine API failed for %s (%.2f, %.2f): %s. "
            "Returning empty DataFrame — check that coordinates are over water.",
            city, lat, lon, exc,
        )
        return pd.DataFrame(columns=["date", "city"] + variables)

    df = _payload_to_dataframe(payload, city)
    logger.info("  ✓ %-15s  %d rows", city, len(df))
    return df


def fetch_historical_chunked(
    city:      str,
    lat:       float,
    lon:       float,
    start:     str,
    end:       str,
    variables: list[str],
    timezone:  str = "auto",
    chunk_years: int = 2,
    delay_between_chunks: float = 2.0,
    timeout:   int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
) -> pd.DataFrame:
    """
    Fetch a long historical range by splitting into yearly chunks.

    Use this when a single fetch triggers HTTP 429 (rate limiting).
    Smaller requests each return faster and make the rate limiter less angry.

    Parameters
    ----------
    chunk_years          : Split the range into chunks of this many years.
                           Default 2 works well for 6-year windows. Use 1
                           if still getting rate-limited.
    delay_between_chunks : Seconds between chunk fetches. Default 2s.

    Returns
    -------
    Single concatenated DataFrame (same shape as fetch_historical).
    """
    _validate_date_range(start, end)

    start_date = date.fromisoformat(start)
    end_date   = date.fromisoformat(end)

    chunks: list[tuple[str, str]] = []
    cur = start_date
    while cur <= end_date:
        # End of this chunk: Dec 31 of (cur.year + chunk_years - 1)
        chunk_end_year = cur.year + chunk_years - 1
        chunk_end = date(chunk_end_year, 12, 31)
        if chunk_end > end_date:
            chunk_end = end_date
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = date(chunk_end_year + 1, 1, 1)

    logger.info(
        "Chunked fetch for %s: %d chunks (%d-year chunks)",
        city, len(chunks), chunk_years,
    )

    pieces: list[pd.DataFrame] = []
    for i, (cs, ce) in enumerate(chunks):
        logger.info("  Chunk %d/%d: %s → %s", i + 1, len(chunks), cs, ce)
        df = fetch_historical(
            city=city, lat=lat, lon=lon,
            start=cs, end=ce, variables=variables,
            timezone=timezone, timeout=timeout, max_retries=max_retries,
        )
        pieces.append(df)
        if i < len(chunks) - 1 and delay_between_chunks > 0:
            time.sleep(delay_between_chunks)

    return pd.concat(pieces, ignore_index=True).drop_duplicates(["city", "date"])


def fetch_all_cities(
    cities:    dict,            # from src.config.CITIES
    start:     str,
    end:       str,
    variables: list[str],
    timezone_key: str = "timezone",
    timeout:   int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
    delay_between_cities: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical weather for every city in the config dict.

    Parameters
    ----------
    cities               : Dict of {name: {lat, lon, timezone, ...}}
    start/end            : ISO date strings
    variables            : Variable names
    delay_between_cities : Seconds to wait between city fetches. Default 2s
                           keeps us well under Open-Meteo free-tier limits
                           (~600 req/min). Increase to 5–10s if you see
                           HTTP 429 responses.

    Returns
    -------
    dict[city_name → pd.DataFrame]

    Notes
    -----
    Failures for individual cities are caught and logged; the loop continues
    so a single unreachable city doesn't abort the whole run.
    """
    results: dict[str, pd.DataFrame] = {}
    failed:  list[str] = []
    city_list = list(cities.items())

    for i, (name, meta) in enumerate(city_list):
        tz = meta.get(timezone_key, "auto")
        try:
            df = fetch_historical(
                city=name,
                lat=meta["lat"],
                lon=meta["lon"],
                start=start,
                end=end,
                variables=variables,
                timezone=tz,
                timeout=timeout,
                max_retries=max_retries,
            )
            results[name] = df
        except Exception as exc:                         # noqa: BLE001
            logger.error("FAILED %s: %s", name, exc)
            failed.append(name)

        # Pause between cities to stay under rate limits
        if i < len(city_list) - 1 and delay_between_cities > 0:
            time.sleep(delay_between_cities)

    if failed:
        logger.warning("Cities with errors: %s", failed)

    logger.info(
        "fetch_all_cities complete: %d/%d succeeded.",
        len(results), len(cities),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Hourly visibility fetch & aggregation (historical-forecast-api)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_historical_forecast_hourly(
    city:      str,
    lat:       float,
    lon:       float,
    start:     str,
    end:       str,
    variables: list[str],
    timezone:  str = "auto",
    timeout:   int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRY,
) -> pd.DataFrame:
    """
    Fetch HOURLY data from the Open-Meteo historical-forecast endpoint.

    This endpoint archives high-resolution forecast model data from ~2022
    onwards and INCLUDES visibility (which the archive endpoint does not).

    Parameters
    ----------
    city       : City name
    lat, lon   : Decimal coordinates
    start, end : ISO date strings 'YYYY-MM-DD'
                 (must be >= _VISIBILITY_AVAILABLE_FROM for visibility)
    variables  : Hourly variable names, e.g. ['visibility']
    timezone   : 'auto' or IANA string

    Returns
    -------
    pd.DataFrame with columns: [datetime, city, <variable>, ...]
    Note: index is hourly, NOT daily. Use aggregate_hourly_visibility()
    to convert to daily.
    """
    _validate_date_range(start, end)

    if pd.Timestamp(start) < pd.Timestamp(_VISIBILITY_AVAILABLE_FROM):
        logger.warning(
            "Requested start date %s is before %s. "
            "Historical-forecast-api may not have data that far back. "
            "Older periods will use fog_proxy feature instead.",
            start, _VISIBILITY_AVAILABLE_FROM,
        )

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     ",".join(variables),
        "timezone":   timezone,
    }

    logger.info("→ Fetching hourly hist-forecast  %-15s  %s → %s  (%d vars)",
                city, start, end, len(variables))

    payload = _http_get(_HISTORICAL_FORECAST_URL, params, timeout, max_retries)

    hourly = payload.get("hourly")
    if not hourly:
        raise RuntimeError(
            f"historical-forecast-api response for '{city}' has no 'hourly' section. "
            f"Keys present: {list(payload.keys())}"
        )

    df = pd.DataFrame(hourly)
    if "time" not in df.columns:
        raise RuntimeError(
            f"'hourly' section for '{city}' has no 'time' column."
        )

    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.insert(0, "city", city)

    # Coerce numerics
    for col in df.columns:
        if col not in ("datetime", "city"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "  ✓ %-15s  %d hourly rows  %s → %s",
        city, len(df),
        df["datetime"].min(), df["datetime"].max(),
    )
    return df.sort_values("datetime").reset_index(drop=True)


def aggregate_hourly_visibility(
    hourly_df:        pd.DataFrame,
    low_vis_threshold: float = 1000.0,
) -> pd.DataFrame:
    """
    Aggregate hourly visibility to daily statistics.

    Parameters
    ----------
    hourly_df        : DataFrame with 'datetime', 'city', 'visibility' columns
                       (output of fetch_historical_forecast_hourly)
    low_vis_threshold: Metres — count hours below this as fog hours

    Returns
    -------
    pd.DataFrame indexed by (date, city) with columns:
        visibility_mean              — mean daily visibility (m)
        visibility_min               — worst hour of the day (m)
        visibility_hours_below_1km   — count of hours below threshold
    """
    if "visibility" not in hourly_df.columns:
        raise ValueError(
            "hourly_df must contain a 'visibility' column. "
            f"Found: {list(hourly_df.columns)}"
        )

    df = hourly_df.copy()
    df["date"] = df["datetime"].dt.normalize()

    daily = (
        df.groupby(["date", "city"])["visibility"]
          .agg(
              visibility_mean="mean",
              visibility_min="min",
              visibility_hours_below_1km=lambda s: int((s < low_vis_threshold).sum()),
          )
          .reset_index()
    )
    daily["visibility_mean"] = daily["visibility_mean"].round(1)
    daily["visibility_min"]  = daily["visibility_min"].round(1)
    return daily


def merge_visibility_into_daily(
    daily_df:       pd.DataFrame,
    visibility_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-merge aggregated daily visibility columns into the main daily weather
    DataFrame on (date, city). Rows outside visibility coverage will have NaN
    in visibility_* columns — this is expected and handled by fog_proxy in Day 4.

    Parameters
    ----------
    daily_df      : Main daily weather DataFrame (from fetch_historical)
                    Must have 'date' and 'city' columns.
    visibility_df : Aggregated daily visibility (from aggregate_hourly_visibility)

    Returns
    -------
    pd.DataFrame with original columns + visibility_mean, visibility_min,
    visibility_hours_below_1km
    """
    if "date" not in daily_df.columns or "city" not in daily_df.columns:
        raise ValueError("daily_df must have 'date' and 'city' columns")

    # Normalize dates to midnight so merge works
    d1 = daily_df.copy()
    d1["date"] = pd.to_datetime(d1["date"]).dt.normalize()
    d2 = visibility_df.copy()
    d2["date"] = pd.to_datetime(d2["date"]).dt.normalize()

    merged = d1.merge(d2, on=["date", "city"], how="left")

    n_covered = merged["visibility_mean"].notna().sum()
    n_total   = len(merged)
    logger.info(
        "Visibility merge: %d/%d rows have visibility data (%.1f%%)",
        n_covered, n_total, n_covered / n_total * 100 if n_total else 0,
    )
    return merged


def fetch_and_merge_visibility(
    daily_df:     pd.DataFrame,
    cities:       dict,
    timezone_key: str = "timezone",
) -> pd.DataFrame:
    """
    End-to-end convenience wrapper:
      1. Determine date range from daily_df
      2. For dates >= _VISIBILITY_AVAILABLE_FROM, fetch hourly visibility
         from historical-forecast-api for each city
      3. Aggregate to daily
      4. Merge back into daily_df

    Returns
    -------
    pd.DataFrame with visibility_mean, visibility_min,
    visibility_hours_below_1km columns added (NaN where unavailable).
    """
    d = daily_df.copy()
    d["date"] = pd.to_datetime(d["date"])

    earliest = d["date"].min()
    latest   = d["date"].max()
    cutoff   = pd.Timestamp(_VISIBILITY_AVAILABLE_FROM)

    if latest < cutoff:
        logger.warning(
            "All dates (%s → %s) are before visibility cutoff (%s). "
            "Returning daily_df unchanged — use fog_proxy for fog risk.",
            earliest.date(), latest.date(), cutoff.date(),
        )
        return d

    vis_start = max(earliest, cutoff).strftime("%Y-%m-%d")
    vis_end   = latest.strftime("%Y-%m-%d")

    all_hourly: list[pd.DataFrame] = []
    for city, meta in cities.items():
        if city not in d["city"].unique():
            continue
        try:
            h = fetch_historical_forecast_hourly(
                city=city,
                lat=meta["lat"],
                lon=meta["lon"],
                start=vis_start,
                end=vis_end,
                variables=["visibility"],
                timezone=meta.get(timezone_key, "auto"),
            )
            all_hourly.append(h)
        except Exception as exc:                             # noqa: BLE001
            logger.error("Visibility fetch FAILED for %s: %s", city, exc)

    if not all_hourly:
        logger.warning("No visibility data fetched. Returning daily_df unchanged.")
        return d

    hourly_combined = pd.concat(all_hourly, ignore_index=True)
    vis_daily       = aggregate_hourly_visibility(hourly_combined)
    return merge_visibility_into_daily(d, vis_daily)


# ══════════════════════════════════════════════════════════════════════════════
# Persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_raw(
    df:        pd.DataFrame,
    name:      str,
    directory: Path,
    fmt:       str = "csv",
) -> Path:
    """
    Persist a DataFrame to data/raw/ as CSV (default) or Parquet (if pyarrow
    is available).

    Parameters
    ----------
    df        : DataFrame to save
    name      : File stem, e.g. 'baku_historical_2015_2024'
    directory : Target folder (will be created if missing)
    fmt       : 'csv' or 'parquet'

    Returns
    -------
    Path to the saved file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
            out = directory / f"{name}.parquet"
            df.to_parquet(out, index=False)
        except ImportError:
            logger.warning("pyarrow not installed — falling back to CSV.")
            fmt = "csv"

    if fmt == "csv":
        out = directory / f"{name}.csv"
        df.to_csv(out, index=False)

    size_kb = out.stat().st_size / 1024
    logger.info("  💾 Saved %-40s  (%.1f KB)", out.name, size_kb)
    return out


def load_raw(
    name:      str,
    directory: Path,
) -> pd.DataFrame:
    """
    Load a previously saved raw file (CSV or Parquet) back to a DataFrame.
    Automatically parses the 'date' column as datetime.
    """
    directory = Path(directory)
    for suffix in (".parquet", ".csv"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            if suffix == ".parquet":
                import pyarrow  # noqa: F401
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=["date"])
            logger.info("  📂 Loaded %s  (%d rows)", path.name, len(df))
            return df

    raise FileNotFoundError(
        f"No raw file found for '{name}' in {directory}. "
        "Run ingestion first."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data audit
# ══════════════════════════════════════════════════════════════════════════════

def audit_dataframe(
    df:             pd.DataFrame,
    city:           str,
    expected_start: str,
    expected_end:   str,
) -> dict:
    """
    Run a comprehensive QA audit on a fetched DataFrame.

    Checks
    ------
    - Row count vs expected calendar days
    - Date range coverage (actual vs requested)
    - Gap detection (missing days in the sequence)
    - Null / NaN counts per variable
    - Duplicate dates
    - Basic numeric sanity (no temperature < -100°C, etc.)

    Returns
    -------
    dict with keys: city, rows, expected_rows, coverage_pct,
                    actual_start, actual_end, gaps, null_counts,
                    duplicates, warnings
    """
    warnings_list: list[str] = []

    # Expected calendar days
    exp_start = pd.Timestamp(expected_start)
    exp_end   = pd.Timestamp(expected_end)
    expected_days = (exp_end - exp_start).days + 1

    # Actual coverage
    actual_start = df["date"].min()
    actual_end   = df["date"].max()
    actual_days  = len(df)
    coverage_pct = actual_days / expected_days * 100

    if coverage_pct < 95:
        warnings_list.append(
            f"Coverage {coverage_pct:.1f}% < 95% threshold."
        )

    # Gap detection
    all_dates  = pd.date_range(actual_start, actual_end, freq="D")
    present    = set(df["date"].dt.normalize())
    missing    = sorted(all_dates.difference(present))
    gap_count  = len(missing)
    gap_sample = [str(d.date()) for d in missing[:5]]

    if gap_count > 0:
        warnings_list.append(
            f"{gap_count} missing day(s) in sequence. "
            f"First few: {gap_sample}"
        )

    # Null counts — only report columns that actually have nulls
    # Visibility columns are expected to be null before 2022-01-01 (the
    # historical-forecast-api doesn't cover earlier dates). These are NOT
    # treated as warnings.
    _EXPECTED_NULL_COLS_BEFORE_2022 = {
        "visibility_mean", "visibility_min", "visibility_hours_below_1km",
    }
    _VISIBILITY_CUTOFF = pd.Timestamp("2022-01-01")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    null_counts  = df[numeric_cols].isnull().sum().to_dict()
    null_counts_nonzero = {k: v for k, v in null_counts.items() if v > 0}

    # Split nulls into expected vs unexpected
    pre_2022_rows = (df["date"] < _VISIBILITY_CUTOFF).sum()
    expected_nulls:   dict[str, int] = {}
    unexpected_nulls: dict[str, int] = {}

    for col, cnt in null_counts_nonzero.items():
        if col in _EXPECTED_NULL_COLS_BEFORE_2022 and cnt <= pre_2022_rows:
            expected_nulls[col] = cnt
        else:
            unexpected_nulls[col] = cnt

    total_nulls            = sum(null_counts_nonzero.values())
    total_unexpected_nulls = sum(unexpected_nulls.values())

    if total_unexpected_nulls > 0:
        top_nulls = sorted(unexpected_nulls.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(
            f"{col}={cnt} ({cnt/len(df)*100:.1f}%)" for col, cnt in top_nulls
        )
        warnings_list.append(
            f"{total_unexpected_nulls} unexpected null(s) across "
            f"{len(unexpected_nulls)} column(s): {top_str}"
        )

    # Duplicate dates
    dup_count = df["date"].duplicated().sum()
    if dup_count > 0:
        warnings_list.append(f"{dup_count} duplicate date rows.")

    # Sanity checks
    if "temperature_2m_max" in df.columns:
        extreme = (df["temperature_2m_max"] > 60) | (df["temperature_2m_max"] < -80)
        if extreme.any():
            warnings_list.append(
                f"{extreme.sum()} extreme temperature values detected."
            )

    if "wind_speed_10m_max" in df.columns:
        extreme_wind = df["wind_speed_10m_max"] > 250
        if extreme_wind.any():
            warnings_list.append(
                f"{extreme_wind.sum()} wind speed values > 250 km/h detected."
            )

    result = {
        "city":              city,
        "rows":              actual_days,
        "expected_rows":     expected_days,
        "coverage_pct":      round(coverage_pct, 2),
        "actual_start":      str(actual_start.date()),
        "actual_end":        str(actual_end.date()),
        "gap_days":          gap_count,
        "gap_sample":        gap_sample,
        "total_nulls":       total_nulls,
        "unexpected_nulls":  total_unexpected_nulls,
        "expected_nulls":    expected_nulls,      # dict: {col: count} — pre-2022 visibility
        "null_by_column":    null_counts_nonzero, # all columns with > 0 nulls
        "duplicate_dates":   dup_count,
        "warnings":          warnings_list,
        "status":            "⚠️  WARN" if warnings_list else "✅ PASS",
    }
    return result


def audit_all(
    dataframes:     dict[str, pd.DataFrame],
    expected_start: str,
    expected_end:   str,
) -> pd.DataFrame:
    """
    Run audit_dataframe for every city and return a summary DataFrame.
    """
    rows = []
    for city, df in dataframes.items():
        r = audit_dataframe(df, city, expected_start, expected_end)
        rows.append({
            "city":             r["city"],
            "status":           r["status"],
            "rows":             r["rows"],
            "expected_rows":    r["expected_rows"],
            "coverage_%":       r["coverage_pct"],
            "gap_days":         r["gap_days"],
            "total_nulls":      r["total_nulls"],
            "unexpected_nulls": r["unexpected_nulls"],
            "dup_dates":        r["duplicate_dates"],
            "warnings":         "; ".join(r["warnings"]) if r["warnings"] else "",
        })
    return pd.DataFrame(rows).set_index("city")


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight SQLite cache  (avoids re-fetching the same city+date range)
# ══════════════════════════════════════════════════════════════════════════════

class IngestionCache:
    """
    Tracks which (city, start, end, source) combinations have been fetched
    so that incremental runs skip already-downloaded data.

    Stored as a lightweight SQLite database in data/raw/ingestion_cache.db
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                city        TEXT    NOT NULL,
                source      TEXT    NOT NULL,
                start_date  TEXT    NOT NULL,
                end_date    TEXT    NOT NULL,
                rows        INTEGER,
                fetched_at  TEXT    NOT NULL,
                file_path   TEXT,
                PRIMARY KEY (city, source, start_date, end_date)
            )
        """)
        self._conn.commit()

    def is_cached(self, city: str, source: str, start: str, end: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM fetch_log WHERE city=? AND source=? "
            "AND start_date=? AND end_date=?",
            (city, source, start, end),
        )
        return cur.fetchone() is not None

    def record(
        self,
        city: str,
        source: str,
        start: str,
        end: str,
        rows: int,
        file_path: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO fetch_log
               (city, source, start_date, end_date, rows, fetched_at, file_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (city, source, start, end, rows,
             datetime.utcnow().isoformat(), file_path),
        )
        self._conn.commit()

    def summary(self) -> pd.DataFrame:
        cur = self._conn.execute(
            "SELECT city, source, start_date, end_date, rows, fetched_at "
            "FROM fetch_log ORDER BY fetched_at DESC"
        )
        cols = ["city", "source", "start_date", "end_date", "rows", "fetched_at"]
        return pd.DataFrame(cur.fetchall(), columns=cols)

    def close(self) -> None:
        self._conn.close()
