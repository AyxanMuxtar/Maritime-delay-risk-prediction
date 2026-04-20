"""
src/ais_client.py
─────────────────
AIS (Automatic Identification System) vessel traffic data loader.

Data sources & access
---------------------
1. **MarineTraffic Public Statistics** (free, aggregated)
   Monthly vessel counts and port call statistics per port.
   URL: https://www.marinetraffic.com/en/ais/home/centerx:49.9/centery:40.4/zoom:10
   → Export available for registered users (free tier).

2. **AISHub** (free for research, requires registration)
   Real-time and historical AIS position data.
   URL: https://www.aishub.net/api

3. **VesselFinder History API** (freemium)
   URL: https://api.vesselfinder.com/

4. **UN Global Platform — AIS data** (free for research)
   Aggregated trade statistics derived from AIS signals.

5. **CSV proxy approach** (what this module implements by default)
   Since full AIS APIs require paid keys, we use the following strategy:
   a. Load any CSV/Parquet AIS export the user has (e.g. from MarineTraffic)
   b. Fall back to synthetic proxy data for development/testing

Vessel traffic as a label proxy
---------------------------------
When AIS density (vessel count or transit count per day/month) drops
sharply relative to a rolling baseline, it is a strong signal that
maritime operations were disrupted. We use:

    traffic_drop_flag = (daily_count < 0.5 × 30-day rolling mean)

This can serve as an *additional* label signal alongside threshold-based
risk days, or be used to calibrate the HIGH_RISK_MONTH_THRESHOLD.

Usage
-----
    from src.ais_client import load_ais_csv, compute_traffic_proxy, AIS_PORTS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Port bounding boxes for AIS filtering ─────────────────────────────────────
# (lat_min, lat_max, lon_min, lon_max)
AIS_PORTS: dict[str, dict] = {
    "Baku":         {"bbox": (40.30, 40.55, 49.75, 50.10), "country": "Azerbaijan"},
    "Aktau":        {"bbox": (43.55, 43.80, 51.00, 51.35), "country": "Kazakhstan"},
    "Anzali":       {"bbox": (37.40, 37.60, 49.35, 49.60), "country": "Iran"},
    "Turkmenbashi": {"bbox": (39.90, 40.15, 52.85, 53.15), "country": "Turkmenistan"},
    "Makhachkala":  {"bbox": (42.85, 43.10, 47.40, 47.70), "country": "Russia"},
}


def load_ais_csv(
    filepath: str | Path,
    port: Optional[str] = None,
    date_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Load an AIS CSV export and return a clean DataFrame.

    Parameters
    ----------
    filepath  : Path to CSV or Parquet file
    port      : If provided, filter rows to this port's bounding box
    date_col  : Column name containing timestamp
    lat_col   : Latitude column name
    lon_col   : Longitude column name

    Returns
    -------
    pd.DataFrame with DatetimeIndex, filtered to port bbox if requested.

    Expected CSV columns (flexible — adjust to your export):
        timestamp, mmsi, vessel_type, lat, lon, speed, course
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"AIS file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Parse datetime
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    # Spatial filter
    if port is not None:
        meta = AIS_PORTS.get(port)
        if meta is None:
            raise ValueError(f"Unknown port '{port}'. Available: {list(AIS_PORTS)}")
        lat_min, lat_max, lon_min, lon_max = meta["bbox"]
        df = df[
            (df[lat_col] >= lat_min) & (df[lat_col] <= lat_max) &
            (df[lon_col] >= lon_min) & (df[lon_col] <= lon_max)
        ]
        logger.info("Filtered to %s bbox: %d AIS records.", port, len(df))

    return df


def compute_traffic_proxy(
    df: pd.DataFrame,
    freq: str = "D",
    rolling_window: int = 30,
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute daily/monthly vessel count and flag traffic-drop days.

    Parameters
    ----------
    df              : AIS DataFrame (DatetimeIndex)
    freq            : Resample frequency — 'D' (daily) or 'ME' (monthly)
    rolling_window  : Days for rolling baseline
    drop_threshold  : Fraction below rolling mean to flag as disruption

    Returns
    -------
    pd.DataFrame with columns:
        vessel_count, rolling_mean, drop_ratio, traffic_disruption (0/1)
    """
    counts = df.resample(freq).size().rename("vessel_count")
    result = counts.to_frame()

    result["rolling_mean"] = (
        result["vessel_count"]
        .rolling(rolling_window, min_periods=7)
        .mean()
    )
    result["drop_ratio"] = result["vessel_count"] / result["rolling_mean"].replace(0, np.nan)
    result["traffic_disruption"] = (result["drop_ratio"] < drop_threshold).astype(int)

    return result


def generate_synthetic_ais(
    start: str,
    end: str,
    city: str,
    base_daily_count: int = 45,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic AIS vessel count data for development & testing.

    Simulates realistic patterns:
    - Weekly seasonality (fewer vessels on weekends)
    - Winter suppression (Nov–Mar: 30% fewer vessels)
    - Random disruption events (matching typical Caspian storm frequency)
    - Gaussian noise

    Parameters
    ----------
    start, end          : ISO date strings
    city                : City name (for reproducible seed offset)
    base_daily_count    : Average vessels per day in calm summer conditions
    seed                : Random seed

    Returns
    -------
    pd.DataFrame (DatetimeIndex) with vessel_count column.

    WARNING: This is SYNTHETIC DATA — for development only.
    Replace with real AIS data before any production use.
    """
    city_seed_offset = sum(ord(c) for c in city)
    rng = np.random.default_rng(seed + city_seed_offset)

    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    # Base signal
    counts = np.full(n, float(base_daily_count))

    # Winter suppression (month 11–3)
    months = pd.DatetimeIndex(dates).month
    winter_mask = np.isin(months, [11, 12, 1, 2, 3])
    counts[winter_mask] *= 0.70

    # Weekend dip
    weekdays = pd.DatetimeIndex(dates).dayofweek
    counts[weekdays >= 5] *= 0.85

    # Gaussian noise
    counts += rng.normal(0, 5, n)

    # Simulate ~4 disruption events/year (multi-day storms)
    n_events = max(1, int(n / 365 * 4))
    for _ in range(n_events):
        start_idx = rng.integers(0, n - 5)
        duration  = rng.integers(2, 7)
        end_idx   = min(start_idx + duration, n)
        counts[start_idx:end_idx] *= rng.uniform(0.1, 0.4)

    counts = np.clip(counts, 0, None).round().astype(int)

    df = pd.DataFrame({"vessel_count": counts}, index=dates)
    df.index.name = "date"
    logger.warning(
        "Returning SYNTHETIC AIS data for %s (%s → %s). "
        "Replace with real MarineTraffic or AISHub data before training.",
        city, start, end
    )
    return df


def monthly_traffic_features(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily vessel count data to monthly features.

    Returns
    -------
    pd.DataFrame with monthly index and columns:
        vessel_count_mean, vessel_count_min, vessel_count_sum,
        disruption_days, disruption_day_pct
    """
    monthly = traffic_df["vessel_count"].resample("ME").agg(
        vessel_count_mean="mean",
        vessel_count_min="min",
        vessel_count_sum="sum",
    ).round(1)

    if "traffic_disruption" in traffic_df.columns:
        monthly["disruption_days"] = (
            traffic_df["traffic_disruption"].resample("ME").sum()
        )
        monthly["disruption_day_pct"] = (
            monthly["disruption_days"]
            / traffic_df["traffic_disruption"].resample("ME").count()
            * 100
        ).round(1)

    return monthly
