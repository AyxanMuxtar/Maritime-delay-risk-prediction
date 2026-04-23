"""
src/api_client.py
─────────────────
DEPRECATED MODULE — Compatibility shim only.

As of Day 2, this module has been superseded by:
  - src/config.py       → all constants (CITIES, variables, thresholds, etc.)
  - src/ingestion.py    → production HTTP client with retries, caching, auditing
  - src/era5_client.py  → marine forecast + SMB wave proxy
  - src/risk_labeler.py → threshold-based risk-day labelling

This file remains ONLY so that older notebook cells that do:
    from src.api_client import CITIES, RISK_THRESHOLDS
still work. It re-exports from src.config to guarantee there is exactly
one source of truth for these values.

**Do not add new code here. Use src.config for constants and
src.ingestion for API calls.**
"""

from __future__ import annotations

import warnings

# Re-export the canonical constants from src.config
# This guarantees that any caller using this module gets the SAME values
# as every other module in the project — no risk of silent divergence.
from src.config import (
    CITIES,
    ALL_VARIABLES as DAILY_VARIABLES,   # Day 1 called it DAILY_VARIABLES
    RISK_THRESHOLDS,
    HIGH_RISK_MONTH_THRESHOLD,
)

# Re-export the fetch functions from src.ingestion for the same reason.
# If a caller does `from src.api_client import fetch_historical`, route it
# to the production implementation.
from src.ingestion import (
    fetch_historical,
    fetch_forecast,
    fetch_all_cities,
)


# Backward-compatible URL constants (unchanged names from Day 1)
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"


# Issue a one-time deprecation warning on import. The warning does not
# interrupt execution — it just makes the old usage pattern visible.
warnings.warn(
    "src.api_client is deprecated. Import from src.config (constants) "
    "and src.ingestion (functions) directly.",
    DeprecationWarning,
    stacklevel=2,
)
