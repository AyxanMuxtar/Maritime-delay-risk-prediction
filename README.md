# Changed Files — Marine / Wave Fix

| File | Goes into |
|------|-----------|
| `era5_client.py` | `src/era5_client.py` |
| `day_01_exploration.ipynb` | `notebooks/day_01_exploration.ipynb` |

## What Was Wrong

Two issues caused the `HTTPError: 400 Bad Request`:

1. **Wrong variable names** — we were sending `daily=wave_height,wave_period,...` but the daily endpoint only accepts `_max`-suffixed versions (`wave_height_max`, `wave_period_max`, etc.)
2. **Marine API is forecast-only** — it does not accept `start_date`/`end_date`. It returns a 7-day forecast only.

## What Changed

### `src/era5_client.py` (rewritten)
- `fetch_marine()` → **deprecated** (raises `DeprecationWarning` if called)
- `fetch_marine_forecast()` → new: 7-day live forecast with correct daily variable names
- `estimate_wave_height_from_wind()` → new: SMB fetch-limited wave formula
- `add_wave_proxy_to_dataframe(df, city)` → new: convenience wrapper that adds a `wave_height` column computed from `wind_speed_10m_max` + `wind_direction_10m_dominant`
- `CASPIAN_FETCH_KM` → new: per-city directional fetch lookup (Baku, Aktau, Anzali, Turkmenbashi, Makhachkala × 8 compass directions)
- Wave output capped at 6.0 m (physical ceiling for the Caspian — max ever observed is ~5-6 m)

### `notebooks/day_01_exploration.ipynb` — Section 6
The old 3 broken cells (that called `fetch_marine()`) are replaced with 5 new cells:
1. Explanation of why we use a proxy (Marine API is forecast-only, ERA5 wave requires CDS account)
2. Apply `add_wave_proxy_to_dataframe()` to Baku 2023
3. Visualise wave height with risk threshold
4. Combined wind+wave risk breakdown
5. Optional live 7-day marine forecast (graceful fallback if offline)

## Why a Proxy Instead of Real Data

Free historical wave data for the Caspian Sea is essentially unavailable:
- Open-Meteo Marine API is forecast-only (7 days)
- Copernicus ERA5 wave requires a CDS account and complex cdsapi setup
- Copernicus Marine has patchy inland-sea coverage

The Sverdrup-Munk-Bretschneider (SMB) formula gives ±30% accuracy vs ERA5 — more than sufficient for threshold-based delay-risk labelling at the 2.5 m operational cutoff.

## Optional Next Step

If you want real ERA5 wave data for the full 2015-2024 range, set up the Copernicus CDS API:
1. Register at https://cds.climate.copernicus.eu
2. `pip install cdsapi`
3. Create `~/.cdsapirc` with your key
4. Replace `add_wave_proxy_to_dataframe()` calls with a CDS fetch

The proxy is fine for the 8-day sprint; swap to real ERA5 later if needed.
