# Caspian Maritime Delay-Risk Forecasting

> A monthly classifier that predicts whether the next calendar month will be a
> **high delay-risk month** for maritime operations across five Caspian Sea
> ports, using historical weather, fog, and wind-derived wave data.

---

## 1. Problem Statement

Maritime operations across the Caspian Sea are routinely disrupted by
storm-force winds, heavy precipitation, dense fog, and rough seas. Vessel
operators, port authorities, and cargo planners need advance notice of
*risky months* so they can:

- adjust scheduling and crew rotations
- buy insurance hedges
- pre-position equipment
- communicate realistic delivery windows to customers
| Parameter          | Value                          |
| ------------------ | ------------------------------ |
| Start date         | 2015-01-01                     |
| End date           | 2024-12-31                     |
| Total span         | 10 years (3,653 days per city) |
| Cities             | 5 Caspian coastal ports        |
| Total observations | ~18,265 city-days              |

Today, this is done informally based on operator experience and short-range
weather forecasts. There is no public, port-level, monthly probability of
weather-driven disruption for the Caspian. **This project builds one.**

The model answers a single question, asked once per month:

> *"For each of our five ports, what is the probability that each day of next month will
> have weather conditions severe enough to disrupt
> normal maritime operations?"*
| Source                             | Endpoint                                 | Coverage               | What it provides                                   |
| ---------------------------------- | ---------------------------------------- | ---------------------- | -------------------------------------------------- |
| Open-Meteo Archive API             | `archive-api.open-meteo.com`             | 2015–2024 (full range) | 15 daily weather variables                         |
| Open-Meteo Historical Forecast API | `historical-forecast-api.open-meteo.com` | 2022–2024 only         | Hourly visibility → aggregated to 3 daily features |
| SMB Wave Proxy                     | Derived from wind variables              | 2015–2024 (full range) | Estimated significant wave height                  |

## Why It Matters

| Stakeholder | Decision the prediction informs |
|-------------|--------------------------------|
| **Port operations** | Staff scheduling, equipment readiness, suspension policies |
| **Cargo planners** | Buffer days in delivery quotes, route alternatives through lower-risk ports |
| **Vessel operators** | Crew rotations, voyage timing, fuel planning for diversions |
| **Insurance underwriters** | Premium calibration, claim-rate forecasting |
| **Government / coastguard** | Search-and-rescue readiness, environmental incident pre-positioning |
### Visibility Feature Consistency

Visibility data is only directly available from 2022 onwards via the Open-Meteo historical-forecast API. Earlier versions of the dataset used a fog proxy (based on humidity and dew point) for 2015–2021.

To ensure feature consistency across the full time range, the synthetic fog proxy will be removed during **Day 4 (Data Cleaning)**. As a result, visibility-related features will only be used where real measurements exist.

This avoids introducing artificial patterns into the model caused by mixing proxy-derived and real observations.

---

The Caspian is a closed inland sea bordered by 5 countries (Azerbaijan,
Kazakhstan, Iran, Turkmenistan, Russia). Its weather and wave climate is
poorly served by global ocean reanalysis products. A purpose-built
classifier — even one based on threshold heuristics — fills a real gap.


## 2: TEAM ANEMOI
### Roles:
| Level                 | Detail                                                        |
| --------------------- | ------------------------------------------------------------- |
| Raw ingestion         | Daily (from Open-Meteo archive API)                           |
| Visibility enrichment | Hourly (from historical-forecast-api), aggregated to daily    |
| Feature engineering   | Daily → monthly aggregates (mean, max, min, std, percentiles) |
| Target variable       | Monthly (one label per city per calendar month)               |

### 1. Məhəmməd Sadıqov, Adil Həsənov - Data Engineer

**Focus:** Getting the data, storing it, and making it accessible.

- **Core Ownership:** Managing the Open-Meteo API requests (both Weather and Marine) and setting up the DuckDB architecture.
- **Key Deliverables:** The `src/` Python scripts that download historical data, handle the hourly-to-daily aggregations, and merge everything into clean tables.
- **Interaction:** Ensure the Data Analyst and ML Engineer have a clean, ready-to-use dataset by Day 2 or 3.

### 2. Ayxan Muxtar, Məhəmməd Sadıqov - Data Analyst

**Focus:** Understanding the data and defining the logic.

- **Core Ownership:** Exploratory Data Analysis (EDA) and defining the business logic for what constitutes a "delay-risk day" (e.g., finding the right wind speed and wave height thresholds).
- **Key Deliverables:** Jupyter notebooks with visualizations (e.g., distribution of high-risk months across Baku vs. Aktau), correlation matrices, and identifying class imbalances.
- **Interaction: P**rovide the ML Engineer with the exact features and thresholds needed to train the model.

### 3. Ayxan Muxtar, Əli Əliqulu - ML Engineer

**Focus:** Building and training the predictive engine.

- **Core Ownership:** Transforming daily data into monthly aggregates, splitting the data chronologically, and training the probability classifier (Logistic Regression/XGBoost).
- **Key Deliverables:** The model training scripts, hyperparameter tuning, and performance evaluation metrics (ROC-AUC, Precision-Recall curves).
- **Interaction:** Rely on the Data Analyst for feature ideas and work with the Integrator to get the model ready for predictions.

### 4. Əli Əliqulu, Adil Həsənov - MLOps Engineer

**Focus:** Tying it all together and making it usable.

- **Core Ownership:** Repository health, dependency management, and building the final inference pipeline (the script that takes the 14-day forecast and outputs the final risk probability).
- **Key Deliverables:** Managing GitHub pull requests, ensuring the final documentation is professional, and writing the final prediction script.
- **Interaction:** Act as the glue, taking the pipeline from the Data Engineer and the model from the ML Engineer to create the final working product.
## 3. Initial Features

### 3.1 Raw Daily Features (19 per city-day)

**Weather variables from Open-Meteo Archive (15):**

| Group         | Variable                      | Unit     | Maritime Relevance                                    |
| ------------- | ----------------------------- | -------- | ----------------------------------------------------- |
| Temperature   | `temperature_2m_max`          | °C       | Extreme cold → ice accretion on vessels               |
|               | `temperature_2m_min`          | °C       | Sub-zero triggers icing on surfaces                   |
|               | `temperature_2m_mean`         | °C       | Background thermal state; drives humidity and fog     |
|               | `apparent_temperature_mean`   | °C       | Wind chill; crew safety in exposed conditions         |
| Wind          | `wind_speed_10m_max`          | km/h     | **Primary risk driver**: >50 km/h suspends operations |
|               | `wind_gusts_10m_max`          | km/h     | Sudden gusts endanger vessel stability                |
|               | `wind_direction_10m_dominant` | °        | Determines wave fetch length over open water          |
| Precipitation | `precipitation_sum`           | mm       | Heavy rain reduces visibility; port flooding          |
|               | `rain_sum`                    | mm       | Liquid precipitation component                        |
|               | `snowfall_sum`                | cm       | Ice risk; port equipment failure                      |
| Atmosphere    | `weather_code`                | WMO code | Encodes storm, fog, blizzard categorically            |
|               | `relative_humidity_2m_mean`   | %        | High humidity + low temp → fog probability            |
|               | `dew_point_2m_mean`           | °C       | Dew point ≈ temperature → fog is imminent             |
|               | `surface_pressure_mean`       | hPa      | Rapid drops signal incoming storms                    |
|               | `shortwave_radiation_sum`     | MJ/m²    | Low values → overcast / storm conditions              |

**Visibility features from Historical Forecast API (3, available 2022+):**

| Variable                     | Unit  | Maritime Relevance                           |
| ---------------------------- | ----- | -------------------------------------------- |
| `visibility_mean`            | m     | Mean daily visibility; <1000 m = fog closure |
| `visibility_min`             | m     | Worst hour; <500 m = severe fog              |
| `visibility_hours_below_1km` | count | Number of fog-closure hours per day          |

**Wave feature from SMB proxy (1):**

| Variable      | Unit | Maritime Relevance                                                               |
| ------------- | ---- | -------------------------------------------------------------------------------- |
| `wave_height` | m    | Estimated via Sverdrup-Munk-Bretschneider formula; >2.5 m = operations suspended |

### 3.2 Delay-Risk Day Thresholds

A day is flagged as a **delay-risk day** if **any** of the following thresholds are breached:

| Variable                     | Threshold | Direction        |
| ---------------------------- | --------- | ---------------- |
| `wind_speed_10m_max`         | 50 km/h   | Above = risk     |
| `wind_gusts_10m_max`         | 75 km/h   | Above = risk     |
| `precipitation_sum`          | 15 mm     | Above = risk     |
| `snowfall_sum`               | 5 cm      | Above = risk     |
| `wave_height`                | 2.5 m     | Above = risk     |
| `visibility_mean`            | 1000 m    | **Below** = risk |
| `visibility_min`             | 500 m     | **Below** = risk |
| `visibility_hours_below_1km` | 4 hours   | Above = risk     |

### 3.3 Planned Engineered Features (Day 4)

Daily risk flags will be aggregated to monthly features for model training:

- **Monthly aggregates**: mean, max, min, std of each raw variable per city-month
- **Risk counts**: number of delay-risk days per month, percentage of risky days
- **Lag features**: previous month's risk count, rolling 3-month averages
- **Seasonal encoding**: month-of-year (sine/cosine cyclical encoding)
- **Fog proxy** (pre-2022): binary flag derived from humidity and dew-point spread
- **City indicator**: one-hot or label encoding

### Threshold Sensitivity and Calibration

The definition of a delay-risk day is based on domain-informed thresholds (e.g., wind speed, precipitation, visibility). These thresholds directly influence the number of risk days and the resulting monthly classification label.

Small changes in threshold values can significantly affect:

- the number of risk days per month,
- the distribution of the target variable,
- and overall class balance.

To address this, **Day 6 (Exploratory Data Analysis)** will include:

- analysis of risk-day distributions across cities and months,
- evaluation of class balance under current thresholds,
- potential calibration of both daily thresholds and the monthly cutoff (≥5 days).

Thresholds are therefore treated as tunable parameters rather than fixed constants.

---

## 3. Target Variable

The model predicts at **daily granularity**, but the user-facing output is **monthly** — for any upcoming calendar month, you get one row per (city × day) telling you whether that specific day will be a delay-risk day.

### Primary target — daily

| Property | Value |
|----------|-------|
| **Name** | `is_risk_day` |
| **Type** | Binary classification (0 / 1) |
| **Definition** | `1` if the day's weather breaches any of the 8 thresholds below |
| **Granularity** | One label per (city × day) |
| **Total training samples** | ~18,265 city-days (5 cities × 3,653 days across 2015–2024) |
| Property      | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Name          | `high_risk_month`                                               |
| Type          | Binary classification (0 or 1)                                  |
| Definition    | 1 if the month has **≥ 5 delay-risk days**, 0 otherwise         |
| Granularity   | One label per city per calendar month                           |
| Total samples | ~600 city-months (5 cities × 10 years × 12 months)              |
| Threshold     | Configurable via `HIGH_RISK_MONTH_THRESHOLD` in `src/config.py` |

**Interpretation**: a month labelled `1` means that at least 5 out of ~30 days (≈16%) had weather conditions severe enough to disrupt maritime operations — through extreme wind, heavy precipitation, high waves, or dense fog.

### Derived target — monthly

| Property | Value |
|----------|-------|
| **Name** | `high_risk_month` |
| **Type** | Binary (0 / 1) computed from daily predictions, not a separate model |
| **Definition** | `1` if predicted risk-days ≥ `HIGH_RISK_MONTH_THRESHOLD` (default 5) |
| **Granularity** | One label per (city × calendar month) |
| **Probability** | Estimated by Monte Carlo over per-day probabilities (Poisson-binomial) |

A **delay-risk day** is any calendar day where *any* of the following thresholds is breached:

| Variable | Threshold | Direction | Operational rationale |
|----------|-----------|-----------|-----------------------|
| `wind_speed_10m_max` | 50 km/h | above | Beaufort 10 / storm force |
| `wind_gusts_10m_max` | 75 km/h | above | Sudden gusts compromise vessel stability |
| `precipitation_sum` | 15 mm | above | Heavy rain reduces visibility, port flooding |
| `snowfall_sum` | 5 cm | above | Ice on equipment, port closures |
| `wave_height` *(SMB proxy)* | 2.5 m | above | Caspian operational suspension threshold |
| `visibility_mean` | 1000 m | **below** | Fog closure threshold |
| `visibility_min` | 500 m | **below** | Severe fog (worst hour of day) |
| `visibility_hours_below_1km` | 4 hours | above | Sustained fog event during day |

All thresholds live in `src/config.py:RISK_THRESHOLDS` (single source of truth).

## 4. Features

### 4.1 Raw Variables (per city × day)

#### Weather variables — Open-Meteo Archive API (ERA5 reanalysis), 2015–2024

| Source variable | Final column name | Units | Aggregation in source |
|----------------|-------------------|-------|----------------------|
| `temperature_2m_max` | `temperature_2m_max` | °C | Daily max |
| `temperature_2m_min` | `temperature_2m_min` | °C | Daily min |
| `temperature_2m_mean` | `temperature_2m_mean` | °C | Daily mean |
| `apparent_temperature_mean` | `apparent_temperature_mean` | °C | Daily mean (wind chill) |
| `wind_speed_10m_max` | `wind_speed_10m_max` | km/h | Daily max |
| `wind_gusts_10m_max` | `wind_gusts_10m_max` | km/h | Daily max |
| `wind_direction_10m_dominant` | `wind_direction_10m_dominant` | degrees | Daily dominant |
| `precipitation_sum` | `precipitation_sum` | mm/day | Daily sum |
| `rain_sum` | `rain_sum` | mm/day | Daily sum |
| `snowfall_sum` | `snowfall_sum` | cm/day | Daily sum |
| `weather_code` | `weather_code` | WMO code | Daily (worst conditions) |
| `relative_humidity_2m_mean` | `relative_humidity_2m_mean` | % | Daily mean |
| `dew_point_2m_mean` | `dew_point_2m_mean` | °C | Daily mean |
| `surface_pressure_mean` | `surface_pressure_mean` | hPa | Daily mean |
| `shortwave_radiation_sum` | `shortwave_radiation_sum` | MJ/m² | Daily sum |

#### Visibility — Open-Meteo Historical Forecast API, 2022–2024 only

Hourly visibility values are aggregated to three daily features:

| Hourly source | Daily feature | Units | Aggregation |
|---------------|--------------|-------|-------------|
| `visibility` | `visibility_mean` | m | Mean over 24 hours |
| `visibility` | `visibility_min` | m | Min (worst hour) |
| `visibility` | `visibility_hours_below_1km` | count | Count of hours below 1000 m |

Pre-2022 visibility rows are NaN at ingestion and median-imputed at cleaning time. A binary flag `visibility_is_known` (1 = real data, 0 = imputed) tells the model how much to trust the visibility features for each row.

#### Wave height — derived from wind, 2015–2024

| Derivation | Final column | Units | Method |
|-----------|--------------|-------|--------|
| Wind speed + direction + per-city fetch lookup | `wave_height` | m | Sverdrup-Munk-Bretschneider (SMB) fetch-limited formula, capped at 6 m |

**Why a proxy?** Free historical wave data for the Caspian is essentially unavailable (Open-Meteo Marine API is forecast-only; Copernicus ERA5-wave requires a CDS account). The SMB formula gives ±30% accuracy vs ERA5 — sufficient for threshold-based labelling at the 2.5 m operational cutoff.

### 4.2 Engineered Features (per city × day)

Computed in `src/features.py:engineer_all_features()`:

| Category | Feature columns | Purpose |
|----------|----------------|---------|
| **Calendar** | `year`, `month`, `quarter`, `day_of_year`, `week_of_year`, `day_of_week`, `season` | Basic temporal context |
| **Cyclical** | `month_sin`, `month_cos`, `doy_sin`, `doy_cos` | Tree models see month 12 ↔ month 1 adjacency |
| **Volatility** | `temp_range_c`, `temp_range_7d` | Daily max-min and its 7-day mean |
| **Energy** | `hdd`, `cdd` | Heating / cooling degree-days, base 18 °C |
| **Rolling** | `<var>_7d_mean`, `<var>_7d_max`, `<var>_30d_mean`, `<var>_30d_max` for temperature, precipitation, wind | Persistence and accumulation |
| **Anomaly** | `<var>_anom` for temperature, wind, precipitation | Per-city per-day-of-year z-score (deviation from climatology) |
| **Lag** | `<var>_lag1`, `<var>_lag2` for same 3 variables | Short-term autocorrelation |
| **Quality flags** | `<var>_is_outlier` for 6 numeric columns | IQR-based per-city outlier marker (rows preserved, not removed) |
| **Visibility provenance** | `visibility_is_known` | 1 if real visibility data, 0 if imputed |

### 4.3 Monthly Aggregates (per city × month — model training table)

Computed in `analytics.monthly_summary` via SQL window functions:

- Risk counts: `risk_days`, `risk_day_pct`, `wind_risk_days`, `gust_risk_days`, `precip_risk_days`, `snow_risk_days`, `wave_risk_days`, `visibility_risk_days`, `severe_fog_days`
- Stats per variable: mean, max, min, std for temperature, wind, precipitation, pressure, humidity, visibility, wave height
- Targets: **`high_risk_month`** (binary), `risk_days` (count, used to compute the binary)

## 5. Prediction Horizon

The output covers a full calendar month, but the prediction strategy splits the month at the 16-day mark:
**1 month** — the model predicts whether the upcoming calendar month will be a high-risk month.

| Aspect       | Detail                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------ |
| Horizon      | 1 calendar month                                                                                 |
| Input window | Previous month's weather features + seasonal context                                             |
| Output       | Probability that the target month has ≥ 5 delay-risk days                                        |
| Use case     | Route planning, cargo scheduling, port operations staffing — decisions made 2–4 weeks in advance |
| Evaluation   | Temporal train/test split: train on 2015–2022, validate on 2023, test on 2024                    |

| Days | Prediction source | Why |
|------|-------------------|-----|
| **1–16** | `DailyClassifier` fed by Open-Meteo's free 16-day forecast (real model output) | Forecast features have genuine atmospheric skill in this window |
| **17–end of month** | `ClimatologyTable` lookup — per-(city, day-of-year) historical positive rate over the 2015–2024 base period, smoothed with a ±7-day rolling window | Beyond ~14 days, free deterministic weather forecasts converge toward climatology anyway. We make this explicit instead of pretending we have skill we don't |

Each prediction row is tagged with a `source` column (`short_horizon` or `climatology`) so downstream consumers know how much to trust each value.

| Aspect | Value |
|--------|-------|
| **Horizon** | Full upcoming calendar month (~30 days) |
| **Cadence** | Monthly cron via GitHub Actions, on the 1st of each month at 06:00 UTC |
| **Output** | `predictions/YYYY-MM/daily.csv` (per-day) + `predictions/YYYY-MM/monthly.csv` (summary) |
| **Use case** | Operational planning for the next 16 days; medium-term planning informed by climatology for days 17+ |

## 6. Dataset

| Property | Value |
|----------|-------|
| **Date range** | 2015-01-01 → 2024-12-31 (10 years) |
| **Granularity (raw)** | Daily |
| **Granularity (target)** | Monthly |
| **Region** | Five Caspian coastal ports |
| **Total daily observations** | ~18,265 city-days |
| **Total monthly labels** | ~600 city-months |
| **Source** | Open-Meteo Archive API + Historical Forecast API |
| **Storage** | Local DuckDB analytical database (`data/caspian_weather.duckdb`) |

### 6.1 Cities

| City | Country | Lat | Lon | Maritime role |
|------|---------|-----|-----|---------------|
| Baku | Azerbaijan | 40.41 | 49.87 | Largest Caspian port; major oil terminal |
| Aktau | Kazakhstan | 43.65 | 51.17 | Trans-Caspian trade hub; ferry terminus |
| Anzali | Iran | 37.47 | 49.46 | Iran's primary Caspian port; southern node |
| Turkmenbashi | Turkmenistan | 40.02 | 52.97 | Eastern Caspian; energy exports |
| Makhachkala | Russia | 42.98 | 47.50 | Northern Russian port; ferry terminus |
## Cities

| City         | Country      | Latitude | Longitude | Role in Caspian Maritime Network           |
| ------------ | ------------ | -------- | --------- | ------------------------------------------ |
| Baku         | Azerbaijan   | 40.41    | 49.87     | Largest Caspian port; major oil terminal   |
| Aktau        | Kazakhstan   | 43.65    | 51.17     | Trans-Caspian trade hub; ferry terminus    |
| Anzali       | Iran         | 37.47    | 49.46     | Iran's primary Caspian port; southern node |
| Turkmenbashi | Turkmenistan | 40.02    | 52.97     | Eastern Caspian; energy exports            |
| Makhachkala  | Russia       | 42.98    | 47.50     | Northern Russian port; ferry terminus      |

## 7. Key Definitions

| Term | Definition |
|------|-----------|
| **Delay-risk day** | A calendar day where ≥ 1 of the 8 threshold variables crosses its threshold (see §3) |
| **High-risk month** | A calendar month containing ≥ 5 delay-risk days |
| **Imputed row** | A row whose visibility was NaN at ingestion and filled with the per-city median; `visibility_is_known = 0` |
| **SMB wave proxy** | Wave height estimated from wind speed via the Sverdrup-Munk-Bretschneider fetch-limited formula |
| **Three-layer database** | DuckDB schemas: `raw` (untouched API output), `staging` (cleaned, type-enforced), `analytics` (feature-engineered, model-ready) |
| **Quality gate** | An automated check that runs after each pipeline stage; can be ABORT, WARN, or FLAG severity |
| **Run history** | The `meta.pipeline_runs` table — one row per pipeline invocation, used for monitoring |

## 8. Architecture

```
                        ┌──────────────────────┐
                        │  GitHub Actions cron │
                        │   1st of every month │
                        └──────────┬───────────┘
                                   │
                                   ▼
       ┌──────────────────────────────────────────────────────┐
       │                src/pipeline.py                       │
       │       --incremental │ full │ dry-run │ since         │
       └──────────┬─────────────────────────────────┬─────────┘
                  ▼                                 │
   ┌─────────┐  ┌──────────┐  ┌────────┐  ┌────────────┐  ┌────────┐
   │ INGEST  │─▶│ LOAD RAW │─▶│ CLEAN  │─▶│  FEATURES  │─▶│ TRAIN  │
   │ Open-   │  │ CSV →    │  │ raw →  │  │ staging →  │  │ model  │
   │ Meteo   │  │ DuckDB   │  │staging │  │ analytics  │  │        │
   └────┬────┘  └─────┬────┘  └────┬───┘  └─────┬──────┘  └────┬───┘
        │             │            │            │              │
      ❬gate❭       ❬gate❭       ❬gate❭       ❬gate❭            ▼
                                                          ┌─────────┐
                                                          │ PREDICT │
                                                          │  next   │
                                                          │  month  │
                                                          └────┬────┘
                                                               ▼
                                                  predictions/YYYY-MM.csv
```

| Stage | Script | Output |
|-------|--------|--------|
| Ingest | `src/ingestion.py` | CSV files in `data/raw/` |
| Load raw | `src/database.py` | `raw.weather_daily`, `raw.visibility_daily` |
| Clean | `src/cleaning.py` | `staging.weather_daily` |
| Features | `src/features.py` | `analytics.daily_enriched`, `analytics.monthly_summary` |
| Train | `src/modeling.py` | `models/latest.pkl` |
| Predict | `src/modeling.py` | `predictions/YYYY-MM.csv` |
| Quality gates | `src/quality_checks.py` | `meta.quality_flags`, log entries |
| Run history | `src/pipeline.py` | `meta.pipeline_runs` |

## 9. Project Status — Daily Incrementals

This is an 8-day sprint. Days 1–5 (Week 1: Data Engineering) are complete.

### April 20th — Project Setup & API Exploration
- Identified five target ports and their coordinates
- Mapped Open-Meteo Archive API endpoints, variables, and rate limits
- Explored visibility availability (discovered ERA5 archive does not serve it for the Caspian)
- Established WMO weather code reference table
- Bootstrapped `src/api_client.py` and the initial `src/config.py`
- **Deliverable**: `notebooks/day_01_exploration.ipynb`

### April 21st — Production Ingestion
- Built `src/ingestion.py` — production HTTP client using stdlib `urllib` only (no `requests` dependency)
- Implemented retry logic with exponential backoff and `Retry-After` header handling for HTTP 429
- Added per-city delay between fetches to stay under Open-Meteo rate limits
- Added `fetch_historical_chunked()` for splitting long date ranges into smaller windows
- Added two-tier visibility strategy: Historical Forecast API (2022+) + median imputation for pre-2022
- Built `audit_dataframe()` to distinguish expected nulls (pre-2022 visibility) from real problems
- Replaced broken Marine API call with the SMB wave proxy in `src/era5_client.py`
- Added per-city directional fetch lookup table (`CASPIAN_FETCH_KM`) for SMB wave estimation
- **Deliverable**: `notebooks/day_02_ingestion.ipynb`, complete raw data in `data/raw/`

### April 22nd — Database Design
- Designed three-layer schema: `raw`, `staging`, `analytics`
- Built `src/database.py` with: `get_connection`, `create_schemas`, `create_raw_tables`, `load_raw_data`, `build_staging`, `build_analytics`, `validate_database`, `run_query`, `build_database`
- Auto-detection of CSV files by naming pattern (weather, hourly visibility, forecasts)
- 8 automated database validation checks (table existence, city count, row counts, date range, no duplicates, no gaps, target label present, risk days present)
- 8 analytical SQL queries demonstrating the database works (avg temp by year, precipitation variance, top hottest days, dry days, risk seasonality, trigger frequency, monthly heatmap, high-risk months by year)
- **Deliverable**: `notebooks/day_03_database.ipynb`

### April 23rd — Cleaning & Feature Engineering
- Built `src/cleaning.py` — 5 functions: `handle_missing_values()`, `flag_outliers()`, `validate_date_continuity()`, `winsorize_by_city()`, `clean_raw_to_staging()`
- Built `src/features.py` — 9 functions covering rolling, seasonal, range, degree-days, anomaly, lag, wave-proxy, full-pipeline orchestrator, analytics-layer builder
- **Decision: per-city precipitation winsorizing** (Anzali = 60mm cap, others 25–40mm) to prevent one wet station from dominating cross-city statistics
- Implemented IQR-based outlier flagging (rows preserved, flags added as boolean columns)
- Decided on a 3-year window (2022–2024) initially to avoid synthetic visibility data
- **Switched to Option B2 mid-day**: extended window back to 2015 with `visibility_is_known` flag + median imputation, gaining 4× training data while avoiding regime-shift bugs
- Removed dead `fog_proxy_flag` code from `src/database.py`
- Patched `src/features.py` to handle below-threshold variable NaN safely (use `1e9` not `0` for visibility-style fillna)
- Wrote `reports/data_quality_report.md` — formal 5-section trust assessment
- **Deliverable**: `notebooks/day_04_cleaning.ipynb`

### April 24th — Pipeline Automation & Quality Gates
- Built `src/pipeline.py` — single-CLI orchestrator with `--mode full`, `--mode incremental`, `--since`, `--dry-run`, `--no-train`, `--no-predict`, `--strict-freshness` flags
- Built `src/quality_checks.py` — 6 automated quality gates: `row_count` (ABORT), `null_ratio` (WARN), `date_continuity` (WARN), `value_ranges` (FLAG), `feature_completeness` (WARN), `freshness_monthly` (WARN)
- Added incremental loading: `INSERT OR REPLACE` in `src/database.py`, per-city max-date detection in pipeline, 3-day overlap window for self-healing
- Added `src/modeling.py` — placeholder `BaselinePredictor` (per-city positive-rate predictor) with `.fit()` / `.predict_proba()` interface; Day 6 will replace internals with real classifier
- Added rotating-file logging at `logs/pipeline.log`, anchored to project root regardless of CWD
- Added `meta.pipeline_runs` audit table (run_id, mode, status, durations, row counts, errors)
- Added `meta.quality_flags` table for FLAG-severity check results
- Built `.github/workflows/monthly-pipeline.yml` — automatic monthly scheduling via GitHub Actions, with state restoration from artifacts, prediction commits back to repo
- **Deliverable**: `notebooks/day_05_pipeline.ipynb`, automatic monthly cron

### Significant Bug Fixes Across the Sprint

| # | Bug | Fix |
|---|-----|-----|
| 1 | `visibility_mean` all-null from ERA5 archive | Switched to `historical-forecast-api` for hourly visibility |
| 2 | Marine API 400 error (wrong variable suffixes + no historical support) | Replaced with SMB wave proxy; renamed deprecated function |
| 3 | `KeyError: 'date'` in Day 1 visualisation | Added if/else guard for DatetimeIndex vs date-column DataFrames |
| 4 | Synthetic visibility >24,140 m physically impossible | Capped noise before clipping; tuned fog frequency to ~6–16 days/year per city |
| 5 | `BinderException: GROUP BY with aliases` in DuckDB | Changed to `EXTRACT(YEAR FROM date)` instead of the alias |
| 6 | Heatmap showed 30/31 risk days for 2015–2021 | Risk-labeling `fillna(0)` on NaN visibility made `0 < 1000` trigger fog risk; switched to `fillna(1e9)` for below-threshold variables |
| 7 | `api_client.RISK_THRESHOLDS` had only 5 keys (missing wave/visibility) | Converted `api_client.py` to a re-export shim from `src.config`; verified with `is` identity |
| 8 | `validate_database()` defaulted to hardcoded `2019-01-01` | Now defaults to `None` and resolves from `config.DATE_RANGE` at call time |
| 9 | `save_raw(data_dir=...)` keyword mismatch | Fixed to `save_raw(directory=...)` |
| 10 | `date.fromisoformat('2026-04-24 00:00:00')` ValueError | DuckDB `MAX(date)` returned a Timestamp; cast to DATE in SQL + slice to `[:10]` |
| 11 | Logs not appearing | Anchored log dir to `PATHS['repo_root']` instead of CWD |

### Days 6–8 (planned)

- **Day 6** — Model training: replace `DailyClassifier`'s baseline internals with XGBoost. Train on `analytics.daily_enriched` with `is_risk_day` as the target. Stratified k-fold CV (group by city to avoid leakage). Class balance assessment. Feature importance analysis. The orchestration code in `predict_next_month()` does NOT need to change — only the classifier internals.
- **Day 7** — Evaluation: ROC-AUC, Brier score, calibration curves, per-city performance, error analysis. Also evaluate the climatology baseline as a benchmark — the daily model needs to clearly beat climatology at its 16-day horizon to justify itself.
- **Day 8** — Deployment polish: confirm the live `fetch_forecast` integration works in production. Consider adding a small static site that renders the latest `predictions/YYYY-MM/daily.csv` as a calendar view.

### Day 5 addendum — Daily-then-Monthly prediction strategy

Mid-Day 5, the prediction strategy was switched from "monthly classifier" to "daily classifier with monthly aggregation":

- The user-facing output is now **monthly with daily breakdown** (one row per city × day for the upcoming month)
- The model itself predicts at daily granularity, trained on `analytics.daily_enriched`
- Days 1–16 use the model with real Open-Meteo forecast features
- Days 17+ use the per-(city, day-of-year) climatology lookup, computed once during the train stage
- The monthly summary (`risk_days_predicted`, `high_risk_month_probability`) is **derived** from daily predictions, not a separate model
- The monthly cron schedule is unchanged



## 10. Repository Structure

```
caspian-maritime-weather/
├── README.md                                ← this file
├── requirements.txt
├── .github/
│   └── workflows/
│       └── monthly-pipeline.yml             ← automatic monthly scheduling
├── notebooks/
│   ├── day_01_exploration.ipynb
│   ├── day_02_ingestion.ipynb
│   ├── day_03_database.ipynb
│   ├── day_04_cleaning.ipynb
│   └── day_05_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            ← single source of truth for all constants
│   ├── api_client.py        ← deprecated shim, re-exports from config
│   ├── ingestion.py         ← Open-Meteo client with retries, caching, auditing
│   ├── era5_client.py       ← marine forecast + SMB wave proxy
│   ├── database.py          ← DuckDB three-layer schema + load + validate
│   ├── cleaning.py          ← raw → staging cleaning pipeline
│   ├── features.py          ← engineered features + analytics builder
│   ├── risk_labeler.py      ← delay-risk day & monthly label logic
│   ├── quality_checks.py    ← 6 automated quality gates
│   ├── modeling.py          ← train + predict (Day 5 stub, Day 6 replaces with XGBoost)
│   └── pipeline.py          ← end-to-end orchestrator (CLI + Python API)
├── data/
│   ├── raw/                 ← API responses (CSV)
│   ├── processed/           ← (intermediate, currently unused)
│   └── caspian_weather.duckdb
├── models/
│   └── latest.pkl
├── predictions/
│   └── YYYY-MM.csv
├── reports/
│   ├── data_quality_report.md
│   └── *.png                ← figures from notebooks
└── logs/
    └── pipeline.log         ← rotating, last 30 runs
```

## 11. Quick Start

```bash
git clone https://github.com/<your-username>/caspian-maritime-weather.git
cd caspian-maritime-weather
pip install -r requirements.txt

# Build everything from scratch (~3 minutes)
python src/pipeline.py --mode full

# Subsequent runs only fetch new days
python src/pipeline.py --mode incremental

# What would happen if I ran this?
python src/pipeline.py --dry-run
```

To enable automatic monthly runs:
1. Push to GitHub
2. Repo **Settings → Actions → General** → allow workflows
3. Settings → Actions → General → "Workflow permissions" → **Read and write**
4. Trigger once manually from the **Actions** tab to verify
5. Cron takes over from there

## 12. Limitations & Honest Assessment

1. **Wave height is a wind-derived proxy**, not a measurement. Accuracy is ±30% vs ERA5 reanalysis. If the model ends up heavily wave-driven, this uncertainty propagates into the risk probability.
2. **Labels are threshold-based, not observational**. The model predicts "weather thresholds will be breached on ≥5 days," not "actual port disruptions will occur." Without AIS or port-log ground truth, this is the closest proxy available.
3. **Pre-2022 visibility is imputed**. ~70% of training rows carry a per-city median for the three visibility columns, with `visibility_is_known = 0` flagging them. Tree-based models handle this natively, but linear models would need explicit missingness strategies.
4. **The Caspian is data-poor**. ERA5 reanalysis has known biases over inland seas. These biases are systematic across all cities, so they affect cross-city comparisons less than absolute values.
5. **3 years of real visibility data** (2022–2024) is a short window for fog climatology. Year-to-year variance in fog days is high.

---


