# Team Info

## Team Name

### Anemoi

## Roles

| Role                    | Members                        | Responsibility                                                   |
| ----------------------- | ------------------------------ | ---------------------------------------------------------------- |
| **Data Engineering**    | Məhəmməd Sadıqov, Adil Həsənov | Open-Meteo ingestion, DuckDB setup, raw/staging data pipeline    |
| **Data Analysis**       | Ayxan Muxtar, Məhəmməd Sadıqov | EDA, thresholds, feature insights, class balance                 |
| **Machine Learning**    | Ayxan Muxtar, Əli Əliqulu      | Model training, model comparison, calibration, evaluation        |
| **MLOps & Integration** | Əli Əliqulu, Adil Həsənov      | Pipeline automation, repo management, inference and website flow |

---

# Project Status — Daily Incrementals

This project was developed as an 8-day sprint, moving from API exploration to a working automated forecasting demo.

## April 20th — Project Setup & API Exploration

- Selected five Caspian port cities and coordinates
- Explored Open-Meteo APIs and available variables
- Identified visibility limitations in ERA5 archive data
- Started project configuration and API client setup

**Deliverable:** `notebooks/day_01_exploration.ipynb`

## April 21st — Production Ingestion

- Built `src/ingestion.py` for historical weather fetching
- Added retry logic, rate-limit handling, and chunked fetching
- Fetched visibility from the Historical Forecast API where available
- Added raw data saving and audit checks

**Deliverable:** `notebooks/day_02_ingestion.ipynb`, raw CSV files in `data/raw/`

## April 22nd — Database Design

- Created DuckDB database with `raw`, `staging`, and `analytics` schemas
- Added CSV loading, validation checks, and SQL analysis queries
- Built the first database-backed version of the project pipeline

**Deliverable:** `notebooks/day_03_database.ipynb`

## April 23rd — Cleaning & Feature Engineering

- Built `src/cleaning.py` for missing values, outliers, and date continuity
- Built `src/features.py` for rolling, seasonal, lag, anomaly, and wave-proxy features
- Added `visibility_is_known` to separate real visibility from imputed values
- Removed synthetic fog logic

**Deliverable:** `notebooks/day_04_cleaning.ipynb`

## April 24th — Pipeline Automation & Quality Gates

- Built `src/pipeline.py` as a single-command pipeline runner
- Added full and incremental modes
- Built `src/quality_checks.py` for automated pass/warning/fail checks
- Added logging to `logs/pipeline.log`

**Deliverable:** `notebooks/day_05_pipeline.ipynb`

## April 27th — Exploratory Data Analysis

- Compared cities by weather patterns and delay-risk frequency
- Reviewed distributions, class balance, and threshold behavior
- Checked dry days, precipitation variance, wind patterns, and seasonal effects
- Prepared plots and tables for explaining the dataset

**Deliverable:** `notebooks/day_06_eda.ipynb`

## April 28th — Statistical Analysis & Feature Review

- Tested relationships between engineered features and risk labels
- Reviewed selected model features and leakage-safe inputs
- Checked city-level differences and possible imbalance
- Prepared statistical outputs for model-readiness discussion

**Deliverable:** `notebooks/day_07_statistical_analysis.ipynb`

## April 29th — Modeling, Evaluation & Demo Output

- Compared climatology, Logistic Regression, tree-based models, and calibrated variants
- Tuned the decision threshold for better F1/recall tradeoff
- Selected the production model using F1, ROC-AUC, Brier score, and recall
- Generated rolling daily predictions for the website demo

**Deliverable:** `notebooks/day_08_modeling.ipynb`, `models/daily_model.pkl`, `reports/day08_model_comparison.csv`

---

# Caspian Maritime Delay-Risk Forecasting

> A weather-based maritime risk forecasting project for Caspian Sea ports.  
> The system estimates future port-day risk levels and turns them into simple operational guidance.

---

# Problem Statement

Maritime operations across the Caspian Sea can be disrupted by:

- strong wind
- heavy precipitation
- dense fog / low visibility
- rough sea conditions

These conditions affect port scheduling, cargo planning, vessel operations, and route decisions.

The goal of this project is to turn historical and forecast weather data into a practical daily risk signal for maritime planning.

---

# Why It Matters

| Stakeholder                | How the prediction helps                              |
| -------------------------- | ----------------------------------------------------- |
| **Port operations**        | Staffing, equipment readiness, delay planning         |
| **Cargo planners**         | Buffer days, delivery estimates, schedule flexibility |
| **Vessel operators**       | Voyage timing, crew planning, manual review triggers  |
| **Insurance / risk teams** | Estimating weather-related operational risk           |

The Caspian Sea is an inland sea with limited public maritime risk tools. This project creates a port-level early-warning pipeline for that gap.

---

# Target Variable

| Property         | Value                                                  |
| ---------------- | ------------------------------------------------------ |
| **Name**         | `target_risk_next_day`                                 |
| **Type**         | Binary classification                                  |
| **Definition**   | `1` if the next day becomes a delay-risk day, else `0` |
| **Granularity**  | One label per city × day                               |
| **Source table** | `analytics.daily_enriched`                             |

A **delay-risk day** is a day where at least one maritime weather-risk threshold is breached.

## Delay-Risk Thresholds

| Variable                     | Threshold | Direction    |
| ---------------------------- | --------- | ------------ |
| `wind_speed_10m_max`         | 50 km/h   | Above = risk |
| `wind_gusts_10m_max`         | 75 km/h   | Above = risk |
| `precipitation_sum`          | 15 mm     | Above = risk |
| `snowfall_sum`               | 5 cm      | Above = risk |
| `wave_height`                | 2.5 m     | Above = risk |
| `visibility_mean`            | 1000 m    | Below = risk |
| `visibility_min`             | 500 m     | Below = risk |
| `visibility_hours_below_1km` | 4 hours   | Above = risk |

Thresholds are stored in `src/config.py` and mirrored in `src/risk_labeler.py`.

---

# Features

## Raw Variables

Daily weather variables are fetched from Open-Meteo APIs.

| Group             | Variables                                              |
| ----------------- | ------------------------------------------------------ |
| **Temperature**   | max, min, mean, apparent temperature                   |
| **Wind**          | max wind speed, gusts, dominant direction              |
| **Precipitation** | total precipitation, rain, snowfall                    |
| **Atmosphere**    | weather code, humidity, dew point, pressure, radiation |

## Visibility Features

Visibility is available from 2022 onward through the Historical Forecast API.

| Feature                      | Meaning                         |
| ---------------------------- | ------------------------------- |
| `visibility_mean`            | Average daily visibility        |
| `visibility_min`             | Worst visibility during the day |
| `visibility_hours_below_1km` | Number of low-visibility hours  |

For older rows, visibility is imputed during cleaning and marked with:

```text
visibility_is_known = 0
```

## Maritime Features

The project also adds maritime-specific features instead of relying only on generic weather values.

| Feature Group             | Purpose                                                       |
| ------------------------- | ------------------------------------------------------------- |
| **Wave proxy**            | Estimates historical wave height from wind and fetch exposure |
| **Wave-history features** | Adds lagged and rolling wave context for model training       |
| **Marine forecast**       | Adds short-horizon offshore sea-state context where available |

Example wave-history features:

```text
wave_height_lag1
wave_height_lag2
wave_height_7d_mean
wave_height_7d_max
wave_height_30d_mean
wave_height_30d_max
wave_height_anom
```

---

# Demo output

The website displays:

- a rolling 30-day forecast window
- one calendar per Caspian port
- risk levels and recommended actions
- forecast vs climatology source coloring
- popup details with key conditions such as wind, precipitation, visibility, and waves

## Risk Levels

| Risk Level    | Probability | Suggested Action              |
| ------------- | ----------- | ----------------------------- |
| **Low**       | `< 10%`     | Proceed normally              |
| **Moderate**  | `10–25%`    | Monitor conditions            |
| **High**      | `25–50%`    | Plan with caution             |
| **Very High** | `50–75%`    | Manual review advised         |
| **Severe**    | `75%+`      | Avoid or postpone if possible |

---

# Running the Pipeline

```bash
python src/pipeline.py --mode full
python src/pipeline.py --mode incremental
```

The incremental pipeline checks whether historical data is already up to date. If no new rows are needed, it skips API fetching but still refreshes analytics and predictions so the rolling calendar can move forward.

---
