# Caspian Maritime Delay-Risk Forecasting

**Predicting the probability that a given month will have a high number of maritime delay-risk days between Caspian coastal cities, based on wind, precipitation, wave height, visibility, and related atmospheric variables.**

---

## 1. Dataset History Length

The project uses **10 years** of historical weather data:

| Parameter          | Value                          |
| ------------------ | ------------------------------ |
| Start date         | 2015-01-01                     |
| End date           | 2024-12-31                     |
| Total span         | 10 years (3,653 days per city) |
| Cities             | 5 Caspian coastal ports        |
| Total observations | ~18,265 city-days              |

The 10-year window captures sufficient seasonal variation (40 full winter storm seasons across 5 cities) for robust monthly-level classification.

**Data sources and their temporal coverage:**

| Source                             | Endpoint                                 | Coverage               | What it provides                                   |
| ---------------------------------- | ---------------------------------------- | ---------------------- | -------------------------------------------------- |
| Open-Meteo Archive API             | `archive-api.open-meteo.com`             | 2015–2024 (full range) | 15 daily weather variables                         |
| Open-Meteo Historical Forecast API | `historical-forecast-api.open-meteo.com` | 2022–2024 only         | Hourly visibility → aggregated to 3 daily features |
| SMB Wave Proxy                     | Derived from wind variables              | 2015–2024 (full range) | Estimated significant wave height                  |

For dates before 2022 where visibility data is unavailable, a derived `fog_proxy_flag` feature (based on relative humidity ≥ 90% and temperature–dew point spread ≤ 2°C) is used as a substitute.

### Visibility Feature Consistency

Visibility data is only directly available from 2022 onwards via the Open-Meteo historical-forecast API. Earlier versions of the dataset used a fog proxy (based on humidity and dew point) for 2015–2021.

To ensure feature consistency across the full time range, the synthetic fog proxy will be removed during **Day 4 (Data Cleaning)**. As a result, visibility-related features will only be used where real measurements exist.

This avoids introducing artificial patterns into the model caused by mixing proxy-derived and real observations.

---

## 2. Dataset Granularity

**Daily granularity** — one observation per city per calendar day.

| Level                 | Detail                                                        |
| --------------------- | ------------------------------------------------------------- |
| Raw ingestion         | Daily (from Open-Meteo archive API)                           |
| Visibility enrichment | Hourly (from historical-forecast-api), aggregated to daily    |
| Feature engineering   | Daily → monthly aggregates (mean, max, min, std, percentiles) |
| Target variable       | Monthly (one label per city per calendar month)               |

The pipeline operates in two stages: daily collection and cleaning, followed by monthly aggregation for model training.

---

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

## 4. Target Variable

| Property      | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Name          | `high_risk_month`                                               |
| Type          | Binary classification (0 or 1)                                  |
| Definition    | 1 if the month has **≥ 5 delay-risk days**, 0 otherwise         |
| Granularity   | One label per city per calendar month                           |
| Total samples | ~600 city-months (5 cities × 10 years × 12 months)              |
| Threshold     | Configurable via `HIGH_RISK_MONTH_THRESHOLD` in `src/config.py` |

**Interpretation**: a month labelled `1` means that at least 5 out of ~30 days (≈16%) had weather conditions severe enough to disrupt maritime operations — through extreme wind, heavy precipitation, high waves, or dense fog.

The threshold of 5 days will be calibrated during EDA (Day 3) by examining the distribution of risk-day counts across all city-months and selecting a cutoff that produces a balanced-enough split for classification.

---

## 5. Prediction Horizon

**1 month** — the model predicts whether the upcoming calendar month will be a high-risk month.

| Aspect       | Detail                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------ |
| Horizon      | 1 calendar month                                                                                 |
| Input window | Previous month's weather features + seasonal context                                             |
| Output       | Probability that the target month has ≥ 5 delay-risk days                                        |
| Use case     | Route planning, cargo scheduling, port operations staffing — decisions made 2–4 weeks in advance |
| Evaluation   | Temporal train/test split: train on 2015–2022, validate on 2023, test on 2024                    |

**Why 1 month?** Maritime logistics decisions (vessel scheduling, insurance quotes, port staffing) are typically planned on a monthly cycle. A monthly risk probability is directly actionable: a high probability triggers schedule adjustments, buffer days, or route alternatives through lower-risk ports.

---

## Cities

| City         | Country      | Latitude | Longitude | Role in Caspian Maritime Network           |
| ------------ | ------------ | -------- | --------- | ------------------------------------------ |
| Baku         | Azerbaijan   | 40.41    | 49.87     | Largest Caspian port; major oil terminal   |
| Aktau        | Kazakhstan   | 43.65    | 51.17     | Trans-Caspian trade hub; ferry terminus    |
| Anzali       | Iran         | 37.47    | 49.46     | Iran's primary Caspian port; southern node |
| Turkmenbashi | Turkmenistan | 40.02    | 52.97     | Eastern Caspian; energy exports            |
| Makhachkala  | Russia       | 42.98    | 47.50     | Northern Russian port; ferry terminus      |

These five cities span all five Caspian littoral states and represent the major maritime trade corridors.

---

## Team & Roles

**Data Engineering** — Məhəmməd Sadıqov, Adil Həsənov

- Data ingestion (Open-Meteo APIs) and DuckDB setup
- Delivered clean, structured datasets

**Data Analysis** — Ayxan Muxtar, Məhəmməd Sadıqov

- EDA, threshold definition, and feature insights
- Analyzed distributions and class balance

**Machine Learning** — Ayxan Muxtar, Əli Əliqulu

- Monthly feature engineering and model training
- Evaluation and tuning (LogReg, XGBoost)

**MLOps & Integration** — Əli Əliqulu, Adil Həsənov

- Pipeline integration and inference script
- Repo management and final documentation
