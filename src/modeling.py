"""
src/modeling.py
───────────────
Daily delay-risk prediction pipeline.

Two-component prediction strategy
---------------------------------
The user-facing output is **monthly** (one CSV per upcoming calendar month
covering all ~30 days), but the model itself works at daily granularity:

    Days 1–16 of horizon  →  DailyClassifier (XGBoost in Day 6, baseline now)
                              fed real Open-Meteo 16-day forecast features
    Days 17 onwards       →  ClimatologyTable lookup
                              per-(city, day-of-year) historical positive rate

Each prediction row is tagged with a `source` column ('short_horizon' or
'climatology') so downstream consumers know how much to trust it.

The monthly summary CSV is **derived** from the daily predictions — it is not
a separate model. It just counts predicted risk days per (city × month) and
estimates P(high_risk_month) via Monte Carlo over the daily probabilities.

**Day 5 status: PLACEHOLDER `DailyClassifier`.** The `BaselinePredictor`
class predicts the per-(city, month) historical positive rate. Day 6 will
swap its internals for XGBoost / RandomForest while keeping the same
.fit(X, y) / .predict_proba(X) interface so the rest of the pipeline
needs no changes.

Public API
----------
    train_model(conn, model_path)                     → dict (metrics)
    build_climatology(conn, climatology_path)         → dict (lookup info)
    predict_next_month(conn, model_path, climatology_path,
                       target_month, ...)             → tuple[DataFrame, DataFrame]
    save_predictions(daily_df, monthly_df, target_month, out_dir)
                                                      → tuple[Path, Path]
    load_model(path)                                   → object | None
"""

from __future__ import annotations

import logging
import pickle
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)




# ══════════════════════════════════════════════════════════════════════════════
# 2. ClimatologyTable — per-(city, day-of-year) historical base rate
# ══════════════════════════════════════════════════════════════════════════════

class ClimatologyTable:
    """
    Lookup of P(is_risk_day = 1) by (city, day-of-year).

    Built once from the full historical analytics.daily_enriched table.
    Used for predictions beyond the 16-day forecast horizon.

    Smoothing: predictions for day-of-year d are computed as the mean over
    a centred ±N-day window in the historical data. This stabilises against
    the high variance you'd otherwise get from individual calendar days.
    """

    def __init__(self, smoothing_window: int = 7):
        self.smoothing_window = smoothing_window
        self.rates: dict[tuple[str, int], float] = {}
        self.global_rate: float = 0.5
        self.trained_on_rows: int = 0

    def fit(self, df: pd.DataFrame) -> "ClimatologyTable":
        """
        df must contain: city, date (or day_of_year), is_risk_day.
        """
        if "city" not in df.columns or "is_risk_day" not in df.columns:
            raise ValueError("df must contain 'city' and 'is_risk_day' columns")

        df = df.copy()
        if "day_of_year" not in df.columns:
            df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
        df["day_of_year"] = df["day_of_year"].astype(int)

        # Per-(city, day_of_year) raw rate
        raw_rates = (
            df.groupby(["city", "day_of_year"])["is_risk_day"]
              .mean()
              .reset_index(name="rate")
        )

        # Apply circular rolling-mean smoothing over ±smoothing_window days
        # (so day 365 wraps around to day 1)
        smoothed_rows = []
        for city, sub in raw_rates.groupby("city"):
            # Build a length-366 array indexed by day_of_year
            arr = np.full(367, np.nan)
            for _, r in sub.iterrows():
                arr[int(r["day_of_year"])] = r["rate"]
            # Forward-fill any NaN holes from the historical data
            mask = np.isnan(arr[1:367])
            if mask.any():
                vals = arr[1:367]
                last_good = np.nan
                for i in range(366):
                    if not np.isnan(vals[i]):
                        last_good = vals[i]
                    elif not np.isnan(last_good):
                        vals[i] = last_good
                arr[1:367] = vals
            # Circular rolling mean
            w = self.smoothing_window
            padded = np.concatenate([arr[1:367][-w:], arr[1:367], arr[1:367][:w]])
            kernel = np.ones(2 * w + 1) / (2 * w + 1)
            smoothed = np.convolve(padded, kernel, mode="valid")
            for doy in range(1, 367):
                smoothed_rows.append({
                    "city":         city,
                    "day_of_year":  doy,
                    "rate":         float(smoothed[doy - 1]),
                })

        smoothed_df = pd.DataFrame(smoothed_rows)
        self.rates = {
            (r["city"], int(r["day_of_year"])): float(r["rate"])
            for _, r in smoothed_df.iterrows()
        }
        self.global_rate = float(df["is_risk_day"].mean())
        self.trained_on_rows = len(df)

        logger.info(
            "  ClimatologyTable built from %d rows; %d entries; "
            "smoothing ±%d days; global rate = %.3f",
            self.trained_on_rows, len(self.rates),
            self.smoothing_window, self.global_rate,
        )
        return self

    def predict_proba(self, city: str, day_of_year: int) -> float:
        return self.rates.get((city, int(day_of_year)), self.global_rate)

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        if "city" not in df.columns:
            raise ValueError("df must contain 'city' column")
        if "day_of_year" not in df.columns:
            df = df.copy()
            df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
        keys = list(zip(df["city"].values, df["day_of_year"].astype(int).values))
        return np.array([self.rates.get(k, self.global_rate) for k in keys],
                        dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Train daily model
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    conn,
    model_path: str | Path = "models/daily_model.pkl",
    feature_table: str = "analytics.daily_enriched",
    decision_threshold: float = 0.10,
) -> dict:
    """
    Train the production next-day maritime delay-risk model.

    Production target:
        features at day t -> target_risk_next_day = is_risk_day at day t+1

    This replaces the old DailyClassifier baseline. The saved object is a
    bundle containing:
        - calibrated sklearn model
        - feature column list
        - target name
        - decision threshold
        - metadata for the app/pipeline
    """
    logger.info("Training calibrated next-day model on %s ...", feature_table)

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    df = conn.execute(f"""
        SELECT *
        FROM {feature_table}
        ORDER BY city, date
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{feature_table} is empty — cannot train")

    if "is_risk_day" not in df.columns:
        raise ValueError(
            f"{feature_table} is missing 'is_risk_day'. "
            "Build analytics layer before training."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Next-day target: today's features predict tomorrow's risk
    target = "target_risk_next_day"
    df[target] = df.groupby("city")["is_risk_day"].shift(-1)

    df_model = df.dropna(subset=[target]).copy()
    df_model[target] = df_model[target].astype(int)

    # These must never be used as model inputs
    leakage_cols = {
    # targets / labels
    "is_risk_day",
    "target_risk_next_day",
    "high_risk_month",
    "risk_days",
    "risk_day_pct",

    # direct same-day threshold variables used to define risk
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "visibility_mean",
    "visibility_min",
    "visibility_hours_below_1km",
    "wave_height",

    # direct risk flags
    "risk_wind",
    "risk_gust",
    "risk_precip",
    "risk_snow",
    "risk_wave",
    "risk_visibility",
    "risk_fog_min",
    "risk_fog_proxy",
}

    # Use Day-7 selected features as source of truth
    try:
        from reports.selected_features import SELECTED_FEATURES

        feature_cols = [
            c for c in SELECTED_FEATURES
            if c in df_model.columns and c not in leakage_cols
        ]

        logger.info("  Using %d Day-7 selected leakage-safe features", len(feature_cols))

    except Exception as exc:
        logger.warning(
            "Could not load reports.selected_features.SELECTED_FEATURES: %s",
            exc,
        )
        logger.warning("Falling back to all leakage-safe non-target columns.")

        excluded = leakage_cols | {"date"}
        feature_cols = [
            c for c in df_model.columns
            if c not in excluded
        ]

    if not feature_cols:
        raise ValueError("No usable feature columns found for training")

    X = df_model[feature_cols].copy()
    y = df_model[target].astype(int).values

    categorical_features = [
        c for c in ["city", "season"]
        if c in feature_cols
    ]

    numeric_features = [
        c for c in feature_cols
        if c not in categorical_features
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )

    base_pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", base_model),
    ])

    # Calibrated model for trustworthy user-facing probabilities
    try:
        production_model = CalibratedClassifierCV(
            estimator=base_pipeline,
            method="isotonic",
            cv=5,
        )
    except TypeError:
        # Older sklearn compatibility
        production_model = CalibratedClassifierCV(
            base_estimator=base_pipeline,
            method="isotonic",
            cv=5,
        )

    production_model.fit(X, y)

    model_bundle = {
        "model": production_model,
        "model_name": "LogisticRegression_calibrated",
        "base_model_name": "LogisticRegression",
        "is_calibrated": True,
        "calibration_method": "isotonic",
        "decision_threshold": float(decision_threshold),
        "target": target,
        "feature_cols": feature_cols,
        "trained_rows": int(len(df_model)),
        "positive_rate": float(y.mean()),
        "description": (
            "Production next-day delay-risk model: "
            "features at day t -> calibrated P(risk on day t+1)"
        ),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    logger.info("  Saved calibrated next-day model → %s", model_path)
    logger.info("  Feature count: %d", len(feature_cols))
    logger.info("  Positive rate: %.4f", float(y.mean()))
    logger.info("  Decision threshold: %.2f", decision_threshold)

    return {
        "rows_trained": int(len(df_model)),
        "n_features": int(len(feature_cols)),
        "positive_rate": round(float(y.mean()), 4),
        "model_path": str(model_path),
        "model_type": "LogisticRegression_calibrated",
        "target": target,
        "decision_threshold": float(decision_threshold),
        "is_calibrated": True,
    }





def build_climatology(
    conn,
    climatology_path: str | Path = "models/climatology.pkl",
    feature_table:    str        = "analytics.daily_enriched",
    smoothing_window: int        = 7,
) -> dict:
    """
    Build the climatology lookup table → save to disk.
    """
    logger.info("Building climatology from %s ...", feature_table)

    df = conn.execute(f"""
        SELECT city, date, is_risk_day
        FROM {feature_table}
        ORDER BY city, date
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{feature_table} is empty — cannot build climatology")

    table = ClimatologyTable(smoothing_window=smoothing_window).fit(df)

    climatology_path = Path(climatology_path)
    climatology_path.parent.mkdir(parents=True, exist_ok=True)
    with climatology_path.open("wb") as f:
        pickle.dump(table, f)
    logger.info("  Saved climatology → %s", climatology_path)

    return {
        "entries":        len(table.rates),
        "rows_used":      int(table.trained_on_rows),
        "smoothing_days": smoothing_window,
        "global_rate":    round(table.global_rate, 4),
        "path":           str(climatology_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Helpers — load + horizon planning
# ══════════════════════════════════════════════════════════════════════════════

def load_model(path: str | Path):
    """Load a pickled model. Returns None if file missing."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_target_month(conn, target_month: Optional[str]) -> str:
    """
    Determine the target month: 'YYYY-MM' for the calendar month after the
    latest month in analytics.monthly_summary.
    """
    if target_month is not None:
        return target_month

    try:
        row = conn.execute("""
            SELECT MAX(year) AS y,
                   MAX(CASE WHEN year = (SELECT MAX(year) FROM analytics.monthly_summary)
                            THEN month END) AS m
            FROM analytics.monthly_summary
        """).fetchone()
        latest_year, latest_month = int(row[0]), int(row[1])
    except Exception:
        # No monthly data yet — predict for next month from today
        today = date.today()
        if today.month == 12:
            return f"{today.year + 1}-01"
        return f"{today.year}-{today.month + 1:02d}"

    if latest_month == 12:
        return f"{latest_year + 1}-01"
    return f"{latest_year}-{latest_month + 1:02d}"


def _all_dates_in_month(target_month: str) -> list[date]:
    """Return list of all calendar dates in 'YYYY-MM'."""
    y, m = map(int, target_month.split("-"))
    n_days = monthrange(y, m)[1]
    return [date(y, m, d) for d in range(1, n_days + 1)]


def _build_forecast_features_with_lookback(
    conn,
    forecast_df: pd.DataFrame,
    feature_cols: list[str],
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Build live prediction features using recent historical context.

    Why:
    - wind_change_1d needs the previous day
    - precip_change_1d needs the previous day
    - 3-day rolling features need recent rows
    - lag features need previous rows

    Flow:
        recent staging history + forecast rows
        -> engineer_all_features()
        -> keep only forecast rows
    """
    from src.features import engineer_all_features

    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame()

    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    min_forecast_date = forecast_df["date"].min()
    lookback_start = min_forecast_date - pd.Timedelta(days=lookback_days)

    history_df = conn.execute("""
        SELECT *
        FROM staging.weather_daily
        WHERE date >= ?
          AND date < ?
        ORDER BY city, date
    """, [lookback_start.date(), min_forecast_date.date()]).fetchdf()

    if history_df is None or history_df.empty:
        logger.warning(
            "No lookback rows found in staging.weather_daily. "
            "Rolling/change features may be less reliable."
        )
        history_df = pd.DataFrame()
    else:
        history_df["date"] = pd.to_datetime(history_df["date"])

    # Align history and forecast columns safely
    all_cols = sorted(set(history_df.columns) | set(forecast_df.columns))

    for col in all_cols:
        if col not in history_df.columns:
            history_df[col] = np.nan
        if col not in forecast_df.columns:
            forecast_df[col] = np.nan

    combined = pd.concat(
        [history_df[all_cols], forecast_df[all_cols]],
        ignore_index=True,
    )

    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)

    # Rebuild the SAME feature pipeline used for analytics/training
    engineered = engineer_all_features(combined)

    # Keep only forecast rows after features are created
    forecast_dates = set(forecast_df["date"].dt.date)

    pred_df = engineered[
        engineered["date"].dt.date.isin(forecast_dates)
    ].copy()

    # Make sure every expected model feature exists
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = np.nan

    return pred_df

# ══════════════════════════════════════════════════════════════════════════════
# 5. Predict — orchestrates short_horizon model + climatology
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_month(
    conn,
    model_path:        str | Path     = "models/daily_model.pkl",
    climatology_path:  str | Path     = "models/climatology.pkl",
    target_month:      Optional[str]  = None,
    forecast_horizon:  Optional[int]  = None,
    cities:            Optional[list[str]] = None,
    threshold:         float          = 0.5,
    n_monte_carlo:     int            = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate per-day predictions for every (city × day) of the target month,
    plus a derived monthly summary.

    Pipeline:
      1. Resolve target_month and the list of dates to predict
      2. Determine which dates fall in the short-horizon window (≤ N days
         from today) vs the climatology window
      3. For short-horizon dates: optionally fetch the Open-Meteo 7/16-day
         forecast and use DailyClassifier on those features. If forecast
         fetching fails, fall back to climatology with a warning.
      4. For climatology dates: ClimatologyTable lookup
      5. Compute monthly summary: Σ predictions, P(high_risk_month) via
         Monte Carlo over per-day probabilities (Poisson-binomial estimate)

    Returns
    -------
    daily_df : DataFrame with columns
        city, date, day_of_month, probability, prediction, source
    monthly_df : DataFrame with columns
        city, target_month, risk_days_predicted,
        high_risk_month_probability, n_short_horizon_days, n_climatology_days
    """
    # Lazy imports
    try:
        from src.config import (
            CITIES,
            FORECAST_HORIZON_DAYS,
            HIGH_RISK_MONTH_THRESHOLD,
            FORECAST_VARIABLES,
        )
    except ImportError:
        CITIES = {}
        FORECAST_HORIZON_DAYS = 16
        HIGH_RISK_MONTH_THRESHOLD = 5
        FORECAST_VARIABLES = []

    forecast_horizon = forecast_horizon if forecast_horizon is not None else FORECAST_HORIZON_DAYS
    cities = cities or list(CITIES.keys())
    if not cities:
        raise ValueError("No cities to predict for")

    # Load both models
    model = load_model(model_path)
    if model is None:
        raise FileNotFoundError(
            f"No model at {model_path} — run train_model() first"
        )
    clim = load_model(climatology_path)
    if clim is None:
        raise FileNotFoundError(
            f"No climatology at {climatology_path} — run build_climatology() first"
        )
    
    # Support production model bundle
    if isinstance(model, dict):
        model_bundle = model
        model_object = model_bundle["model"]
        model_feature_cols = model_bundle.get("feature_cols", [])
        threshold = model_bundle.get("decision_threshold", threshold)
    else:
        # Legacy fallback, kept only so older pickles do not instantly break
        model_bundle = None
        model_object = model
        model_feature_cols = []

    # Plan the horizon
    target_month = _resolve_target_month(conn, target_month)
    target_dates = _all_dates_in_month(target_month)
    today = date.today()
    short_horizon_end = today + timedelta(days=forecast_horizon)

    # Try to fetch short-horizon forecast features (best-effort)
    forecast_df = _try_fetch_forecast(
        cities, CITIES, today, short_horizon_end, FORECAST_VARIABLES,
    )

    # Build forecast features using a 7-day lookback window.
    # This is required for lag/change/rolling features.
    forecast_features_df = None

    if forecast_df is not None and model_feature_cols:
        forecast_features_df = _build_forecast_features_with_lookback(
            conn=conn,
            forecast_df=forecast_df,
            feature_cols=model_feature_cols,
            lookback_days=7,
        )


    # Build the full prediction DataFrame
    rows = []
    for city in cities:
        for d in target_dates:
            day_of_year = d.timetuple().tm_yday

            # Decide source
            use_short_horizon = (
                forecast_df is not None
                and today <= d <= short_horizon_end
            )

            if use_short_horizon:
                if forecast_features_df is not None and len(forecast_features_df) > 0:
                    fc_row = forecast_features_df[
                        (forecast_features_df["city"] == city)
                        & (pd.to_datetime(forecast_features_df["date"]).dt.date == d)
                    ]
                else:
                    fc_row = pd.DataFrame()

                if len(fc_row) == 0:
                    # Forecast did not cover this day or features could not be built
                    p = clim.predict_proba(city, day_of_year)
                    src = "climatology"
                else:
                    fc_row = fc_row.copy()

                    if model_feature_cols:
                        for col in model_feature_cols:
                            if col not in fc_row.columns:
                                fc_row[col] = np.nan

                        X_row = fc_row[model_feature_cols].copy()
                        p = float(model_object.predict_proba(X_row)[:, 1][0])
                    else:
                        # Legacy fallback
                        if "month" not in fc_row.columns:
                            fc_row["month"] = d.month
                        p = float(model_object.predict_proba(fc_row)[:, 1][0])

                    src = "short_horizon"
            else:
                p = clim.predict_proba(city, day_of_year)
                src = "climatology"

            rows.append({
                "city":         city,
                "date":         d.isoformat(),
                "day_of_month": d.day,
                "probability":  round(p, 4),
                "prediction":   int(p >= threshold),
                "source":       src,
            })

    daily_df = pd.DataFrame(rows).sort_values(["city", "date"]).reset_index(drop=True)

    # Monthly summary
    monthly_df = _summarise_monthly(
        daily_df, target_month, threshold, n_monte_carlo, HIGH_RISK_MONTH_THRESHOLD,
    )

    logger.info(
        "  %d daily predictions for %s across %d cities "
        "(short_horizon=%d, climatology=%d)",
        len(daily_df), target_month, len(cities),
        int((daily_df["source"] == "short_horizon").sum()),
        int((daily_df["source"] == "climatology").sum()),
    )

    return daily_df, monthly_df


# ── Helpers used inside predict_next_month ───────────────────────────────────

def _try_fetch_forecast(
    cities:         list[str],
    cities_meta:    dict,
    start_date:     date,
    end_date:       date,
    variables:      list[str],
) -> Optional[pd.DataFrame]:
    """
    Attempt to fetch the Open-Meteo forecast for each city.
    Returns a combined DataFrame, or None if any fetch fails (in which case
    the caller falls back to climatology for the entire window).
    """
    try:
        from src.ingestion import fetch_forecast
    except ImportError:
        logger.warning(
            "  src.ingestion.fetch_forecast not importable — "
            "all dates will use climatology"
        )
        return None

    if not cities_meta:
        logger.warning("  No CITIES metadata — skipping forecast fetch")
        return None

    n_days = (end_date - start_date).days + 1
    pieces = []
    for city in cities:
        meta = cities_meta.get(city)
        if not meta:
            logger.warning("  No metadata for %s — skipping", city)
            continue
        try:
            df = fetch_forecast(
                city=city,
                lat=meta["lat"],
                lon=meta["lon"],
                variables=variables,
                forecast_days=min(n_days, 16),
                timezone=meta.get("timezone", "auto"),
            )
            if df is not None and not df.empty:
                pieces.append(df)
                logger.info("  Forecast fetched for %s (%d days)", city, len(df))
        except Exception as exc:
            logger.warning("  Forecast fetch FAILED for %s: %s", city, exc)
            return None   # one failure → climatology for everyone

    if not pieces:
        return None
    combined = pd.concat(pieces, ignore_index=True)
    return combined


def _summarise_monthly(
    daily_df:                  pd.DataFrame,
    target_month:              str,
    threshold:                 float,
    n_monte_carlo:             int,
    high_risk_month_threshold: int,
) -> pd.DataFrame:
    """
    Per-city monthly summary derived from daily predictions.

    risk_days_predicted          = count of days where prediction == 1
    high_risk_month_probability  = P(sum of bernoulli(p_i) ≥ threshold)
                                   estimated by Monte Carlo over the
                                   per-day probabilities (Poisson-binomial)
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    for city, sub in daily_df.groupby("city"):
        probs = sub["probability"].values
        risk_days_pred = int(sub["prediction"].sum())

        # Monte-Carlo: P(Σ Bernoulli(p_i) ≥ threshold)
        samples = rng.binomial(1, probs[None, :], size=(n_monte_carlo, len(probs)))
        sums = samples.sum(axis=1)
        p_high_risk = float((sums >= high_risk_month_threshold).mean())

        n_short = int((sub["source"] == "short_horizon").sum())
        n_clim  = int((sub["source"] == "climatology").sum())

        rows.append({
            "city":                         city,
            "target_month":                 target_month,
            "risk_days_predicted":          risk_days_pred,
            "high_risk_month_probability":  round(p_high_risk, 4),
            "n_short_horizon_days":         n_short,
            "n_climatology_days":           n_clim,
        })
    return pd.DataFrame(rows).sort_values("city").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save predictions
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(
    daily_df:     pd.DataFrame,
    monthly_df:   pd.DataFrame,
    target_month: Optional[str]   = None,
    out_dir:      str | Path      = "predictions",
) -> tuple[Path, Path]:
    """
    Write predictions to:
        <out_dir>/<target_month>/daily.csv
        <out_dir>/<target_month>/monthly.csv

    Returns (daily_path, monthly_path).
    """
    if target_month is None:
        if "target_month" in monthly_df.columns and len(monthly_df) > 0:
            target_month = str(monthly_df["target_month"].iloc[0])
        else:
            raise ValueError("target_month not provided and not derivable")

    out_dir = Path(out_dir) / target_month
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path   = out_dir / "daily.csv"
    monthly_path = out_dir / "monthly.csv"
    daily_df.to_csv(daily_path,   index=False)
    monthly_df.to_csv(monthly_path, index=False)

    logger.info("  Saved daily predictions   → %s", daily_path)
    logger.info("  Saved monthly summary     → %s", monthly_path)
    return daily_path, monthly_path
