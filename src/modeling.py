"""
src/modeling.py
───────────────
Model training + prediction for the Caspian Maritime Delay-Risk project.

**Day 5 version: PLACEHOLDER implementation.**

The functions have the correct signatures and produce the correct output
shape — but the model itself is a trivial baseline (predicts the long-run
per-city positive-class rate). Day 6 will replace the internals with a
real XGBoost / Random Forest classifier, keeping the same signatures so
the pipeline doesn't need to change.

Public API
----------
    train_model(conn, model_path)                → dict   (metrics)
    predict_next_month(conn, model_path, month)  → pd.DataFrame (predictions)
    load_model(model_path)                       → object (pickled model)

The pipeline calls these in sequence:
    metrics = train_model(conn, "models/latest.pkl")
    preds   = predict_next_month(conn, "models/latest.pkl", next_month)
    preds.to_csv("predictions/YYYY-MM.csv")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Placeholder model class
# ══════════════════════════════════════════════════════════════════════════════

class BaselinePredictor:
    """
    Per-city base-rate predictor: predicts the long-run fraction of
    high-risk months observed in training.

    A real model (XGBoost / RandomForest) will replace this class in Day 6.
    The interface is the same: .fit(X, y), .predict_proba(X).
    """

    def __init__(self):
        self.city_rates: dict[str, float] = {}
        self.global_rate: float = 0.5
        self.trained_on_rows: int = 0
        self.trained_on_months: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselinePredictor":
        """Learn per-city positive-class rate from the training set."""
        if "city" not in X.columns:
            raise ValueError("X must contain a 'city' column")

        df = X[["city"]].copy()
        df["_y"] = y.values
        self.city_rates = df.groupby("city")["_y"].mean().to_dict()
        self.global_rate = float(df["_y"].mean())
        self.trained_on_rows = len(df)
        self.trained_on_months = len(df)

        logger.info("  BaselinePredictor fit: %d months across %d cities",
                    self.trained_on_rows, len(self.city_rates))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row probability of class=1."""
        if "city" not in X.columns:
            raise ValueError("X must contain a 'city' column")
        probs = X["city"].map(self.city_rates).fillna(self.global_rate).values
        # sklearn convention: shape (n, 2) with [P(0), P(1)]
        return np.column_stack([1 - probs, probs])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Train
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    conn,
    model_path:      str | Path = "models/latest.pkl",
    training_table:  str        = "analytics.monthly_summary",
) -> dict:
    """
    Train a classifier on analytics.monthly_summary → save to disk.

    Returns a metrics dict:
        {"rows_trained", "cities", "positive_rate", "city_rates",
         "model_path", "model_type"}
    """
    logger.info("Training model on %s ...", training_table)

    # Pull the monthly training data
    df = conn.execute(f"""
        SELECT * FROM {training_table} ORDER BY city, year, month
    """).fetchdf()

    if "high_risk_month" not in df.columns:
        raise ValueError(
            f"{training_table} is missing the 'high_risk_month' target column"
        )

    if len(df) == 0:
        raise ValueError(f"{training_table} is empty — cannot train")

    # Split features and target (we keep 'city' as a feature for the baseline)
    y = df["high_risk_month"].astype(int)
    X = df.drop(columns=["high_risk_month"])

    # Fit
    model = BaselinePredictor().fit(X, y)

    # Persist
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("  Saved model → %s", model_path)

    metrics = {
        "rows_trained":  int(model.trained_on_rows),
        "cities":        len(model.city_rates),
        "positive_rate": round(model.global_rate, 4),
        "city_rates":    {k: round(v, 4) for k, v in model.city_rates.items()},
        "model_path":    str(model_path),
        "model_type":    "BaselinePredictor (Day-5 stub — replace in Day 6)",
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3. Load
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str | Path = "models/latest.pkl"):
    """Load a pickled model. Returns None if file missing."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Predict
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_month(
    conn,
    model_path: str | Path  = "models/latest.pkl",
    target_month: Optional[str] = None,
    feature_table: str      = "analytics.monthly_summary",
) -> pd.DataFrame:
    """
    Generate one prediction per city for `target_month`.

    Since we don't have truly forward-looking features yet, we use the
    most recent month's features as a proxy for "next month" — Day 6/8
    will replace this with proper forecast-feature handling.

    Parameters
    ----------
    target_month : 'YYYY-MM' string; defaults to the calendar month AFTER
                   the latest month in `feature_table`.

    Returns
    -------
    pd.DataFrame with columns: city, target_month, probability, prediction
    """
    model = load_model(model_path)
    if model is None:
        raise FileNotFoundError(
            f"No model at {model_path} — run train_model() first"
        )

    # Pull each city's most recent month as the feature row
    latest = conn.execute(f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY city ORDER BY year DESC, month DESC
                   ) AS rn
            FROM {feature_table}
        )
        SELECT * FROM ranked WHERE rn = 1
    """).fetchdf()
    if "rn" in latest.columns:
        latest = latest.drop(columns=["rn"])

    if len(latest) == 0:
        raise ValueError(f"{feature_table} is empty — cannot predict")

    # Determine target month
    if target_month is None:
        latest_year  = int(latest["year"].max())
        latest_month = int(latest[latest["year"] == latest_year]["month"].max())
        # Add 1 month (rolling over December)
        if latest_month == 12:
            tgt_year, tgt_month = latest_year + 1, 1
        else:
            tgt_year, tgt_month = latest_year, latest_month + 1
        target_month = f"{tgt_year:04d}-{tgt_month:02d}"

    # Predict
    proba = model.predict_proba(latest)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "city":         latest["city"].values,
        "target_month": target_month,
        "probability":  np.round(proba, 4),
        "prediction":   pred,
    }).sort_values("city").reset_index(drop=True)

    logger.info(
        "  Generated predictions for %s across %d cities",
        target_month, len(out),
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save predictions to CSV (convenience)
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(preds: pd.DataFrame, out_dir: str | Path = "predictions") -> Path:
    """
    Save predictions to `<out_dir>/YYYY-MM.csv`.
    Returns the written path.
    """
    target_month = preds["target_month"].iloc[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_month}.csv"
    preds.to_csv(out_path, index=False)
    logger.info("  Saved predictions → %s", out_path)
    return out_path
