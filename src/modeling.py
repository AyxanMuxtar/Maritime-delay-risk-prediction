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
    preferred_model: Optional[str] = None,
) -> dict:
    """
    Train, evaluate, select, and save the production next-day model.

    If preferred_model is provided, this function still evaluates all candidates
    for reporting, but it retrains/saves that specific model family. This lets
    Day 5/pipeline retrain the Day 8-selected winner instead of running a
    separate model-selection contest.

    IMPORTANT SINGLE-WORKFLOW DESIGN
    --------------------------------
    This function is the only source of truth for production model selection.
    The pipeline calls this function, and Day 8 should call this function too.

    It writes BOTH:
      - models/daily_model.pkl
      - reports/day08_model_comparison.csv

    So Day 5, Day 8, and the pipeline all read the same production artifact.
    """
    logger.info("Training/evaluating production model on %s ...", feature_table)

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        brier_score_loss,
    )

    try:
        from src.config import PATHS
        reports_dir = Path(PATHS["repo_root"]) / "reports"
    except Exception:
        reports_dir = Path(model_path).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = reports_dir / "day08_model_comparison.csv"

    df = conn.execute(f"""
        SELECT *
        FROM {feature_table}
        ORDER BY city, date
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{feature_table} is empty — cannot train")

    if "is_risk_day" not in df.columns:
        raise ValueError(
            f"{feature_table} is missing 'is_risk_day'. Build analytics first."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Production target: today's features predict tomorrow's risk.
    target = "target_risk_next_day"
    df[target] = df.groupby("city")["is_risk_day"].shift(-1)

    df_model = df.dropna(subset=[target]).copy()
    df_model[target] = df_model[target].astype(int)

    # Do NOT feed direct same-day threshold variables into the model.
    # Those variables are used to define is_risk_day and would cause leakage.
    leakage_cols = {
        "is_risk_day", "target_risk_next_day", "high_risk_month",
        "risk_days", "risk_day_pct",
        "wind_speed_10m_max", "wind_gusts_10m_max", "precipitation_sum",
        "rain_sum", "snowfall_sum", "visibility_mean", "visibility_min",
        "visibility_hours_below_1km", "wave_height",
        "risk_wind", "risk_gust", "risk_precip", "risk_snow",
        "risk_wave", "risk_visibility", "risk_fog_min", "risk_fog_proxy",
    }

    maritime_context_candidates = [
        "wave_height_lag1", "wave_height_lag2",
        "wave_height_7d_mean", "wave_height_7d_max",
        "wave_height_30d_mean", "wave_height_30d_max",
        "wave_height_anom",
    ]

    try:
        from reports.selected_features import SELECTED_FEATURES
        feature_cols = [
            c for c in SELECTED_FEATURES
            if c in df_model.columns and c not in leakage_cols
        ]
        logger.info("  Loaded %d Day-7 selected leakage-safe features", len(feature_cols))
    except Exception as exc:
        logger.warning(
            "Could not load reports.selected_features.SELECTED_FEATURES: %s", exc
        )
        excluded = leakage_cols | {"date"}
        feature_cols = [c for c in df_model.columns if c not in excluded]

    added_maritime = []
    for c in maritime_context_candidates:
        if c in df_model.columns and c not in leakage_cols and c not in feature_cols:
            feature_cols.append(c)
            added_maritime.append(c)

    if added_maritime:
        logger.info("  Added maritime context features: %s", added_maritime)

    if not feature_cols:
        raise ValueError("No usable feature columns found for training")

    X = df_model[feature_cols].copy()
    y = df_model[target].astype(int).values

    categorical_features = [c for c in ["city", "season"] if c in feature_cols]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    def _make_preprocess(scale_numeric: bool = True) -> ColumnTransformer:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))

        numeric_transformer = Pipeline(steps=num_steps)
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

    def _pipeline(base_estimator, scale_numeric: bool = True):
        return Pipeline(steps=[
            ("preprocess", _make_preprocess(scale_numeric=scale_numeric)),
            ("model", base_estimator),
        ])

    def _calibrated_pipeline(base_estimator, scale_numeric: bool = True):
        base_pipeline = _pipeline(base_estimator, scale_numeric=scale_numeric)
        try:
            return CalibratedClassifierCV(
                estimator=base_pipeline,
                method="sigmoid",
                cv=3,
            )
        except TypeError:
            return CalibratedClassifierCV(
                base_estimator=base_pipeline,
                method="sigmoid",
                cv=3,
            )

    def _tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> dict:
        thresholds = np.arange(0.01, 0.81, 0.01)
        rows = []
        for t in thresholds:
            pred = (probs >= t).astype(int)
            rows.append({
                "threshold": float(round(t, 2)),
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "predicted_positive_rate": float(pred.mean()),
            })
        return sorted(
            rows,
            key=lambda r: (
                r["f1"],
                r["recall"],
                r["precision"],
                -abs(r["threshold"] - decision_threshold),
            ),
            reverse=True,
        )[0]

    def _safe_auc(y_true, probs) -> float:
        try:
            return float(roc_auc_score(y_true, probs))
        except Exception:
            return float("nan")

    def _safe_brier(y_true, probs) -> float:
        try:
            return float(brier_score_loss(y_true, probs))
        except Exception:
            return float("nan")

    def _adjust_validation_probs(port_probs: np.ndarray, rows_df: pd.DataFrame) -> np.ndarray:
        adjusted = []
        for p_port, (_, hist_row) in zip(port_probs, rows_df.iterrows()):
            try:
                offshore_info = _offshore_risk_from_row(hist_row)
                p_sea = float(offshore_info.get("offshore_sea_probability", 0.0) or 0.0)
                adjusted.append(
                    _adjust_maritime_probability(
                        port_weather_probability=float(p_port),
                        offshore_sea_probability=p_sea,
                    )
                )
            except Exception:
                adjusted.append(float(p_port))
        return np.asarray(adjusted, dtype=float)

    # Use calendar-year 2024 for the evaluation report if available, because
    # this is what Day 8 historically reported. If 2024 is unavailable, use a
    # temporal last-20% split.
    years = pd.to_datetime(df_model["date"]).dt.year
    if (years == 2024).sum() >= 50 and len(np.unique(y[(years == 2024).values])) >= 2:
        train_mask = years < 2024
        val_mask = years == 2024
        validation_note = "2024 holdout evaluation set"
    else:
        unique_dates = np.array(sorted(pd.to_datetime(df_model["date"]).dt.normalize().unique()))
        split_idx = int(len(unique_dates) * 0.80)
        split_idx = max(1, min(split_idx, len(unique_dates) - 1))
        split_date = unique_dates[split_idx]
        train_mask = pd.to_datetime(df_model["date"]).dt.normalize() < split_date
        val_mask = ~train_mask
        validation_note = f"temporal holdout from {pd.Timestamp(split_date).date()} onward"

    # Defensive fallback if the temporal split does not contain both classes.
    if (
        train_mask.sum() < 100 or val_mask.sum() < 50 or
        len(np.unique(y[train_mask.values])) < 2 or
        len(np.unique(y[val_mask.values])) < 2
    ):
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df_model))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
        train_mask = pd.Series(False, index=df_model.index)
        val_mask = pd.Series(False, index=df_model.index)
        train_mask.iloc[train_idx] = True
        val_mask.iloc[val_idx] = True
        validation_note = "stratified random holdout evaluation set"

    X_train = X.loc[train_mask].copy()
    y_train = y[train_mask.values]
    X_val = X.loc[val_mask].copy()
    y_val = y[val_mask.values]
    df_train = df_model.loc[train_mask].copy()
    df_val = df_model.loc[val_mask].copy()

    logger.info(
        "  Model selection/evaluation using %s: train=%d, eval=%d, eval positive rate=%.3f",
        validation_note, len(X_train), len(X_val), float(y_val.mean()),
    )

    selection_rows = []
    production_candidates = {}

    # Baseline: per-(city, day-of-year) historical next-day risk rate.
    try:
        clim_train = df_train[["city", "date", target]].copy()
        clim_train = clim_train.rename(columns={target: "is_risk_day"})
        clim = ClimatologyTable(smoothing_window=7).fit(clim_train)
        p_clim = clim.predict_proba_df(df_val[["city", "date"]].copy())
        best_t = _tune_threshold(y_val, p_clim)
        selection_rows.append({
            "Model": "Climatology",
            "Threshold": best_t["threshold"],
            "Accuracy": round(best_t["accuracy"], 3),
            "Precision": round(best_t["precision"], 3),
            "Recall": round(best_t["recall"], 3),
            "F1": round(best_t["f1"], 3),
            "ROC-AUC": round(_safe_auc(y_val, p_clim), 3),
            "Brier": round(_safe_brier(y_val, p_clim), 3),
            "Predicted positive rate": round(best_t["predicted_positive_rate"], 3),
            "Eligible": False,
            "Selected": False,
            "Production score": np.nan,
            "Notes": "Baseline: per-(city,doy) historical next-day risk rate | threshold tuned for F1",
        })
    except Exception as exc:
        logger.warning("  Climatology baseline evaluation failed: %s", exc)

    model_specs = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver="lbfgs"),
            True,
            False,
            "Linear; interpretable coefficients",
        ),
        (
            "LogisticRegression_calibrated",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver="lbfgs"),
            True,
            True,
            "Logistic Regression with sigmoid probability calibration",
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=4,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            ),
            False,
            False,
            "Bagged decision trees; non-linear interactions",
        ),
        (
            "RandomForest_calibrated",
            RandomForestClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=4,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            ),
            False,
            True,
            "RandomForest with sigmoid probability calibration",
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=4,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            False,
            False,
            "Extremely randomized trees; non-linear baseline",
        ),
        (
            "ExtraTrees_calibrated",
            ExtraTreesClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=4,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            False,
            True,
            "ExtraTrees with sigmoid probability calibration",
        ),
    ]

    # Optional XGBoost if installed. Do not fail the pipeline if unavailable.
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model_specs.extend([
            ("XGBoost", xgb, False, False, "Tree boosting model; non-linear; regularised"),
            ("XGBoost_calibrated", clone(xgb), False, True, "Boosting model with sigmoid probability calibration"),
        ])
    except Exception:
        pass

    # Fast production path used by Day 5 / pipeline:
    # Day 8 has already evaluated candidates and written the selected winner to
    # reports/day08_model_comparison.csv. In that case we do NOT run a second
    # model-selection contest here. We only retrain the Day-8 selected model
    # family on the latest full analytics table, then save daily_model.pkl.
    spec_by_name = {
        name: {
            "estimator": estimator,
            "scale_numeric": scale_numeric,
            "calibrate": calibrate,
            "note": note,
        }
        for name, estimator, scale_numeric, calibrate, note in model_specs
    }

    if preferred_model:
        preferred_model = str(preferred_model)
        if preferred_model not in spec_by_name:
            available = ", ".join(sorted(spec_by_name))
            raise ValueError(
                f"Day 8 selected {preferred_model!r}, but that model family is not available in src.modeling. "
                f"Available candidates: {available}"
            )

        comparison_df = None
        selected_report_row = None
        selected_threshold = float(decision_threshold)

        if comparison_path.exists():
            try:
                comparison_df = pd.read_csv(comparison_path)
                if "Model" in comparison_df.columns:
                    matches = comparison_df[comparison_df["Model"].astype(str) == preferred_model]
                    if matches.empty:
                        norm_pref = preferred_model.lower().replace(" ", "").replace("-", "_")
                        norm_col = (
                            comparison_df["Model"].astype(str)
                            .str.lower()
                            .str.replace(" ", "", regex=False)
                            .str.replace("-", "_", regex=False)
                        )
                        matches = comparison_df[norm_col == norm_pref]
                    if not matches.empty:
                        selected_report_row = matches.iloc[0]
                        if "Threshold" in selected_report_row and pd.notna(selected_report_row["Threshold"]):
                            selected_threshold = float(selected_report_row["Threshold"])

                        # Repair the report if Selected=True was missing, so Day 5 summaries stay consistent.
                        if "Selected" not in comparison_df.columns:
                            comparison_df["Selected"] = False
                        comparison_df["Selected"] = False
                        comparison_df.loc[matches.index[0], "Selected"] = True
                        comparison_df.to_csv(comparison_path, index=False)
            except Exception as exc:
                logger.warning("  Could not read/repair Day 8 comparison report %s: %s", comparison_path, exc)

        spec = spec_by_name[preferred_model]
        if bool(spec["calibrate"]):
            production_model = _calibrated_pipeline(
                clone(spec["estimator"]),
                scale_numeric=bool(spec["scale_numeric"]),
            )
        else:
            production_model = _pipeline(
                clone(spec["estimator"]),
                scale_numeric=bool(spec["scale_numeric"]),
            )

        logger.info(
            "  Retraining Day-8 selected production model only: %s  threshold=%.2f",
            preferred_model,
            selected_threshold,
        )
        production_model.fit(X, y)

        def _row_float(key: str, default=float("nan")) -> float:
            try:
                if selected_report_row is not None and key in selected_report_row and pd.notna(selected_report_row[key]):
                    return float(selected_report_row[key])
            except Exception:
                pass
            return default

        model_bundle = {
            "model": production_model,
            "model_name": preferred_model,
            "base_model_name": preferred_model.replace("_calibrated", ""),
            "is_calibrated": preferred_model.endswith("_calibrated"),
            "calibration_method": "sigmoid" if preferred_model.endswith("_calibrated") else "none",
            "maritime_probability_calibrator": None,
            "maritime_calibration": {
                "enabled": False,
                "method": None,
                "note": "Disabled: final displayed score uses raw maritime adjustment; classification threshold is tuned separately.",
            },
            "maritime_adjustment_weight": OFFSHORE_ADJUSTMENT_WEIGHT,
            "decision_threshold": selected_threshold,
            "threshold_tuning": {
                "threshold": selected_threshold,
                "accuracy": _row_float("Accuracy"),
                "precision": _row_float("Precision"),
                "recall": _row_float("Recall"),
                "f1": _row_float("F1"),
                "roc_auc": _row_float("ROC-AUC"),
                "brier": _row_float("Brier"),
            },
            "model_selection_policy": (
                "Day 8 selects the winning algorithm and writes reports/day08_model_comparison.csv. "
                "Day 5/pipeline retrains that same selected algorithm on the latest full analytics data."
            ),
            "comparison_report_path": str(comparison_path),
            "selection_source": "day8_selected_model",
            "target": target,
            "feature_cols": feature_cols,
            "trained_rows": int(len(df_model)),
            "positive_rate": float(y.mean()),
            "description": (
                "Production model retrained from the Day-8 selected model family; "
                "features at day t -> P(risk on day t+1)."
            ),
        }

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as f:
            pickle.dump(model_bundle, f)

        logger.info("  Saved Day-8 selected production model → %s", model_path)
        logger.info("  Feature count: %d", len(feature_cols))
        logger.info("  Positive rate: %.4f", float(y.mean()))

        return {
            "rows_trained": int(len(df_model)),
            "n_features": int(len(feature_cols)),
            "positive_rate": round(float(y.mean()), 4),
            "model_path": str(model_path),
            "model_type": preferred_model,
            "target": target,
            "decision_threshold": selected_threshold,
            "is_calibrated": preferred_model.endswith("_calibrated"),
            "calibration_method": "sigmoid" if preferred_model.endswith("_calibrated") else "none",
            "comparison_report_path": str(comparison_path),
            "selection_source": "day8_selected_model",
            "validation_f1": round(_row_float("F1"), 4) if not pd.isna(_row_float("F1")) else None,
            "validation_precision": round(_row_float("Precision"), 4) if not pd.isna(_row_float("Precision")) else None,
            "validation_recall": round(_row_float("Recall"), 4) if not pd.isna(_row_float("Recall")) else None,
            "validation_roc_auc": round(_row_float("ROC-AUC"), 4) if not pd.isna(_row_float("ROC-AUC")) else None,
            "validation_brier": round(_row_float("Brier"), 4) if not pd.isna(_row_float("Brier")) else None,
            "added_maritime_features": added_maritime,
        }

    for name, estimator, scale_numeric, calibrate, note in model_specs:
        try:
            if calibrate:
                candidate_model = _calibrated_pipeline(clone(estimator), scale_numeric=scale_numeric)
            else:
                candidate_model = _pipeline(clone(estimator), scale_numeric=scale_numeric)

            candidate_model.fit(X_train, y_train)
            port_probs = candidate_model.predict_proba(X_val)[:, 1]
            probs = _adjust_validation_probs(port_probs, df_val)
            best_t = _tune_threshold(y_val, probs)

            auc = _safe_auc(y_val, probs)
            brier = _safe_brier(y_val, probs)

            row = {
                "Model": name,
                "Threshold": best_t["threshold"],
                "Accuracy": round(best_t["accuracy"], 3),
                "Precision": round(best_t["precision"], 3),
                "Recall": round(best_t["recall"], 3),
                "F1": round(best_t["f1"], 3),
                "ROC-AUC": round(auc, 3),
                "Brier": round(brier, 3),
                "Predicted positive rate": round(best_t["predicted_positive_rate"], 3),
                "Eligible": False,
                "Selected": False,
                "Production score": np.nan,
                "Notes": f"{note} | threshold tuned for F1",
                "_estimator": estimator,
                "_scale_numeric": scale_numeric,
                "_calibrate": calibrate,
                "_raw_f1": float(best_t["f1"]),
                "_raw_recall": float(best_t["recall"]),
                "_raw_auc": float(auc) if not np.isnan(auc) else np.nan,
                "_raw_brier": float(brier) if not np.isnan(brier) else np.nan,
            }
            selection_rows.append(row)
            production_candidates[name] = row

            logger.info(
                "  Candidate %-28s F1=%.3f  P=%.3f  R=%.3f  AUC=%.3f  Brier=%.3f  threshold=%.2f",
                name, row["F1"], row["Precision"], row["Recall"], row["ROC-AUC"], row["Brier"], row["Threshold"],
            )
        except Exception as exc:
            logger.warning("  Candidate %s failed: %s", name, exc)

    if not production_candidates:
        raise RuntimeError("All production model candidates failed")

    comparison_df = pd.DataFrame(selection_rows)

    if "Climatology" in set(comparison_df["Model"]):
        baseline = comparison_df[comparison_df["Model"] == "Climatology"].iloc[0]
        baseline_f1 = float(baseline["F1"])
        baseline_auc = float(baseline["ROC-AUC"])
        baseline_brier = float(baseline["Brier"])
    else:
        baseline_f1 = 0.0
        baseline_auc = 0.0
        baseline_brier = 1.0

    # Production eligibility: must improve over climatology as a probability
    # forecast and not collapse recall. This prevents the bad situation where a
    # model with poor Brier or tiny recall wins because of a single metric.
    production_rows = comparison_df[comparison_df["Model"] != "Climatology"].copy()
    production_rows["Eligible"] = (
        (production_rows["F1"].astype(float) >= baseline_f1) &
        (production_rows["ROC-AUC"].astype(float) >= baseline_auc) &
        (production_rows["Brier"].astype(float) <= baseline_brier + 0.005) &
        (production_rows["Recall"].astype(float) >= 0.25)
    )

    # Composite production score: detection + recall + ranking + probability quality.
    production_rows["Production score"] = (
        0.35 * production_rows["F1"].astype(float) +
        0.20 * production_rows["Recall"].astype(float) +
        0.20 * production_rows["ROC-AUC"].astype(float) +
        0.25 * (1.0 - production_rows["Brier"].astype(float))
    )

    eligible = production_rows[production_rows["Eligible"]].copy()
    if eligible.empty:
        logger.warning(
            "  No candidate passed baseline/Brier/recall guardrails; falling back to highest production score."
        )
        eligible = production_rows.copy()

    selection_source = "internal_production_score"

    if preferred_model:
        preferred_model = str(preferred_model)
        if preferred_model not in production_candidates:
            logger.warning(
                "  Preferred model from Day 8 (%s) is not available in current candidates; "
                "falling back to internal production score.",
                preferred_model,
            )
            winner_row = eligible.sort_values(
                by=["Production score", "Brier", "F1", "Recall"],
                ascending=[False, True, False, False],
            ).iloc[0]
        else:
            winner_matches = production_rows[production_rows["Model"].astype(str) == preferred_model]
            if winner_matches.empty:
                raise RuntimeError(f"Preferred model {preferred_model!r} was found in candidates but not in report rows")
            winner_row = winner_matches.iloc[0]
            selection_source = "day8_selected_model"
            logger.info("  Using Day 8-selected model for production retraining: %s", preferred_model)
    else:
        winner_row = eligible.sort_values(
            by=["Production score", "Brier", "F1", "Recall"],
            ascending=[False, True, False, False],
        ).iloc[0]

    selected_model_name = str(winner_row["Model"])
    selected_threshold = float(winner_row["Threshold"])
    selected_spec = production_candidates[selected_model_name]

    logger.info(
        "  Selected production model: %s  threshold=%.2f  F1=%.3f  Recall=%.3f  AUC=%.3f  Brier=%.3f  source=%s",
        selected_model_name,
        selected_threshold,
        float(winner_row["F1"]),
        float(winner_row["Recall"]),
        float(winner_row["ROC-AUC"]),
        float(winner_row["Brier"]),
        selection_source,
    )

    # Fit selected model on ALL available rows for the actual production pickle.
    if bool(selected_spec["_calibrate"]):
        production_model = _calibrated_pipeline(
            clone(selected_spec["_estimator"]),
            scale_numeric=bool(selected_spec["_scale_numeric"]),
        )
    else:
        production_model = _pipeline(
            clone(selected_spec["_estimator"]),
            scale_numeric=bool(selected_spec["_scale_numeric"]),
        )
    production_model.fit(X, y)

    # Clean report columns and mark selected row.
    comparison_df = comparison_df.drop(columns=[c for c in comparison_df.columns if c.startswith("_")], errors="ignore")
    comparison_df.loc[comparison_df["Model"].isin(production_rows[production_rows["Eligible"]]["Model"]), "Eligible"] = True
    comparison_df.loc[comparison_df["Model"] == selected_model_name, "Selected"] = True
    for _, pr in production_rows.iterrows():
        comparison_df.loc[comparison_df["Model"] == pr["Model"], "Production score"] = round(float(pr["Production score"]), 4)

    comparison_df = comparison_df.sort_values(
        by=["Selected", "Eligible", "Production score", "F1"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("  Saved model comparison report → %s", comparison_path)

    model_bundle = {
        "model": production_model,
        "model_name": selected_model_name,
        "base_model_name": selected_model_name.replace("_calibrated", ""),
        "is_calibrated": selected_model_name.endswith("_calibrated"),
        "calibration_method": "sigmoid" if selected_model_name.endswith("_calibrated") else "none",
        "maritime_probability_calibrator": None,
        "maritime_calibration": {
            "enabled": False,
            "method": None,
            "note": "Disabled: final displayed score uses raw maritime adjustment; classification threshold is tuned separately.",
        },
        "maritime_adjustment_weight": OFFSHORE_ADJUSTMENT_WEIGHT,
        "decision_threshold": selected_threshold,
        "threshold_tuning": {
            "threshold": selected_threshold,
            "accuracy": float(winner_row["Accuracy"]),
            "precision": float(winner_row["Precision"]),
            "recall": float(winner_row["Recall"]),
            "f1": float(winner_row["F1"]),
            "roc_auc": float(winner_row["ROC-AUC"]),
            "brier": float(winner_row["Brier"]),
        },
        "model_selection_policy": (
            "Day 8 selects the winning algorithm. If preferred_model is passed from the Day 8 report, "
            "the pipeline retrains that same algorithm on the latest data; otherwise train_model() falls back "
            "to its internal weighted production score."
        ),
        "model_selection_source": selection_source,
        "preferred_model": preferred_model,
        "model_selection": comparison_df.to_dict(orient="records"),
        "comparison_report_path": str(comparison_path),
        "validation_note": validation_note,
        "target": target,
        "feature_cols": feature_cols,
        "trained_rows": int(len(df_model)),
        "positive_rate": float(y.mean()),
        "description": (
            "Tuned next-day maritime delay-risk model selected by the same train_model() "
            "workflow used by Day 8 and the production pipeline."
        ),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    logger.info("  Saved production model → %s", model_path)
    logger.info("  Feature count: %d", len(feature_cols))
    logger.info("  Positive rate: %.4f", float(y.mean()))

    return {
        "rows_trained": int(len(df_model)),
        "n_features": int(len(feature_cols)),
        "positive_rate": round(float(y.mean()), 4),
        "model_path": str(model_path),
        "model_type": selected_model_name,
        "target": target,
        "decision_threshold": selected_threshold,
        "is_calibrated": selected_model_name.endswith("_calibrated"),
        "calibration_method": "sigmoid" if selected_model_name.endswith("_calibrated") else "none",
        "validation_note": validation_note,
        "comparison_report_path": str(comparison_path),
        "validation_f1": round(float(winner_row["F1"]), 4),
        "validation_precision": round(float(winner_row["Precision"]), 4),
        "validation_recall": round(float(winner_row["Recall"]), 4),
        "validation_roc_auc": round(float(winner_row["ROC-AUC"]), 4),
        "validation_brier": round(float(winner_row["Brier"]), 4),
        "added_maritime_features": added_maritime,
        "model_selection_source": selection_source,
        "preferred_model": preferred_model,
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


def _rolling_dates(start_date: Optional[date] = None, days_ahead: int = 30) -> list[date]:
    """
    Return a rolling inclusive date window.

    Example:
        start_date = May 1, days_ahead = 30
        returns May 1 → May 31 inclusive

        start_date = May 2, days_ahead = 30
        returns May 2 → June 1 inclusive
    """
    if start_date is None:
        start_date = date.today()

    end_date = start_date + timedelta(days=days_ahead)
    return [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]


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
# 4b. Offshore sea-state adjustment helpers
# ══════════════════════════════════════════════════════════════════════════════

OFFSHORE_ADJUSTMENT_WEIGHT = 0.45


def _safe_float(row, col: str) -> Optional[float]:
    """Read a numeric value safely from a Series/dict-like row."""
    try:
        if row is None:
            return None
        value = row.get(col, None)
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _threshold_score(value: Optional[float], threshold: float, cap: float = 0.85) -> float:
    """
    Convert a physical sea-state value into a probability-like risk score.

    The score rises smoothly around the operational threshold instead of
    switching abruptly from 0 to 1. This makes the offshore adjustment useful
    even when waves are close to, but not above, the formal cutoff.
    """
    if value is None or threshold <= 0:
        return 0.0

    ratio = max(float(value), 0.0) / threshold
    score = cap / (1.0 + np.exp(-6.0 * (ratio - 1.0)))
    return float(np.clip(score, 0.0, cap))


def _offshore_risk_from_row(row) -> dict:
    """
    Estimate offshore sea-state risk from marine forecast/proxy columns.

    Uses Caspian operational thresholds:
      - significant wave height around 2.5m is high risk
      - wind-wave and swell components add supporting evidence
    """
    wave = (
        _safe_float(row, "wave_height_max")
        or _safe_float(row, "wave_height")
        or _safe_float(row, "wave_height_estimated")
    )
    wind_wave = _safe_float(row, "wind_wave_height_max")
    swell = _safe_float(row, "swell_wave_height_max")
    period = _safe_float(row, "wave_period_max") or _safe_float(row, "swell_wave_period_max")

    candidates = []
    if wave is not None:
        candidates.append(("offshore wave height", _threshold_score(wave, 2.5, cap=0.85), wave))
    if wind_wave is not None:
        candidates.append(("wind-wave height", _threshold_score(wind_wave, 2.0, cap=0.75), wind_wave))
    if swell is not None:
        candidates.append(("swell height", _threshold_score(swell, 1.5, cap=0.65), swell))

    if not candidates:
        return {
            "offshore_sea_probability": 0.0,
            "offshore_driver": "offshore sea-state data unavailable",
            "offshore_wave_height_m": np.nan,
            "offshore_source": "not_available",
        }

    driver, base_score, driver_value = max(candidates, key=lambda x: x[1])

    # Longer-period waves can be operationally uncomfortable even when height is
    # moderate, but they should not dominate the score alone.
    period_bonus = 0.0
    if period is not None and period >= 6.0 and wave is not None and wave >= 1.0:
        period_bonus = 0.08

    offshore_prob = float(np.clip(base_score + period_bonus, 0.0, 0.90))

    if wave is not None:
        wave_out = round(float(wave), 2)
    else:
        wave_out = np.nan

    return {
        "offshore_sea_probability": round(offshore_prob, 4),
        "offshore_driver": driver,
        "offshore_driver_value": round(float(driver_value), 2),
        "offshore_wave_height_m": wave_out,
        "offshore_source": "marine_forecast",
    }


def _adjust_maritime_probability(port_weather_probability: float,
                                 offshore_sea_probability: float,
                                 offshore_weight: float = OFFSHORE_ADJUSTMENT_WEIGHT) -> float:
    """
    Combine port-weather risk and offshore sea-state risk as separate failure modes.

    We weight the offshore component so that it adjusts the ML risk rather than
    replacing it. This keeps the model probability central while surfacing a
    maritime-specific hazard that a normal weather forecast does not provide.
    """
    p_port = float(np.clip(port_weather_probability, 0.0, 1.0))
    p_sea = float(np.clip(offshore_sea_probability, 0.0, 1.0))
    effective_sea = float(np.clip(offshore_weight * p_sea, 0.0, 1.0))
    return float(np.clip(1.0 - (1.0 - p_port) * (1.0 - effective_sea), 0.0, 0.95))



def _calibrate_maritime_probability(probability: float, model_bundle: Optional[dict]) -> float:
    """Apply the saved calibration mapping to the final adjusted maritime score."""
    p = float(np.clip(probability, 0.0, 1.0))

    if not isinstance(model_bundle, dict):
        return p

    calibrator = model_bundle.get("maritime_probability_calibrator")
    if calibrator is None:
        return p

    try:
        calibrated = float(calibrator.predict(np.array([p], dtype=float))[0])
        return float(np.clip(calibrated, 0.0, 0.95))
    except Exception as exc:
        logger.warning("  Maritime probability calibration failed at prediction time: %s", exc)
        return p


def _try_fetch_offshore_marine_forecast(
    cities: list[str],
    cities_meta: dict,
    forecast_days: int = 7,
) -> dict[tuple[str, str], dict]:
    """
    Fetch live offshore marine forecast for each city using its offshore point.

    Returns a lookup keyed by (city, 'YYYY-MM-DD'). Failures are non-fatal; the
    caller can fall back to historical wave climatology.
    """
    try:
        from src.era5_client import fetch_marine_forecast, MARINE_DAILY_VARIABLES
    except Exception as exc:
        logger.warning("  Offshore marine forecast unavailable: %s", exc)
        return {}

    lookup: dict[tuple[str, str], dict] = {}
    forecast_days = int(max(1, min(forecast_days, 7)))

    for city in cities:
        meta = cities_meta.get(city, {}) if cities_meta else {}
        offshore = meta.get("offshore", {}) if isinstance(meta, dict) else {}
        lat = offshore.get("lat", meta.get("lat"))
        lon = offshore.get("lon", meta.get("lon"))
        timezone = meta.get("timezone", "auto")

        if lat is None or lon is None:
            logger.warning("  No offshore coordinates for %s — skipping marine forecast", city)
            continue

        try:
            marine_df = fetch_marine_forecast(
                lat=lat,
                lon=lon,
                variables=MARINE_DAILY_VARIABLES,
                forecast_days=forecast_days,
                timezone=timezone,
            )

            if marine_df is None or marine_df.empty:
                continue

            marine_df = marine_df.reset_index() if "date" not in marine_df.columns else marine_df.copy()
            marine_df["date"] = pd.to_datetime(marine_df["date"]).dt.date

            for _, row in marine_df.iterrows():
                info = _offshore_risk_from_row(row)
                info["offshore_source"] = "marine_forecast"
                lookup[(city, row["date"].isoformat())] = info

            logger.info("  Offshore marine forecast fetched for %s (%d days)", city, len(marine_df))

        except Exception as exc:
            logger.warning("  Offshore marine fetch failed for %s: %s", city, exc)
            continue

    return lookup


def _build_offshore_wave_climatology(conn, cities: list[str]) -> dict[tuple[str, int], dict]:
    """
    Build fallback offshore sea-risk lookup from historical/proxy wave_height.

    The analytics table should contain wave_height from the SMB fetch-limited
    proxy. This turns that project-specific maritime feature into a daily
    climatological offshore risk adjustment for dates beyond the marine forecast.
    """
    try:
        df = conn.execute("""
            SELECT city, date, wave_height
            FROM analytics.daily_enriched
            WHERE wave_height IS NOT NULL
            ORDER BY city, date
        """).fetchdf()
    except Exception as exc:
        logger.warning("  Could not build offshore wave climatology: %s", exc)
        return {}

    if df is None or df.empty:
        logger.warning("  No historical wave_height available for offshore climatology")
        return {}

    df = df[df["city"].isin(cities)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)

    probs = []
    for _, row in df.iterrows():
        info = _offshore_risk_from_row(row)
        probs.append(info["offshore_sea_probability"])
    df["offshore_sea_probability"] = probs

    grouped = (
        df.groupby(["city", "day_of_year"])
          .agg(
              offshore_sea_probability=("offshore_sea_probability", "mean"),
              offshore_wave_height_m=("wave_height", "mean"),
          )
          .reset_index()
    )

    lookup: dict[tuple[str, int], dict] = {}
    for _, row in grouped.iterrows():
        lookup[(row["city"], int(row["day_of_year"]))] = {
            "offshore_sea_probability": round(float(row["offshore_sea_probability"]), 4),
            "offshore_driver": "historical offshore wave exposure",
            "offshore_wave_height_m": round(float(row["offshore_wave_height_m"]), 2),
            "offshore_source": "wave_climatology",
        }

    logger.info("  Built offshore wave climatology with %d city/day entries", len(lookup))
    return lookup


def _offshore_info_for_day(city: str,
                           d: date,
                           live_lookup: dict[tuple[str, str], dict],
                           clim_lookup: dict[tuple[str, int], dict]) -> dict:
    """Return offshore sea-state info for one city/date."""
    key_live = (city, d.isoformat())
    if key_live in live_lookup:
        return live_lookup[key_live]

    key_clim = (city, int(d.timetuple().tm_yday))
    if key_clim in clim_lookup:
        return clim_lookup[key_clim]

    return {
        "offshore_sea_probability": 0.0,
        "offshore_driver": "offshore sea-state data unavailable",
        "offshore_wave_height_m": np.nan,
        "offshore_source": "not_available",
    }


def _build_risk_reason(reason_row,
                       source: str,
                       port_weather_probability: float,
                       offshore_info: dict,
                       adjusted_probability: float,
                       threshold: float) -> str:
    """Build a compact explanation for the adjusted maritime-risk row."""
    src = (source or "").lower()
    drivers = []

    offshore_prob = float(offshore_info.get("offshore_sea_probability", 0.0) or 0.0)
    offshore_driver = offshore_info.get("offshore_driver", "")
    wave = offshore_info.get("offshore_wave_height_m", np.nan)

    if offshore_prob >= 0.25 and offshore_driver and offshore_driver != "offshore sea-state data unavailable":
        if pd.notna(wave):
            drivers.append(f"{offshore_driver} ({wave:.2f}m wave estimate)")
        else:
            drivers.append(offshore_driver)

    if "climatology" in src:
        if not drivers:
            drivers.append("historical city/date risk pattern")
    else:
        wind = _safe_float(reason_row, "wind_speed_10m_max")
        gust = _safe_float(reason_row, "wind_gusts_10m_max")
        precip = _safe_float(reason_row, "precipitation_sum")
        snow = _safe_float(reason_row, "snowfall_sum")
        visibility_mean = _safe_float(reason_row, "visibility_mean")
        visibility_min = _safe_float(reason_row, "visibility_min")

        if gust is not None and gust >= 75:
            drivers.append("strong wind gusts")
        elif wind is not None and wind >= 50:
            drivers.append("high wind")

        if precip is not None and precip >= 15:
            drivers.append("heavy precipitation")

        if snow is not None and snow >= 5:
            drivers.append("snowfall")

        if visibility_min is not None and visibility_min <= 500:
            drivers.append("very low visibility")
        elif visibility_mean is not None and visibility_mean <= 1000:
            drivers.append("low visibility")

    if drivers:
        # Preserve order but remove duplicates.
        unique = []
        for dvr in drivers:
            if dvr not in unique:
                unique.append(dvr)
        return ", ".join(unique[:3])

    if adjusted_probability >= threshold and port_weather_probability >= threshold:
        return "model signal from recent weather trends"

    if adjusted_probability >= threshold and offshore_prob > 0:
        return "offshore sea-state adjustment raised the risk"

    return "no major maritime risk driver"

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
        city, date, day_of_month, port_weather_probability, offshore_sea_probability, uncalibrated_maritime_probability, probability, probability_calibrated, prediction, source, offshore_source, risk_reason
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

    # Plan the rolling forecast window
    today = date.today()
    target_dates = _rolling_dates(start_date=today, days_ahead=30)

    window_start = target_dates[0]
    window_end = target_dates[-1]
    target_window = f"{window_start.isoformat()}_{window_end.isoformat()}"

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

    # Maritime-specific adjustment: live offshore marine forecast where available,
    # historical/proxy wave climatology for the rest of the rolling window.
    offshore_live_lookup = _try_fetch_offshore_marine_forecast(
        cities=cities,
        cities_meta=CITIES,
        forecast_days=min(7, len(target_dates)),
    )
    offshore_clim_lookup = _build_offshore_wave_climatology(conn, cities)

    # Build the full prediction DataFrame
    rows = []
    for city in cities:
        for d in target_dates:
            day_of_year = d.timetuple().tm_yday
            fc_row = pd.DataFrame()

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

            port_weather_probability = float(np.clip(p, 0.0, 1.0))
            offshore_info = _offshore_info_for_day(
                city=city,
                d=d,
                live_lookup=offshore_live_lookup,
                clim_lookup=offshore_clim_lookup,
            )
            offshore_sea_probability = float(
                offshore_info.get("offshore_sea_probability", 0.0) or 0.0
            )
            uncalibrated_maritime_probability = _adjust_maritime_probability(
                port_weather_probability=port_weather_probability,
                offshore_sea_probability=offshore_sea_probability,
            )
            adjusted_probability = _calibrate_maritime_probability(
                probability=uncalibrated_maritime_probability,
                model_bundle=model_bundle,
            )

            reason_row = fc_row.iloc[0] if len(fc_row) > 0 else pd.Series(dtype="object")
            risk_reason = _build_risk_reason(
                reason_row=reason_row,
                source=src,
                port_weather_probability=port_weather_probability,
                offshore_info=offshore_info,
                adjusted_probability=adjusted_probability,
                threshold=threshold,
            )

            rows.append({
                "city": city,
                "date": d.isoformat(),
                "day_of_month": d.day,
                "port_weather_probability": round(port_weather_probability, 4),
                "offshore_sea_probability": round(offshore_sea_probability, 4),
                "uncalibrated_maritime_probability": round(uncalibrated_maritime_probability, 4),
                "probability": round(adjusted_probability, 4),
                "probability_calibrated": int(
                    isinstance(model_bundle, dict)
                    and model_bundle.get("maritime_probability_calibrator") is not None
                ),
                "prediction": int(adjusted_probability >= threshold),
                "source": src,
                "offshore_source": offshore_info.get("offshore_source", "not_available"),
                "offshore_driver": offshore_info.get("offshore_driver", ""),
                "offshore_wave_height_m": offshore_info.get("offshore_wave_height_m", np.nan),
                "risk_reason": risk_reason,
            })

    daily_df = pd.DataFrame(rows).sort_values(["city", "date"]).reset_index(drop=True)

    # Rolling-window summary
    monthly_df = _summarise_window(
        daily_df=daily_df,
        window_start=window_start,
        window_end=window_end,
        threshold=threshold,
        n_monte_carlo=n_monte_carlo,
        high_risk_month_threshold=HIGH_RISK_MONTH_THRESHOLD,
    )

    logger.info(
        "  %d daily predictions for %s → %s across %d cities "
        "(short_horizon=%d, climatology=%d)",
        len(daily_df), window_start, window_end, len(cities),
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

def _summarise_window(
    daily_df:                  pd.DataFrame,
    window_start:              date,
    window_end:                date,
    threshold:                 float,
    n_monte_carlo:             int,
    high_risk_month_threshold: int,
) -> pd.DataFrame:
    """
    Per-city summary derived from the rolling daily prediction window.

    risk_days_predicted          = count of days where prediction == 1
    high_risk_window_probability = P(sum of Bernoulli(p_i) >= threshold)
                                   estimated by Monte Carlo over the
                                   per-day probabilities.
    """
    rng = np.random.default_rng(seed=42)
    rows = []

    for city, sub in daily_df.groupby("city"):
        probs = sub["probability"].values
        risk_days_pred = int(sub["prediction"].sum())

        samples = rng.binomial(1, probs[None, :], size=(n_monte_carlo, len(probs)))
        sums = samples.sum(axis=1)
        p_high_risk = float((sums >= high_risk_month_threshold).mean())

        n_short = int((sub["source"] == "short_horizon").sum())
        n_clim = int((sub["source"] == "climatology").sum())

        rows.append({
            "city":                         city,
            "window_start":                 window_start.isoformat(),
            "window_end":                   window_end.isoformat(),
            "risk_days_predicted":          risk_days_pred,
            "high_risk_window_probability": round(p_high_risk, 4),
            "n_short_horizon_days":         n_short,
            "n_climatology_days":           n_clim,
        })

    return pd.DataFrame(rows).sort_values("city").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save predictions
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(
    daily_df:   pd.DataFrame,
    monthly_df: pd.DataFrame,
    target_month: Optional[str] = None,  # kept for compatibility with pipeline.py
    out_dir:    str | Path = "predictions",
) -> tuple[Path, Path]:
    """
    Write latest rolling-window predictions to:

        <out_dir>/latest/daily.csv
        <out_dir>/latest/monthly.csv
        <out_dir>/latest.json

    The name monthly.csv is kept for website compatibility, but its contents
    now summarize the rolling forecast window, not a calendar month.
    """
    out_dir = Path(out_dir)
    latest_dir = out_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    daily_path = latest_dir / "daily.csv"
    monthly_path = latest_dir / "monthly.csv"
    latest_json_path = out_dir / "latest.json"

    daily_df.to_csv(daily_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    if len(daily_df) > 0:
        window_start = str(daily_df["date"].min())
        window_end = str(daily_df["date"].max())
    else:
        window_start = ""
        window_end = ""

    metadata = {
        "window_start": window_start,
        "window_end": window_end,
        "days": int(daily_df["date"].nunique()) if "date" in daily_df.columns else 0,
        "daily_path": "latest/daily.csv",
        "monthly_path": "latest/monthly.csv",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    import json
    with latest_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("  Saved daily predictions   → %s", daily_path)
    logger.info("  Saved window summary      → %s", monthly_path)
    logger.info("  Saved latest metadata     → %s", latest_json_path)

    return daily_path, monthly_path
