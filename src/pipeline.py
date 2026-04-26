"""
src/pipeline.py
───────────────
End-to-end pipeline orchestrator for the Caspian Maritime project.

Runs ingestion → cleaning → feature engineering → training → prediction
with quality gates between each stage and full logging.

CLI
---
    python src/pipeline.py --mode full                 # re-ingest everything
    python src/pipeline.py --mode incremental          # fetch only new days
    python src/pipeline.py --mode incremental \\
                          --since 2024-10-01            # manual start date
    python src/pipeline.py --dry-run                   # plan only, no writes
    python src/pipeline.py --no-predict                # skip model stage
    python src/pipeline.py --no-train                  # skip train + predict
    python src/pipeline.py --strict-freshness          # fail hard on stale data

Exit codes
----------
    0  success
    1  quality gate ABORTED the run
    2  unhandled exception
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import pandas as pd

# Make sure src is importable whether called as `python src/pipeline.py`
# or `python -m src.pipeline`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import (
    CITIES, ALL_VARIABLES, DATE_RANGE, PATHS, API,
)
from src.ingestion import (
    fetch_historical, fetch_all_cities, save_raw,
)
from src.database import (
    get_connection, create_schemas, create_raw_tables,
    create_raw_tables_if_absent, load_raw_data, run_query,
)
from src.cleaning import clean_raw_to_staging
from src.features import build_analytics_layer
from src.quality_checks import (
    run_all_checks, format_check_report, any_aborting,
    ABORT, WARN, FLAG,
)
from src.modeling import train_model, predict_next_month, save_predictions


# ══════════════════════════════════════════════════════════════════════════════
# 1. Logging setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(run_id: str, log_dir: Optional[str | Path] = None) -> str:
    """
    Configure two handlers:
      - rotating file at <repo_root>/logs/pipeline.log (last 30 runs × 2MB)
      - stdout (INFO+)

    Every log line is prefixed with the run_id so one run's lines can
    be grepped out of a multi-run log file.

    The log directory is anchored to the project root (not the current
    working directory), so logs land in the same place whether the
    pipeline is invoked from the repo root, from notebooks/, or from a
    GitHub Actions runner.
    """
    if log_dir is None:
        log_dir = PATHS["repo_root"] / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    fmt = logging.Formatter(
        fmt=f"%(asctime)s  [{run_id}]  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove old handlers so repeated setup_logging() calls don't duplicate
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=30,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(sh)

    return str(log_file)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Run history table
# ══════════════════════════════════════════════════════════════════════════════

def _init_run_history(conn) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.pipeline_runs (
            run_id         VARCHAR PRIMARY KEY,
            mode           VARCHAR,
            start_time     TIMESTAMP,
            end_time       TIMESTAMP,
            duration_sec   DOUBLE,
            status         VARCHAR,
            rows_ingested  INTEGER,
            rows_cleaned   INTEGER,
            rows_analytics INTEGER,
            predictions_for VARCHAR,
            errors         VARCHAR,
            warnings_count INTEGER
        )
    """)


def _record_run(conn, summary: dict) -> None:
    _init_run_history(conn)
    conn.execute("""
        INSERT OR REPLACE INTO meta.pipeline_runs VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        summary["run_id"],
        summary["mode"],
        summary["start_time"],
        summary["end_time"],
        summary["duration_sec"],
        summary["status"],
        summary.get("rows_ingested", 0),
        summary.get("rows_cleaned",  0),
        summary.get("rows_analytics", 0),
        summary.get("predictions_for", ""),
        summary.get("errors", "")[:2000],   # truncate long tracebacks
        summary.get("warnings_count", 0),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 3. Incremental-window helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_date_per_city(conn) -> dict[str, Optional[str]]:
    """
    Return {city: 'YYYY-MM-DD' | None} for raw.weather_daily.
    None means the city has no rows yet.
    """
    # Check if raw table exists; if not, return Nones for all cities
    exists = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'raw' AND table_name = 'weather_daily'
    """).fetchone()[0]

    out = {c: None for c in CITIES}
    if not exists:
        return out

    # CAST to DATE in SQL so DuckDB returns a date, not a datetime.
    # str(datetime.date)     → '2026-04-24'           ✅ parseable
    # str(datetime.datetime) → '2026-04-24 00:00:00'  ❌ NOT parseable by date.fromisoformat()
    rows = conn.execute("""
        SELECT city, CAST(MAX(date) AS DATE) AS max_date
        FROM raw.weather_daily
        GROUP BY city
    """).fetchdf()
    for _, r in rows.iterrows():
        md = r["max_date"]
        # Handle pandas Timestamp / datetime.datetime / datetime.date / str
        if hasattr(md, "date"):
            md = md.date()
        out[r["city"]] = str(md)[:10]   # always YYYY-MM-DD
    return out


def resolve_window(mode: str, conn, since: Optional[str] = None,
                   overlap_days: int = 3) -> tuple[str, str]:
    """
    Decide the (start, end) date range to fetch.

    Full mode        : (DATE_RANGE.start, today-1)
    Incremental mode : (max(max_per_city) - overlap_days, today-1)
    Manual since     : (since, today-1)
    """
    today = date.today()
    end = (today - timedelta(days=1)).isoformat()  # yesterday — today's data isn't final yet

    if mode == "full":
        return DATE_RANGE["start"], end

    if since:
        return since, end

    # Incremental: start from oldest max_date across cities
    latest = get_latest_date_per_city(conn)
    max_dates = [d for d in latest.values() if d is not None]
    if not max_dates:
        # No existing data → behaves like full mode
        return DATE_RANGE["start"], end

    oldest_max = min(max_dates)
    # Defensive: strip any time component before parsing
    oldest_max_date = date.fromisoformat(oldest_max[:10])
    start = (oldest_max_date - timedelta(days=overlap_days)).isoformat()
    return start, end


# ══════════════════════════════════════════════════════════════════════════════
# 4. Stage runners
# ══════════════════════════════════════════════════════════════════════════════

def stage_ingest(conn, start: str, end: str, data_dir: Path,
                 dry_run: bool = False) -> dict:
    log = logging.getLogger("stage.ingest")
    log.info("=== STAGE 1: Ingest  (%s → %s) ===", start, end)

    if dry_run:
        log.info("  [dry-run] would fetch %d cities × %d vars for %s to %s",
                 len(CITIES), len(ALL_VARIABLES), start, end)
        return {"rows_fetched": 0, "files_written": 0}

    dfs = fetch_all_cities(
        cities=CITIES, start=start, end=end, variables=ALL_VARIABLES,
        delay_between_cities=2.0,
    )

    rows_total, files = 0, 0
    for city, df in dfs.items():
        if df is None or df.empty:
            log.warning("  %s: 0 rows returned", city)
            continue
        fname = f"{city.lower()}_historical_{start[:4]}_{end[:4]}"
        save_raw(df, fname, directory=data_dir)
        rows_total += len(df)
        files += 1

    log.info("  Fetched %d rows across %d files", rows_total, files)
    return {"rows_fetched": rows_total, "files_written": files}


def stage_load_raw(conn, data_dir: Path, incremental: bool = False,
                   dry_run: bool = False) -> dict:
    log = logging.getLogger("stage.load_raw")
    log.info("=== STAGE 2: Load raw  (from %s, incremental=%s) ===",
             data_dir, incremental)

    if dry_run:
        return {"rows_loaded": 0}

    create_schemas(conn)

    if incremental:
        # Keep existing data — use IF NOT EXISTS + INSERT OR REPLACE
        create_raw_tables_if_absent(conn)
    else:
        # Full mode: wipe and rebuild
        create_raw_tables(conn)

    counts = load_raw_data(conn, data_dir)
    rows = counts.get("raw.weather_daily", 0)
    log.info("  Loaded %s", counts)
    return {"rows_loaded": rows, "by_table": counts}


def stage_clean(conn, dry_run: bool = False) -> dict:
    log = logging.getLogger("stage.clean")
    log.info("=== STAGE 3: Clean raw → staging ===")

    if dry_run:
        return {"rows_out": 0}

    summary = clean_raw_to_staging(conn)
    log.info("  %s", summary)
    return summary


def stage_features(conn, dry_run: bool = False) -> dict:
    log = logging.getLogger("stage.features")
    log.info("=== STAGE 4: Features + analytics ===")

    if dry_run:
        return {"daily_enriched_rows": 0, "monthly_summary_rows": 0}

    result = build_analytics_layer(conn)
    log.info("  %s", result)
    return result


def stage_train(conn, model_path: Path, dry_run: bool = False) -> dict:
    log = logging.getLogger("stage.train")
    log.info("=== STAGE 5: Train model ===")

    if dry_run:
        return {"rows_trained": 0, "model_path": str(model_path)}

    metrics = train_model(conn, model_path=model_path)
    log.info("  %s", metrics)
    return metrics


def stage_predict(conn, model_path: Path, preds_dir: Path,
                  dry_run: bool = False) -> dict:
    log = logging.getLogger("stage.predict")
    log.info("=== STAGE 6: Predict next month ===")

    if dry_run:
        return {"target_month": None, "rows": 0}

    preds = predict_next_month(conn, model_path=model_path)
    saved = save_predictions(preds, out_dir=preds_dir)
    target = preds["target_month"].iloc[0]
    log.info("  %d predictions for %s → %s", len(preds), target, saved)
    return {"target_month": target, "rows": len(preds), "file": str(saved)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Quality gate runner
# ══════════════════════════════════════════════════════════════════════════════

def run_gate(conn, stage_name: str, monthly: bool = True) -> tuple[list[dict], bool]:
    """Run quality checks and return (results, should_abort)."""
    log = logging.getLogger("quality")
    log.info("--- Quality gate: %s ---", stage_name)
    results = run_all_checks(conn, stage=stage_name, monthly=monthly)
    report = format_check_report(results)
    for line in report.splitlines():
        log.info("  %s", line)
    return results, any_aborting(results)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main pipeline function
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    mode:              str  = "incremental",
    since:             Optional[str] = None,
    dry_run:           bool = False,
    skip_predict:      bool = False,
    skip_train:        bool = False,
    strict_freshness:  bool = False,
    db_path:           Optional[Path] = None,
    data_dir:          Optional[Path] = None,
) -> dict:
    """
    Orchestrate the full pipeline. Returns a summary dict.

    If any ABORT-severity check fails, aborts the pipeline and records
    the failure in meta.pipeline_runs.
    """
    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(run_id)
    log = logging.getLogger("pipeline")

    db_path  = Path(db_path)  if db_path  else PATHS["repo_root"] / "data" / "caspian_weather.duckdb"
    data_dir = Path(data_dir) if data_dir else PATHS["data_raw"]
    model_path = PATHS["models"] / "latest.pkl"
    preds_dir  = PATHS["repo_root"] / "predictions"

    start_time = datetime.now()
    summary = {
        "run_id":     run_id,
        "mode":       mode,
        "start_time": start_time,
        "since":      since,
        "dry_run":    dry_run,
    }

    log.info("╔" + "═" * 68 + "╗")
    log.info("║  Pipeline run: %s  (mode=%s, dry_run=%s)",
             run_id, mode, dry_run)
    log.info("╚" + "═" * 68 + "╝")

    conn = None
    all_check_results: list[dict] = []

    try:
        conn = get_connection(db_path)
        _init_run_history(conn)

        # Resolve the fetch window
        fetch_start, fetch_end = resolve_window(mode, conn, since=since)
        log.info("Fetch window: %s → %s", fetch_start, fetch_end)

        if mode == "incremental":
            current_max = get_latest_date_per_city(conn)
            log.info("Current max dates: %s", current_max)
            # If nothing to fetch, short-circuit
            # Compare just the date portion (current_max values are 'YYYY-MM-DD' after fix)
            if all(d is not None and d[:10] >= fetch_end for d in current_max.values()):
                log.info("✅ Data already up to date — no fetch needed")
                summary["status"] = "UP_TO_DATE"
                summary["end_time"]     = datetime.now()
                summary["duration_sec"] = (summary["end_time"] - start_time).total_seconds()
                summary["rows_ingested"] = 0
                summary["rows_cleaned"]  = 0
                summary["rows_analytics"] = 0
                summary["warnings_count"] = 0
                _record_run(conn, summary)
                return summary

        # ── STAGE 1: Ingest ───────────────────────────────────────────────────
        ingest = stage_ingest(conn, fetch_start, fetch_end, data_dir, dry_run)
        summary["rows_ingested"] = ingest["rows_fetched"]

        # ── STAGE 2: Load raw ─────────────────────────────────────────────────
        load = stage_load_raw(conn, data_dir,
                              incremental=(mode == "incremental"),
                              dry_run=dry_run)
        summary["rows_raw"] = load["rows_loaded"]

        # Gate: raw
        if not dry_run:
            raw_results, abort = run_gate(conn, "raw", monthly=not strict_freshness)
            all_check_results.extend(raw_results)
            if abort:
                raise RuntimeError("Raw-layer quality gate ABORTED the run")

        # ── STAGE 3: Clean ────────────────────────────────────────────────────
        clean = stage_clean(conn, dry_run)
        summary["rows_cleaned"] = clean.get("rows_out", 0)

        # Gate: staging
        if not dry_run:
            stg_results, abort = run_gate(conn, "staging", monthly=not strict_freshness)
            all_check_results.extend(stg_results)
            if abort:
                raise RuntimeError("Staging-layer quality gate ABORTED the run")

        # ── STAGE 4: Features ────────────────────────────────────────────────
        feats = stage_features(conn, dry_run)
        summary["rows_analytics"] = feats.get("daily_enriched_rows", 0)

        # Gate: analytics
        if not dry_run:
            ana_results, abort = run_gate(conn, "analytics", monthly=not strict_freshness)
            all_check_results.extend(ana_results)
            if abort:
                raise RuntimeError("Analytics-layer quality gate ABORTED the run")

        # ── STAGE 5: Train ────────────────────────────────────────────────────
        if not skip_train and not dry_run:
            train_metrics = stage_train(conn, model_path, dry_run)
            summary["train_metrics"] = train_metrics

        # ── STAGE 6: Predict ─────────────────────────────────────────────────
        if not skip_predict and not skip_train and not dry_run:
            pred = stage_predict(conn, model_path, preds_dir, dry_run)
            summary["predictions_for"] = pred.get("target_month", "")

        summary["status"] = "SUCCESS"

    except Exception as exc:
        log.error("Pipeline FAILED: %s", exc)
        log.error(traceback.format_exc())
        summary["status"] = "FAILED"
        summary["errors"] = f"{type(exc).__name__}: {exc}"

    finally:
        summary["end_time"] = datetime.now()
        summary["duration_sec"] = round(
            (summary["end_time"] - start_time).total_seconds(), 1
        )
        summary["warnings_count"] = sum(
            1 for r in all_check_results if r["status"] in ("WARN", "FAIL", "FLAG")
        )
        summary["check_results"] = all_check_results

        if conn is not None:
            try:
                _record_run(conn, summary)
            except Exception as e:
                log.error("Failed to record run history: %s", e)
            conn.close()

        log.info("═" * 70)
        log.info("Pipeline finished in %.1fs — status=%s",
                 summary["duration_sec"], summary["status"])
        log.info("═" * 70)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Caspian Maritime Delay-Risk — end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("full", "incremental"),
                   default="incremental",
                   help="full = re-ingest everything from DATE_RANGE.start; "
                        "incremental = fetch only new days (default)")
    p.add_argument("--since", type=str, default=None,
                   help="Override the incremental start date (YYYY-MM-DD)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run; make no changes")
    p.add_argument("--no-train", action="store_true",
                   help="Skip the training stage (also skips prediction)")
    p.add_argument("--no-predict", action="store_true",
                   help="Skip only the prediction stage")
    p.add_argument("--strict-freshness", action="store_true",
                   help="Use the 2-day freshness check instead of 35-day")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    summary = run_pipeline(
        mode=args.mode,
        since=args.since,
        dry_run=args.dry_run,
        skip_predict=args.no_predict,
        skip_train=args.no_train,
        strict_freshness=args.strict_freshness,
    )

    # Exit codes per docstring
    if summary["status"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
