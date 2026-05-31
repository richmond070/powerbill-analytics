import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BRONZE_DATASETS = [
    "billing_payments",
    "commercial_industries_consumption",
    "customers_complaint",
    "grid_load",
    "power_flow",
    "retail_tariffs",
]


def _resolve_root(project_root: Optional[str]) -> Path:
    """Resolve project root — same priority chain as run_bronze.py."""
    if project_root:
        return Path(project_root).resolve()
    if os.getenv("PROJECT_ROOT"):
        return Path(os.getenv("PROJECT_ROOT")).resolve()
    return Path(__file__).parent.parent.parent.resolve()


def _check_audit_table(
    config_path: str,
    datasets: List[str],
    hours: int,
) -> Tuple[bool, Dict[str, str]]:
    """
    Verify each dataset has a SUCCESS row in bronze_ingestion_audit
    within the last N hours.

    Returns:
        (all_ok, {dataset_name: status_string})
    """
    root_str = str(Path(__file__).parent.parent.parent.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from bronze.observer.db_pool import pg_connection

    query = """
        SELECT DISTINCT ON (dataset_name)
            dataset_name,
            status,
            row_count,
            duration_ms,
            execution_time
        FROM bronze_ingestion_audit
        WHERE dataset_name = ANY(%s)
          AND execution_time > NOW() - INTERVAL '%s hours'
        ORDER BY dataset_name, execution_time DESC;
    """

    found: Dict[str, dict] = {}
    with pg_connection(config_path) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (datasets, hours))
            for row in cur.fetchall():
                found[row[0]] = {
                    "status":         row[1],
                    "row_count":      row[2],
                    "duration_ms":    row[3],
                    "execution_time": row[4],
                }

    results: Dict[str, str] = {}
    all_ok = True

    for ds in datasets:
        if ds not in found:
            results[ds] = "NO_RECENT_RUN"
            all_ok = False
        elif found[ds]["status"] != "SUCCESS":
            results[ds] = found[ds]["status"]
            all_ok = False
        else:
            results[ds] = "SUCCESS"

    return all_ok, results


def _check_metrics_table(
    config_path: str,
    datasets: List[str],
) -> Tuple[bool, Dict[str, str]]:
    """
    Verify each dataset has a metrics row for today in bronze_ingestion_metrics.

    Returns:
        (all_ok, {dataset_name: "PRESENT" | "MISSING"})
    """
    from bronze.observer.db_pool import pg_connection

    today = date.today()
    query = """
        SELECT dataset_name
        FROM bronze_ingestion_metrics
        WHERE dataset_name = ANY(%s)
          AND metric_date = %s;
    """

    with pg_connection(config_path) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (datasets, today))
            present = {row[0] for row in cur.fetchall()}

    results: Dict[str, str] = {}
    all_ok = True

    for ds in datasets:
        if ds in present:
            results[ds] = "PRESENT"
        else:
            results[ds] = "MISSING"
            all_ok = False

    return all_ok, results


def validate_audit_metrics(
    project_root: str = None,
    audit_hours: int = 2,
    **kwargs,
) -> dict:

    root        = _resolve_root(project_root)
    config_path = str(root / "databricks" / "databricks.cfg")

    # Log Airflow runtime context for audit correlation
    dag_run_id      = None
    execution_date  = None
    if kwargs.get("dag_run"):
        dag_run_id = kwargs["dag_run"].run_id
    if kwargs.get("execution_date"):
        execution_date = kwargs["execution_date"].isoformat()

    logger.info(
        f"validate_audit_metrics | "
        f"dag_run_id={dag_run_id} | "
        f"execution_date={execution_date}"
    )

    failures: List[str] = []

    # ------------------------------------------------------------------
    # Check 1: bronze_ingestion_audit
    # ------------------------------------------------------------------
    audit_ok, audit_results = _check_audit_table(
        config_path=config_path,
        datasets=BRONZE_DATASETS,
        hours=audit_hours,
    )

    if not audit_ok:
        bad = {k: v for k, v in audit_results.items() if v != "SUCCESS"}
        for ds, status in bad.items():
            failures.append(
                f"audit: {ds} — status='{status}'"
            )

    # ------------------------------------------------------------------
    # Check 2: bronze_ingestion_metrics
    # ------------------------------------------------------------------
    metrics_ok, metrics_results = _check_metrics_table(
        config_path=config_path,
        datasets=BRONZE_DATASETS,
    )

    if not metrics_ok:
        missing = [k for k, v in metrics_results.items() if v == "MISSING"]
        for ds in missing:
            failures.append(
                f"metrics: {ds} — no metrics row for today "
                f"({date.today().isoformat()})"
            )

    # ------------------------------------------------------------------
    # Raise if anything failed
    # ------------------------------------------------------------------
    if failures:
        failure_block = "\n".join(f"  - {f}" for f in failures)
        raise Exception(
            f"Observability validation failed "
            f"({len(failures)} issue(s)):\n{failure_block}"
        )

    logger.info(
        "Observability validation passed — "
        "audit and metrics records complete for all datasets"
    )

    return {
        "status":         "passed",
        "dag_run_id":     dag_run_id,
        "execution_date": execution_date,
        "audit":          audit_results,
        "metrics":        metrics_results,
    }