import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# All 6 bronze datasets — expected row counts match bronze_ingestion_contract.json
_EXPECTED_ROWS = {
    "billing_payments":                  200_000,
    "commercial_industries_consumption": 220_000,
    "customers_complaint":               100_000,
    "grid_load":                         200_000,
    "power_flow":                        200_000,
    "retail_tariffs":                     90_000,
}

# Acceptable row count variance (+-20% of contract total_rows)
ROW_COUNT_TOLERANCE = 0.20


def _resolve_root(project_root: Optional[str]) -> Path:
    """Resolve project root — same priority chain as run_bronze.py."""
    if project_root:
        return Path(project_root).resolve()
    if os.getenv("PROJECT_ROOT"):
        return Path(os.getenv("PROJECT_ROOT")).resolve()
    # airflow/utils/quality_checks.py -> airflow/ -> project root
    return Path(__file__).parent.parent.parent.resolve()


def _load_expected_rows(root: Path) -> Dict[str, int]:
    """
    Read expected row counts from the live contract file.
    Falls back to hardcoded _EXPECTED_ROWS if the contract is missing.
    """
    contract_path = root / "bronze_metadata" / "bronze_ingestion_contract.json"
    if not contract_path.exists():
        logger.warning(
            f"Contract not found at {contract_path}. "
            "Using hardcoded expected row counts."
        )
        return _EXPECTED_ROWS.copy()

    with open(contract_path, "r") as f:
        contract = json.load(f)

    return {
        ds["dataset_name"]: ds["total_rows"]
        for ds in contract.get("datasets", [])
        if "dataset_name" in ds and "total_rows" in ds
    }


def _get_recent_audit_rows(
    config_path: str,
    datasets: List[str],
    hours: int = 2,
) -> Dict[str, dict]:
    root_str = str(Path(__file__).parent.parent.parent.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from bronze.observer.db_pool import pg_connection

    query = """
        SELECT DISTINCT ON (dataset_name)
            dataset_name,
            status,
            row_count,
            error_message,
            execution_time
        FROM bronze_ingestion_audit
        WHERE dataset_name = ANY(%s)
          AND execution_time > NOW() - INTERVAL '%s hours'
        ORDER BY dataset_name, execution_time DESC;
    """

    results = {}
    with pg_connection(config_path) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (datasets, hours))
            for row in cur.fetchall():
                results[row[0]] = {
                    "status":         row[1],
                    "row_count":      row[2],
                    "error_message":  row[3],
                    "execution_time": row[4],
                }

    return results


def bronze_quality_gate(
    project_root: str = None,
    **kwargs,
) -> dict:
    root        = _resolve_root(project_root)
    config_path = str(root / "databricks" / "databricks.cfg")
    datasets    = list(_EXPECTED_ROWS.keys())

    # Read expected rows from the live contract
    expected_rows = _load_expected_rows(root)

    logger.info(f"Running bronze quality gate for {len(datasets)} datasets")

    # Fetch recent audit rows in one query
    audit_rows = _get_recent_audit_rows(
        config_path=config_path,
        datasets=datasets,
        hours=2,
    )

    failures: List[str] = []
    results:  Dict[str, dict] = {}

    for dataset in datasets:
        row = audit_rows.get(dataset)

        # Check 1: Audit row must exist for this run window
        if row is None:
            failures.append(
                f"{dataset}: no audit row found in the last 2 hours"
            )
            results[dataset] = {"check": "MISSING"}
            continue

        # Check 2: Status must be SUCCESS
        if row["status"] != "SUCCESS":
            failures.append(
                f"{dataset}: status={row['status']} "
                f"error='{row['error_message']}'"
            )
            results[dataset] = {"check": "FAILED_STATUS", **row}
            continue

        # Check 3: Zero-row protection
        row_count = row.get("row_count") or 0
        if row_count == 0:
            failures.append(
                f"{dataset}: ingested 0 rows — possible empty source feed "
                "or failed COPY INTO / MERGE"
            )
            results[dataset] = {"check": "ZERO_ROWS", **row}
            continue

        # Check 4: Row count within tolerance of contract total_rows
        expected = expected_rows.get(dataset, 0)
        if expected > 0:
            lower = expected * (1 - ROW_COUNT_TOLERANCE)
            upper = expected * (1 + ROW_COUNT_TOLERANCE)
            if not (lower <= row_count <= upper):
                failures.append(
                    f"{dataset}: row_count={row_count:,} is outside "
                    f"+-{int(ROW_COUNT_TOLERANCE*100)}% of expected "
                    f"{expected:,} "
                    f"(acceptable range: {int(lower):,}–{int(upper):,})"
                )
                results[dataset] = {"check": "ROW_COUNT_ANOMALY", **row}
                continue

        # All checks passed for this dataset
        results[dataset] = {"check": "PASSED", **row}
        logger.info(
            f"  OK {dataset}: status={row['status']} rows={row_count:,}"
        )

    # Raise if any dataset failed
    if failures:
        failure_block = "\n".join(f"  - {f}" for f in failures)
        raise Exception(
            f"Bronze quality gate failed for "
            f"{len(failures)}/{len(datasets)} dataset(s):\n{failure_block}"
        )

    logger.info(
        f"Bronze quality gate passed — all {len(datasets)} datasets OK"
    )

    return {
        "status":           "passed",
        "datasets_checked": len(datasets),
        "results":          results,
    }