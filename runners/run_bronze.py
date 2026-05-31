import os
import sys
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def _resolve_project_root(project_root: str = None) -> Path:
    if project_root:
        resolved = Path(project_root).resolve()
    elif os.getenv("PROJECT_ROOT"):
        resolved = Path(os.getenv("PROJECT_ROOT")).resolve()
    else:
        # runners/run_bronze.py is one level below project root
        resolved = Path(__file__).parent.parent.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Project root not found at: {resolved}\n"
            f"Set PROJECT_ROOT environment variable or pass project_root explicitly."
        )

    return resolved


def main(
    project_root: str = None,
    datasets: list = None,
    download: bool = True,
    optimize: bool = False,
    dry_run: bool = False,
    **kwargs,   # absorbs any extra Airflow context kwargs
) -> dict:
    start_time = datetime.utcnow()
    root = _resolve_project_root(project_root)

    contract_path = root / "bronze" / "bronze_metadata" / "bronze_ingestion_contract.json"
    config_path   = root / "databricks" / "databricks.cfg"

    if not contract_path.exists():
        raise FileNotFoundError(
            f"Bronze ingestion contract not found: {contract_path}\n"
            "Run extraction/runner.py first to generate the contract."
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Databricks config not found: {config_path}\n"
            "Create databricks/databricks.cfg with workspace_url, token, warehouse_id."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from bronze.bronze_orchestrator import BronzeLayerOrchestrator

    logger.info("=" * 60)
    logger.info("BRONZE PIPELINE — START")
    logger.info("=" * 60)
    logger.info(f"Project root  : {root}")
    logger.info(f"Contract      : {contract_path}")
    logger.info(f"Config        : {config_path}")
    logger.info(f"Datasets      : {datasets or 'ALL'}")
    logger.info(f"Download      : {download}")
    logger.info(f"Optimize      : {optimize}")
    logger.info(f"Dry run       : {dry_run}")

    orchestrator = BronzeLayerOrchestrator(
        contract_path=str(contract_path),
        config_path=str(config_path),
        catalog=os.getenv("DATABRICKS_CATALOG", "main"),
        schema=os.getenv("DATABRICKS_SCHEMA", "bronze"),
        # staging_root and delta_root use orchestrator defaults:
        # /mnt/staging/raw  and  /mnt/delta/bronze
    )

    orchestrator.run_full_pipeline(
        datasets=datasets,
        download=download,
        optimize=optimize,
        dry_run=dry_run,
    )

    duration = (datetime.utcnow() - start_time).total_seconds()

    summary = {
        "status":              "success",
        "duration_seconds":    round(duration, 2),
        "datasets_processed":  datasets or "all",
        "dry_run":             dry_run,
    }

    logger.info("=" * 60)
    logger.info(f"BRONZE PIPELINE — COMPLETE ({duration:.1f}s)")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = main()
    print(f"\nRun complete: {result}")