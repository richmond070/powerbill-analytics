#!/usr/bin/env python3
"""
Bronze Layer Entry Point
========================
Runs the bronze layer orchestration pipeline.

This script is the single CLI entrypoint for the bronze layer.
It resolves all paths relative to the project root so it works
correctly whether called from:
  - the project root        :  python runners/run_bronze.py
  - the runners/ directory  :  python run_bronze.py
  - GitHub Actions CI       :  python runners/run_bronze.py --dry-run
  - Airflow BashOperator    :  python /opt/airflow/runners/run_bronze.py

Path layout this script expects:
  <project_root>/
    bronze_metadata/
      bronze_ingestion_contract.json   ← contract (required)
    databricks/
      databricks.cfg                   ← Databricks + Postgres config (required)
    bronze/
      bronze_orchestrator.py           ← orchestrator module

Environment variable overrides (all optional):
  DATABRICKS_CATALOG   default: main
  DATABRICKS_SCHEMA    default: bronze
  STAGING_ROOT         default: /tmp/staging/raw
  DELTA_ROOT           default: /tmp/delta/bronze
  DRY_RUN              set to "true" to force dry-run regardless of --dry-run flag

Usage:
  # Dry run — generates and prints SQL, no Databricks execution
  python runners/run_bronze.py --dry-run

  # Real run — executes SQL against Databricks SQL warehouse
  python runners/run_bronze.py

  # Real run, specific datasets only
  python runners/run_bronze.py --datasets billing_payments grid_load

  # Real run with table optimisation after ingestion
  python runners/run_bronze.py --optimize

  # Skip downloading raw files (re-use previously staged files)
  python runners/run_bronze.py --no-download
"""

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1 — Resolve project root and make bronze/ importable
#
# runners/run_bronze.py lives one level below the project root.
# We walk up one directory from this file's location to reach the root,
# then insert it at the front of sys.path so `from bronze.xxx import ...`
# resolves correctly without any relative imports.
# ---------------------------------------------------------------------------
RUNNERS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = RUNNERS_DIR.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now safe to import — bronze/ is resolvable as an absolute package
from bronze.bronze_orchestrator import BronzeLayerOrchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Step 2 — Path constants (all relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "bronze_metadata" / "bronze_ingestion_contract.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "databricks" / "databricks.cfg"


# ---------------------------------------------------------------------------
# Step 3 — Path resolution with clear error messages
# ---------------------------------------------------------------------------
def resolve_contract_path() -> str:
    """
    Resolve the bronze ingestion contract path.
    Checks the standard location: bronze_metadata/bronze_ingestion_contract.json

    Returns:
        Absolute path string to the contract file.

    Raises:
        FileNotFoundError: If the contract file cannot be found.
    """
    if DEFAULT_CONTRACT_PATH.exists():
        return str(DEFAULT_CONTRACT_PATH)

    raise FileNotFoundError(
        f"bronze_ingestion_contract.json not found.\n"
        f"Expected location: {DEFAULT_CONTRACT_PATH}\n"
        f"Run the extraction pipeline first to generate the contract:\n"
        f"  python extraction/runner.py"
    )


def resolve_config_path() -> str:
    """
    Resolve the Databricks config path.
    Checks the standard location: databricks/databricks.cfg

    Returns:
        Absolute path string to the config file.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    if DEFAULT_CONFIG_PATH.exists():
        return str(DEFAULT_CONFIG_PATH)

    raise FileNotFoundError(
        f"databricks.cfg not found.\n"
        f"Expected location: {DEFAULT_CONFIG_PATH}\n"
        f"Create the file with:\n"
        f"\n"
        f"  [DEFAULT]\n"
        f"  token         = dapi...\n"
        f"  workspace_url = https://dbc-xxxxx.cloud.databricks.com\n"
        f"  warehouse_id  = xxxxxxxxxxxxxxxx\n"
        f"\n"
        f"  [POSTGRES]\n"
        f"  host     = localhost\n"
        f"  port     = 5432\n"
        f"  dbname   = your_db\n"
        f"  user     = postgres\n"
        f"  password = your_password\n"
    )


# ---------------------------------------------------------------------------
# Step 4 — CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    All flags are optional — sensible defaults are used when not provided.
    """
    parser = argparse.ArgumentParser(
        prog="run_bronze",
        description="Bronze layer ingestion pipeline — orchestrates metadata-driven "
        "Databricks SQL ingestion from bronze_ingestion_contract.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("DRY_RUN", "false").lower() == "true",
        help=(
            "Generate SQL and print execution plan without running anything "
            "against Databricks. Safe to use in CI and local testing. "
            "[env: DRY_RUN=true]"
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="DATASET_NAME",
        default=None,
        help=(
            "Space-separated list of dataset names to process. "
            "If omitted, all datasets in the contract are processed. "
            "Example: --datasets billing_payments grid_load"
        ),
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help=(
            "Run OPTIMIZE + VACUUM on Delta tables after ingestion. "
            "Adds time but improves query performance. Skipped by default."
        ),
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help=(
            "Skip downloading raw parquet files and use previously staged files. "
            "Useful when re-running ingestion after a partial failure."
        ),
    )

    parser.add_argument(
        "--catalog",
        default=os.getenv("DATABRICKS_CATALOG", "main"),
        help="Unity Catalog name. [env: DATABRICKS_CATALOG] (default: main)",
    )

    parser.add_argument(
        "--schema",
        default=os.getenv("DATABRICKS_SCHEMA", "bronze"),
        help="Databricks schema / database name. [env: DATABRICKS_SCHEMA] (default: bronze)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 5 — Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point for the bronze layer pipeline.

    Execution order:
      1. Parse CLI arguments
      2. Resolve all required file paths
      3. Print run configuration
      4. Instantiate BronzeLayerOrchestrator
      5. Execute run_full_pipeline()
      6. Report outcome and exit with appropriate code
    """
    start_time = datetime.utcnow()

    print("=" * 80)
    print("BRONZE LAYER INGESTION PIPELINE")
    print("=" * 80)
    print(f"Start Time : {start_time.isoformat()}Z")
    print(f"Project    : {PROJECT_ROOT}")
    print()

    # ── Step 1: Parse arguments ───────────────────────────────────────────
    args = parse_args()

    # ── Step 2: Resolve paths ─────────────────────────────────────────────
    print("Resolving paths...")
    try:
        contract_path = resolve_contract_path()
        config_path = resolve_config_path()
    except FileNotFoundError as e:
        print(f"\nERROR — Missing required file:\n{e}")
        sys.exit(1)

    #staging_root = os.getenv("STAGING_ROOT", str(PROJECT_ROOT / "staging" / "raw")).strip().replace("\\", "/")
    wait_timeout = int(os.getenv("SQL_WAIT_TIMEOUT", "900")) 
    databricks_staging = os.getenv(
        "DATABRICKS_STAGING",
        "/Volumes/main/bronze/staging/raw"
    )
    delta_root   = os.getenv("DELTA_ROOT",   "")

    # ── Step 3: Print run configuration ──────────────────────────────────
    print(f"  Contract     : {contract_path}")
    print(f"  Config       : {config_path}")
    print(f"  Catalog      : {args.catalog}")
    print(f"  Schema       : {args.schema}")
    #print(f"  Staging Root : {staging_root}")
    print(f"  Delta Root   : {delta_root}")
    print()
    print("Run options:")
    print(f"  Dry Run      : {args.dry_run}")
    print(f"  Download     : {not args.no_download}")
    print(f"  Optimize     : {args.optimize}")
    print(f"  Datasets     : {', '.join(args.datasets) if args.datasets else 'ALL'}")
    print()

    if args.dry_run:
        print(" DRY RUN MODE — SQL will be generated but NOT executed against Databricks")
        print()

    # ── Step 4: Instantiate orchestrator ─────────────────────────────────
    print("Initializing orchestrator...")
    try:
        orchestrator = BronzeLayerOrchestrator(
            contract_path=contract_path,
            config_path=config_path,
            catalog=args.catalog,
            schema=args.schema,
            #staging_root=staging_root,
            delta_root=delta_root,
            #databricks_staging=databricks_staging
        )
        print("Orchestrator initialized.\n")
    except Exception as e:
        print(f"\nERROR — Failed to initialize orchestrator:\n{e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 5: Execute pipeline ──────────────────────────────────────────
    try:
        orchestrator.run_full_pipeline(
            datasets=args.datasets,
            download=not args.no_download,
            optimize=args.optimize,
            dry_run=args.dry_run,
        )
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        print(f"\n{'=' * 80}")
        print("PIPELINE FAILED")
        print(f"{'=' * 80}")
        print(f"Error    : {e}")
        print(f"Duration : {duration:.1f}s")
        print(f"End Time : {datetime.utcnow().isoformat()}Z")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 6: Report success ────────────────────────────────────────────
    duration = (datetime.utcnow() - start_time).total_seconds()
    print(f"{'=' * 80}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    print(f"Duration : {duration:.1f}s")
    print(f"End Time : {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
