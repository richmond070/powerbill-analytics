import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


# =============================================================================
# LOGGER
# =============================================================================
log = logging.getLogger(__name__)


# =============================================================================
# PATHS & CONFIG
# All values come from environment variables injected by docker-compose.yaml.
# Safe defaults let the DAG load at parse time even if a variable is missing —
# tasks will fail with a clear error at runtime, not silently at parse time.
# =============================================================================

# extraction/api_config.json — list of {name, url} dicts, one per dataset
API_CONFIG_PATH = os.getenv(
    "EXTRACTION_API_CONFIG_PATH",
    "/opt/airflow/extraction/api_config.json"
)

# Where runner.py writes the finished contract
CONTRACT_PATH = os.getenv(
    "BRONZE_CONTRACT_PATH",
    "/opt/airflow/bronze/bronze_metadata/bronze_ingestion_contract.json"
)

# Databricks credentials
DATABRICKS_CFG = os.getenv(
    "DATABRICKS_CFG_PATH",
    "/opt/airflow/databricks/databricks.cfg"
)

# dbt
DBT_PROJECT_DIR  = os.getenv("DBT_PROJECT_DIR",  "/opt/airflow/dbt_project")
DBT_PROFILES_DIR = os.getenv("DBT_PROFILES_DIR", "/opt/airflow/dbt_project")

# Databricks target
CATALOG      = os.getenv("DATABRICKS_CATALOG",   "main")
SCHEMA       = os.getenv("DATABRICKS_SCHEMA",    "bronze")
STAGING_ROOT = os.getenv("BRONZE_STAGING_ROOT",  "/tmp/bronze_staging")
DELTA_ROOT   = os.getenv("BRONZE_DELTA_ROOT",    "/mnt/delta/bronze")

# DAG schedule
SCHEDULE = os.getenv("BRONZE_PIPELINE_SCHEDULE", "0 1 * * *")


# =============================================================================
# DATASET LIST
# Resolved from api_config.json at parse time so Airflow can build the task
# graph dynamically. Falls back to an empty list if the file is missing —
# the DAG still loads and the validate_env task will surface the problem.
# =============================================================================

def _load_dataset_names() -> List[str]:
    """Read dataset names from extraction/api_config.json."""
    try:
        with open(API_CONFIG_PATH, "r") as fh:
            config = json.load(fh)
        return [d["name"] for d in config["datasets"]]
    except FileNotFoundError:
        log.warning(
            "api_config.json not found at %s — using empty dataset list.",
            API_CONFIG_PATH
        )
        return []
    except Exception as exc:
        log.error("Failed to parse api_config.json: %s", exc)
        return []


DATASETS: List[str] = _load_dataset_names()


# =============================================================================
# DEFAULT TASK ARGS
# =============================================================================

DEFAULT_ARGS = {
    "owner"                    : "data-engineering",
    "depends_on_past"          : False,
    "email_on_failure"         : False,   # flip True once SMTP is configured
    "email_on_retry"           : False,
    "retries"                  : 2,
    "retry_delay"              : timedelta(minutes=5),
    "retry_exponential_backoff": True,    # 5m → 10m → fail
}


# =============================================================================
# TASK CALLABLES
# =============================================================================

# -----------------------------------------------------------------------------
# PHASE 0 — validate_env
# -----------------------------------------------------------------------------

def validate_env(**kwargs) -> None:
    """
    Verify that every required file and environment variable exists before
    any work begins. Fail fast here rather than inside a dataset task where
    the error message is harder to attribute.

    Checks:
      - api_config.json is present and parseable
      - databricks.cfg is present
      - bronze/bronze_metadata/ output directory exists (creates it if not)
      - DATABRICKS_CATALOG and DATABRICKS_SCHEMA are set
    """
    errors = []

    # api_config.json
    if not Path(API_CONFIG_PATH).exists():
        errors.append(f"api_config.json not found: {API_CONFIG_PATH}")
    else:
        try:
            with open(API_CONFIG_PATH) as fh:
                cfg = json.load(fh)
            names = [d["name"] for d in cfg["datasets"]]
            log.info("api_config.json — %d datasets: %s", len(names), names)
        except Exception as exc:
            errors.append(f"api_config.json parse error: {exc}")

    # databricks.cfg
    if not Path(DATABRICKS_CFG).exists():
        errors.append(f"databricks.cfg not found: {DATABRICKS_CFG}")
    else:
        log.info("databricks.cfg found: %s", DATABRICKS_CFG)

    # bronze_metadata output directory
    contract_dir = Path(CONTRACT_PATH).parent
    contract_dir.mkdir(parents=True, exist_ok=True)
    log.info("Contract output directory: %s", contract_dir)

    # catalog / schema
    if not CATALOG:
        errors.append("DATABRICKS_CATALOG env var is not set")
    if not SCHEMA:
        errors.append("DATABRICKS_SCHEMA env var is not set")

    if errors:
        for err in errors:
            log.error("VALIDATION FAILED: %s", err)
        raise RuntimeError(
            f"validate_env failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    log.info("validate_env passed. Pipeline is clear to run.")


# -----------------------------------------------------------------------------
# PHASE 1 — extract_dataset  (one task per dataset)
# -----------------------------------------------------------------------------

def extract_dataset(dataset_name: str, **kwargs) -> None:
    """
    Run the full extraction pipeline for a single dataset.

    What this task does (mirrors extraction/runner.py logic, per dataset):
      1. Reads api_config.json to find the HuggingFace API URL for this dataset.
      2. Calls HuggingFaceDatasetResolver.resolve(api_url)
           → queries the HuggingFace metadata API
           → parses the response (handles all 3 response formats)
           → returns a list of {url, filename, size_bytes} dicts
      3. Calls ParquetValidator.validate_remote_parquet(url, filename) for each file
           → downloads the parquet file fully
           → reads PyArrow schema: num_rows, num_columns, column types
           → returns validation metadata dict
      4. Builds a complete dataset_entry dict (same structure as contract datasets[])
      5. Pushes the dataset_entry to XCom under key f"entry__{dataset_name}"
         so finalise_contract can collect all 6 entries and write one contract.

    Why XCom instead of writing the contract directly?
      Six parallel tasks writing to the same JSON file would produce a race
      condition. XCom is Airflow's safe mechanism for passing data between tasks.
      finalise_contract is the single writer.

    Args:
        dataset_name: The dataset to extract (injected via op_kwargs).
    """
    log.info("=== EXTRACT: %s ===", dataset_name)

    # Import inside the callable — keeps scheduler memory lean
    from extraction.resolver  import HuggingFaceDatasetResolver
    from extraction.validator import ParquetValidator

    # Load api_config to find this dataset's URL
    with open(API_CONFIG_PATH, "r") as fh:
        config = json.load(fh)

    dataset_config = next(
        (d for d in config["datasets"] if d["name"] == dataset_name),
        None
    )
    if dataset_config is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in api_config.json. "
            f"Available: {[d['name'] for d in config['datasets']]}"
        )

    api_url = dataset_config["url"]
    log.info("Resolving API endpoint: %s", api_url)

    # Step 1 — resolve parquet file URLs from HuggingFace API
    resolver     = HuggingFaceDatasetResolver()
    parquet_files = resolver.resolve(api_url)
    log.info("Resolved %d parquet file(s) for %s", len(parquet_files), dataset_name)

    # Step 2 — validate each parquet file (downloads + reads PyArrow metadata)
    validator       = ParquetValidator()
    validated_files = []

    for file_info in parquet_files:
        log.info("Validating: %s", file_info["filename"])
        metadata = validator.validate_remote_parquet(
            url      = file_info["url"],
            filename = file_info["filename"]
        )
        # Merge {url, filename, size_bytes} with {num_rows, columns, ...}
        validated_files.append({**file_info, **metadata})

    # Step 3 — build the dataset entry (same structure as contract datasets[])
    dataset_entry = {
        "dataset_name" : dataset_name,
        "api_endpoint" : api_url,
        "file_count"   : len(validated_files),
        "total_rows"   : sum(f.get("num_rows", 0) for f in validated_files),
        "files"        : validated_files,
    }

    log.info(
        "Extraction complete: %s — %d file(s), %s total rows",
        dataset_name,
        dataset_entry["file_count"],
        f"{dataset_entry['total_rows']:,}"
    )

    # Step 4 — push to XCom for finalise_contract to collect
    kwargs["ti"].xcom_push(
        key   = f"entry__{dataset_name}",
        value = dataset_entry
    )


# -----------------------------------------------------------------------------
# PHASE 1 fanin — finalise_contract
# -----------------------------------------------------------------------------

def finalise_contract(**kwargs) -> None:
    """
    Collect all 6 dataset entries from XCom and write one complete
    bronze_ingestion_contract.json.

    This is the single writer for the contract file, which eliminates
    any race condition that would occur if 6 parallel extract tasks
    each tried to write the same file directly.

    After writing, it validates the contract is complete and all datasets
    have validation_status == 'success' before allowing downstream tasks
    to proceed.
    """
    log.info("=== FINALISE CONTRACT ===")

    ti = kwargs["ti"]

    # Pull each dataset entry from XCom (written by extract_dataset tasks)
    dataset_entries = []
    failed_datasets = []

    for dataset_name in DATASETS:
        entry = ti.xcom_pull(
            task_ids = f"extract__{dataset_name}",
            key      = f"entry__{dataset_name}"
        )
        if entry is None:
            log.error("No XCom entry found for dataset: %s", dataset_name)
            failed_datasets.append(dataset_name)
            continue

        # Check each file's validation_status
        for file_info in entry.get("files", []):
            if file_info.get("validation_status") != "success":
                log.error(
                    "Validation failed for %s / %s: %s",
                    dataset_name,
                    file_info.get("filename"),
                    file_info.get("error", "unknown error")
                )
                failed_datasets.append(dataset_name)
                break

        dataset_entries.append(entry)
        log.info(
            "  ✓ %s — %s rows across %d file(s)",
            dataset_name,
            f"{entry['total_rows']:,}",
            entry["file_count"]
        )

    if failed_datasets:
        raise RuntimeError(
            f"Extraction failed for {len(failed_datasets)} dataset(s): "
            f"{failed_datasets}. Contract will not be written."
        )

    # Build and write the contract
    contract = {
        "generated_at" : datetime.utcnow().isoformat() + "Z",
        "datasets"     : dataset_entries,
    }

    contract_path = Path(CONTRACT_PATH)
    contract_path.parent.mkdir(parents=True, exist_ok=True)

    with open(contract_path, "w") as fh:
        json.dump(contract, fh, indent=2)

    log.info(
        "Contract written to %s — %d datasets, generated_at=%s",
        CONTRACT_PATH,
        len(dataset_entries),
        contract["generated_at"]
    )

    # Push contract summary to XCom for audit
    kwargs["ti"].xcom_push(
        key   = "contract_generated_at",
        value = contract["generated_at"]
    )
    kwargs["ti"].xcom_push(
        key   = "contract_dataset_names",
        value = [e["dataset_name"] for e in dataset_entries]
    )


# -----------------------------------------------------------------------------
# PHASE 2 — create_bronze_table  (one task per dataset)
# -----------------------------------------------------------------------------

def create_bronze_table(dataset_name: str, **kwargs) -> None:
    """
    Run Phase 2 for a single dataset: CREATE TABLE IF NOT EXISTS in Databricks.

    Reads the contract written by finalise_contract, resolves the partition
    strategy for this dataset, generates the CREATE TABLE SQL via
    BronzeSQLGenerator, and executes it via DatabricksSQLClient.

    Args:
        dataset_name: The dataset to process (injected via op_kwargs).
    """
    log.info("=== CREATE TABLE: %s ===", dataset_name)

    from bronze.ingestion.bronze_orchestrator import BronzeLayerOrchestrator

    orchestrator = BronzeLayerOrchestrator(
        contract_path = CONTRACT_PATH,
        config_path   = DATABRICKS_CFG,
        catalog       = CATALOG,
        schema        = SCHEMA,
        staging_root  = STAGING_ROOT,
        delta_root    = DELTA_ROOT,
    )

    orchestrator.create_bronze_tables(
        datasets = [dataset_name],
        dry_run  = False,
    )

    log.info("CREATE TABLE complete: %s", dataset_name)


# -----------------------------------------------------------------------------
# PHASE 3 — ingest_dataset  (one task per dataset)
# -----------------------------------------------------------------------------

def ingest_dataset(dataset_name: str, **kwargs) -> None:
    """
    Run Phase 3 for a single dataset: download raw parquet + COPY INTO / MERGE.

    The parquet file is downloaded from the URL recorded in the contract
    (written in Phase 1) to the local staging area, then loaded into the
    Delta table via Databricks SQL API.

    Args:
        dataset_name: The dataset to process (injected via op_kwargs).
    """
    log.info("=== INGEST: %s ===", dataset_name)

    from bronze.ingestion.bronze_orchestrator import BronzeLayerOrchestrator

    orchestrator = BronzeLayerOrchestrator(
        contract_path = CONTRACT_PATH,
        config_path   = DATABRICKS_CFG,
        catalog       = CATALOG,
        schema        = SCHEMA,
        staging_root  = STAGING_ROOT,
        delta_root    = DELTA_ROOT,
    )

    orchestrator.ingest_data(
        datasets = [dataset_name],
        download = True,
        dry_run  = False,
    )

    log.info("INGEST complete: %s", dataset_name)


# -----------------------------------------------------------------------------
# PHASE 5 — notify_complete
# -----------------------------------------------------------------------------

def notify_complete(**kwargs) -> None:
    """
    Log a final pipeline summary. Extend this with a Slack webhook post,
    email, or a write to bronze_ingestion_audit when ready.
    """
    ti           = kwargs["ti"]
    run_id       = kwargs["run_id"]
    logical_date = kwargs["logical_date"]

    # Pull audit data from XCom
    generated_at = ti.xcom_pull(
        task_ids = "finalise_contract",
        key      = "contract_generated_at"
    )
    dataset_names = ti.xcom_pull(
        task_ids = "finalise_contract",
        key      = "contract_dataset_names"
    )

    log.info("=" * 70)
    log.info("BRONZE PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info("Run ID              : %s", run_id)
    log.info("Logical date        : %s", logical_date)
    log.info("Contract written at : %s", generated_at)
    log.info("Datasets ingested   : %s", dataset_names)
    log.info("Catalog target      : %s.%s", CATALOG, SCHEMA)
    log.info("=" * 70)


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id            = "bronze_pipeline",
    description       = (
        "Full Bronze pipeline: HuggingFace extraction → contract generation → "
        "Databricks Delta ingestion → dbt staging models. "
        "Nigerian Energy & Utilities Analytics project."
    ),
    default_args      = DEFAULT_ARGS,
    schedule_interval = SCHEDULE,
    start_date        = days_ago(1),
    catchup           = False,       # no backfill on first deploy
    max_active_runs   = 1,           # prevent overlapping pipeline runs
    tags              = ["bronze", "extraction", "ingestion", "databricks", "dbt"],
) as dag:

    # =========================================================================
    # PHASE 0 — validate_env
    # =========================================================================
    task_validate_env = PythonOperator(
        task_id         = "validate_env",
        python_callable = validate_env,
        doc_md          = (
            "Verify api_config.json, databricks.cfg, output directories, "
            "and required env vars exist before any work begins."
        ),
    )

    # =========================================================================
    # PHASE 1 — extract tasks  (one per dataset, all parallel)
    # =========================================================================
    extract_tasks = []

    for dataset_name in DATASETS:
        t = PythonOperator(
            task_id         = f"extract__{dataset_name}",
            python_callable = extract_dataset,
            op_kwargs       = {"dataset_name": dataset_name},
            doc_md          = (
                f"Resolve HuggingFace API endpoint for {dataset_name}, "
                f"download and validate parquet metadata via PyArrow, "
                f"push dataset entry to XCom for finalise_contract."
            ),
        )
        extract_tasks.append(t)

    # =========================================================================
    # PHASE 1 fanin — finalise_contract
    # =========================================================================
    task_finalise_contract = PythonOperator(
        task_id         = "finalise_contract",
        python_callable = finalise_contract,
        doc_md          = (
            "Collect all 6 dataset entries from XCom. "
            "Validate all extraction runs succeeded. "
            "Write the single merged bronze_ingestion_contract.json. "
            "This is the sole writer — avoids race conditions from parallel tasks."
        ),
    )

    # =========================================================================
    # PHASE 2 — create_table tasks  (one per dataset, all parallel)
    # =========================================================================
    create_table_tasks = []

    for dataset_name in DATASETS:
        t = PythonOperator(
            task_id         = f"create_table__{dataset_name}",
            python_callable = create_bronze_table,
            op_kwargs       = {"dataset_name": dataset_name},
            doc_md          = (
                f"CREATE TABLE IF NOT EXISTS bronze_{dataset_name} in "
                f"{CATALOG}.{SCHEMA} using the schema resolved from the "
                f"freshly written contract."
            ),
        )
        create_table_tasks.append(t)

    # =========================================================================
    # PHASE 3 — ingest tasks  (one per dataset, each after its own create_table)
    # =========================================================================
    ingest_tasks = []

    for dataset_name in DATASETS:
        t = PythonOperator(
            task_id         = f"ingest__{dataset_name}",
            python_callable = ingest_dataset,
            op_kwargs       = {"dataset_name": dataset_name},
            doc_md          = (
                f"Download {dataset_name} parquet from the URL in the contract "
                f"to the local staging area, then execute COPY INTO / MERGE "
                f"into bronze_{dataset_name} via Databricks SQL API."
            ),
        )
        ingest_tasks.append(t)

    # =========================================================================
    # PHASE 4 — dbt run + test
    # =========================================================================
    task_dbt_run = BashOperator(
        task_id      = "dbt_run_staging",
        bash_command = (
            "dbt run "
            f"--project-dir  {DBT_PROJECT_DIR} "
            f"--profiles-dir {DBT_PROFILES_DIR} "
            "--select staging.* "
            "--target dev "
            "--no-use-colors "
            "--log-format json"
        ),
        doc_md = (
            "Run all dbt models under models/staging/ after all Bronze "
            "ingestion tasks complete. dbt is installed inside the Airflow "
            "container via airflow/requirements.txt."
        ),
    )

    task_dbt_test = BashOperator(
        task_id      = "dbt_test_staging",
        bash_command = (
            "dbt test "
            f"--project-dir  {DBT_PROJECT_DIR} "
            f"--profiles-dir {DBT_PROFILES_DIR} "
            "--select staging.* "
            "--target dev "
            "--no-use-colors "
            "--log-format json"
        ),
        doc_md = "Run dbt tests on all staging models after dbt run completes.",
    )

    # =========================================================================
    # PHASE 5 — notify
    # =========================================================================
    task_notify = PythonOperator(
        task_id         = "notify_complete",
        python_callable = notify_complete,
        doc_md          = (
            "Log final pipeline summary including contract timestamp and "
            "dataset list pulled from XCom. Extend with Slack/email alert."
        ),
        trigger_rule    = "all_success",
    )


    # Phase 0 → Phase 1: validate_env fans out to all extract tasks
    task_validate_env >> extract_tasks

    # Phase 1 → Phase 1 fanin: all extract tasks must finish before contract is written
    extract_tasks >> task_finalise_contract

    # Phase 1 fanin → Phase 2: contract fans out to all create_table tasks
    task_finalise_contract >> create_table_tasks

    # Phase 2 → Phase 3: each create_table feeds only its own ingest task
    for ct_task, ing_task in zip(create_table_tasks, ingest_tasks):
        ct_task >> ing_task

    # Phase 3 → Phase 4: all ingest tasks fan in to dbt
    ingest_tasks >> task_dbt_run

    # Phase 4 chain → Phase 5
    task_dbt_run >> task_dbt_test >> task_notify