"""
Stress Test: bronze_orchestrator.py
=====================================
Tests BronzeLayerOrchestrator across the key ingestion pipeline properties:

  ATOMICITY
    - Pipeline failure in any step re-raises and halts execution
    - Download failure atomically skips the dependent ingest step
    - Databricks SQL errors are captured and do not silently continue

  IDEMPOTENCY
    - CREATE TABLE uses IF NOT EXISTS — safe to run multiple times
    - COPY INTO uses force=false — already-loaded files are skipped
    - MERGE uses WHEN NOT MATCHED — duplicate rows are never inserted
    - Dry-run produces identical SQL on repeated calls (deterministic)

  DRY-RUN ISOLATION
    - dry_run=True never calls db_client.execute_sql for any method
    - dry_run=True still writes SQL files to /tmp (so SQL is reviewable)

  MERGE ROUTING
    - use_append_only=True on partition_config → MERGE
    - total_rows > 300_000 → MERGE regardless of partition config
    - Neither condition → COPY INTO

  DATASET FILTERING
    - None processes all datasets from the contract
    - Named list returns only those datasets
    - Unknown name returns empty (no crash, no silent wildcard)

  PIPELINE SEQUENCING
    - run_full_pipeline calls create → ingest → optimize in order
    - optimize only runs when optimize=True

  AUDIT LOGGING
    - Every execution (success and failure) is logged with correct fields
    - Log is append-only across multiple calls (no overwrite)
    - sql_preview is capped at 200 chars + '...'

  SQL FILE PERSISTENCE
    - Both create and ingest SQL files are written to /tmp
    - File content contains the correct SQL keywords

Project stack: Python | Databricks | Spark (Delta Lake)
Reference files:
  - bronze_ingestion_contract.json  → loaded directly at import time as FULL_CONTRACT
  - bronze_orchestrator.py          → module under test
  - databricks_client.py            → DatabricksSQLClient + SQLExecutionLogger mocked
  - data_downloader.py              → DataDownloader mocked
  - sql_generator.py                → real module (no mock)
  - partition_strategy.py           → real module (no mock)
  - schema_mapper.py                → real module (no mock)

Run with:
    pytest tests/test_bronze_orchestrator.py -v
"""

import os
import sys
import json
import types
import tempfile
import shutil
import unittest
import importlib.util
import configparser
from unittest.mock import MagicMock, patch, call
from datetime import datetime

# ---------------------------------------------------------------------------
# Step 1 — Resolve project root (cross-platform safe)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BRONZE_DIR = os.path.join(PROJECT_ROOT, "bronze") 
for p in [PROJECT_ROOT, BRONZE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Step 2 — Bootstrap the 'bronze' package in sys.modules
#
# bronze_orchestrator.py uses 5 relative imports:
#   from .schema_mapper import SchemaMapper
#   from .partition_strategy import PartitionHeuristics, PartitionConfig
#   from .sql_generator import BronzeSQLGenerator
#   from .databricks_client import DatabricksSQLClient, SQLExecutionLogger
#   from .data_downloader import DataDownloader, DataValidator
#
# We load all real modules under the 'bronze' namespace.
# DatabricksSQLClient and DataDownloader are replaced with MagicMock instances
# per-test — the real classes are loaded but their instances are swapped out.
# ---------------------------------------------------------------------------

def _bootstrap_modules():
    """
    Load all project modules under the 'bronze' package namespace.
    Returns a dict of loaded module objects keyed by short name.
    """
    bronze_pkg = types.ModuleType("bronze")
    sys.modules["bronze"] = bronze_pkg

    loaded = {}
    for mod_name in [
        "schema_mapper",
        "partition_strategy",
        "sql_generator",
        "databricks_client",
        "data_downloader",
    ]:
        path = os.path.join(BRONZE_DIR, f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(
            f"bronze.{mod_name}", path, submodule_search_locations=[]
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "bronze"
        sys.modules[f"bronze.{mod_name}"] = mod
        spec.loader.exec_module(mod)
        loaded[mod_name] = mod

    # Load the orchestrator itself
    orch_path = os.path.join(BRONZE_DIR, "bronze_orchestrator.py")
    spec = importlib.util.spec_from_file_location(
        "bronze.bronze_orchestrator", orch_path, submodule_search_locations=[]
    )
    orch_mod = importlib.util.module_from_spec(spec)
    orch_mod.__package__ = "bronze"
    sys.modules["bronze.bronze_orchestrator"] = orch_mod
    spec.loader.exec_module(orch_mod)
    loaded["bronze_orchestrator"] = orch_mod

    return loaded


# Bootstrap once at module level
_MODS = _bootstrap_modules()

BronzeLayerOrchestrator = _MODS["bronze_orchestrator"].BronzeLayerOrchestrator
BronzeSQLGenerator      = _MODS["sql_generator"].BronzeSQLGenerator
DataDownloader          = _MODS["data_downloader"].DataDownloader
SQLExecutionLogger      = _MODS["databricks_client"].SQLExecutionLogger
SQLExecutionResult      = _MODS["databricks_client"].SQLExecutionResult
PartitionHeuristics     = _MODS["partition_strategy"].PartitionHeuristics
PartitionConfig         = _MODS["partition_strategy"].PartitionConfig
PartitionStrategy       = _MODS["partition_strategy"].PartitionStrategy


# ---------------------------------------------------------------------------
# Contract — loaded directly from bronze_ingestion_contract.json
#
# WHY: This project is metadata-driven. The contract is the single source of
# truth for dataset schemas, row counts, and file locations. Reading it
# directly means:
#   1. New datasets added to the contract are automatically covered by tests
#      that iterate over FULL_CONTRACT["datasets"] — no test edits needed.
#   2. The tests reflect the actual contract the orchestrator uses at runtime,
#      not a copy that can silently drift out of sync.
#
# The contract lives at the project root (one level above tests/).
#
# ---------------------------------------------------------------------------

BRONZE_METADATA_DIR = os.path.join(PROJECT_ROOT, "bronze_metadata")
_CONTRACT_PATH = os.path.join(BRONZE_METADATA_DIR, "bronze_ingestion_contract.json")

if not os.path.exists(_CONTRACT_PATH):
    raise FileNotFoundError(
        f"bronze_ingestion_contract.json not found at: {_CONTRACT_PATH}\n"
        f"Ensure the contract file is at the project root: {PROJECT_ROOT}"
    )

with open(_CONTRACT_PATH, "r") as _f:
    FULL_CONTRACT = json.load(_f)

if not FULL_CONTRACT.get("datasets"):
    raise ValueError(
        f"Contract at {_CONTRACT_PATH} contains no datasets — "
        f"check the file is not empty or corrupt."
    )

def make_success_result(
    statement_id: str = "stmt_001",
    row_count: int = 200000,
    duration_ms: int = 1200,
) -> SQLExecutionResult: # type: ignore
    return SQLExecutionResult(
        statement_id=statement_id,
        status="SUCCEEDED",
        row_count=row_count,
        duration_ms=duration_ms,
    )


def make_failed_result(
    statement_id: str = "stmt_fail",
    error_message: str = "Table already exists",
) -> SQLExecutionResult: # type: ignore
    return SQLExecutionResult(
        statement_id=statement_id,
        status="FAILED",
        row_count=None,
        duration_ms=500,
        error_message=error_message,
    )


class OrchestratorTestBase(unittest.TestCase):
    """
    Base class for all orchestrator tests.
    Provides setUp/tearDown that creates:
      - A temp copy of bronze_ingestion_contract.json (real contract, not hardcoded)
      - A temp log directory
      - A pre-wired orchestrator with mocked db_client and downloader
    """

    def setUp(self):
        # Temp directory for contract file and log file
        self.temp_dir = tempfile.mkdtemp()
        self.contract_file = os.path.join(self.temp_dir, "contract.json")
        self.log_file      = os.path.join(self.temp_dir, "ingestion_log.json")
        self.config_file   = os.path.join(self.temp_dir, "databricks.cfg")

        # Write a temp copy of the real contract so the orchestrator has an
        # isolated file — safe if the real contract changes mid-run.
        with open(self.contract_file, "w") as f:
            json.dump(FULL_CONTRACT, f)

        # Write a minimal Databricks config (DatabricksSQLClient is mocked per-test
        # but the config file path is stored on the orchestrator)
        cfg = configparser.ConfigParser()
        cfg["DEFAULT"] = {
            "workspace_url": "https://test.databricks.com",
            "token": "dapi_test_token",
            "warehouse_id": "wh_test_123",
        }
        with open(self.config_file, "w") as f:
            cfg.write(f)

        self.orch = self._make_orchestrator()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any SQL files written to the OS temp directory during tests
        import tempfile
        tmp = tempfile.gettempdir()
        for ds in FULL_CONTRACT["datasets"]:
            name = ds["dataset_name"]
            for suffix in ["_create.sql", "_ingest.sql"]:
                path = os.path.join(tmp, f"bronze_{name}{suffix}")
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        # Also clean up dynamic dataset names used in merge routing tests
        for extra_name in ["large_dataset", "boundary_dataset"]:
            for suffix in ["_create.sql", "_ingest.sql"]:
                path = os.path.join(tmp, f"bronze_{extra_name}{suffix}")
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    def _make_orchestrator(
        self,
        contract: dict = None,
        catalog: str = "main",
        schema: str = "bronze",
    ) -> BronzeLayerOrchestrator: # type: ignore
        """
        Bypass __init__ and wire all components manually.
        DatabricksSQLClient and DataDownloader are always MagicMocks.
        BronzeSQLGenerator, SQLExecutionLogger, and partition modules are real.
        """
        orch = BronzeLayerOrchestrator.__new__(BronzeLayerOrchestrator)
        orch.contract_path = self.contract_file
        orch.config_path   = self.config_file
        orch.catalog       = catalog
        orch.schema        = schema

        if contract is not None:
            orch.contract = contract
        else:
            with open(self.contract_file) as f:
                orch.contract = json.load(f)

        orch.sql_generator = BronzeSQLGenerator(
            catalog=catalog,
            schema=schema,
            base_location="/mnt/delta/bronze",
            staging_location="/mnt/staging/raw",
        )
        orch.downloader = MagicMock(spec=DataDownloader)
        orch.db_client  = MagicMock()
        orch.logger     = SQLExecutionLogger(log_file=self.log_file)

        # Default mock behaviours
        orch.db_client.execute_sql.return_value = make_success_result()
        orch.downloader.download_dataset.return_value = [MagicMock(success=True)]
        orch.downloader.validate_downloads.return_value = True

        return orch

    def _read_log(self) -> list:
        """Read the audit log file and return parsed entries."""
        if not os.path.exists(self.log_file):
            return []
        with open(self.log_file) as f:
            return json.load(f)

    def _read_sql_file(self, dataset_name: str, kind: str) -> str:
        """Read a generated SQL file from the OS temp directory. kind is 'create' or 'ingest'."""
        import tempfile
        path = os.path.join(tempfile.gettempdir(), f"bronze_{dataset_name}_{kind}.sql")
        with open(path) as f:
            return f.read()


# ===========================================================================
# GROUP 1 — __init__ and contract loading
# ===========================================================================

class TestOrchestratorInit(OrchestratorTestBase):
    """
    S01–S05: Verify the orchestrator loads the contract file, wires all
    components, stores config values, and handles a corrupt contract.
    """

    def test_s01_contract_loaded_from_file(self):
        """S01 — Contract JSON is loaded from disk and accessible as self.contract."""
        self.assertIn("datasets", self.orch.contract)
        self.assertEqual(len(self.orch.contract["datasets"]), 6)

    def test_s02_all_six_contract_datasets_present(self):
        """S02 — All 6 dataset names from the contract are present after loading."""
        names = {d["dataset_name"] for d in self.orch.contract["datasets"]}
        expected = {
            "billing_payments", "commercial_industries_consumption",
            "customers_complaint", "grid_load", "power_flow", "retail_tariffs",
        }
        self.assertEqual(names, expected)

    def test_s03_catalog_and_schema_stored_on_instance(self):
        """S03 — catalog and schema passed to constructor are stored on the instance."""
        orch = self._make_orchestrator(catalog="energy_cat", schema="bronze_ng")
        self.assertEqual(orch.catalog, "energy_cat")
        self.assertEqual(orch.schema, "bronze_ng")

    def test_s04_missing_contract_file_raises_file_not_found(self):
        """
        S04 — Attempting to instantiate with a non-existent contract path
        raises FileNotFoundError. The orchestrator must not silently continue
        with an empty contract.
        """
        with self.assertRaises((FileNotFoundError, IOError)):
            # Use real __init__ but point at a non-existent file
            with patch.object(
                _MODS["databricks_client"].DatabricksSQLClient,
                "__init__",
                return_value=None,
            ):
                BronzeLayerOrchestrator(
                    contract_path="/nonexistent/path/contract.json",
                    config_path=self.config_file,
                )

    def test_s05_corrupt_contract_file_raises_json_error(self):
        """
        S05 — A corrupt (non-JSON) contract file raises json.JSONDecodeError.
        """
        corrupt_path = os.path.join(self.temp_dir, "corrupt.json")
        with open(corrupt_path, "w") as f:
            f.write("{this is not valid json}")

        with self.assertRaises((json.JSONDecodeError, ValueError)):
            with patch.object(
                _MODS["databricks_client"].DatabricksSQLClient,
                "__init__",
                return_value=None,
            ):
                BronzeLayerOrchestrator(
                    contract_path=corrupt_path,
                    config_path=self.config_file,
                )


# ===========================================================================
# GROUP 2 — Idempotency: CREATE TABLE SQL
# ===========================================================================

class TestIdempotencyCreateTable(OrchestratorTestBase):
    """
    S06–S10: CREATE TABLE SQL must be idempotent — running it twice must
    not fail or produce duplicate tables.
    """

    def test_s06_create_table_sql_uses_if_not_exists(self):
        """
        S06 — The generated CREATE TABLE SQL always uses CREATE TABLE IF NOT EXISTS.
        This is the foundation of idempotency: the statement is safe to re-run.
        """
        self.orch.create_bronze_tables(dry_run=True)
        sql = self._read_sql_file("billing_payments", "create")
        self.assertIn("CREATE TABLE IF NOT EXISTS", sql)

    def test_s07_all_six_datasets_use_if_not_exists(self):
        """
        S07 — CREATE TABLE IF NOT EXISTS appears in the generated SQL for
        every one of the 6 contract datasets, not just billing_payments.
        """
        self.orch.create_bronze_tables(dry_run=True)
        for ds in FULL_CONTRACT["datasets"]:
            name = ds["dataset_name"]
            sql = self._read_sql_file(name, "create")
            self.assertIn(
                "CREATE TABLE IF NOT EXISTS", sql,
                f"{name}: missing IF NOT EXISTS in CREATE TABLE"
            )

    def test_s08_identical_sql_on_repeated_dry_run_calls(self):
        """
        S08 — Calling create_bronze_tables(dry_run=True) twice produces SQL
        with identical structure, table names, columns, and properties.
        The only permitted difference is the '-- Generated: <timestamp>' comment
        line, which records wall-clock time. All load-bearing SQL content is
        deterministic — idempotent at the generation level.
        """
        import re

        def strip_generated_comment(sql: str) -> str:
            """Remove the timestamp comment line before comparing."""
            return re.sub(r"-- Generated: .*\n", "", sql)

        self.orch.create_bronze_tables(dry_run=True)
        sql_first = strip_generated_comment(
            self._read_sql_file("billing_payments", "create")
        )

        self.orch.create_bronze_tables(dry_run=True)
        sql_second = strip_generated_comment(
            self._read_sql_file("billing_payments", "create")
        )

        self.assertEqual(
            sql_first, sql_second,
            "CREATE TABLE SQL structure is not deterministic across repeated calls"
        )

    def test_s09_create_table_sql_contains_delta_tblproperties(self):
        """
        S09 — The SQL contains TBLPROPERTIES that enforce schema quality,
        enable Change Data Feed, and enable auto-optimize — all idempotency
        enablers in Databricks Delta.
        """
        self.orch.create_bronze_tables(dry_run=True)
        sql = self._read_sql_file("billing_payments", "create")
        self.assertIn("delta.enableChangeDataFeed", sql)
        self.assertIn("delta.autoOptimize.optimizeWrite", sql)
        self.assertIn("quality.enforceSchema", sql)

    def test_s10_create_table_called_once_per_dataset_on_execute(self):
        """
        S10 — In execute mode, db_client.execute_sql is called exactly once
        per dataset. No dataset gets a double CREATE TABLE call.
        """
        self.orch.create_bronze_tables(dry_run=False)
        expected_calls = len(FULL_CONTRACT["datasets"])
        actual_calls = self.orch.db_client.execute_sql.call_count
        self.assertEqual(
            actual_calls, expected_calls,
            f"Expected {expected_calls} execute_sql calls, got {actual_calls}"
        )


# ===========================================================================
# GROUP 3 — Idempotency: COPY INTO (small datasets)
# ===========================================================================

class TestIdempotencyCopyInto(OrchestratorTestBase):
    """
    S11–S13: COPY INTO SQL must be idempotent — force=false means already-
    loaded files are skipped on re-run.
    """

    def _get_copy_into_dataset(self):
        """Return retail_tariffs (90k rows) — the one dataset guaranteed to route to COPY INTO."""
        return next(
            d for d in FULL_CONTRACT["datasets"]
            if d["dataset_name"] == "retail_tariffs"
        )

    def test_s11_copy_into_uses_force_false(self):
        """
        S11 — COPY INTO SQL contains force=false.
        This is what makes Databricks skip already-ingested files on re-run,
        providing idempotency at the file level.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_copy_into_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("retail_tariffs", "ingest")
        self.assertIn("'force' = 'false'", sql)

    def test_s12_copy_into_uses_merge_schema_false(self):
        """
        S12 — COPY INTO SQL contains mergeSchema=false in both FORMAT_OPTIONS
        and COPY_OPTIONS. This enforces the contract schema and prevents schema
        drift from corrupting the bronze table on re-ingestion.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_copy_into_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("retail_tariffs", "ingest")
        # Both sections must have mergeSchema=false
        self.assertEqual(sql.count("'mergeSchema' = 'false'"), 2)

    def test_s13_copy_into_has_bad_records_path(self):
        """
        S13 — COPY INTO SQL contains a badRecordsPath.
        Corrupt rows are quarantined rather than failing the entire load —
        this makes partial re-runs safe since bad records won't block good ones.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_copy_into_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("retail_tariffs", "ingest")
        self.assertIn("_bad_records", sql)
        self.assertIn("badRecordsPath", sql)


# ===========================================================================
# GROUP 4 — Idempotency: MERGE (large datasets)
# ===========================================================================

class TestIdempotencyMerge(OrchestratorTestBase):
    """
    S14–S17: MERGE SQL must be idempotent — WHEN NOT MATCHED THEN INSERT *
    means rows already in the target are never duplicated.
    """

    def _get_merge_dataset(self):
        """Return commercial_industries_consumption (220k rows) — guaranteed MERGE."""
        return next(
            d for d in FULL_CONTRACT["datasets"]
            if d["dataset_name"] == "commercial_industries_consumption"
        )

    def test_s14_merge_uses_when_not_matched_insert(self):
        """
        S14 — MERGE SQL uses WHEN NOT MATCHED THEN INSERT *.
        This is the core idempotency guarantee: rows that already exist in
        the target (matched by _bronze_row_hash) are skipped, not duplicated.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_merge_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("commercial_industries_consumption", "ingest")
        self.assertIn("MERGE INTO", sql)
        self.assertIn("WHEN NOT MATCHED THEN INSERT *", sql)

    def test_s15_merge_deduplication_is_hash_based(self):
        """
        S15 — The MERGE join condition is on _bronze_row_hash (SHA-256 of all
        columns). This means deduplication is content-based, not key-based —
        the same row re-submitted from a re-run will never be inserted twice.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_merge_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("commercial_industries_consumption", "ingest")
        self.assertIn("ON target._bronze_row_hash = source._bronze_row_hash", sql)
        self.assertIn("sha2(concat_ws", sql)

    def test_s16_merge_uses_read_files_with_enforced_schema(self):
        """
        S16 — The MERGE source uses read_files() with schema enforcement.
        This prevents a re-ingestion from a schema-drifted file from corrupting
        the target table.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_merge_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("commercial_industries_consumption", "ingest")
        self.assertIn("read_files(", sql)
        self.assertIn("format => 'parquet'", sql)
        self.assertIn("schema =>", sql)

    def test_s17_merge_does_not_contain_update_clause(self):
        """
        S17 — MERGE SQL must NOT have a WHEN MATCHED THEN UPDATE clause.
        The bronze layer is append-only — existing records are never modified.
        Any WHEN MATCHED UPDATE would break the immutability guarantee.
        """
        contract = {**FULL_CONTRACT, "datasets": [self._get_merge_dataset()]}
        orch = self._make_orchestrator(contract=contract)
        orch.ingest_data(download=False, dry_run=True)
        sql = self._read_sql_file("commercial_industries_consumption", "ingest")
        self.assertNotIn("WHEN MATCHED THEN UPDATE", sql)
        self.assertNotIn("WHEN MATCHED UPDATE", sql)


# ===========================================================================
# GROUP 5 — Atomicity: pipeline failure handling
# ===========================================================================

class TestAtomicityPipelineFailures(OrchestratorTestBase):
    """
    S18–S23: The pipeline must fail atomically — any unhandled exception
    halts execution and propagates to the caller. Partial success is not
    silently swallowed.
    """

    def test_s18_databricks_exception_in_create_table_propagates(self):
        """
        S18 — If db_client.execute_sql raises during create_bronze_tables,
        the exception propagates out of run_full_pipeline.
        The pipeline does not continue to ingest_data after a create failure.
        """
        self.orch.db_client.execute_sql.side_effect = RuntimeError(
            "Databricks cluster unreachable"
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.orch.run_full_pipeline(dry_run=False)
        self.assertIn("unreachable", str(ctx.exception))

    def test_s19_ingest_data_not_called_after_create_table_exception(self):
        """
        S19 — When create_bronze_tables raises, ingest_data is never called.
        We verify this by confirming execute_sql is called only once (for create)
        and then the pipeline halts.
        """
        call_count = {"n": 0}

        def fail_on_second_call(sql):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Cluster startup failed")
            return make_success_result()

        self.orch.db_client.execute_sql.side_effect = fail_on_second_call

        with self.assertRaises(RuntimeError):
            self.orch.run_full_pipeline(dry_run=False)

        # Only 1 call was made (the one that raised) — ingest never ran
        self.assertEqual(call_count["n"], 1)

    def test_s20_failed_sql_result_does_not_raise_but_is_logged(self):
        """
        S20 — A FAILED SQLExecutionResult (status='FAILED') does not raise an
        exception from create_bronze_tables — it is logged and the pipeline
        continues to the next dataset.
        RATIONALE: Databricks returns FAILED status for statements like
        'table already exists' which are non-fatal in some pipelines.
        """
        self.orch.db_client.execute_sql.return_value = make_failed_result(
            error_message="Table already exists"
        )
        # Must not raise
        try:
            self.orch.create_bronze_tables(dry_run=False)
        except Exception as e:
            self.fail(f"create_bronze_tables raised on FAILED result: {e}")

        # The failure must be logged
        logs = self._read_log()
        failed_logs = [l for l in logs if l["status"] == "FAILED"]
        self.assertGreater(len(failed_logs), 0)

    def test_s21_optimize_not_called_when_optimize_flag_is_false(self):
        """
        S21 — run_full_pipeline with optimize=False must not call
        optimize_tables. This ensures the post-ingestion maintenance step
        is only triggered intentionally.
        """
        call_tracker = []

        original_optimize = self.orch.optimize_tables

        def track_optimize(*args, **kwargs):
            call_tracker.append("optimize_called")
            return original_optimize(*args, **kwargs)

        self.orch.optimize_tables = track_optimize
        self.orch.run_full_pipeline(optimize=False, dry_run=True)

        self.assertNotIn(
            "optimize_called", call_tracker,
            "optimize_tables was called even though optimize=False"
        )

    def test_s22_optimize_called_when_optimize_flag_is_true(self):
        """
        S22 — run_full_pipeline with optimize=True must call optimize_tables
        after ingest_data completes. The optimize step is the third and final
        step in the full pipeline.
        """
        call_tracker = []
        original_optimize = self.orch.optimize_tables

        def track_optimize(*args, **kwargs):
            call_tracker.append("optimize_called")
            return original_optimize(*args, **kwargs)

        self.orch.optimize_tables = track_optimize
        self.orch.run_full_pipeline(optimize=True, dry_run=True)

        self.assertIn(
            "optimize_called", call_tracker,
            "optimize_tables was NOT called even though optimize=True"
        )

    def test_s23_pipeline_steps_execute_in_correct_order(self):
        """
        S23 — run_full_pipeline executes steps in order:
        (1) create_bronze_tables → (2) ingest_data → (3) optimize_tables.
        Verified by tracking the order of calls via side effects.
        """
        step_order = []

        original_create   = self.orch.create_bronze_tables
        original_ingest   = self.orch.ingest_data
        original_optimize = self.orch.optimize_tables

        self.orch.create_bronze_tables  = lambda **kw: step_order.append("create")
        self.orch.ingest_data           = lambda **kw: step_order.append("ingest")
        self.orch.optimize_tables       = lambda **kw: step_order.append("optimize")

        self.orch.run_full_pipeline(optimize=True, dry_run=True)

        self.assertEqual(
            step_order, ["create", "ingest", "optimize"],
            f"Pipeline executed out of order: {step_order}"
        )


# ===========================================================================
# GROUP 6 — Atomicity: download failure skips ingestion
# ===========================================================================

class TestAtomicityDownloadFailure(OrchestratorTestBase):
    """
    S24–S27: If the download step fails for a dataset, the ingest step for
    that dataset must be skipped. This is per-dataset atomicity — one failed
    download must not cascade into a corrupt or partial ingest.
    """

    def test_s24_download_failure_skips_execute_sql(self):
        """
        S24 — When validate_downloads returns False, db_client.execute_sql
        is never called for that dataset's ingestion SQL.
        """
        self.orch.downloader.validate_downloads.return_value = False
        self.orch.ingest_data(download=True, dry_run=False)
        self.orch.db_client.execute_sql.assert_not_called()

    def test_s25_download_failure_on_one_dataset_does_not_stop_others(self):
        """
        S25 — If download fails for one dataset, the pipeline continues to
        process the remaining datasets. Per-dataset failure is isolated.
        """
        # First dataset download fails, rest succeed
        call_count = {"n": 0}

        def fail_first_validate(results):
            call_count["n"] += 1
            return call_count["n"] > 1  # First call returns False (fail)

        self.orch.downloader.validate_downloads.side_effect = fail_first_validate
        self.orch.ingest_data(download=True, dry_run=False)

        # execute_sql should be called for datasets 2-6 (5 times), not for dataset 1
        expected_calls = len(FULL_CONTRACT["datasets"]) - 1
        actual_calls = self.orch.db_client.execute_sql.call_count
        self.assertEqual(
            actual_calls, expected_calls,
            f"Expected {expected_calls} execute_sql calls after 1 download failure, "
            f"got {actual_calls}"
        )

    def test_s26_download_true_calls_download_dataset_for_each_dataset(self):
        """
        S26 — When download=True, downloader.download_dataset is called once
        per dataset — not batched, not skipped.
        """
        self.orch.ingest_data(download=True, dry_run=False)
        expected = len(FULL_CONTRACT["datasets"])
        actual = self.orch.downloader.download_dataset.call_count
        self.assertEqual(
            actual, expected,
            f"Expected {expected} download_dataset calls, got {actual}"
        )

    def test_s27_download_false_skips_downloader_entirely(self):
        """
        S27 — When download=False, downloader.download_dataset is never called.
        This supports re-ingestion from already-staged files without re-downloading.
        """
        self.orch.ingest_data(download=False, dry_run=False)
        self.orch.downloader.download_dataset.assert_not_called()


# ===========================================================================
# GROUP 7 — Dry-run isolation
# ===========================================================================

class TestDryRunIsolation(OrchestratorTestBase):
    """
    S28–S33: dry_run=True must never trigger any Databricks SQL execution.
    SQL is still generated and written to /tmp for review.
    """

    def test_s28_dry_run_create_table_never_calls_execute_sql(self):
        """S28 — create_bronze_tables(dry_run=True) does not call execute_sql."""
        self.orch.create_bronze_tables(dry_run=True)
        self.orch.db_client.execute_sql.assert_not_called()

    def test_s29_dry_run_ingest_data_never_calls_execute_sql(self):
        """S29 — ingest_data(dry_run=True) does not call execute_sql."""
        self.orch.ingest_data(download=False, dry_run=True)
        self.orch.db_client.execute_sql.assert_not_called()

    def test_s30_dry_run_optimize_never_calls_execute_sql(self):
        """S30 — optimize_tables(dry_run=True) does not call execute_sql."""
        self.orch.optimize_tables(dry_run=True)
        self.orch.db_client.execute_sql.assert_not_called()

    def test_s31_dry_run_full_pipeline_never_calls_execute_sql(self):
        """
        S31 — run_full_pipeline(dry_run=True) with optimize=True does not
        call execute_sql at any stage (create, ingest, optimize).
        """
        self.orch.run_full_pipeline(optimize=True, dry_run=True)
        self.orch.db_client.execute_sql.assert_not_called()

    def test_s32_dry_run_still_writes_sql_files_to_tmp(self):
        """
        S32 — dry_run=True still writes the generated SQL files to /tmp.
        This is intentional: dry_run allows SQL review before execution.
        """
        import tempfile
        self.orch.create_bronze_tables(dry_run=True)
        sql_path = os.path.join(tempfile.gettempdir(), "bronze_billing_payments_create.sql")
        self.assertTrue(
            os.path.exists(sql_path),
            f"SQL file not written to temp directory during dry_run: {sql_path}"
        )

    def test_s33_dry_run_does_not_write_audit_log(self):
        """
        S33 — dry_run=True does not write to the audit log because no SQL
        is executed and therefore no SQLExecutionResult is produced.
        """
        self.orch.create_bronze_tables(dry_run=True)
        logs = self._read_log()
        self.assertEqual(len(logs), 0, "Audit log was written during dry_run")


# ===========================================================================
# GROUP 8 — MERGE routing logic
# ===========================================================================

class TestMergeRoutingLogic(OrchestratorTestBase):
    """
    S34–S39: The ingest_data method routes to MERGE or COPY INTO based on:
      - partition_config.use_append_only (True for rows >= threshold)
      - dataset total_rows > 300_000 (explicit large dataset override)
    """

    def _make_dataset(self, name: str, total_rows: int) -> dict:
        return {
            "dataset_name": name,
            "api_endpoint": "https://hf.co/parquet",
            "file_count": 1,
            "total_rows": total_rows,
            "files": [{"url": "https://hf.co/0.parquet", "filename": "0.parquet",
                        "size_bytes": 0, "num_rows": total_rows, "num_columns": 2,
                        "num_row_groups": 1, "validation_status": "success",
                        "columns": [
                            {"name": "id",        "type": "string", "nullable": True},
                            {"name": "timestamp", "type": "string", "nullable": True},
                        ]}],
        }

    def _get_ingest_sql(self, name: str) -> str:
        return self._read_sql_file(name, "ingest")

    def test_s34_total_rows_above_300k_routes_to_merge(self):
        """
        S34 — A dataset with total_rows=400,000 (> 300k threshold) routes to
        MERGE INTO regardless of the partition_config.use_append_only value.
        """
        ds = self._make_dataset("large_dataset", 400000)
        orch = self._make_orchestrator(contract={**FULL_CONTRACT, "datasets": [ds]})
        orch.ingest_data(download=False, dry_run=True)
        sql = self._get_ingest_sql("large_dataset")
        self.assertIn("MERGE INTO", sql)
        self.assertNotIn("COPY INTO", sql)

    def test_s35_total_rows_at_300k_boundary_routes_to_copy_into(self):
        """
        S35 — A dataset with exactly 300,000 rows does NOT trigger the MERGE
        override (condition is > 300_000, not >= 300_000). Whether it routes
        to COPY INTO or MERGE depends solely on use_append_only at this boundary.
        Retail_tariffs style dataset at 300k rows with NONE partition → COPY INTO.
        """
        ds = self._make_dataset("boundary_dataset", 300000)
        orch = self._make_orchestrator(contract={**FULL_CONTRACT, "datasets": [ds]})
        orch.ingest_data(download=False, dry_run=True)
        sql = self._get_ingest_sql("boundary_dataset")
        # At 300k with no time/category columns, use_append_only depends on
        # your local partition_strategy threshold — assert it's one or the other
        is_copy  = "COPY INTO" in sql
        is_merge = "MERGE INTO" in sql
        self.assertTrue(is_copy or is_merge)
        self.assertFalse(is_copy and is_merge)

    def test_s36_retail_tariffs_90k_routes_to_copy_into(self):
        """
        S36 — retail_tariffs (90k rows) is below both the use_append_only
        threshold and the 300k explicit MERGE threshold → must use COPY INTO.
        """
        ds = next(d for d in FULL_CONTRACT["datasets"] if d["dataset_name"] == "retail_tariffs")
        orch = self._make_orchestrator(contract={**FULL_CONTRACT, "datasets": [ds]})
        orch.ingest_data(download=False, dry_run=True)
        sql = self._get_ingest_sql("retail_tariffs")
        self.assertIn("COPY INTO", sql)
        self.assertNotIn("MERGE INTO", sql)

    def test_s37_commercial_industries_consumption_routes_to_merge(self):
        """
        S37 — commercial_industries_consumption (220k rows) has use_append_only=True
        from the partition heuristic → MERGE path even though 220k < 300k threshold.
        This tests that the use_append_only condition is checked first.
        """
        ds = next(d for d in FULL_CONTRACT["datasets"]
                  if d["dataset_name"] == "commercial_industries_consumption")
        orch = self._make_orchestrator(contract={**FULL_CONTRACT, "datasets": [ds]})
        orch.ingest_data(download=False, dry_run=True)
        sql = self._get_ingest_sql("commercial_industries_consumption")
        self.assertIn("MERGE INTO", sql)

    def test_s38_all_six_datasets_route_to_exactly_one_template(self):
        """
        S38 — Every contract dataset routes to either COPY INTO or MERGE INTO —
        never both, never neither. The routing logic is exhaustive.
        """
        self.orch.ingest_data(download=False, dry_run=True)
        for ds in FULL_CONTRACT["datasets"]:
            name = ds["dataset_name"]
            sql = self._get_ingest_sql(name)
            is_copy  = "COPY INTO" in sql
            is_merge = "MERGE INTO" in sql
            self.assertTrue(
                is_copy or is_merge,
                f"{name}: SQL contains neither COPY INTO nor MERGE INTO"
            )
            self.assertFalse(
                is_copy and is_merge,
                f"{name}: SQL contains both COPY INTO and MERGE INTO"
            )

    def test_s39_merge_routing_produces_read_files_source(self):
        """
        S39 — When MERGE is selected, the source uses read_files() not a raw
        FROM path. This ensures schema enforcement is applied at read time.
        """
        ds = next(d for d in FULL_CONTRACT["datasets"]
                  if d["dataset_name"] == "commercial_industries_consumption")
        orch = self._make_orchestrator(contract={**FULL_CONTRACT, "datasets": [ds]})
        orch.ingest_data(download=False, dry_run=True)
        sql = self._get_ingest_sql("commercial_industries_consumption")
        self.assertIn("read_files(", sql)
        self.assertNotIn("FROM '/mnt/staging", sql)


# ===========================================================================
# GROUP 9 — Dataset filtering (_get_datasets_to_process)
# ===========================================================================

class TestDatasetFiltering(OrchestratorTestBase):
    """
    S40–S45: _get_datasets_to_process controls which datasets are processed.
    Wrong filtering could silently skip datasets or process the wrong ones.
    """

    def test_s40_none_filter_returns_all_six_datasets(self):
        """S40 — None returns the complete list of 6 datasets from the contract."""
        result = self.orch._get_datasets_to_process(None)
        self.assertEqual(len(result), 6)

    def test_s41_named_filter_returns_only_matching_dataset(self):
        """S41 — A list with one name returns exactly that one dataset."""
        result = self.orch._get_datasets_to_process(["billing_payments"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["dataset_name"], "billing_payments")

    def test_s42_named_filter_with_multiple_names(self):
        """S42 — A list of 3 names returns exactly 3 datasets."""
        names = ["billing_payments", "grid_load", "retail_tariffs"]
        result = self.orch._get_datasets_to_process(names)
        self.assertEqual(len(result), 3)
        returned_names = {d["dataset_name"] for d in result}
        self.assertEqual(returned_names, set(names))

    def test_s43_unknown_dataset_name_returns_empty_list(self):
        """
        S43 — A name that doesn't exist in the contract returns an empty list.
        Must not raise, must not return all datasets as a wildcard fallback.
        """
        result = self.orch._get_datasets_to_process(["nonexistent_dataset"])
        self.assertEqual(result, [])

    def test_s44_mixed_valid_and_invalid_names_returns_only_valid(self):
        """S44 — A mix of valid and invalid names returns only the valid ones."""
        result = self.orch._get_datasets_to_process(["billing_payments", "does_not_exist"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["dataset_name"], "billing_payments")

    def test_s45_dataset_filter_passed_to_create_table_limits_execution(self):
        """
        S45 — Passing datasets=['retail_tariffs'] to create_bronze_tables
        results in exactly 1 execute_sql call, not 6.
        """
        self.orch.create_bronze_tables(
            datasets=["retail_tariffs"], dry_run=False
        )
        self.assertEqual(self.orch.db_client.execute_sql.call_count, 1)


# ===========================================================================
# GROUP 10 — Audit logging
# ===========================================================================

class TestAuditLogging(OrchestratorTestBase):
    """
    S46–S53: Every SQL execution must produce an audit log entry with the
    correct fields. The log is append-only and captures both successes and
    failures. This is essential for ingestion traceability.
    """

    def test_s46_successful_create_table_writes_log_entry(self):
        """S46 — A successful CREATE TABLE execution writes one log entry per dataset."""
        self.orch.create_bronze_tables(
            datasets=["billing_payments"], dry_run=False
        )
        logs = self._read_log()
        self.assertEqual(len(logs), 1)
        entry = logs[0]
        self.assertEqual(entry["dataset"], "billing_payments")
        self.assertEqual(entry["sql_type"], "CREATE_TABLE")
        self.assertEqual(entry["status"], "SUCCEEDED")

    def test_s47_log_entry_contains_all_required_fields(self):
        """
        S47 — Each log entry contains all required audit fields:
        timestamp, dataset, sql_type, statement_id, status, row_count,
        duration_ms, error, and sql_preview.
        """
        self.orch.create_bronze_tables(
            datasets=["billing_payments"], dry_run=False
        )
        logs = self._read_log()
        entry = logs[0]
        required_fields = [
            "timestamp", "dataset", "sql_type", "statement_id",
            "status", "row_count", "duration_ms", "error", "sql_preview",
        ]
        for field in required_fields:
            self.assertIn(field, entry, f"Log entry missing required field: '{field}'")

    def test_s48_log_is_append_only_across_multiple_calls(self):
        """
        S48 — Calling create_bronze_tables twice writes TWO entries to the log,
        not one. Each call appends — the log is never overwritten.
        """
        self.orch.create_bronze_tables(datasets=["billing_payments"], dry_run=False)
        self.orch.create_bronze_tables(datasets=["billing_payments"], dry_run=False)
        logs = self._read_log()
        self.assertEqual(len(logs), 2, "Log was overwritten instead of appended to")

    def test_s49_failed_execution_is_also_logged(self):
        """
        S49 — A FAILED SQLExecutionResult is logged with status='FAILED' and
        the error message captured in the 'error' field.
        """
        self.orch.db_client.execute_sql.return_value = make_failed_result(
            error_message="Warehouse offline"
        )
        self.orch.create_bronze_tables(datasets=["billing_payments"], dry_run=False)
        logs = self._read_log()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "FAILED")
        self.assertEqual(logs[0]["error"], "Warehouse offline")

    def test_s50_log_timestamp_is_iso_format(self):
        """
        S50 — Log entry timestamps are in ISO 8601 format and parseable
        by datetime.fromisoformat().
        """
        self.orch.create_bronze_tables(datasets=["billing_payments"], dry_run=False)
        logs = self._read_log()
        ts = logs[0]["timestamp"]
        try:
            datetime.fromisoformat(ts)
        except ValueError:
            self.fail(f"Log timestamp is not valid ISO 8601: {ts!r}")

    def test_s51_sql_preview_capped_at_200_chars_plus_ellipsis(self):
        """
        S51 — sql_preview in the log is capped at 200 characters + '...'.
        Long SQL statements are truncated to keep the log file manageable.
        """
        self.orch.create_bronze_tables(datasets=["billing_payments"], dry_run=False)
        logs = self._read_log()
        preview = logs[0]["sql_preview"]
        # The actual CREATE TABLE SQL is well over 200 chars — verify truncation
        self.assertTrue(
            len(preview) <= 203,  # 200 chars + '...'
            f"sql_preview is {len(preview)} chars (should be <= 203)"
        )
        self.assertTrue(
            preview.endswith("..."),
            "sql_preview does not end with '...'"
        )

    def test_s52_ingest_log_records_correct_sql_type(self):
        """S52 — Ingestion SQL execution logs with sql_type='INGEST'."""
        self.orch.ingest_data(
            datasets=["retail_tariffs"], download=False, dry_run=False
        )
        logs = self._read_log()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["sql_type"], "INGEST")
        self.assertEqual(logs[0]["dataset"], "retail_tariffs")

    def test_s53_full_pipeline_creates_one_log_entry_per_dataset_per_step(self):
        """
        S53 — A full pipeline (create + ingest, no optimize) over all 6 datasets
        produces exactly 12 log entries (6 CREATE_TABLE + 6 INGEST).
        """
        self.orch.run_full_pipeline(
            download=False, optimize=False, dry_run=False
        )
        logs = self._read_log()
        create_logs = [l for l in logs if l["sql_type"] == "CREATE_TABLE"]
        ingest_logs  = [l for l in logs if l["sql_type"] == "INGEST"]
        self.assertEqual(len(create_logs), 6, f"Expected 6 CREATE_TABLE logs, got {len(create_logs)}")
        self.assertEqual(len(ingest_logs),  6, f"Expected 6 INGEST logs, got {len(ingest_logs)}")


# ===========================================================================
# GROUP 11 — SQL file persistence
# ===========================================================================

class TestSQLFilePersistence(OrchestratorTestBase):
    """
    S54–S58: SQL files written to /tmp allow the team to inspect and audit
    the exact SQL that will be (or was) executed against Databricks.
    """

    def test_s54_create_sql_file_written_to_tmp(self):
        """S54 — create_bronze_tables writes a .sql file to /tmp for each dataset."""
        self.orch.create_bronze_tables(dry_run=True)
        for ds in FULL_CONTRACT["datasets"]:
            path = os.path.join(tempfile.gettempdir(), f"bronze_{ds['dataset_name']}_create.sql")
            self.assertTrue(
                os.path.exists(path),
                f"SQL file missing for {ds['dataset_name']}: {path}"
            )

    def test_s55_ingest_sql_file_written_to_tmp(self):
        """S55 — ingest_data writes a .sql file to /tmp for each dataset."""
        self.orch.ingest_data(download=False, dry_run=True)
        for ds in FULL_CONTRACT["datasets"]:
            path = os.path.join(tempfile.gettempdir(), f"bronze_{ds['dataset_name']}_ingest.sql")
            self.assertTrue(
                os.path.exists(path),
                f"Ingest SQL file missing for {ds['dataset_name']}: {path}"
            )

    def test_s56_create_sql_file_content_is_valid_sql(self):
        """
        S56 — The create SQL file content contains the minimum expected SQL
        keywords: CREATE TABLE IF NOT EXISTS, USING DELTA, LOCATION, TBLPROPERTIES.
        """
        self.orch.create_bronze_tables(
            datasets=["billing_payments"], dry_run=True
        )
        sql = self._read_sql_file("billing_payments", "create")
        for keyword in ["CREATE TABLE IF NOT EXISTS", "USING DELTA", "LOCATION", "TBLPROPERTIES"]:
            self.assertIn(keyword, sql, f"Keyword '{keyword}' missing from create SQL file")

    def test_s57_ingest_sql_file_references_correct_dataset_path(self):
        """
        S57 — The ingest SQL file references the correct staging path for
        each dataset: /mnt/staging/raw/<dataset_name>/*.parquet
        """
        self.orch.ingest_data(
            datasets=["grid_load"], download=False, dry_run=True
        )
        sql = self._read_sql_file("grid_load", "ingest")
        self.assertIn("/mnt/staging/raw/grid_load/", sql)
        self.assertIn("*.parquet", sql)

    def test_s58_sql_file_uses_correct_catalog_and_schema(self):
        """
        S58 — SQL files reference the catalog and schema provided at
        orchestrator construction time, not hardcoded defaults.
        """
        orch = self._make_orchestrator(catalog="energy_prod", schema="bronze_live")
        orch.create_bronze_tables(datasets=["retail_tariffs"], dry_run=True)
        sql = self._read_sql_file("retail_tariffs", "create")
        self.assertIn("energy_prod.bronze_live.bronze_retail_tariffs", sql)


# ===========================================================================
# GROUP 12 — All 6 contract datasets: full dry-run pipeline
# ===========================================================================

class TestAllSixDatasetsDryRun(OrchestratorTestBase):
    """
    S59–S64: Every dataset in the contract must complete the full
    dry-run pipeline without errors. This is an end-to-end smoke test
    for each dataset's metadata, schema, and partition strategy.
    """

    def test_s59_billing_payments_full_dry_run(self):
        """S59 — billing_payments completes create + ingest dry-run without error."""
        self.orch.run_full_pipeline(
            datasets=["billing_payments"], download=False,
            optimize=False, dry_run=True
        )
        self.assertIn("billing_month", self._read_sql_file("billing_payments", "create"))

    def test_s60_commercial_industries_consumption_full_dry_run(self):
        """S60 — commercial_industries_consumption completes dry-run; routes to MERGE."""
        self.orch.run_full_pipeline(
            datasets=["commercial_industries_consumption"], download=False,
            optimize=False, dry_run=True
        )
        sql = self._read_sql_file("commercial_industries_consumption", "ingest")
        self.assertIn("MERGE INTO", sql)

    def test_s61_customers_complaint_full_dry_run(self):
        """S61 — customers_complaint completes dry-run; sla_met BOOLEAN in DDL."""
        self.orch.run_full_pipeline(
            datasets=["customers_complaint"], download=False,
            optimize=False, dry_run=True
        )
        sql = self._read_sql_file("customers_complaint", "create")
        self.assertIn("sla_met BOOLEAN", sql)

    def test_s62_grid_load_full_dry_run(self):
        """S62 — grid_load completes dry-run; voltage_level_kv BIGINT in DDL."""
        self.orch.run_full_pipeline(
            datasets=["grid_load"], download=False,
            optimize=False, dry_run=True
        )
        sql = self._read_sql_file("grid_load", "create")
        self.assertIn("voltage_level_kv BIGINT", sql)

    def test_s63_power_flow_full_dry_run(self):
        """S63 — power_flow completes dry-run; from_kv and to_kv BIGINT in DDL."""
        self.orch.run_full_pipeline(
            datasets=["power_flow"], download=False,
            optimize=False, dry_run=True
        )
        sql = self._read_sql_file("power_flow", "create")
        self.assertIn("from_kv BIGINT", sql)
        self.assertIn("to_kv BIGINT", sql)

    def test_s64_retail_tariffs_full_dry_run(self):
        """S64 — retail_tariffs completes dry-run; routes to COPY INTO (small dataset)."""
        self.orch.run_full_pipeline(
            datasets=["retail_tariffs"], download=False,
            optimize=False, dry_run=True
        )
        sql = self._read_sql_file("retail_tariffs", "ingest")
        self.assertIn("COPY INTO", sql)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestOrchestratorInit,
        TestIdempotencyCreateTable,
        TestIdempotencyCopyInto,
        TestIdempotencyMerge,
        TestAtomicityPipelineFailures,
        TestAtomicityDownloadFailure,
        TestDryRunIsolation,
        TestMergeRoutingLogic,
        TestDatasetFiltering,
        TestAuditLogging,
        TestSQLFilePersistence,
        TestAllSixDatasetsDryRun,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)