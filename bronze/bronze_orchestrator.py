"""
Bronze Layer Orchestrator  (refactored — observability-integrated)
===================================================================
Main orchestration script for metadata-driven bronze layer ingestion.
Python orchestrates only — Spark / Databricks SQL does all processing.

Observability changes vs original
----------------------------------
Every ingestion run now carries a UUID trace_id that flows through:

  Phase                        | Observability action
  -----------------------------|-----------------------------------------------------
  Orchestrator init            | configure_logging() + ensure_observability_tables()
  Per-dataset: start           | generate trace_id → AuditWriter.insert_running()
  Per-dataset: SQL generated   | BronzeLogger.log_sql_generated()
  Per-dataset: SQL executed    | BronzeLogger.log_sql_executed()
                               | AuditWriter.update_completed()
                               | MetricsAggregator.record_ingestion()
                               | ObservabilityRuleEvaluator.evaluate()
  Per-dataset: exception       | BronzeLogger.log_sql_failed()
                               | AuditWriter.mark_failed()
                               | MetricsAggregator.record_ingestion(success=False)
  Shutdown                     | close_pool()

No distributed tracing / Prometheus / OpenTelemetry — see §11 of
bronze_observability.md for scope boundaries.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import uuid

# Core pipeline modules (unchanged)
from .partition_strategy import PartitionHeuristics
from .sql_generator      import BronzeSQLGenerator
from .databricks_client  import DatabricksSQLClient, SQLExecutionLogger
from .data_downloader    import DataDownloader

# Observability layer (new)
from .observer import (
    configure_logging,
    ensure_observability_tables,
    BronzeLogger,
    AuditWriter,
    MetricsAggregator,
    ObservabilityContractParser,
    ObservabilityRuleEvaluator,
    close_pool,
)


class BronzeLayerOrchestrator:
    """
    Orchestrates metadata-driven bronze layer ingestion.
    All processing is pushed to Spark / Databricks SQL.
    Observability (audit, metrics, structured logging) is handled via
    the bronze_observability package backed by PostgreSQL.
    """

    def __init__(
        self,
        contract_path:  str,
        config_path:    str,
        catalog:        str = "main",
        schema:         str = "bronze",
        staging_root:   str = "/mnt/staging/raw",
        delta_root:     str = "/mnt/delta/bronze",
    ):
        """
        Initialize the orchestrator.

        Args:
            contract_path: Path to bronze_ingestion_contract.json.
            config_path:   Path to databricks.cfg (shared by Databricks + Postgres).
            catalog:       Unity Catalog name.
            schema:        Schema / database name.
            staging_root:  Root path for staging raw files.
            delta_root:    Root path for Delta tables.
        """
        self.contract_path = contract_path
        self.config_path   = config_path
        self.catalog       = catalog
        self.schema        = schema

        configure_logging()

        with open(contract_path, "r") as f:
            self.contract = json.load(f)

        self._obs_rules = ObservabilityContractParser.parse_all(self.contract)

        ensure_observability_tables(config_path)

        self._audit    = AuditWriter(config_path)
        self._metrics  = MetricsAggregator(config_path)

        # Instantiate pipeline components (unchanged from original)
        self.sql_generator = BronzeSQLGenerator(
            catalog          = catalog,
            schema           = schema,
            base_location    = delta_root,
            staging_location = staging_root,
        )
        self.downloader = DataDownloader(staging_root=staging_root)
        self.db_client  = DatabricksSQLClient(
            config_path = config_path,
            catalog     = catalog,
            schema      = schema,
        )
        self.logger     = SQLExecutionLogger()   # original file-based logger kept

        print(f"\n{'='*80}")
        print("Bronze Layer Orchestrator Initialized")
        print(f"{'='*80}")
        print(f"Contract:  {contract_path}")
        print(f"Config:    {config_path}")
        print(f"Catalog:   {catalog}")
        print(f"Schema:    {schema}")
        print(f"Datasets:  {len(self.contract['datasets'])}")
        print(f"{'='*80}\n")

   
    def create_bronze_tables(
        self,
        datasets:  Optional[List[str]] = None,
        dry_run:   bool = False,
    ) -> None:
        """
        Create bronze Delta tables from contract metadata.

        A trace_id is generated per dataset.  The CREATE TABLE statement
        is treated as an auditable ingestion event so the audit + metrics
        tables remain consistent with every Databricks API call.

        Args:
            datasets: Dataset names to process (None = all).
            dry_run:  Generate SQL but do not execute.
        """
        print(f"\n{'='*80}")
        print("STEP 1: CREATE BRONZE TABLES")
        print(f"{'='*80}\n")

        datasets_to_process = self._get_datasets_to_process(datasets)
        timestamp = datetime.utcnow().isoformat()

        for i, dataset in enumerate(datasets_to_process, 1):
            dataset_name = dataset["dataset_name"]
            print(f"\n[{i}/{len(datasets_to_process)}] Processing: {dataset_name}")
            print(f"   Rows: {dataset['total_rows']:,}")

            # ── Observability: generate trace_id ────
            trace_id = uuid.uuid4()
            blog     = BronzeLogger(dataset_name)

            partition_config = PartitionHeuristics.determine_strategy(
                dataset_name = dataset_name,
                total_rows   = dataset["total_rows"],
                columns      = dataset["files"][0]["columns"],
                file_count   = dataset["file_count"],
            )
            print(f"   Strategy:   {partition_config.strategy.value}")
            if partition_config.partition_columns:
                print(f"   Partitions: {', '.join(partition_config.partition_columns)}")
            print(f"   Reason:     {partition_config.reason}")

            # ── Observability: audit RUNNING ────
            audit_id = None
            if not dry_run:
                audit_id = self._audit.insert_running(
                    trace_id           = trace_id,
                    dataset_name       = dataset_name,
                    partition_strategy = partition_config.strategy.value,
                )

            create_sql = self.sql_generator.generate_create_table_sql(
                dataset_metadata = dataset,
                partition_config = partition_config,
                timestamp        = timestamp,
            )

            # ── Observability: log SQL generated ───
            blog.log_sql_generated(
                trace_id           = trace_id,
                partition_strategy = partition_config.strategy.value,
                sql_type           = "CREATE_TABLE",
            )

            sql_file = os.path.join(
                tempfile.gettempdir(), f"bronze_{dataset_name}_create.sql"
            )
            with open(sql_file, "w") as f:
                f.write(create_sql)
            print(f"   SQL saved: {sql_file}")

            if dry_run:
                print("   [DRY RUN] Skipping execution")
                continue

            # ── Execute via Databricks SQL API ─────────────────────────
            print("   Executing CREATE TABLE…")
            start_ms = int(time.time() * 1000)
            try:
                result = self.db_client.execute_sql(create_sql)

                # ── Observability: log execution result ─────────────
                blog.log_sql_executed(
                    trace_id     = trace_id,
                    statement_id = result.statement_id,
                    status       = result.status,
                    row_count    = result.row_count,
                    duration_ms  = result.duration_ms,
                )

                # ── Observability: audit update ──────────────────────
                self._audit.update_completed(
                    audit_id      = audit_id,
                    trace_id      = trace_id,
                    statement_id  = result.statement_id,
                    status        = result.status,
                    row_count     = result.row_count,
                    duration_ms   = result.duration_ms,
                    error_message = result.error_message,
                )

                # ── Observability: metrics update ────────────────────
                self._metrics.record_ingestion(
                    trace_id    = trace_id,
                    dataset_name= dataset_name,
                    success     = result.status == "SUCCEEDED",
                    row_count   = result.row_count,
                    duration_ms = result.duration_ms,
                )

                # ── Original file-based logger (kept for compatibility)
                self.logger.log_execution(
                    dataset_name   = dataset_name,
                    sql_type       = "CREATE_TABLE",
                    result         = result,
                    sql_statement  = create_sql,
                )

                if result.status == "SUCCEEDED":
                    print(f"Table created ({result.duration_ms}ms)")
                else:
                    print(f"Failed: {result.error_message}")

            except Exception as exc:
                duration_ms = int(time.time() * 1000) - start_ms
                blog.log_sql_failed(trace_id=trace_id, error_message=str(exc))
                self._audit.mark_failed(
                    audit_id      = audit_id,
                    trace_id      = trace_id,
                    error_message = str(exc),
                    duration_ms   = duration_ms,
                )
                self._metrics.record_ingestion(
                    trace_id     = trace_id,
                    dataset_name = dataset_name,
                    success      = False,
                    row_count    = None,
                    duration_ms  = duration_ms,
                )
                raise

        print(f"\n{'='*80}")
        print("Table Creation Complete")
        print(f"{'='*80}\n")

    def ingest_data(
        self,
        datasets:  Optional[List[str]] = None,
        download:  bool = True,
        dry_run:   bool = False,
    ) -> None:
        """
        Ingest data into bronze Delta tables.

        A fresh trace_id is generated per dataset so logs, audit rows,
        and metrics are independently traceable per ingestion run.

        Args:
            datasets: Dataset names to process (None = all).
            download: Download raw Parquet files before ingesting.
            dry_run:  Generate SQL but do not execute.
        """
        print(f"\n{'='*80}")
        print("STEP 2: INGEST DATA")
        print(f"{'='*80}\n")

        datasets_to_process = self._get_datasets_to_process(datasets)

        for i, dataset in enumerate(datasets_to_process, 1):
            dataset_name = dataset["dataset_name"]
            print(f"\n[{i}/{len(datasets_to_process)}] Ingesting: {dataset_name}")

            # ── Observability: generate trace_id ──────────────────────
            trace_id = uuid.uuid4()
            blog     = BronzeLogger(dataset_name)
            rules    = self._obs_rules.get(dataset_name)
            evaluator= ObservabilityRuleEvaluator(rules) if rules else None

            # ── Download raw data ──────────────────────────────────────
            if download:
                download_results = self.downloader.download_dataset(dataset)
                if not self.downloader.validate_downloads(download_results):
                    print("   Download failed, skipping ingestion")
                    # Audit the skip as a failure
                    if not dry_run:
                        audit_id = self._audit.insert_running(
                            trace_id           = trace_id,
                            dataset_name       = dataset_name,
                            partition_strategy = "N/A",
                        )
                        self._audit.mark_failed(
                            audit_id      = audit_id,
                            trace_id      = trace_id,
                            error_message = "Download failed before ingestion",
                        )
                        self._metrics.record_ingestion(
                            trace_id     = trace_id,
                            dataset_name = dataset_name,
                            success      = False,
                            row_count    = None,
                            duration_ms  = 0,
                        )
                    continue

            # ── Determine partition strategy ───────────────────────────
            partition_config = PartitionHeuristics.determine_strategy(
                dataset_name = dataset_name,
                total_rows   = dataset["total_rows"],
                columns      = dataset["files"][0]["columns"],
                file_count   = dataset["file_count"],
            )

            # ── Observability: audit RUNNING ───────────────────────────
            audit_id = None
            if not dry_run:
                audit_id = self._audit.insert_running(
                    trace_id           = trace_id,
                    dataset_name       = dataset_name,
                    partition_strategy = partition_config.strategy.value,
                )

            # ── Generate ingestion SQL ─────────────────────────────────
            use_merge  = partition_config.use_append_only or dataset["total_rows"] > 300_000
            ingest_sql = self.sql_generator.generate_ingestion_sql(
                dataset_metadata = dataset,
                partition_config = partition_config,
                use_merge        = use_merge,
            )

            # ── Observability: log SQL generated ──────────────────────
            sql_type = "MERGE" if use_merge else "COPY_INTO"
            blog.log_sql_generated(
                trace_id           = trace_id,
                partition_strategy = partition_config.strategy.value,
                sql_type           = sql_type,
            )

            sql_file = os.path.join(
                tempfile.gettempdir(), f"bronze_{dataset_name}_ingest.sql"
            )
            with open(sql_file, "w") as f:
                f.write(ingest_sql)
            print(f"   SQL saved: {sql_file}")

            if dry_run:
                print("   [DRY RUN] Skipping execution")
                continue

            # ── Execute via Databricks SQL API ─────────────────────────
            print(f"   Executing {sql_type}…")
            start_ms = int(time.time() * 1000)
            try:
                result = self.db_client.execute_sql(ingest_sql)
                duration_ms = int(time.time() * 1000) - start_ms

                # ── Observability: log execution result ─────────────
                blog.log_sql_executed(
                    trace_id     = trace_id,
                    statement_id = result.statement_id,
                    status       = result.status,
                    row_count    = result.row_count,
                    duration_ms  = result.duration_ms,
                )

                # ── Observability: audit update ──────────────────────
                self._audit.update_completed(
                    audit_id      = audit_id,
                    trace_id      = trace_id,
                    statement_id  = result.statement_id,
                    status        = result.status,
                    row_count     = result.row_count,
                    duration_ms   = result.duration_ms,
                    error_message = result.error_message,
                )

                success = result.status == "SUCCEEDED"

                # ── Observability: metrics update ────────────────────
                self._metrics.record_ingestion(
                    trace_id     = trace_id,
                    dataset_name = dataset_name,
                    success      = success,
                    row_count    = result.row_count,
                    duration_ms  = result.duration_ms,
                )

                # ── Observability: rule evaluation ───────────────────
                if evaluator and success:
                    evaluator.evaluate(
                        trace_id    = trace_id,
                        row_count   = result.row_count,
                        duration_ms = result.duration_ms,
                    )

                # ── Original file-based logger (kept for compatibility)
                self.logger.log_execution(
                    dataset_name  = dataset_name,
                    sql_type      = "INGEST",
                    result        = result,
                    sql_statement = ingest_sql,
                )

                if success:
                    row_info = f"{result.row_count:,} rows" if result.row_count else "completed"
                    print(f"   ✓ Ingestion complete: {row_info} ({result.duration_ms}ms)")
                else:
                    print(f"   ✗ Failed: {result.error_message}")

            except Exception as exc:
                duration_ms = int(time.time() * 1000) - start_ms
                blog.log_sql_failed(trace_id=trace_id, error_message=str(exc))
                self._audit.mark_failed(
                    audit_id      = audit_id,
                    trace_id      = trace_id,
                    error_message = str(exc),
                    duration_ms   = duration_ms,
                )
                self._metrics.record_ingestion(
                    trace_id     = trace_id,
                    dataset_name = dataset_name,
                    success      = False,
                    row_count    = None,
                    duration_ms  = duration_ms,
                )
                raise

        print(f"\n{'='*80}")
        print("Data Ingestion Complete")
        print(f"{'='*80}\n")

    def optimize_tables(
        self,
        datasets: Optional[List[str]] = None,
        dry_run:  bool = False,
    ) -> None:
        """
        Optimize bronze Delta tables (OPTIMIZE + VACUUM).
        Optimisation runs are not audited individually — they share no
        ingested rows — but execution errors are logged via BronzeLogger.

        Args:
            datasets: Dataset names to process (None = all).
            dry_run:  Generate SQL but do not execute.
        """
        print(f"\n{'='*80}")
        print("STEP 3: OPTIMIZE TABLES")
        print(f"{'='*80}\n")

        datasets_to_process = self._get_datasets_to_process(datasets)

        for i, dataset in enumerate(datasets_to_process, 1):
            dataset_name = dataset["dataset_name"]
            print(f"\n[{i}/{len(datasets_to_process)}] Optimizing: {dataset_name}")

            trace_id     = uuid.uuid4()
            blog         = BronzeLogger(dataset_name)
            optimize_sql = self.sql_generator.generate_optimization_sql(dataset_name)

            blog.log_sql_generated(
                trace_id           = trace_id,
                partition_strategy = "N/A",
                sql_type           = "OPTIMIZE",
            )

            if dry_run:
                print("   [DRY RUN] Skipping execution")
                continue

            print("   Executing OPTIMIZE…")
            try:
                result = self.db_client.execute_sql(optimize_sql)
                blog.log_sql_executed(
                    trace_id     = trace_id,
                    statement_id = result.statement_id,
                    status       = result.status,
                    row_count    = result.row_count,
                    duration_ms  = result.duration_ms,
                )
                if result.status == "SUCCEEDED":
                    print(f"   ✓ Optimization complete ({result.duration_ms}ms)")
                else:
                    print(f"   ✗ Failed: {result.error_message}")
            except Exception as exc:
                blog.log_sql_failed(trace_id=trace_id, error_message=str(exc))
                raise

        print(f"\n{'='*80}")
        print("Table Optimization Complete")
        print(f"{'='*80}\n")

    def run_full_pipeline(
        self,
        datasets: Optional[List[str]] = None,
        download: bool = True,
        optimize: bool = False,
        dry_run:  bool = False,
    ) -> None:
        """
        Run the complete bronze layer pipeline end-to-end.

        Args:
            datasets: Dataset names to process (None = all).
            download: Download raw Parquet files before ingesting.
            optimize: Run OPTIMIZE + VACUUM after ingestion.
            dry_run:  Generate SQL but do not execute.
        """
        print(f"\n{'#'*80}")
        print("BRONZE LAYER PIPELINE — FULL RUN")
        print(f"{'#'*80}\n")
        print(f"Mode:      {'DRY RUN' if dry_run else 'EXECUTE'}")
        print(f"Datasets:  {', '.join(datasets) if datasets else 'ALL'}")
        print(f"Download:  {download}")
        print(f"Optimize:  {optimize}")

        start_time = datetime.utcnow()

        try:
            self.create_bronze_tables(datasets=datasets, dry_run=dry_run)
            self.ingest_data(datasets=datasets, download=download, dry_run=dry_run)
            if optimize:
                self.optimize_tables(datasets=datasets, dry_run=dry_run)

            duration = (datetime.utcnow() - start_time).total_seconds()
            print(f"\n{'#'*80}")
            print("PIPELINE COMPLETE")
            print(f"{'#'*80}")
            print(f"Duration: {duration:.1f}s")
            print(f"{'#'*80}\n")

        except Exception as exc:
            print(f"\n{'#'*80}")
            print("PIPELINE FAILED")
            print(f"{'#'*80}")
            print(f"Error: {exc}")
            print(f"{'#'*80}\n")
            raise

        finally:
            # Close the Postgres connection pool cleanly on pipeline exit
            close_pool()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_datasets_to_process(
        self, dataset_names: Optional[List[str]]
    ) -> List[Dict]:
        """Return contract datasets filtered by the optional name list."""
        all_datasets = self.contract["datasets"]
        if dataset_names is None:
            return all_datasets
        return [d for d in all_datasets if d["dataset_name"] in dataset_names]


# ---------------------------------------------------------------------------
# CLI / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("""
Bronze Layer Orchestrator (observability-integrated)
=====================================================

Usage:
    orchestrator = BronzeLayerOrchestrator(
        contract_path = 'bronze_ingestion_contract.json',
        config_path   = 'databricks/databricks.cfg',
        catalog       = 'main',
        schema        = 'bronze',
    )

    # Full pipeline
    orchestrator.run_full_pipeline(
        datasets = ['billing_payments'],   # None = all datasets
        download = True,
        optimize = False,
        dry_run  = True,                   # Set False to execute
    )

    # Individual steps
    orchestrator.create_bronze_tables(dry_run=True)
    orchestrator.ingest_data(download=True, dry_run=True)
    orchestrator.optimize_tables(dry_run=True)

databricks/databricks.cfg — add a [POSTGRES] section:
    [DEFAULT]
    workspace_url = https://xxx.cloud.databricks.com
    token         = dapi...
    warehouse_id  = abc123

    [POSTGRES]
    host     = localhost
    port     = 5432
    dbname   = bronze_control
    user     = pipeline_user
    password = secret

Per-dataset observability rules (optional, add to contract):
    {
      "dataset_name": "billing_payments",
      ...
      "observability": {
        "alert_on_zero_rows":        true,
        "max_expected_duration_sec": 120,
        "expected_min_rows":         1000
      }
    }
    """)
