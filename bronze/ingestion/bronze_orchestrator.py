import json
import os
import time
import threading
import tempfile
import uuid
from datetime import date, datetime, timezone
from typing import Dict, List, Literal, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .partition_strategy import PartitionHeuristics
from .sql_generator import BronzeSQLGenerator
from ...databricks.databricks_client import DatabricksSQLClient, SQLExecutionLogger
from .data_downloader import DataDownloader

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
from .observer.db_pool import pg_connection


# ---------------------------------------------------------------------------
# Rerun mode type alias
# ---------------------------------------------------------------------------

RerunMode = Literal["skip_completed", "failed_only", "full"]


# ---------------------------------------------------------------------------
# Thread-safe file logger wrapper
# ---------------------------------------------------------------------------

class _ThreadSafeExecutionLogger:
    """
    Wraps SQLExecutionLogger with a lock so concurrent threads never
    interleave their JSON writes to the audit log file.
    """

    def __init__(self, log_file: str = "bronze_ingestion_log.json"):
        self._logger = SQLExecutionLogger(log_file=log_file)
        self._lock = threading.Lock()

    def log_execution(self, **kwargs):
        with self._lock:
            self._logger.log_execution(**kwargs)


# ---------------------------------------------------------------------------
# Per-dataset ingestion result
# ---------------------------------------------------------------------------

class _DatasetResult:
    """Carries the outcome of one dataset's ingestion thread."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.success: bool = False
        self.skipped: bool = False
        self.skip_reason: Optional[str] = None
        self.error: Optional[str] = None
        self.row_count: Optional[int] = None
        self.duration_ms: int = 0


# ---------------------------------------------------------------------------
# Checkpoint reader
# ---------------------------------------------------------------------------

class _CheckpointReader:
    """
    Reads bronze_ingestion_audit to determine which datasets to skip.
    All queries run in the main thread before the worker pool starts.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path

    def get_completed_today(self, run_date: Optional[date] = None) -> Set[str]:
        """
        Return dataset names whose most recent run today ended in SUCCESS.
        A later FAILED run on the same day overrides an earlier SUCCESS.
        """
        today = run_date or datetime.now(timezone.utc).date()
        sql = """
            SELECT DISTINCT ON (dataset_name)
                dataset_name,
                status
            FROM bronze_ingestion_audit
            WHERE execution_time::date = %s
            ORDER BY dataset_name, execution_time DESC;
        """
        completed: Set[str] = set()
        try:
            with pg_connection(self.config_path) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (today,))
                    for dataset_name, status in cur.fetchall():
                        if status == "SUCCESS":
                            completed.add(dataset_name)
        except Exception as exc:
            # Fail open — if we cannot read the checkpoint, process everything
            print(f"  [WARN] Checkpoint query failed ({exc}). Processing all datasets.")
        return completed

    def get_failed_today(self, run_date: Optional[date] = None) -> Set[str]:
        """
        Return dataset names whose most recent run today ended in FAILED or RUNNING.
        RUNNING rows indicate a crashed run that never reached a terminal state.
        """
        today = run_date or datetime.now(timezone.utc).date()
        sql = """
            SELECT DISTINCT ON (dataset_name)
                dataset_name,
                status
            FROM bronze_ingestion_audit
            WHERE execution_time::date = %s
            ORDER BY dataset_name, execution_time DESC;
        """
        needs_rerun: Set[str] = set()
        try:
            with pg_connection(self.config_path) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (today,))
                    for dataset_name, status in cur.fetchall():
                        if status in ("FAILED", "RUNNING"):
                            needs_rerun.add(dataset_name)
        except Exception as exc:
            print(f"  [WARN] Checkpoint query failed ({exc}). Processing all datasets.")
        return needs_rerun


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class BronzeLayerOrchestrator:
    """
    Metadata-driven bronze layer orchestrator.
    No data bytes pass through this machine.
    Datasets are ingested concurrently with checkpoint-based skip logic.
    """

    def __init__(
        self,
        contract_path: str,
        config_path: str,
        catalog: str = "main",
        schema: str = "bronze",
        delta_root: str = "",
        wait_timeout: int = 900,
        max_workers: int = 3,
        max_failures: int = 2,
    ):
        """
        Args:
            contract_path: Path to bronze_ingestion_contract.json.
            config_path:   Path to databricks.cfg (Databricks + Postgres).
            catalog:       Unity Catalog name.
            schema:        Schema / database name.
            delta_root:    Base location for Delta tables. Empty string
                           lets Unity Catalog manage storage automatically.
            wait_timeout:  Seconds to wait for each Databricks SQL statement.
            max_workers:   Maximum concurrent dataset threads.
                           Keep at 3 or below to avoid warehouse queue pressure.
            max_failures:  Halt the pipeline if this many datasets fail.
        """
        self.contract_path = contract_path
        self.config_path = config_path
        self.catalog = catalog
        self.schema = schema
        self.wait_timeout = wait_timeout
        self.max_workers = max_workers
        self.max_failures = max_failures

        configure_logging()

        with open(contract_path, "r") as f:
            self.contract = json.load(f)

        self._obs_rules = ObservabilityContractParser.parse_all(self.contract)
        ensure_observability_tables(config_path)

        self._audit = AuditWriter(config_path)
        self._metrics = MetricsAggregator(config_path)
        self._file_logger = _ThreadSafeExecutionLogger()
        self._checkpoint = _CheckpointReader(config_path)

        self.sql_generator = BronzeSQLGenerator(
            catalog=catalog,
            schema=schema,
            base_location=delta_root,
        )
        self.downloader = DataDownloader(max_retries=3)

        self._failure_count = 0
        self._failure_lock = threading.Lock()

        print(f"\n{'='*80}")
        print("Bronze Layer Orchestrator  [cloud-native | concurrent | resilient]")
        print(f"{'='*80}")
        print(f"Contract:     {contract_path}")
        print(f"Config:       {config_path}")
        print(f"Catalog:      {catalog}  |  Schema: {schema}")
        print(f"Datasets:     {len(self.contract['datasets'])}")
        print(f"Workers:      {max_workers}")
        print(f"Timeout:      {wait_timeout}s per statement")
        print(f"Max failures: {max_failures}")
        print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # Public pipeline steps
    # ------------------------------------------------------------------

    def create_bronze_tables(
        self,
        datasets: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> None:
        """
        Create bronze Delta tables concurrently (DDL only).
        CREATE TABLE IF NOT EXISTS makes this safe to re-run at any time.
        """
        print(f"\n{'='*80}")
        print("STEP 1: CREATE BRONZE TABLES")
        print(f"{'='*80}\n")

        datasets_to_process = self._get_datasets_to_process(datasets)
        timestamp = datetime.utcnow().isoformat()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._create_one_table, dataset, timestamp, dry_run
                ): dataset["dataset_name"]
                for dataset in datasets_to_process
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"  [ERROR] create_bronze_tables: {name} — {exc}")

        print(f"\n{'='*80}")
        print("Table Creation Complete")
        print(f"{'='*80}\n")

    def ingest_data(
        self,
        datasets: Optional[List[str]] = None,
        download: bool = True,
        dry_run: bool = False,
        rerun_mode: RerunMode = "skip_completed",
        run_date: Optional[date] = None,
    ) -> None:
        """
        Ingest datasets concurrently with checkpoint-based skip logic.

        Args:
            datasets:    Names to process (None = all from contract).
            download:    When True, HEAD-verify HuggingFace URLs before
                         submitting to Databricks (~200ms, no data moved).
                         When False, use URLs from the contract directly.
            dry_run:     Generate SQL but do not execute.
            rerun_mode:  Controls which datasets are processed:

                "skip_completed" (default)
                    Skip datasets that succeeded today. Safe for daily reruns.

                "failed_only"
                    Process only FAILED or RUNNING datasets from today.
                    Use after fixing the root cause of a specific failure.

                "full"
                    Ignore prior run status. COPY INTO force=false keeps
                    this idempotent — no rows are double-inserted.

            run_date:    Date for checkpoint queries (UTC). Default = today.
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: INGEST DATA  [mode={rerun_mode} | concurrent]")
        print(f"{'='*80}\n")

        datasets_to_process = self._get_datasets_to_process(datasets)
        self._failure_count = 0

        # Checkpoint runs in the main thread once, before the pool starts
        datasets_to_run, datasets_to_skip = self._apply_checkpoint(
            datasets_to_process, rerun_mode, run_date
        )

        for name, reason in datasets_to_skip:
            print(f"  [SKIP] {name} — {reason}")

        if not datasets_to_run:
            print("  All datasets already completed. Nothing to do.")
            print(f"\n{'='*80}")
            print("Data Ingestion Complete")
            print(f"{'='*80}\n")
            return

        print(f"  Queuing {len(datasets_to_run)} dataset(s)...\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._ingest_one_dataset, dataset, download, dry_run
                ): dataset["dataset_name"]
                for dataset in datasets_to_run
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result: _DatasetResult = future.result()
                    if result.skipped:
                        print(f"  [SKIP]   {name} — {result.skip_reason}")
                    elif result.success:
                        rows = f"{result.row_count:,} rows" if result.row_count else "completed"
                        print(f"  [OK]     {name} — {rows} in {result.duration_ms}ms")
                    else:
                        print(f"  [FAILED] {name} — {result.error}")
                except Exception as exc:
                    print(f"  [ERROR]  {name} — unhandled: {exc}")

        print(f"\n{'='*80}")
        print("Data Ingestion Complete")
        print(f"{'='*80}\n")

    def optimize_tables(
        self,
        datasets: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> None:
        """OPTIMIZE + VACUUM all tables concurrently."""
        print(f"\n{'='*80}")
        print("STEP 3: OPTIMIZE TABLES  [concurrent]")
        print(f"{'='*80}\n")

        datasets_to_process = self._get_datasets_to_process(datasets)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._optimize_one_table, dataset, dry_run
                ): dataset["dataset_name"]
                for dataset in datasets_to_process
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"  [ERROR] optimize: {name} — {exc}")

        print(f"\n{'='*80}")
        print("Table Optimization Complete")
        print(f"{'='*80}\n")

    def run_full_pipeline(
        self,
        datasets: Optional[List[str]] = None,
        download: bool = True,
        optimize: bool = False,
        dry_run: bool = False,
        rerun_mode: RerunMode = "skip_completed",
        run_date: Optional[date] = None,
    ) -> None:
        """Full pipeline: create → ingest → (optionally) optimize."""
        print(f"\n{'#'*80}")
        print("BRONZE LAYER PIPELINE  [cloud-native | concurrent | resilient]")
        print(f"{'#'*80}\n")
        print(f"Mode:         {'DRY RUN' if dry_run else 'EXECUTE'}")
        print(f"Rerun mode:   {rerun_mode}")
        print(f"Datasets:     {', '.join(datasets) if datasets else 'ALL'}")
        print(f"URL verify:   {download}")
        print(f"Optimize:     {optimize}")

        start_time = datetime.utcnow()

        try:
            self.create_bronze_tables(datasets=datasets, dry_run=dry_run)
            self.ingest_data(
                datasets=datasets,
                download=download,
                dry_run=dry_run,
                rerun_mode=rerun_mode,
                run_date=run_date,
            )
            if optimize:
                self.optimize_tables(datasets=datasets, dry_run=dry_run)

            duration = (datetime.utcnow() - start_time).total_seconds()
            print(f"\n{'#'*80}")
            print("PIPELINE COMPLETE")
            print(f"Duration: {duration:.1f}s")
            print(f"{'#'*80}\n")

        except Exception as exc:
            print(f"\n{'#'*80}")
            print("PIPELINE FAILED")
            print(f"Error: {exc}")
            print(f"{'#'*80}\n")
            raise

        finally:
            close_pool()

    # ------------------------------------------------------------------
    # Thread workers
    # ------------------------------------------------------------------

    def _create_one_table(self, dataset: Dict, timestamp: str, dry_run: bool) -> None:
        """Thread worker: CREATE TABLE IF NOT EXISTS for one dataset."""
        dataset_name = dataset["dataset_name"]
        trace_id = uuid.uuid4()
        blog = BronzeLogger(dataset_name)

        print(f"  >> CREATE TABLE: {dataset_name}  ({dataset['total_rows']:,} rows)")

        partition_config = PartitionHeuristics.determine_strategy(
            dataset_name=dataset_name,
            total_rows=dataset["total_rows"],
            columns=dataset["files"][0]["columns"],
            file_count=dataset["file_count"],
        )

        audit_id = None
        if not dry_run:
            audit_id = self._audit.insert_running(
                trace_id=trace_id,
                dataset_name=dataset_name,
                partition_strategy=partition_config.strategy.value,
            )

        create_sql = self.sql_generator.generate_create_table_sql(
            dataset_metadata=dataset,
            partition_config=partition_config,
            timestamp=timestamp,
        )
        blog.log_sql_generated(
            trace_id=trace_id,
            partition_strategy=partition_config.strategy.value,
            sql_type="CREATE_TABLE",
        )

        sql_file = os.path.join(
            tempfile.gettempdir(), f"bronze_{dataset_name}_create.sql"
        )
        with open(sql_file, "w") as f:
            f.write(create_sql)

        if dry_run:
            print(f"  [DRY RUN] {dataset_name} CREATE TABLE -> {sql_file}")
            return

        db_client = self._make_db_client()
        start_ms = int(time.time() * 1000)

        try:
            result = db_client.execute_sql(create_sql, wait_timeout=self.wait_timeout)
            blog.log_sql_executed(
                trace_id=trace_id,
                statement_id=result.statement_id,
                status=result.status,
                row_count=result.row_count,
                duration_ms=result.duration_ms,
            )
            self._audit.update_completed(
                audit_id=audit_id,
                trace_id=trace_id,
                statement_id=result.statement_id,
                status=result.status,
                row_count=result.row_count,
                duration_ms=result.duration_ms,
                error_message=result.error_message,
            )
            self._metrics.record_ingestion(
                trace_id=trace_id,
                dataset_name=dataset_name,
                success=result.status == "SUCCEEDED",
                row_count=result.row_count,
                duration_ms=result.duration_ms,
            )
            self._file_logger.log_execution(
                dataset_name=dataset_name,
                sql_type="CREATE_TABLE",
                result=result,
                sql_statement=create_sql,
            )
            status_str = "[OK]" if result.status == "SUCCEEDED" else f"[FAILED] {result.error_message}"
            print(f"  {status_str} CREATE TABLE {dataset_name} ({result.duration_ms}ms)")

        except Exception as exc:
            duration_ms = int(time.time() * 1000) - start_ms
            blog.log_sql_failed(trace_id=trace_id, error_message=str(exc))
            self._audit.mark_failed(
                audit_id=audit_id, trace_id=trace_id,
                error_message=str(exc), duration_ms=duration_ms,
            )
            self._metrics.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=False, row_count=None, duration_ms=duration_ms,
            )
            raise

    def _ingest_one_dataset(
        self,
        dataset: Dict,
        download: bool,
        dry_run: bool,
    ) -> _DatasetResult:
        """
        Thread worker: resolve URL → generate SQL → execute ingestion.

        Routing:
          COPY INTO — partition_config.use_append_only is False.
                      Faster for bounded snapshots; no row comparison.
          MERGE     — partition_config.use_append_only is True.
                      Used for large/streaming datasets where deduplication
                      across partial loads is genuinely required.
        """
        dataset_name = dataset["dataset_name"]
        result_obj = _DatasetResult(dataset_name)
        trace_id = uuid.uuid4()
        blog = BronzeLogger(dataset_name)
        rules = self._obs_rules.get(dataset_name)
        evaluator = ObservabilityRuleEvaluator(rules) if rules else None

        print(f"  >> INGEST: {dataset_name}")

        # ── Step 1: Resolve source URL ─────────────────────────────────
        source_url: Optional[str] = None

        if download:
            url_results = self.downloader.resolve_dataset_urls(dataset)
            if not self.downloader.validate_urls(url_results):
                msg = "URL verification failed"
                result_obj.error = msg
                self._record_skip(trace_id, dataset_name, msg)
                self._increment_failures(dataset_name)
                return result_obj
            source_url = url_results[0].volume_path
        else:
            files = dataset.get("files", [])
            if not files or not files[0].get("url"):
                result_obj.error = "No URL in contract"
                return result_obj
            source_url = files[0]["url"]

        print(f"     URL: .../{source_url.split('/')[-1]}")

        # ── Step 2: Partition strategy ─────────────────────────────────
        partition_config = PartitionHeuristics.determine_strategy(
            dataset_name=dataset_name,
            total_rows=dataset["total_rows"],
            columns=dataset["files"][0]["columns"],
            file_count=dataset["file_count"],
        )

        # ── Step 3: Audit RUNNING ──────────────────────────────────────
        audit_id = None
        if not dry_run:
            audit_id = self._audit.insert_running(
                trace_id=trace_id,
                dataset_name=dataset_name,
                partition_strategy=partition_config.strategy.value,
            )

        # ── Step 4: Generate SQL — let partition heuristics decide ─────
        use_merge = partition_config.use_append_only
        sql_type = "MERGE" if use_merge else "COPY_INTO"

        ingest_sql = self.sql_generator.generate_ingestion_sql(
            dataset_metadata=dataset,
            partition_config=partition_config,
            source_url=source_url,
            use_merge=use_merge,
        )
        blog.log_sql_generated(
            trace_id=trace_id,
            partition_strategy=partition_config.strategy.value,
            sql_type=sql_type,
        )

        sql_file = os.path.join(
            tempfile.gettempdir(), f"bronze_{dataset_name}_ingest.sql"
        )
        with open(sql_file, "w") as f:
            f.write(ingest_sql)

        if dry_run:
            print(f"  [DRY RUN] {dataset_name} {sql_type} -> {sql_file}")
            result_obj.success = True
            return result_obj

        # ── Step 5: Execute ────────────────────────────────────────────
        db_client = self._make_db_client()
        start_ms = int(time.time() * 1000)

        try:
            result = db_client.execute_sql(ingest_sql, wait_timeout=self.wait_timeout)
            duration_ms = int(time.time() * 1000) - start_ms

            blog.log_sql_executed(
                trace_id=trace_id,
                statement_id=result.statement_id,
                status=result.status,
                row_count=result.row_count,
                duration_ms=result.duration_ms,
            )
            self._audit.update_completed(
                audit_id=audit_id, trace_id=trace_id,
                statement_id=result.statement_id, status=result.status,
                row_count=result.row_count, duration_ms=result.duration_ms,
                error_message=result.error_message,
            )

            success = result.status == "SUCCEEDED"

            self._metrics.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=success, row_count=result.row_count,
                duration_ms=result.duration_ms,
            )
            if evaluator and success:
                evaluator.evaluate(
                    trace_id=trace_id,
                    row_count=result.row_count,
                    duration_ms=result.duration_ms,
                )
            self._file_logger.log_execution(
                dataset_name=dataset_name, sql_type="INGEST",
                result=result, sql_statement=ingest_sql,
            )

            result_obj.success = success
            result_obj.row_count = result.row_count
            result_obj.duration_ms = result.duration_ms
            if not success:
                result_obj.error = result.error_message
                self._increment_failures(dataset_name)
            return result_obj

        except Exception as exc:
            duration_ms = int(time.time() * 1000) - start_ms
            blog.log_sql_failed(trace_id=trace_id, error_message=str(exc))
            self._audit.mark_failed(
                audit_id=audit_id, trace_id=trace_id,
                error_message=str(exc), duration_ms=duration_ms,
            )
            self._metrics.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=False, row_count=None, duration_ms=duration_ms,
            )
            result_obj.error = str(exc)
            self._increment_failures(dataset_name)
            return result_obj

    def _optimize_one_table(self, dataset: Dict, dry_run: bool) -> None:
        """Thread worker: OPTIMIZE + VACUUM one table."""
        dataset_name = dataset["dataset_name"]
        trace_id = uuid.uuid4()
        blog = BronzeLogger(dataset_name)

        optimize_sql = self.sql_generator.generate_optimization_sql(dataset_name)
        blog.log_sql_generated(
            trace_id=trace_id, partition_strategy="N/A", sql_type="OPTIMIZE"
        )

        if dry_run:
            print(f"  [DRY RUN] {dataset_name} OPTIMIZE skipped")
            return

        db_client = self._make_db_client()
        try:
            result = db_client.execute_sql(optimize_sql, wait_timeout=self.wait_timeout)
            blog.log_sql_executed(
                trace_id=trace_id, statement_id=result.statement_id,
                status=result.status, row_count=result.row_count,
                duration_ms=result.duration_ms,
            )
            status_str = "[OK]" if result.status == "SUCCEEDED" else f"[FAILED] {result.error_message}"
            print(f"  {status_str} OPTIMIZE {dataset_name} ({result.duration_ms}ms)")
        except Exception as exc:
            blog.log_sql_failed(trace_id=trace_id, error_message=str(exc))
            raise

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _apply_checkpoint(
        self,
        datasets: List[Dict],
        rerun_mode: RerunMode,
        run_date: Optional[date],
    ) -> Tuple[List[Dict], List[Tuple[str, str]]]:
        """
        Partition datasets into (to_run, to_skip) based on rerun_mode.

        Returns:
            to_run:  Dataset dicts that should be processed.
            to_skip: (dataset_name, reason) pairs for logging.
        """
        if rerun_mode == "full":
            return datasets, []

        if rerun_mode == "skip_completed":
            completed = self._checkpoint.get_completed_today(run_date)
            to_run, to_skip = [], []
            for ds in datasets:
                name = ds["dataset_name"]
                if name in completed:
                    to_skip.append((name, "already succeeded today"))
                else:
                    to_run.append(ds)
            return to_run, to_skip

        if rerun_mode == "failed_only":
            needs_rerun = self._checkpoint.get_failed_today(run_date)
            to_run, to_skip = [], []
            for ds in datasets:
                name = ds["dataset_name"]
                if name in needs_rerun:
                    to_run.append(ds)
                else:
                    to_skip.append((name, "not in failed set — skipping"))
            return to_run, to_skip

        return datasets, []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_db_client(self) -> DatabricksSQLClient:
        """Fresh DatabricksSQLClient per thread — no shared HTTP session."""
        return DatabricksSQLClient(
            config_path=self.config_path,
            catalog=self.catalog,
            schema=self.schema,
        )

    def _record_skip(
        self,
        trace_id: uuid.UUID,
        dataset_name: str,
        error_msg: str,
    ) -> None:
        """Write a FAILED audit + metrics record for an error-skipped dataset."""
        try:
            audit_id = self._audit.insert_running(
                trace_id=trace_id,
                dataset_name=dataset_name,
                partition_strategy="N/A",
            )
            self._audit.mark_failed(
                audit_id=audit_id, trace_id=trace_id, error_message=error_msg,
            )
            self._metrics.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=False, row_count=None, duration_ms=0,
            )
        except Exception:
            pass  # observability must not crash the pipeline

    def _increment_failures(self, dataset_name: str) -> None:
        """Increment failure counter; raise if max_failures is breached."""
        with self._failure_lock:
            self._failure_count += 1
            count = self._failure_count

        if count >= self.max_failures:
            raise RuntimeError(
                f"Pipeline halted: {count} dataset(s) failed "
                f"(threshold: {self.max_failures}). "
                f"Last failure: {dataset_name}"
            )

    def _get_datasets_to_process(
        self, dataset_names: Optional[List[str]]
    ) -> List[Dict]:
        """Filter contract datasets by the optional name list."""
        all_datasets = self.contract["datasets"]
        if dataset_names is None:
            return all_datasets
        return [d for d in all_datasets if d["dataset_name"] in dataset_names]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("""
Bronze Layer Orchestrator — cloud-native + concurrent + resilient
=================================================================

Normal daily run:
    orchestrator.run_full_pipeline(
        rerun_mode = "skip_completed",   # skip datasets that already succeeded today
        download   = True,
        dry_run    = False,
    )

Partial rerun after fixing a failure:
    orchestrator.run_full_pipeline(
        rerun_mode = "failed_only",      # only FAILED/RUNNING datasets today
        download   = True,
        dry_run    = False,
    )

Forced full re-ingestion (safe — COPY INTO force=false prevents double-insert):
    orchestrator.run_full_pipeline(
        rerun_mode = "full",
        download   = True,
        dry_run    = False,
    )
    """)