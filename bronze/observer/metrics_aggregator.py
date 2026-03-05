"""
Metrics Aggregator
Maintains daily per-dataset counters in `bronze_ingestion_metrics`.
Uses a PostgreSQL INSERT … ON CONFLICT DO UPDATE (upsert) so every ingestion
run contributes to the running day-level totals without a preceding SELECT.

Metrics tracked (bronze_observability.md §6.1):
  - ingestion_success_total        : count of successful runs today
  - ingestion_failures_total       : count of failed runs today
  - ingestion_rows_total           : cumulative rows ingested today
  - ingestion_duration_seconds     : cumulative wall-clock seconds today
  - schema_evolution_count         : times schema drift was detected today

Called at ingestion completion (success OR failure).

Referenced by: bronze_orchestrator.py
"""

import logging
from datetime import date, datetime, timezone
from typing import Optional
from uuid import UUID

from .db_pool import pg_connection

logger = logging.getLogger("bronze.observability")


class MetricsAggregator:
    """
    Upserts daily metrics for a dataset into `bronze_ingestion_metrics`.

    Args:
        config_path: Path to databricks.cfg (forwarded to db_pool).
    """

    def __init__(self, config_path: str = "databricks/databricks.cfg") -> None:
        self.config_path = config_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_ingestion(
        self,
        trace_id: UUID,
        dataset_name: str,
        success: bool,
        row_count: Optional[int],
        duration_ms: int,
        schema_evolved: bool = False,
        metric_date: Optional[date] = None,
    ) -> None:
        """
        Upsert today's metric counters for the given dataset.

        Increments the relevant success/failure counter, adds rows and
        duration to the running daily totals, and bumps schema_evolution_count
        if the caller detected schema drift.

        Args:
            trace_id:       Run correlation UUID (for log correlation).
            dataset_name:   Dataset being tracked.
            success:        True if Databricks returned SUCCEEDED.
            row_count:      Rows inserted/merged (None treated as 0).
            duration_ms:    Wall-clock duration in milliseconds.
            schema_evolved: True if schema drift was detected this run.
            metric_date:    Override date (defaults to UTC today). Useful in tests.
        """
        today = metric_date or datetime.now(timezone.utc).date()
        duration_seconds = duration_ms / 1000.0
        rows = row_count or 0

        success_delta  = 1 if success else 0
        failure_delta  = 0 if success else 1
        schema_delta   = 1 if schema_evolved else 0

        # Upsert: insert a fresh row for the (dataset, date) pair,
        # or add the run's contribution to existing day-level counters.
        upsert_sql = """
            INSERT INTO bronze_ingestion_metrics
                (dataset_name, metric_date,
                 ingestion_success_total, ingestion_failures_total,
                 ingestion_rows_total, ingestion_duration_seconds,
                 schema_evolution_count)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_name, metric_date) DO UPDATE SET
                ingestion_success_total    = bronze_ingestion_metrics.ingestion_success_total    + EXCLUDED.ingestion_success_total,
                ingestion_failures_total   = bronze_ingestion_metrics.ingestion_failures_total   + EXCLUDED.ingestion_failures_total,
                ingestion_rows_total       = bronze_ingestion_metrics.ingestion_rows_total       + EXCLUDED.ingestion_rows_total,
                ingestion_duration_seconds = bronze_ingestion_metrics.ingestion_duration_seconds + EXCLUDED.ingestion_duration_seconds,
                schema_evolution_count     = bronze_ingestion_metrics.schema_evolution_count     + EXCLUDED.schema_evolution_count;
        """
        with pg_connection(self.config_path) as conn:
            with conn.cursor() as cur:
                cur.execute(upsert_sql, (
                    dataset_name,
                    today,
                    success_delta,
                    failure_delta,
                    rows,
                    duration_seconds,
                    schema_delta,
                ))

        logger.info(
            "Metrics updated",
            extra={
                "event":            "bronze_metrics_updated",
                "trace_id":         str(trace_id),
                "dataset_name":     dataset_name,
                "metric_date":      str(today),
                "success":          success,
                "row_count":        rows,
                "duration_seconds": round(duration_seconds, 3),
                "schema_evolved":   schema_evolved,
            },
        )
