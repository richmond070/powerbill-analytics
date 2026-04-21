"""
Audit Writer  (thread-safe)
===========================
Manages lifecycle writes to `bronze_ingestion_audit` in PostgreSQL.
Acts as the Bronze layer's "black box recorder" (bronze_observability.md §4).

Thread-safety
-------------
Every public method calls pg_connection() which borrows one connection
from the ThreadedConnectionPool for the duration of the with-block and
returns it immediately after commit/rollback.  No connection object is
stored on the instance, so concurrent threads never share a connection.

Lifecycle (per ingestion run):
  1. insert_running()   — called immediately after trace_id is generated
  2. update_completed() — called after Databricks SQL API returns
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from .db_pool import pg_connection

logger = logging.getLogger("bronze.observability")


class AuditWriter:
    """
    Writes ingestion lifecycle records to `bronze_ingestion_audit`.

    Thread-safe: each method opens and closes its own database connection
    via pg_connection().  No state is stored between calls.

    Args:
        config_path: Path to databricks.cfg (forwarded to db_pool).
    """

    def __init__(self, config_path: str = "databricks/databricks.cfg") -> None:
        # Store config path only — no connection held on the instance.
        # This is intentional: holding a connection on self would mean
        # all threads share the same connection object, causing
        # "connection already used by another thread" errors under psycopg2.
        self.config_path = config_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_running(
        self,
        trace_id: UUID,
        dataset_name: str,
        partition_strategy: str,
    ) -> int:
        """
        Insert a RUNNING audit record immediately after trace_id generation.

        Thread-safety: borrows a fresh connection from the pool for this
        call only.  Safe to call from multiple threads simultaneously.

        Args:
            trace_id:           Run correlation UUID.
            dataset_name:       Dataset being ingested.
            partition_strategy: Strategy resolved by PartitionHeuristics.

        Returns:
            audit_id (SERIAL PK) — pass to update_completed().
        """
        sql = """
            INSERT INTO bronze_ingestion_audit
                (trace_id, dataset_name, partition_strategy, status)
            VALUES
                (%s, %s, %s, 'RUNNING')
            RETURNING id;
        """
        # pg_connection() borrows a connection, commits on exit, returns it.
        # Each thread that calls this method gets a different connection.
        with pg_connection(self.config_path) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(trace_id), dataset_name, partition_strategy))
                audit_id: int = cur.fetchone()[0]

        logger.debug(
            "Audit record inserted (RUNNING)",
            extra={
                "event":        "audit_insert_running",
                "trace_id":     str(trace_id),
                "dataset_name": dataset_name,
                "audit_id":     audit_id,
            },
        )
        return audit_id

    def update_completed(
        self,
        audit_id: int,
        trace_id: UUID,
        statement_id: str,
        status: str,
        row_count: Optional[int],
        duration_ms: int,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the audit record with final execution state.

        Thread-safety: borrows a fresh connection for this call only.

        Args:
            audit_id:      PK returned by insert_running().
            trace_id:      Run correlation UUID (for log correlation).
            statement_id:  Databricks statement ID.
            status:        "SUCCEEDED" | "FAILED" | "CANCELED".
            row_count:     Rows inserted/merged (None if unavailable).
            duration_ms:   Total wall-clock duration in milliseconds.
            error_message: Error description when status != "SUCCEEDED".
        """
        # Normalise Databricks statuses → audit table CHECK constraint values
        audit_status = "SUCCESS" if status == "SUCCEEDED" else "FAILED"

        sql = """
            UPDATE bronze_ingestion_audit
            SET
                statement_id  = %s,
                status        = %s,
                row_count     = %s,
                duration_ms   = %s,
                error_message = %s
            WHERE id = %s;
        """
        with pg_connection(self.config_path) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    statement_id,
                    audit_status,
                    row_count,
                    duration_ms,
                    error_message,
                    audit_id,
                ))

        logger.debug(
            "Audit record updated (%s)", audit_status,
            extra={
                "event":       "audit_update_completed",
                "trace_id":    str(trace_id),
                "audit_id":    audit_id,
                "status":      audit_status,
                "row_count":   row_count,
                "duration_ms": duration_ms,
            },
        )

    def mark_failed(
        self,
        audit_id: int,
        trace_id: UUID,
        error_message: str,
        duration_ms: int = 0,
    ) -> None:
        """
        Convenience wrapper — marks a RUNNING record as FAILED.
        Use when an exception is raised before the Databricks API returns.

        Thread-safety: borrows a fresh connection for this call only.
        """
        self.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id="N/A",
            status="FAILED",
            row_count=None,
            duration_ms=duration_ms,
            error_message=error_message,
        )