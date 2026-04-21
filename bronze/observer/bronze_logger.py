"""
Bronze Structured Logger  (thread-safe)
========================================
Emits machine-readable JSON log entries for the three key observability
events defined in bronze_observability.md §5:

  Event                  | When emitted
  -----------------------|----------------------------------------------
  bronze_sql_generated   | After metadata-driven SQL is built (§5.1)
  bronze_sql_executed    | After Databricks SQL API returns (§5.2)
  bronze_sql_failed      | When ingestion raises an exception (§5.3)

Thread-safety
-------------
Python's stdlib `logging` module is fully thread-safe by default.
Each `Logger.info/error/warning()` call acquires an internal lock before
writing to the handler.  No additional locking is needed here.

Every BronzeLogger instance uses the SAME underlying logger
("bronze.observability") — this is intentional.  All threads write to
the same JSON stream (stdout), and the logging lock ensures their entries
are never interleaved mid-line.  The trace_id in each entry is sufficient
to separate per-dataset events during analysis.

Each log entry carries:
  - trace_id       (UUID — unique per dataset per run)
  - dataset_name
  - event          (one of the three event types above)
  - timestamp      (ISO-8601 UTC, set by _JsonFormatter)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """
    Renders every log record as a single-line JSON object.

    Thread-safety: format() is called while the logging handler holds
    its lock, so this method is never called concurrently for the same
    handler.  No shared mutable state exists on this class.
    """

    # Keys present on every LogRecord that we do NOT want to forward
    # as extra fields — they are either internal Python logging machinery
    # or already captured in the top-level payload.
    _SKIP = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
        }

        # Attach caller-injected extra fields (trace_id, dataset_name, etc.)
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Module-level logging setup
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to emit JSON to stdout.

    Safe to call multiple times — the guard prevents duplicate handlers
    being added on repeated calls (e.g. in tests or when the orchestrator
    is instantiated more than once in a process).

    Call once in BronzeLayerOrchestrator.__init__ before any log is written.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(level)

    # Only add the handler if no StreamHandler is already present.
    # This prevents double-logging when the function is called from
    # multiple threads simultaneously at startup.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# BronzeLogger — typed convenience wrapper
# ---------------------------------------------------------------------------

class BronzeLogger:
    """
    Typed log emitter for the three Bronze observability events.

    Thread-safety: each instance delegates to stdlib logging which
    serialises concurrent writes internally.  Multiple BronzeLogger
    instances (one per dataset thread) can log simultaneously without
    any additional locking.

    Args:
        dataset_name: Fixed for the lifetime of this logger instance.
                      Injected into every log entry as a field.
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        # All instances share the same underlying logger.
        # stdlib logging is thread-safe — the shared logger is intentional.
        self._log = logging.getLogger("bronze.observability")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_sql_generated(
        self,
        trace_id: UUID,
        partition_strategy: str,
        sql_type: str,
    ) -> None:
        """
        §5.1 — Log SQL generation.

        Args:
            trace_id:           Run correlation UUID.
            partition_strategy: e.g. "time_based", "hybrid", "none".
            sql_type:           "CREATE_TABLE" | "MERGE" | "COPY_INTO" | "OPTIMIZE".
        """
        self._log.info(
            "Bronze SQL generated",
            extra={
                "event":              "bronze_sql_generated",
                "trace_id":           str(trace_id),
                "dataset_name":       self.dataset_name,
                "partition_strategy": partition_strategy,
                "sql_type":           sql_type,
            },
        )

    def log_sql_executed(
        self,
        trace_id: UUID,
        statement_id: str,
        status: str,
        row_count: Optional[int],
        duration_ms: int,
    ) -> None:
        """
        §5.2 — Log Databricks SQL API execution result.

        Args:
            trace_id:     Run correlation UUID.
            statement_id: Statement ID returned by Databricks SQL API.
            status:       "SUCCEEDED" | "FAILED" | "CANCELED".
            row_count:    Rows affected (None if not available).
            duration_ms:  Wall-clock time for the statement.
        """
        self._log.info(
            "Bronze SQL executed",
            extra={
                "event":        "bronze_sql_executed",
                "trace_id":     str(trace_id),
                "dataset_name": self.dataset_name,
                "statement_id": statement_id,
                "status":       status,
                "row_count":    row_count,
                "duration_ms":  duration_ms,
            },
        )

    def log_sql_failed(
        self,
        trace_id: UUID,
        error_message: str,
    ) -> None:
        """
        §5.3 — Log ingestion failure.

        Args:
            trace_id:      Run correlation UUID.
            error_message: Exception or API error description.
        """
        self._log.error(
            "Bronze SQL failed",
            extra={
                "event":         "bronze_sql_failed",
                "trace_id":      str(trace_id),
                "dataset_name":  self.dataset_name,
                "error_message": error_message,
            },
        )

    def log_observability_warning(
        self,
        trace_id: UUID,
        rule: str,
        detail: str,
    ) -> None:
        """
        Log a metadata-driven observability rule violation (§8).

        Args:
            trace_id: Run correlation UUID.
            rule:     Rule name, e.g. "zero_rows", "max_duration_exceeded".
            detail:   Human-readable description of the violation.
        """
        self._log.warning(
            "Observability rule violated",
            extra={
                "event":        "bronze_observability_warning",
                "trace_id":     str(trace_id),
                "dataset_name": self.dataset_name,
                "rule":         rule,
                "detail":       detail,
            },
        )