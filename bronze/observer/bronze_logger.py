"""
Bronze Structured Logger
Emits machine-readable JSON log entries for the three key observability events
defined in bronze_observability.md §5:

  Event                  | When emitted
  -----------------------|----------------------------------------------
  bronze_sql_generated   | After metadata-driven SQL is built (§5.1)
  bronze_sql_executed    | After Databricks SQL API returns (§5.2)
  bronze_sql_failed      | When ingestion raises an exception (§5.3)

Every entry carries:
  - trace_id       (UUID, generated at ingestion start, flows through all layers)
  - dataset_name
  - event          (one of the three above)
  - timestamp      (ISO-8601 UTC)

JSON format is chosen so entries can be shipped directly to any log aggregator
(CloudWatch, Datadog, ELK) without post-processing.

Referenced by: bronze_orchestrator.py (via BronzeLogger)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID


# ---------------------------------------------------------------------------
# JSON Formatter — converts LogRecord to a flat JSON string
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Renders every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "level":       record.levelname,
            "logger":      record.name,
            "message":     record.getMessage(),
        }
        # Attach any extra fields the caller injected (e.g. trace_id)
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                # Skip standard LogRecord keys to avoid clutter
                if key not in (
                    "args", "created", "exc_info", "exc_text", "filename",
                    "funcName", "levelname", "levelno", "lineno", "message",
                    "module", "msecs", "msg", "name", "pathname", "process",
                    "processName", "relativeCreated", "stack_info", "thread",
                    "threadName",
                ):
                    payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Module-level logger setup
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger to emit JSON to stdout.
    Call once in bronze_orchestrator.__init__ before any log is written.

    Args:
        level: Python logging level (default INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid double-adding handlers on repeated calls (e.g. in tests)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# BronzeLogger — typed convenience wrapper
# ---------------------------------------------------------------------------

class BronzeLogger:
    """
    Typed log emitter for the three Bronze observability events.
    Each method writes a single structured JSON entry via stdlib logging.

    Args:
        dataset_name: Fixed for the lifetime of this logger instance.
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
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
        Validates metadata resolution and documents the partition strategy chosen.

        Args:
            trace_id:           Run correlation UUID.
            partition_strategy: e.g. "time_based", "hybrid", "none".
            sql_type:           "CREATE_TABLE" | "INGEST" | "OPTIMIZE".
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
        Correlates with the audit table via trace_id + statement_id.

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
        Supports alerting logic and incident root-cause analysis.

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
        Does NOT alert directly — alerting is Phase 2 scope.

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
