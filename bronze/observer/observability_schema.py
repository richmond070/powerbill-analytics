"""
Observability Schema Bootstrap
Idempotently creates the two Bronze observability tables in PostgreSQL:
  - bronze_ingestion_audit   : per-run execution history ("black box recorder")
  - bronze_ingestion_metrics : daily aggregated metrics per dataset

Call `ensure_observability_tables()` once at pipeline startup.
Both CREATE statements use IF NOT EXISTS — safe to run on every deploy.

Referenced by: bronze_orchestrator.py, audit_writer.py, metrics_aggregator.py
"""

import logging
from .db_pool import pg_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL Statements
# ---------------------------------------------------------------------------

# Mirrors the schema specified in bronze_observability.md §4.1
_DDL_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS bronze_ingestion_audit (
    id                 SERIAL PRIMARY KEY,
    trace_id           UUID        NOT NULL,
    dataset_name       TEXT        NOT NULL,
    partition_strategy TEXT,
    statement_id       TEXT,
    status             TEXT        CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED')),
    row_count          BIGINT,
    duration_ms        BIGINT,
    execution_time     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    error_message      TEXT
);
"""

# Mirrors the schema specified in bronze_observability.md §6.2
_DDL_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS bronze_ingestion_metrics (
    dataset_name               TEXT,
    metric_date                DATE,
    ingestion_success_total    BIGINT           DEFAULT 0,
    ingestion_failures_total   BIGINT           DEFAULT 0,
    ingestion_rows_total       BIGINT           DEFAULT 0,
    ingestion_duration_seconds DOUBLE PRECISION DEFAULT 0,
    schema_evolution_count     BIGINT           DEFAULT 0,
    PRIMARY KEY (dataset_name, metric_date)
);
"""

# Index on trace_id for fast log correlation queries
_DDL_AUDIT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_audit_trace_id
    ON bronze_ingestion_audit (trace_id);
"""

# Index to speed up dataset-level metric dashboard queries
_DDL_METRICS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metrics_dataset_date
    ON bronze_ingestion_metrics (dataset_name, metric_date);
"""


def ensure_observability_tables(config_path: str = "databricks/databricks.cfg") -> None:
    """
    Idempotently create observability tables and supporting indexes in PostgreSQL.

    This is safe to call on every pipeline startup — all statements use
    IF NOT EXISTS. No data is mutated.

    Args:
        config_path: Path to shared config file used by db_pool.

    Raises:
        psycopg2.Error: If the DDL statements fail for any reason.
    """
    ddl_steps = [
        ("bronze_ingestion_audit table",        _DDL_AUDIT_TABLE),
        ("bronze_ingestion_metrics table",      _DDL_METRICS_TABLE),
        ("audit trace_id index",                _DDL_AUDIT_INDEX),
        ("metrics dataset/date index",          _DDL_METRICS_INDEX),
    ]

    with pg_connection(config_path) as conn:
        with conn.cursor() as cur:
            for description, ddl in ddl_steps:
                logger.debug("Ensuring schema object: %s", description)
                cur.execute(ddl)

    logger.info(
        "Observability schema ready — "
        "bronze_ingestion_audit + bronze_ingestion_metrics verified."
    )
