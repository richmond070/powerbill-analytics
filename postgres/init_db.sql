-- =============================================================================
-- Postgres Initialization Script
-- Location: postgres/init_db.sql
-- Runs ONCE automatically when the postgres container starts for the first time.
-- Docker mounts this via: ./postgres:/docker-entrypoint-initdb.d/
--
-- This script creates TWO logical databases inside ONE Postgres container:
--   1. airflow_db     → Airflow metadata (DAG runs, task states, logs index)
--   2. bronze_control → Bronze layer control plane (audit + metrics tables)
--
-- Why two databases in one container?
--   Saves ~350MB vs spinning up a second postgres container.
--   They are logically isolated but share one process.
-- =============================================================================
-- -----------------------------------------------------------------------------
-- 1. Create bronze_control database
--    (airflow_db is already created by POSTGRES_DB env var in docker-compose)
-- -----------------------------------------------------------------------------
CREATE DATABASE bronze_control;
-- -----------------------------------------------------------------------------
-- 2. Grant the airflow user full access to bronze_control
--    The same 'airflow' user is used for simplicity in dev/local environments.
--    In production, create a dedicated bronze_user with least-privilege access.
-- -----------------------------------------------------------------------------
GRANT ALL PRIVILEGES ON DATABASE bronze_control TO airflow;
-- -----------------------------------------------------------------------------
-- 3. Connect to bronze_control and create the Bronze observability schema
--    These tables are defined in the Bronze Observability Guide (Section 4 + 6)
-- -----------------------------------------------------------------------------
\ connect bronze_control -- -----------------------------------------------------------------------------
-- 3a. bronze_ingestion_audit
--     Acts as the "black box recorder" for every ingestion run.
--     One row inserted at START (status=RUNNING), updated at END.
--     Referenced in: bronze/observer/audit_writer.py
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze_ingestion_audit (
    id SERIAL PRIMARY KEY,
    trace_id UUID NOT NULL,
    dataset_name TEXT NOT NULL,
    partition_strategy TEXT,
    statement_id TEXT,
    status TEXT CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED')),
    row_count BIGINT,
    duration_ms BIGINT,
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT
);
-- Index for fast lookup by dataset and trace
CREATE INDEX IF NOT EXISTS idx_audit_dataset_name ON bronze_ingestion_audit (dataset_name);
CREATE INDEX IF NOT EXISTS idx_audit_trace_id ON bronze_ingestion_audit (trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_status ON bronze_ingestion_audit (status);
-- -----------------------------------------------------------------------------
-- 3b. bronze_ingestion_metrics
--     Daily aggregated counters per dataset.
--     Upserted on every ingestion completion.
--     Referenced in: bronze/observer/metrics_aggregator.py
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze_ingestion_metrics (
    dataset_name TEXT NOT NULL,
    metric_date DATE NOT NULL,
    ingestion_success_total BIGINT NOT NULL DEFAULT 0,
    ingestion_failures_total BIGINT NOT NULL DEFAULT 0,
    ingestion_rows_total BIGINT NOT NULL DEFAULT 0,
    ingestion_duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    schema_evolution_count BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset_name, metric_date)
);
-- Index for date-range queries (dashboards, trend detection)
CREATE INDEX IF NOT EXISTS idx_metrics_metric_date ON bronze_ingestion_metrics (metric_date);
-- -----------------------------------------------------------------------------
-- Done. Summary of what was created:
--   Database : bronze_control
--   Tables   : bronze_ingestion_audit, bronze_ingestion_metrics
--   Indexes  : 4 total (3 on audit, 1 on metrics)
-- -----------------------------------------------------------------------------