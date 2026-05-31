# ⚡ Nigerian Energy & Utilities — Data Pipeline

A metadata-driven, multi-layer data pipeline for ingesting, transforming, and serving Nigerian energy and utilities data. Built on **Python**, **Apache Spark**, and **Databricks SQL**, with a **Postgres-backed control plane** for observability.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Pipeline Layers](#pipeline-layers)
  - [Bronze Layer](#bronze-layer)
  - [Observability Layer](#observability-layer)
- [Datasets](#datasets)
- [Bronze Layer Deep Dive](#bronze-layer-deep-dive)
  - [Ingestion Contract](#ingestion-contract)
  - [Partition Strategy](#partition-strategy)
  - [Schema Mapping](#schema-mapping)
  - [SQL Generation](#sql-generation)
  - [Databricks Client](#databricks-client)
  - [Data Downloader](#data-downloader)
  - [Orchestrator](#orchestrator)
- [Observability](#observability)
  - [Audit Table](#audit-table)
  - [Metrics Table](#metrics-table)
  - [Logging Strategy](#logging-strategy)
  - [Trace Correlation](#trace-correlation)
  - [Alerting](#alerting)
- [Testing](#testing)
- [Setup & Configuration](#setup--configuration)
- [Running the Pipeline](#running-the-pipeline)

---

## Project Overview

This pipeline ingests six raw datasets from the Nigerian energy sector — covering billing, grid load, power flow, commercial consumption, customer complaints, and retail tariffs — into a curated **Bronze Delta Lake** layer on Databricks.

The design philosophy is **metadata-driven**: the ingestion contract (`bronze_ingestion_contract.json`) is the single source of truth. Python orchestrates; Databricks SQL / Spark does all heavy processing. Nothing is hardcoded.

---

## Architecture

```
External APIs (HuggingFace)
        │
        ▼
  DataDownloader          ← Python downloads raw Parquet files to staging
        │
        ▼
  PartitionHeuristics     ← Determines optimal Delta Lake partition strategy
        │
        ▼
  SchemaMapper            ← Maps contract types to Databricks SQL types
        │
        ▼
  BronzeSQLGenerator      ← Generates CREATE TABLE / COPY INTO / MERGE SQL
        │
        ▼
  DatabricksSQLClient     ← Executes SQL via Databricks SQL API (async polling)
        │
        ▼
  Bronze Delta Tables     ← Partitioned, schema-enforced Delta Lake tables
        │
        ▼
  Observer Layer          ← Postgres-backed audit, metrics, structured logging
```

---

## Tech Stack

| Component           | Technology                                 |
| ------------------- | ------------------------------------------ |
| Orchestration       | Python 3.x                                 |
| Processing Engine   | Apache Spark / Databricks SQL              |
| Storage Format      | Delta Lake (Parquet)                       |
| Catalog             | Unity Catalog                              |
| Control Plane       | PostgreSQL                                 |
| Observability       | Structured JSON logging + Postgres metrics |
| Workflow Scheduler  | Apache Airflow                             |
| Data Transformation | dbt                                        |
| Dashboard           | Plotly Dash                                |
| Testing             | pytest                                     |
| Containerization    | Docker / docker-compose                    |

---

## Repository Structure

```
.
├── bronze/                         # Bronze layer ingestion module
│
├── Ingestion/
│   ├── bronze_orchestrator.py      # Main orchestration entry point
│   ├── databricks_client.py        # Databricks SQL API client
│   ├── data_downloader.py          # Raw file downloader (staging)
│   ├── partition_strategy.py       # Partition heuristics engine
│   ├── schema_mapper.py            # Contract-to-SQL type mapper
│   ├── sql_generator.py            # SQL template generator
│   ├── __init__.py
│   └── observer/                   # Observability sub-module
│       ├── audit_writer.py         # Postgres audit table writer
│       ├── bronze_logger.py        # Structured JSON logger
│       ├── db_pool.py              # Postgres connection pool
│       ├── metrics_aggregator.py   # Daily metrics aggregator
│       ├── observability_contract.py
│       ├── observability_schema.py  # Postgres DDL definitions
│       └── __init__.py
│
├── bronze_metadata/
│   └── bronze_ingestion_contract.json   # Source-of-truth ingestion contract
│
│
├── extraction/                     # API resolution and runner
│   ├── api_config.json
│   ├── resolver.py
│   ├── runner.py
│   ├── validator.py
│   └── scripts/
│       └── run_ingests.sh

```

---

## Pipeline Layers

### Bronze Layer

The Bronze layer ingests **raw, unmodified data** from external sources into schema-enforced Delta Lake tables. It is the first trusted internal representation of the data.

**Key Responsibilities:**

- Download raw Parquet files from HuggingFace dataset endpoints to a staging area
- Determine the optimal Delta Lake partition strategy per dataset using heuristics
- Map contract-defined types to Databricks SQL types (strict — prevents type drift)
- Generate `CREATE TABLE`, `COPY INTO`, and `MERGE` SQL dynamically from templates
- Execute SQL via the Databricks SQL API with async polling and retry logic
- Write audit records and metrics to a Postgres control plane

### Observability Layer

A lightweight, Postgres-backed observability layer lives inside `bronze/observer/`. It provides:

- **Audit trail** — every ingestion run is recorded (start, success/failure, row count, duration)
- **Daily metrics** — aggregated counters for successes, failures, rows, duration, schema evolution
- **Structured logging** — JSON logs with `trace_id` correlation on every event
- **Alerting hooks** — failure detection for Airflow email / Slack webhook integration

---

## Datasets

All six datasets are defined in `bronze/bronze_metadata/bronze_ingestion_contract.json`.

| Dataset                             | Rows    | Columns | Description                                      |
| ----------------------------------- | ------- | ------- | ------------------------------------------------ |
| `billing_payments`                  | 200,000 | 10      | Customer billing and payment records per month   |
| `commercial_industries_consumption` | 220,000 | 11      | Commercial/industrial site power consumption     |
| `customers_complaint`               | 100,000 | 9       | Customer complaint tickets and SLA outcomes      |
| `grid_load`                         | 200,000 | 10      | Substation grid load and weather readings        |
| `power_flow`                        | 200,000 | 10      | Power flow between substations (line-level)      |
| `retail_tariffs`                    | 90,000  | 6       | Retail electricity tariff rates by band and hour |

All datasets are sourced from `electricsheepafrica` on HuggingFace and ingested as Parquet files.

---

## Bronze Layer Deep Dive

### Ingestion Contract

**File:** `bronze/bronze_metadata/bronze_ingestion_contract.json`

The contract is the single source of truth for the entire Bronze layer. It defines for each dataset:

- `dataset_name` — internal canonical name
- `api_endpoint` — HuggingFace API endpoint for discovery
- `files[]` — list of Parquet file URLs with row/column metadata
- `columns[]` — full column schema with name, type, and nullable flag
- `total_rows`, `file_count`, `validation_status`

Nothing in the pipeline is hardcoded — all SQL, partition decisions, and schema definitions are derived from this contract at runtime.

---

### Partition Strategy

**File:** `bronze/partition_strategy.py`

The `PartitionHeuristics` class automatically determines the optimal Delta Lake partition strategy based on dataset size and column names. No manual configuration is needed.

**Decision Rules:**

| Condition                                  | Strategy                   |
| ------------------------------------------ | -------------------------- |
| `rows < 100,000`                           | `NONE` — no partitioning   |
| `rows >= 500,000` + time + category column | `HYBRID` — time + category |
| `rows >= 500,000` + time column only       | `TIME_BASED`               |
| `rows >= 500,000` + category column only   | `CATEGORY_BASED`           |
| Medium datasets with time column           | `TIME_BASED`               |
| No suitable columns found                  | `NONE`                     |

**Column Detection:**

Time columns are detected by pattern matching against: `timestamp`, `created_time`, `billing_month`, `as_of_date`, `date`.

Category columns are detected against: `disco`, `region`, `state`, `country`, `category`, `type`, `site_type`.

The `disco` (distribution company) column is always prioritised as the primary category partition when present.

Large datasets (`>= 200,000 rows`) automatically use `use_append_only=True`, which switches ingestion from `COPY INTO` to an idempotent `MERGE` pattern.

---

### Schema Mapping

**File:** `bronze/schema_mapper.py`

The `SchemaMapper` class converts contract-defined types to Databricks SQL types with strict enforcement. Unknown types raise a `ValueError` immediately — preventing silent type drift.

**Type Map:**

| Contract Type | Databricks SQL Type |
| ------------- | ------------------- |
| `string`      | `STRING`            |
| `double`      | `DOUBLE`            |
| `bool`        | `BOOLEAN`           |
| `int64`       | `BIGINT`            |
| `int32`       | `INT`               |
| `float`       | `FLOAT`             |
| `timestamp`   | `TIMESTAMP`         |
| `date`        | `DATE`              |
| `binary`      | `BINARY`            |

The mapper also generates Spark schema strings for enforced `read_files()` calls in `MERGE` SQL, preventing corrupt Parquet files from crashing ingestion.

---

### SQL Generation

**File:** `bronze/sql_generator.py`

The `BronzeSQLGenerator` class produces schema-safe SQL from three templates, filled entirely from metadata — never from hardcoded values.

**Templates:**

**1. `CREATE TABLE`** — creates a Delta Lake table with:

- Schema-enforced columns from the contract
- Three Bronze metadata columns: `_bronze_ingestion_timestamp`, `_bronze_source_file`, `_bronze_row_hash`
- `PARTITIONED BY` clause (from `PartitionHeuristics`)
- Delta table properties: Change Data Feed, auto-optimise, schema enforcement, source lineage tags

**2. `COPY INTO`** — used for small/medium datasets:

- Incremental, idempotent file loading (Databricks tracks loaded files)
- `mergeSchema = false` to enforce schema and prevent drift
- `badRecordsPath` for quarantining corrupt records
- SHA-256 row hash computed inline

**3. `MERGE UPSERT`** — used for large datasets or append-only tables:

- Reads via `read_files()` with enforced Spark schema
- Deduplicates on `_bronze_row_hash`
- `WHEN NOT MATCHED THEN INSERT *` — append-only pattern

**Optimization SQL** — `OPTIMIZE ... ZORDER BY (_bronze_ingestion_timestamp)` + `VACUUM` with 168-hour retention.

---

### Databricks Client

**File:** `bronze/databricks_client.py`

The `DatabricksSQLClient` class executes SQL against a Databricks SQL Warehouse via the REST API (`/api/2.0/sql/statements`).

**Execution Flow:**

1. Submit statement with `wait_timeout: 0s` (returns immediately with a `statement_id`)
2. Poll `/sql/statements/{statement_id}` with exponential backoff (1s → 10s max)
3. Return `SQLExecutionResult` on `SUCCEEDED`, `FAILED`, `CANCELED`, or `CLOSED`
4. Timeout after configurable `wait_timeout` seconds (default 300s)

Configuration is loaded from `databricks/databricks.cfg`:

```ini
[DEFAULT]
workspace_url = https://xxx.cloud.databricks.com
token = dapi...
warehouse_id = abc123...
```

The `SQLExecutionLogger` class appends structured JSON execution records to `bronze_ingestion_log.json` for every SQL run.

---

### Data Downloader

**File:** `bronze/data_downloader.py`

The `DataDownloader` class downloads raw Parquet files from HuggingFace endpoints to a local staging area before Databricks ingestion.

**Features:**

- Streaming download in configurable chunks (default 8 KB) for large file support
- Exponential backoff retry logic (up to 3 attempts by default)
- Skip-if-exists caching (avoids re-downloading on re-runs)
- Partial download cleanup on failure
- `DataValidator` sub-class uses `pyarrow` to validate Parquet metadata (row count, column count) without loading data into memory

---

### Orchestrator

**File:** `bronze/bronze_orchestrator.py`

The `BronzeLayerOrchestrator` is the main entry point. It wires all components together and exposes three pipeline steps:

**Step 1 — `create_bronze_tables()`**

- Loads datasets from the contract
- Runs `PartitionHeuristics.determine_strategy()` per dataset
- Generates `CREATE TABLE` SQL via `BronzeSQLGenerator`
- Executes via `DatabricksSQLClient`
- Logs via `SQLExecutionLogger`

**Step 2 — `ingest_data()`**

- Optionally downloads raw files via `DataDownloader`
- Generates `COPY INTO` or `MERGE` SQL based on dataset size
- Executes ingestion SQL
- Logs result

**Step 3 — `optimize_tables()`** _(optional)_

- Generates and executes `OPTIMIZE` + `VACUUM` SQL per table

**`run_full_pipeline()`** — runs all three steps in sequence with a single call.

All steps support:

- `datasets` filter — run a subset by name
- `dry_run=True` — generates SQL and saves to disk without executing
- `download=True/False` — control whether raw files are fetched

The pipeline entry point is `runners/run_bronze.py`.

---

## Observability

The observability layer lives in `bronze/observer/` and is backed by a Postgres control plane. It is intentionally lightweight (MVP) — no Prometheus, Grafana, or OpenTelemetry.

### Audit Table

**File:** `bronze/observer/audit_writer.py`

Every ingestion run writes to `bronze_ingestion_audit` in Postgres:

```sql
CREATE TABLE bronze_ingestion_audit (
    id                 SERIAL PRIMARY KEY,
    trace_id           UUID NOT NULL,
    dataset_name       TEXT NOT NULL,
    partition_strategy TEXT,
    statement_id       TEXT,
    status             TEXT CHECK (status IN ('RUNNING','SUCCESS','FAILED')),
    row_count          BIGINT,
    duration_ms        BIGINT,
    execution_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message      TEXT
);
```

**Lifecycle:** On ingestion start → `INSERT` with `status = 'RUNNING'`. On completion → `UPDATE` with final status, row count, duration, and any error message.

### Metrics Table

**File:** `bronze/observer/metrics_aggregator.py`

Daily aggregated counters are stored in `bronze_ingestion_metrics`:

```sql
CREATE TABLE bronze_ingestion_metrics (
    dataset_name              TEXT,
    metric_date               DATE,
    ingestion_success_total   BIGINT,
    ingestion_failures_total  BIGINT,
    ingestion_rows_total      BIGINT,
    ingestion_duration_seconds DOUBLE PRECISION,
    schema_evolution_count    BIGINT,
    PRIMARY KEY (dataset_name, metric_date)
);
```

On each ingestion completion, counters are incremented via `INSERT ... ON CONFLICT DO UPDATE` (upsert). Metrics aggregate per dataset per day.

**Tracked Metrics:**

| Metric                       | Purpose                        |
| ---------------------------- | ------------------------------ |
| `ingestion_success_total`    | Measure health ratio           |
| `ingestion_failures_total`   | Detect instability trends      |
| `ingestion_rows_total`       | Detect data volume anomalies   |
| `ingestion_duration_seconds` | Detect performance degradation |
| `schema_evolution_count`     | Track structural drift         |

### Logging Strategy

**File:** `bronze/observer/bronze_logger.py`

All log entries are structured JSON with mandatory fields:

- `trace_id` — correlation ID for the run
- `dataset_name` — dataset being ingested
- `event` — one of `bronze_sql_generated`, `bronze_sql_executed`, `bronze_sql_failed`

Three event types are logged:

| Event                  | Trigger                      | Key Fields                                           |
| ---------------------- | ---------------------------- | ---------------------------------------------------- |
| `bronze_sql_generated` | After SQL template is filled | `partition_strategy`                                 |
| `bronze_sql_executed`  | After Databricks API returns | `statement_id`, `status`, `row_count`, `duration_ms` |
| `bronze_sql_failed`    | On any ingestion failure     | `error_message`                                      |

### Trace Correlation

**File:** `bronze/observer/observability_contract.py`

Each ingestion run generates a single UUID `trace_id` at startup. This ID is propagated to:

- The audit table row
- All structured log entries
- The metrics update

This allows full reconstruction of any ingestion execution — SQL generation → execution → result → metrics — without distributed tracing infrastructure.

### Alerting

Alerting is minimal for MVP. Alerts are triggered on:

- `FAILED` ingestion status
- Zero rows ingested when `alert_on_zero_rows = true` in metadata
- Duration exceeding `max_expected_duration_sec`

Supported channels: Airflow email notifications, Slack webhook, SMTP email.

---

## Testing

Tests live in `tests/` and are run with `pytest`.

```bash
pytest tests/
```

**Test modules:**

| File                          | Coverage                                                         |
| ----------------------------- | ---------------------------------------------------------------- |
| `test_partition_strategy.py`  | Partition heuristic rules, column detection, strategy selection  |
| `test_sql_generator.py`       | SQL template rendering, column DDL, COPY INTO vs MERGE selection |
| `test_data_downloader.py`     | Download retry logic, caching, validation                        |
| `test_bronze_orchestrator.py` | Full pipeline orchestration, dataset filtering, dry-run mode     |
| `db_test.py`                  | Postgres connectivity and schema validation                      |

**Observability test strategy** (defined in `docs/`):

- **Unit tests** — metric delta logic, date handling, log structure validation
- **Integration tests** — Postgres upsert correctness, counter accumulation across multiple runs
- **Concurrency tests** — 10 parallel threads writing to the same dataset, no lost increments
- **Failure simulation** — Postgres down, audit insert constraint violation, partial failure (Databricks succeeded, metrics write failed)
- **Edge cases** — midnight boundary, large row counts (`10,000,000+`), 8-hour duration floats, schema evolution storms, dataset rename isolation

---

## Setup & Configuration

### Prerequisites

- Python 3.9+
- Docker & docker-compose
- Databricks workspace with a running SQL Warehouse
- PostgreSQL (via docker-compose or managed instance)

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=pipeline_control
POSTGRES_USER=pipeline
POSTGRES_PASSWORD=your_password
```

### Databricks Configuration

Edit `databricks/databricks.cfg`:

```ini
[DEFAULT]
workspace_url = https://your-workspace.cloud.databricks.com
token = dapi...
warehouse_id = your_warehouse_id
```

### Start Infrastructure

```bash
docker-compose up -d     # Starts Postgres + Airflow + Dashboard
```

---

## Running the Pipeline

### Full Pipeline (All Datasets)

```python
from bronze.ingestion.bronze_orchestrator import BronzeLayerOrchestrator

orchestrator = BronzeLayerOrchestrator(
    contract_path='bronze/bronze_metadata/bronze_ingestion_contract.json',
    config_path='databricks/databricks.cfg',
    catalog='main',
    schema='bronze'
)

orchestrator.run_full_pipeline(
    download=True,
    optimize=False,
    dry_run=False   # Set True to preview SQL without executing
)
```

### Single Dataset

```python
orchestrator.run_full_pipeline(
    datasets=['billing_payments'],
    download=True,
    dry_run=True
)
```

### Individual Steps

```python
orchestrator.create_bronze_tables(dry_run=True)
orchestrator.ingest_data(download=True, dry_run=True)
orchestrator.optimize_tables(dry_run=True)
```

### CLI Runner

```bash
python runners/run_bronze.py
```

---

## Design Decisions

**Why metadata-driven?** The contract is the single source of truth. Adding a new dataset requires only a new entry in `bronze_ingestion_contract.json` — no code changes needed.

**Why Python orchestrates but Spark processes?** Python is used only for coordination, SQL generation, and control plane writes. All data processing is pushed to the Databricks SQL engine where it scales natively.

**Why MERGE for large datasets?** `COPY INTO` is idempotent at the file level but cannot deduplicate rows across re-ingested files. For large datasets, a hash-based `MERGE` guarantees row-level idempotency.

**Why Postgres for observability instead of Prometheus?** This is an MVP. Postgres is already part of the control plane (Airflow uses it). Introducing Prometheus + Grafana at this stage would be premature complexity.

---

## Evolution Roadmap

**Phase 2**

- Prometheus metrics export
- Grafana dashboards
- OpenTelemetry tracing
- Data quality validation checks on Bronze output

**Phase 3**

- SLA monitoring per dataset
- Automated anomaly detection on row count trends
- Cross-layer lineage observability (Bronze → Silver → Gold)
- dbt Silver/Gold model completion
