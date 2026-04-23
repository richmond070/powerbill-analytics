# Nigerian Energy & Utilities — Data Pipeline

> A production-grade, metadata-driven data pipeline for ingesting, transforming, and serving Nigerian energy and utilities data. Built on **Python**, **Apache Spark**, and **Databricks SQL** with a **PostgreSQL-backed control plane** for observability and auditability.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Who This Document Is For](#2-who-this-document-is-for)
3. [Repository Structure](#3-repository-structure)
4. [Overall Data Architecture](#4-overall-data-architecture)
5. [Extraction Layer](#5-extraction-layer)
6. [Bronze Layer](#6-bronze-layer)
7. [How Extraction Connects to Bronze](#7-how-extraction-connects-to-bronze)
8. [Trade-offs and Engineering Decisions](#8-trade-offs-and-engineering-decisions)
9. [Datasets](#9-datasets)
10. [Testing](#10-testing)
11. [Setup and Configuration](#11-setup-and-configuration)
12. [Running the Pipeline](#12-running-the-pipeline)
13. [Roadmap](#13-roadmap)

---

## 1. Project Overview

This pipeline ingests six raw datasets from the Nigerian energy sector — spanning billing, grid load, power flow, commercial consumption, customer complaints, and retail tariffs — and processes them into a curated, query-ready Delta Lake data platform on Databricks.

The pipeline is built in layers following the **Medallion Architecture** pattern:

```
Extraction  →  Bronze  →  Silver (planned)  →  Gold (planned)
  (raw API)    (ingest)    (clean/model)         (serve)
```

Each layer has a clearly defined responsibility, a contract with the layer above it, and independent observability. This document covers the **Extraction** and **Bronze** layers in full. Silver and Gold documentation will be added as those layers are built.

---

## 2. Who This Document Is For

This document is written for an engineer picking up this codebase for the first time. It assumes you are comfortable with Python and SQL but may be unfamiliar with Databricks, Delta Lake, or how this specific pipeline is structured.

By the end of this document you should understand:

- What each layer does and why it exists
- How data flows from a raw API endpoint to a structured Delta table
- Every design decision made in the Bronze layer and why
- How to run the pipeline yourself

---

## 3. Repository Structure

```
.
├── bronze/ingestion                       # Bronze layer — core ingestion module
│   ├── bronze_orchestrator.py       # Main pipeline orchestrator
│   ├── databricks_client.py         # Databricks SQL API client
│   ├── data_downloader.py           # Staging file downloader
│   ├── partition_strategy.py        # Automatic partitioning heuristics
│   ├── schema_mapper.py             # Contract type → Databricks SQL type mapper
│   ├── sql_generator.py             # SQL template engine
│   ├── __init__.py
│   └── observer/                    # Observability sub-module
│       ├── audit_writer.py          # Per-run audit table writer
│       ├── bronze_logger.py         # Structured JSON event logger
│       ├── db_pool.py               # PostgreSQL connection pool
│       ├── metrics_aggregator.py    # Daily metrics aggregator
│       ├── observability_contract.py
│       ├── observability_schema.py  # PostgreSQL DDL bootstrap
│       └── __init__.py
│
│    └── bronze_metadata/
│       ├── bronze_ingestion_contract.json  # The hand-off contract between layers
│    └──extraction/                      # Extraction layer — API resolution and validation
│       ├── api_config.json              # Dataset endpoint configuration
│       ├── resolver.py                  # HuggingFace API resolver
│       ├── runner.py                    # Extraction entry point
│       ├── validator.py                 # Remote Parquet file validator
│       ├── __init__.py
│       └── scripts/
│           └── run_ingests.sh
│
├── airflow/                         # Airflow DAGs and scheduler config
│   ├── dags/
│   ├── config/
│   └── plugins/
│
├── databricks/                      # Databricks workspace configuration
│   ├── databricks.cfg
│   └── dbfs_uploader.py
│
├── dbt_project/                     # dbt Silver / Gold transformations (planned)
│   ├── models/
│   │   ├── staging/
│   │   └── marts/
│   └── profiles.yml.example
│
├── dashboard/                       # Observability dashboard
│   ├── app.py
│   └── Dockerfile
│
├── runners/
│   └── run_bronze.py                # CLI entry point for the Bronze pipeline
│
├── tests/                           # pytest test suite
│   ├── test_bronze_orchestrator.py
│   ├── test_data_downloader.py
│   ├── test_partition_strategy.py
│   ├── test_sql_generator.py
│   └── db_test.py
│
├── docs/
├── docker-compose.yaml
├── pytest.ini
└── requirements.txt
```

---

## 4. Overall Data Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTRACTION LAYER                         │
│                                                                 │
│  api_config.json ──► HuggingFaceResolver ──► ParquetValidator  │
│                                  │                              │
│                                  ▼                              │
│                   bronze_ingestion_contract.json                │
└────────────────────────────┬────────────────────────────────────┘
                             │  (contract hand-off)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BRONZE LAYER                            │
│                                                                 │
│  Contract ──► PartitionHeuristics ──► SchemaMapper             │
│                        │                   │                    │
│                        └────────┬──────────┘                   │
│                                 ▼                               │
│                       BronzeSQLGenerator                        │
│                                 │                               │
│                                 ▼                               │
│                     DataDownloader (staging)                    │
│                                 │                               │
│                                 ▼                               │
│                    DatabricksSQLClient ──► Delta Lake Tables    │
│                                 │                               │
│                                 ▼                               │
│                  Observer (Audit + Metrics + Logs)              │
└────────────────────────┬────────────────────────────────────────┘
                         │  (planned)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SILVER LAYER                            │
│              dbt models — clean, typed, deduplicated            │
└────────────────────────┬────────────────────────────────────────┘
                         │  (planned)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                          GOLD LAYER                             │
│              Aggregated marts — analytics-ready                 │
└─────────────────────────────────────────────────────────────────┘
```

The pipeline follows **strict separation of concerns** at every boundary:

- The **Extraction layer** knows about external APIs. It does not know about Databricks.
- The **Bronze layer** knows about Databricks and Delta Lake. It does not know about HuggingFace.
- The **contract JSON file** is the only shared artefact between the two layers.

---

## 5. Extraction Layer

### 5.1 What It Does

The Extraction layer is the first point of contact with the external world. Its responsibility is narrow and deliberate: **resolve external dataset API endpoints into concrete, validated Parquet file URLs and capture their schema metadata**. Nothing more.

It does not load data into a database. It does not transform anything. It produces a single output — the **ingestion contract** — and hands that to the Bronze layer.

### 5.2 How It Works

```
api_config.json
      │
      ▼
HuggingFaceDatasetResolver.resolve(url)
      │
      │  Queries the HuggingFace dataset metadata API
      │  Handles three response formats:
      │    • List of URL strings
      │    • List of file metadata dicts
      │    • Dict with a "files" key
      │
      ▼
List of { url, filename, size_bytes }
      │
      ▼
ParquetValidator.validate_remote_parquet(url)
      │
      │  Downloads each file into memory (no data rows loaded)
      │  Uses PyArrow to inspect metadata only
      │  Extracts: num_rows, num_columns, num_row_groups, column schemas
      │
      ▼
Validated file metadata
      │
      ▼
runner.py assembles bronze_ingestion_contract.json
```

**Entry point:** `extraction/runner.py` → `run_bronze_ingestion()`

### 5.3 Key Components

#### `extraction/resolver.py` — `HuggingFaceDatasetResolver`

HuggingFace dataset API endpoints are **logical directory URLs** — they describe what files exist but are not direct download links. The resolver queries each endpoint and extracts the actual Parquet file URLs.

It handles three different response formats that HuggingFace returns depending on dataset version:

| Response Format                                | Description                   |
| ---------------------------------------------- | ----------------------------- |
| `["https://...file1.parquet", ...]`            | Direct list of URL strings    |
| `[{"path": "...", "url": "...", "size": ...}]` | List of file metadata objects |
| `{"files": [{...}]}`                           | Dict with a `files` key       |

If no Parquet files are found or the response format is unrecognised, `resolve()` raises `ValueError` — failing early rather than producing an empty or invalid contract downstream.

#### `extraction/validator.py` — `ParquetValidator`

After resolving URLs, every file is validated before being written to the contract. The validator downloads each Parquet file into an in-memory buffer and uses **PyArrow** to inspect its metadata without loading any data rows.

For each file it extracts:

- `num_rows` — total row count
- `num_columns` — column count
- `num_row_groups` — internal Parquet structure
- `columns[]` — full column schema with name, type, and nullable flag
- `validation_status` — `"success"` or `"failed"`

If validation fails for one file, the failure is recorded in the contract and the pipeline continues to the next dataset. A single bad dataset never blocks the others.

#### `extraction/api_config.json` — Dataset Registry

```json
{
  "datasets": [
    { "name": "billing_payments", "url": "https://huggingface.co/api/..." },
    { "name": "grid_load", "url": "https://huggingface.co/api/..." }
  ]
}
```

The only place external URLs are stored. Adding a new dataset means adding one entry here and re-running the extraction layer — no code changes anywhere else.

### 5.4 The Ingestion Contract — The Hand-off

The extraction layer's only output is `bronze/bronze_metadata/bronze_ingestion_contract.json`. This is the **formal contract** between the Extraction and Bronze layers.

```json
{
  "generated_at": "2026-01-29T00:10:36.222155Z",
  "datasets": [
    {
      "dataset_name": "billing_payments",
      "api_endpoint": "https://huggingface.co/api/...",
      "file_count": 1,
      "total_rows": 200000,
      "files": [
        {
          "url": "https://huggingface.co/.../0.parquet",
          "filename": "0.parquet",
          "num_rows": 200000,
          "num_columns": 10,
          "columns": [
            { "name": "customer_id", "type": "string", "nullable": true },
            { "name": "kwh", "type": "double", "nullable": true }
          ],
          "validation_status": "success"
        }
      ]
    }
  ]
}
```

**Everything the Bronze layer needs is in this file.** The Bronze layer never calls the HuggingFace API. It never needs to know what the source system looks like. It reads the contract and proceeds entirely from there.

---

## 6. Bronze Layer

### 6.1 Design Philosophy

The Bronze layer is built around four principles:

**1. Metadata-driven, not code-driven.**
No dataset-specific logic exists anywhere in the codebase. Every decision — partition strategy, column types, table names, SQL statements — is derived from the ingestion contract at runtime. Adding or modifying a dataset requires no code changes.

**2. Python orchestrates. Spark processes.**
Python reads the contract, makes decisions, generates SQL, and writes to the control plane. It does not touch the data. All data processing — loading, deduplication, schema enforcement, partitioning — is delegated entirely to Databricks SQL and Apache Spark. This keeps the orchestrator lightweight and the processing horizontally scalable.

**3. Idempotency is non-negotiable.**
Every operation in the Bronze layer is safe to re-run. `CREATE TABLE IF NOT EXISTS` is always used. `COPY INTO` with `force=false` skips already-loaded files. `MERGE` with `WHEN NOT MATCHED` never duplicates rows. A re-run after failure produces exactly the same result as a first-time run.

**4. Schema enforcement at every boundary.**
The ingestion contract defines the schema. The `SchemaMapper` converts it to Databricks SQL types and raises `ValueError` on any unknown type. Generated SQL uses `mergeSchema = false` in `COPY INTO` and enforced schemas in `read_files()` for `MERGE`. Schema drift is caught at ingestion time, not discovered downstream.

---

### 6.2 Architecture

```
bronze_ingestion_contract.json
              │
              ▼
  ┌──────────────────────────┐
  │  BronzeLayerOrchestrator │  ← Entry point. Reads contract. Wires all components.
  └────────────┬─────────────┘
               │
     ┌─────────┼─────────────┐
     │         │             │
     ▼         ▼             ▼
DataDownloader  PartitionHeuristics  SchemaMapper
(staging I/O)   (strategy)           (types)
     │         │             │
     └─────────┴──────┬──────┘
                      │
                      ▼
            BronzeSQLGenerator
            (templates → SQL)
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        CREATE TABLE     COPY INTO / MERGE
             │                 │
             └────────┬────────┘
                      │
                      ▼
          DatabricksSQLClient
          (async REST API execution)
                      │
             ┌────────┴───────────────┐
             │                        │
             ▼                        ▼
       Delta Lake Tables         Observer Layer
       (Unity Catalog)           (Audit + Metrics + Logs)
```

---

### 6.3 Design System

The Bronze layer is a **monolithic metadata-driven pipeline**. Each concern is isolated in its own module with a single, clear responsibility. No module reaches into another's concern.

| Module                   | Single Responsibility                          | Knows About                         |
| ------------------------ | ---------------------------------------------- | ----------------------------------- |
| `bronze_orchestrator.py` | Wires components, drives execution flow        | Everything                          |
| `partition_strategy.py`  | Decides how tables should be partitioned       | Column names, row counts            |
| `schema_mapper.py`       | Maps contract types to SQL/Spark types         | Type definitions only               |
| `sql_generator.py`       | Fills SQL templates from metadata              | Templates, schema, partition config |
| `data_downloader.py`     | Downloads Parquet files to staging             | HTTP, file system                   |
| `databricks_client.py`   | Executes SQL via Databricks REST API           | HTTP, Databricks API spec           |
| `observer/`              | Writes audit records, metrics, structured logs | PostgreSQL, trace IDs               |

This isolation means every module is independently testable and replaceable without touching the others.

---

### 6.4 How the Bronze Layer Works — Step by Step

When `run_full_pipeline()` is called, the following sequence executes for each dataset in the contract.

#### Step 1 — Load the Contract

The orchestrator opens `bronze_ingestion_contract.json` and loads all dataset metadata into memory. This is the only file read at startup — no database queries, no API calls.

#### Step 2 — Determine Partition Strategy

`PartitionHeuristics.determine_strategy()` analyses each dataset's row count and column names to decide the optimal Delta Lake partition strategy automatically.

```
billing_payments (200,000 rows, columns include "billing_month" and "disco")
        │
        ▼
Row count not < 100,000  →  not small
Row count not >= 500,000  →  medium band
Has time column ("billing_month")  →  TIME_BASED
200,000 >= 200,000 threshold  →  use_append_only = True
        │
        ▼
PartitionConfig(strategy=TIME_BASED, partition_columns=["billing_month"], use_append_only=True)
```

Full decision tree:

```
total_rows < 100,000               →  NONE            (no partitioning needed)
total_rows >= 500,000
  + time column + category column  →  HYBRID           (partition on both)
  + time column only               →  TIME_BASED
  + category column only           →  CATEGORY_BASED
  + no matching columns            →  NONE             (use_append_only=True)
Medium range (100,000 – 499,999)
  + time column                    →  TIME_BASED
  + no matching columns            →  NONE
```

#### Step 3 — Map Schema Types

`SchemaMapper.generate_ddl_columns()` converts each column definition to a Databricks SQL DDL fragment:

```
{ "name": "kwh",         "type": "double", "nullable": true  }  →  kwh DOUBLE
{ "name": "paid_on_time","type": "bool",   "nullable": true  }  →  paid_on_time BOOLEAN
{ "name": "customer_id", "type": "string", "nullable": false }  →  customer_id STRING NOT NULL
```

Any type not in the `TYPE_MAP` raises `ValueError` immediately — no silent defaults, no fallback types.

#### Step 4 — Generate CREATE TABLE SQL

`BronzeSQLGenerator.generate_create_table_sql()` fills the `CREATE_TABLE_TEMPLATE` with the dataset name, mapped columns, partition clause, and table location.

The generated SQL always includes:

- `CREATE TABLE IF NOT EXISTS` — safe to re-run at any time
- All source columns from the contract, fully type-mapped and nullable-correct
- Three Bronze metadata columns added automatically to every table:
  - `_bronze_ingestion_timestamp TIMESTAMP` — when the row was loaded
  - `_bronze_source_file STRING` — which Parquet file the row came from
  - `_bronze_row_hash STRING` — SHA-256 hash of all column values concatenated
- `PARTITIONED BY (...)` if strategy is not NONE
- `USING DELTA` — Delta Lake table format
- `LOCATION` — explicit storage path in the Delta root
- `TBLPROPERTIES` — schema enforcement, Change Data Feed, auto-optimise, source lineage tags

Every generated SQL file is also written to `/tmp/bronze_<dataset>_create.sql` before execution so the exact SQL can always be inspected.

#### Step 5 — Download Raw Files to Staging

`DataDownloader.download_dataset()` downloads each Parquet file from the URL in the contract to `/mnt/staging/raw/<dataset_name>/`.

- Skips files that already exist in staging — idempotent by default
- Streams in 8 KB chunks to handle large files without memory pressure
- Retries up to 3 times with exponential backoff (2s, 4s, 8s)
- Cleans up partial files on failure — no corrupt Parquet lands in staging
- A single dataset download failure skips that dataset and continues to the next

#### Step 6 — Generate Ingestion SQL

The ingestion method is selected based on the partition config and dataset size:

```
partition_config.use_append_only = True   →  MERGE UPSERT
total_rows > 300,000                      →  MERGE UPSERT  (explicit override)
Otherwise                                 →  COPY INTO
```

**COPY INTO** — used for small datasets (e.g. `retail_tariffs` at 90,000 rows):

```sql
COPY INTO main.bronze.bronze_retail_tariffs
FROM (
    SELECT
        as_of_date, disco, customer_class, tariff_band, hour, price_ngn_kwh,
        current_timestamp()              AS _bronze_ingestion_timestamp,
        _metadata.file_path              AS _bronze_source_file,
        sha2(concat_ws('||', ...), 256)  AS _bronze_row_hash
    FROM '/mnt/staging/raw/retail_tariffs/*.parquet'
)
FILEFORMAT = PARQUET
FORMAT_OPTIONS ('mergeSchema' = 'false', 'badRecordsPath' = '.../bad_records')
COPY_OPTIONS  ('mergeSchema' = 'false', 'force' = 'false');
```

`force = 'false'` means Databricks tracks which source files have already been loaded. Re-running will not re-insert data from files already processed.

**MERGE UPSERT** — used for large/append-only datasets (e.g. `billing_payments` at 200,000 rows):

```sql
MERGE INTO main.bronze.bronze_billing_payments AS target
USING (
    SELECT
        customer_id, disco, billing_month, ...,
        current_timestamp()              AS _bronze_ingestion_timestamp,
        _metadata.file_path              AS _bronze_source_file,
        sha2(concat_ws('||', ...), 256)  AS _bronze_row_hash
    FROM read_files(
        '/mnt/staging/raw/billing_payments/*.parquet',
        format => 'parquet',
        schema => 'customer_id string, disco string, billing_month string, ...'
    )
) AS source
ON target._bronze_row_hash = source._bronze_row_hash
WHEN NOT MATCHED THEN INSERT *;
```

`read_files()` with an enforced schema string catches corrupt or schema-drifted Parquet files at read time. `WHEN NOT MATCHED THEN INSERT *` means rows already in the target are never modified — the Bronze layer is **append-only by design**.

#### Step 7 — Execute via Databricks SQL API

`DatabricksSQLClient.execute_sql()` submits each SQL statement to the Databricks SQL Statements REST API and polls asynchronously for completion:

```
POST /api/2.0/sql/statements         →  { statement_id: "abc123" }
GET  /api/2.0/sql/statements/abc123  →  { status: { state: "RUNNING" } }
GET  /api/2.0/sql/statements/abc123  →  { status: { state: "SUCCEEDED" },
                                          manifest: { row_count: 200000 } }
```

The statement is submitted with `wait_timeout: 0s` so no HTTP connection is held open during execution. Polling starts at 1-second intervals and grows with exponential backoff to a 10-second maximum. A configurable timeout (default 300 seconds) prevents infinite waits.

#### Step 8 — Write Observability Records

After execution, three writes happen to the PostgreSQL control plane:

1. **Audit table** — the `RUNNING` row inserted at ingestion start is updated to `SUCCESS` or `FAILED` with the Databricks `statement_id`, row count, wall-clock duration, and any error message.
2. **Metrics table** — daily counters are incremented via a PostgreSQL upsert.
3. **Structured log** — a JSON log entry is written with the `trace_id` linking it to the audit row and the metrics record.

#### Step 9 (Optional) — Optimize

```sql
OPTIMIZE main.bronze.bronze_billing_payments
ZORDER BY (_bronze_ingestion_timestamp);

VACUUM main.bronze.bronze_billing_payments RETAIN 168 HOURS;
```

`ZORDER BY` physically orders data by ingestion timestamp to accelerate time-range queries in downstream models. `VACUUM` with 168-hour retention removes old file versions while preserving the Delta Lake time-travel window. This step is separated from ingestion so it can be scheduled on a less frequent cadence.

---

### 6.5 Key Components

#### `bronze/partition_strategy.py` — `PartitionHeuristics`

Determines the Delta Lake partition strategy from dataset characteristics with no manual configuration.

**Time column patterns:** `timestamp`, `created_time`, `resolved_time`, `billing_month`, `as_of_date`, `date`, `datetime`

**Category column patterns:** `disco`, `region`, `state`, `country`, `department`, `category`, `status`, `type`, `site_type`

`disco` (distribution company) is always prioritised as the primary category partition — it is the most universally useful filter dimension across all energy datasets.

Returns a `PartitionConfig` dataclass with `strategy`, `partition_columns`, `reason`, and `use_append_only`.

---

#### `bronze/schema_mapper.py` — `SchemaMapper`

Maps contract type strings to Databricks SQL types and Spark type strings. Strict by design — unknown types raise `ValueError` rather than defaulting silently.

| Contract Type | Databricks SQL | Spark Schema |
| ------------- | -------------- | ------------ |
| `string`      | `STRING`       | `string`     |
| `double`      | `DOUBLE`       | `double`     |
| `bool`        | `BOOLEAN`      | `boolean`    |
| `int64`       | `BIGINT`       | `bigint`     |
| `int32`       | `INT`          | `int`        |
| `float`       | `FLOAT`        | `float`      |
| `timestamp`   | `TIMESTAMP`    | `timestamp`  |
| `date`        | `DATE`         | `date`       |
| `binary`      | `BINARY`       | `binary`     |

Exposes three methods: `map_type()`, `generate_ddl_columns()`, and `generate_spark_schema_string()`.

---

#### `bronze/sql_generator.py` — `BronzeSQLGenerator`

Generates all SQL from three templates using Python's `str.format()`. No string concatenation. No dynamic f-strings. The template structure is fully auditable and the output is deterministic — the same contract always produces the same SQL.

| Template                | Used For                                                 |
| ----------------------- | -------------------------------------------------------- |
| `CREATE_TABLE_TEMPLATE` | Creating Delta tables with schema enforcement            |
| `COPY_INTO_TEMPLATE`    | Incremental idempotent file loading for small datasets   |
| `MERGE_UPSERT_TEMPLATE` | Append-only hash-deduplicated loading for large datasets |

---

#### `bronze/data_downloader.py` — `DataDownloader`

Downloads Parquet files from contract URLs to a staging area. Python handles the file transfer; Databricks reads from staging. This separation means the download step can be retried, inspected, or replaced without touching any SQL logic.

---

#### `bronze/databricks_client.py` — `DatabricksSQLClient`

A thin wrapper around the Databricks SQL Statements REST API. Submits SQL with `wait_timeout: 0s` then polls until the statement reaches a terminal state.

Configuration is read from `databricks/databricks.cfg`:

```ini
[DEFAULT]
workspace_url = https://your-workspace.cloud.databricks.com
token         = dapi...
warehouse_id  = abc123...
```

---

### 6.6 Observability

The observability layer in `bronze/observer/` is backed by PostgreSQL and designed to be lightweight and proportional to the current scale of the pipeline. It intentionally excludes Prometheus, Grafana, and OpenTelemetry — those are Phase 2 scope.

#### Trace Correlation

Every ingestion run generates a UUID `trace_id` at the very start of execution. This ID is propagated to the audit table, every structured log entry, and the metrics upsert for that run. Given any `trace_id`, you can reconstruct the complete lifecycle of a run across all observability surfaces without distributed tracing infrastructure.

#### Audit Table (`bronze_ingestion_audit`)

```sql
CREATE TABLE bronze_ingestion_audit (
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
```

A `RUNNING` row is inserted **before** any SQL is submitted. It is updated to `SUCCESS` or `FAILED` on completion. If the pipeline crashes mid-execution, the unresolved `RUNNING` record makes the orphaned run immediately visible without additional monitoring.

#### Metrics Table (`bronze_ingestion_metrics`)

```sql
CREATE TABLE bronze_ingestion_metrics (
    dataset_name               TEXT,
    metric_date                DATE,
    ingestion_success_total    BIGINT,
    ingestion_failures_total   BIGINT,
    ingestion_rows_total       BIGINT,
    ingestion_duration_seconds DOUBLE PRECISION,
    schema_evolution_count     BIGINT,
    PRIMARY KEY (dataset_name, metric_date)
);
```

Updated via `INSERT ... ON CONFLICT DO UPDATE`. Each run contributes atomically to the running daily totals — concurrency-safe with no lost increments under parallel execution.

#### Structured Logging

All log events are JSON with mandatory `trace_id`, `dataset_name`, and `event` fields. Three event types are defined:

| Event                  | Trigger                       | Additional Fields                                    |
| ---------------------- | ----------------------------- | ---------------------------------------------------- |
| `bronze_sql_generated` | After SQL template is filled  | `partition_strategy`, `sql_type`                     |
| `bronze_sql_executed`  | After Databricks API responds | `statement_id`, `status`, `row_count`, `duration_ms` |
| `bronze_sql_failed`    | On any unhandled exception    | `error_message`                                      |

#### Per-Dataset Observability Rules

Thresholds can be configured per dataset in the ingestion contract:

```json
{
  "dataset_name": "billing_payments",
  "observability": {
    "alert_on_zero_rows": true,
    "max_expected_duration_sec": 120,
    "expected_min_rows": 1000
  }
}
```

If the block is absent, conservative defaults apply. `ObservabilityRuleEvaluator` checks the result against these rules after each run and emits a warning log entry for any violation — no hardcoded thresholds anywhere in code.

---

## 7. How Extraction Connects to Bronze

The two layers are deliberately decoupled. They share exactly one artefact: `bronze/bronze_metadata/bronze_ingestion_contract.json`.

```
Extraction Layer                           Bronze Layer
─────────────────                          ─────────────────────────────
api_config.json                            (never runs)
resolver.py                                (never runs)
validator.py                               (never runs)
runner.py              ──writes──►         bronze_ingestion_contract.json
                                                       │
                                                       ▼  reads
                                           bronze_orchestrator.py
                                           partition_strategy.py
                                           schema_mapper.py
                                           sql_generator.py
                                           data_downloader.py
                                           databricks_client.py
```

**The Extraction layer runs first, once**, to produce the contract. The contract is committed to the repository and version-controlled. The **Bronze layer runs repeatedly** — daily, on schedule, or on demand — reading the same contract every time.

This decoupling delivers three concrete benefits:

- The Bronze layer can be re-run any number of times without hitting the HuggingFace API
- The contract can be inspected, audited, and manually corrected if a dataset changes shape
- The two layers evolve independently — changing the extraction mechanism requires no changes to any Bronze code, as long as the contract format is preserved

---

## 8. Trade-offs and Engineering Decisions

### A JSON file contract rather than a database table

The contract is a file, not a database row, because it needs to be **version-controlled**. When a dataset's schema changes, the git diff shows exactly what changed, who changed it, and when. A database table cannot provide this without a dedicated schema audit mechanism. The file is also simpler to bootstrap — no shared database schema is required for two layers to communicate.

### Python for orchestration rather than a Spark job

The orchestrator's role is coordination — reading the contract, making decisions, generating SQL, calling APIs. None of these tasks need Spark. Putting orchestration logic in a Spark job would make it harder to test in isolation, slower to iterate on locally, and unnecessarily expensive to run for what are control-plane operations. Spark runs where it provides value: data processing at scale.

### MERGE rather than INSERT OVERWRITE for large datasets

`INSERT OVERWRITE` destroys existing data before re-loading. A failure mid-run leaves the target table empty. `MERGE` with `WHEN NOT MATCHED THEN INSERT *` is append-only — existing data is never touched. A failure mid-merge leaves the table in its previous complete state plus however many rows were successfully processed. This is a materially safer failure mode for a layer that is the durable record of raw source data.

### SHA-256 row hash for deduplication rather than a natural key

Not all datasets have a stable natural primary key. Rather than building dataset-specific key logic, a SHA-256 hash of all column values concatenated is computed and stored as `_bronze_row_hash`. This is a universal, schema-agnostic deduplication mechanism that works identically across all six datasets and any future dataset added to the contract. The trade-off is that two records with identical data but different business meaning will be deduplicated at Bronze — this is accepted at this layer, and addressed at Silver.

### PostgreSQL for observability rather than a Delta table

Observability data needs to be available immediately after each run, even if the Databricks cluster is shut down. PostgreSQL is always running (it is also the Airflow backend) and provides transactional writes. A Delta table would require a running Spark cluster to query, making observability dependent on the very system being observed.

### `force = false` in COPY INTO

Databricks maintains an internal record of which source files have already been loaded into each target table. Re-running `COPY INTO` with `force = false` will not re-insert data from files already processed, making ingestion inherently idempotent without application-level state management.

### `mergeSchema = false` in COPY INTO and enforced schema in MERGE

A schema mismatch between a new source file and the existing Bronze table is caught at load time rather than discovered downstream. Rejecting the mismatched file and routing it to `badRecordsPath` is safer than silently accepting schema drift that could corrupt Silver models.

### No Prometheus or distributed tracing at this stage

This is an MVP. The pipeline runs six datasets on a daily schedule. Introducing a full observability stack at this scale adds infrastructure complexity without solving a real operational problem yet. The PostgreSQL metrics table can serve a dashboard today and be migrated to Prometheus when query volume or team growth justifies it.

---

## 9. Datasets

All six datasets are sourced from the `electricsheepafrica` organisation on HuggingFace and defined in `bronze/bronze_metadata/bronze_ingestion_contract.json`.

| Dataset                             | Rows    | Columns | Partition Strategy           | Ingestion Method |
| ----------------------------------- | ------- | ------- | ---------------------------- | ---------------- |
| `billing_payments`                  | 200,000 | 10      | TIME_BASED (`billing_month`) | MERGE            |
| `commercial_industries_consumption` | 220,000 | 11      | TIME_BASED (`timestamp`)     | MERGE            |
| `customers_complaint`               | 100,000 | 9       | TIME_BASED (`created_time`)  | MERGE            |
| `grid_load`                         | 200,000 | 10      | TIME_BASED (`timestamp`)     | MERGE            |
| `power_flow`                        | 200,000 | 10      | TIME_BASED (`timestamp`)     | MERGE            |
| `retail_tariffs`                    | 90,000  | 6       | NONE (below threshold)       | COPY INTO        |

---

## 10. Testing

```bash
pytest tests/ -v
```

| File                          | What It Tests                                                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_partition_strategy.py`  | All 6 real contract datasets + 9 synthetic edge cases: boundary rows, no time column, no category column, zero rows, partial name matches, very large multi-file datasets                                                       |
| `test_sql_generator.py`       | CREATE TABLE, COPY INTO, and MERGE SQL for all 6 datasets. Invalid metadata, corrupt metadata, SQL injection pass-through, unfilled placeholder detection, SchemaMapper integration (NOT NULL, type mapping, Spark type casing) |
| `test_data_downloader.py`     | Retry logic, exponential backoff timing, cache hit/skip, partial file cleanup on failure, HTTP 404 handling, PyArrow validation                                                                                                 |
| `test_bronze_orchestrator.py` | Full pipeline orchestration — idempotency, atomicity, dry-run isolation, MERGE routing logic, dataset name filtering, audit log correctness, SQL file persistence                                                               |
| `db_test.py`                  | PostgreSQL connectivity and observability schema validation                                                                                                                                                                     |

### Observability Test Strategy

| Category           | Scenarios                                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Unit tests         | Metric delta logic, date handling, UTC enforcement, log structure                                                                        |
| Integration tests  | First ingestion of day, multiple runs same day (counter accumulation), failure run, schema evolution run, multiple datasets on same date |
| Concurrency tests  | 10 parallel threads writing to same dataset — no lost increments, no deadlocks                                                           |
| Failure simulation | Postgres down mid-run, audit constraint violation, Databricks success with metrics write failure                                         |
| Edge cases         | Midnight boundary, 10M+ row count overflow, 8-hour duration float precision, schema evolution storm, dataset rename isolation            |

---

## 11. Setup and Configuration

### Prerequisites

- Python 3.9+
- Docker and docker-compose
- A Databricks workspace with a running SQL Warehouse
- PostgreSQL (provided via docker-compose)

### Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root:

```env
PG_HOST=localhost
PG_PORT=5432
PG_DB=pipeline_control
PG_USER=pipeline
PG_PASSWORD=your_password
```

### Databricks Configuration

Edit `databricks/databricks.cfg`:

```ini
[DEFAULT]
workspace_url = https://your-workspace.cloud.databricks.com
token         = dapi...
warehouse_id  = your_warehouse_id

[POSTGRES]
host     = localhost
port     = 5432
dbname   = pipeline_control
user     = pipeline
password = your_password
```

### Start Infrastructure

```bash
docker-compose up -d
```

---

## 12. Running the Pipeline

### Step 1 — Run the Extraction Layer

Queries the HuggingFace API and produces the ingestion contract. Only needs to be re-run when datasets change.

```bash
python -m extraction.runner
```

Output: `bronze/bronze_metadata/bronze_ingestion_contract.json`

### Step 2 — Run the Bronze Layer

**Full pipeline, all datasets:**

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
    dry_run=True        # Set to False to execute against Databricks
)
```

**Single dataset:**

```python
orchestrator.run_full_pipeline(
    datasets=['billing_payments'],
    download=True,
    dry_run=False
)
```

**Individual steps:**

```python
orchestrator.create_bronze_tables(dry_run=True)
orchestrator.ingest_data(download=True, dry_run=True)
orchestrator.optimize_tables(dry_run=True)
```

**Via CLI:**

```bash
python runners/run_bronze.py
```

### Dry Run

`dry_run=True` generates all SQL, writes it to `/tmp/bronze_<dataset>_create.sql` and `/tmp/bronze_<dataset>_ingest.sql`, and exits without executing anything against Databricks. This is the recommended way to verify what the pipeline will do before running it for the first time or after a contract change.

---

## 13. Roadmap

### Silver Layer _(planned)_

The Silver layer will consume Bronze Delta tables via dbt and apply type casting, null handling, business-key deduplication, and standardised column naming. Built with dbt models in `dbt_project/models/staging/`.

### Gold Layer _(planned)_

The Gold layer will aggregate Silver models into analytics-ready business marts: revenue by DISCO and tariff band, grid load actual vs forecast, complaint SLA adherence, and retail tariff trends. Built with dbt models in `dbt_project/models/marts/`.

### Observability Phase 2

- Prometheus metrics export
- Grafana dashboards
- OpenTelemetry distributed tracing
- Automated data quality assertions on Bronze output tables

### Observability Phase 3

- SLA monitoring per dataset per run
- Automated anomaly detection on row count and duration trends
- Cross-layer lineage observability from Bronze through Gold
