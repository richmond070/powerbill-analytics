# Nigerian Energy & Utilities — Data Pipeline

> A production-grade, metadata-driven data pipeline for ingesting, transforming, and serving Nigerian energy and utilities data. Built on **Python**, **Apache Spark**, and **Databricks SQL** with a **PostgreSQL-backed control plane** for observability and auditability.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Who This Document Is For](#2-who-this-document-is-for)
3. [Project Setup](#3-project-setup)
4. [Pipeline Layers](#4-pipeline-layers)
5. [Testing Overview](#5-testing-overview)
6. [Observability System](#6-observability-system)

## System Architecture 


---

## 1. Project Overview

This pipeline ingests six raw datasets from the Nigerian energy sector — spanning billing, grid load, power flow, commercial consumption, customer complaints, and retail tariffs — and processes them into a curated, query-ready Delta Lake data platform on Databricks.

The pipeline is built in layers following the **Medallion Architecture** pattern:

```
Extraction  →  Bronze  →  Silver     →      Gold
  (raw API)    (ingest)   (clean/model)  (serve)
```

Each layer has a clearly defined responsibility, a contract with the layer above it, and independent observability. This document covers the **Extraction**, **Bronze**, **Silver** and **Gold** layers in full.

**Data Source:** 6 Nigerian energy datasets with 1.01 million total rows

**Key Technologies:**

- Python 3.9+
- Databricks / Apache Spark / Delta Lake / DBT
- PostgreSQL (for observability)
- Pytest (testing framework)

---

## 2. Who This Document Is For

This document is written for an engineer picking up this codebase for the first time. It assumes you are comfortable with Python and SQL but may be unfamiliar with Databricks, Delta Lake, or how this specific pipeline is structured.

By the end of this document you should understand:

- What each layer does and why it exists
- How data flows from a raw API endpoint to a structured Delta table
- How to run the pipeline yourself

---

The pipeline follows **strict separation of concerns** at every boundary:

- The **Extraction layer** knows about external APIs. It does not know about Databricks.
- The **Bronze layer** knows about Databricks and Delta Lake. It does not know about HuggingFace.
- The **contract JSON file** is the only shared artefact between the two layers.

---

## 3. Project Setup

### Prerequisites

```bash
# System requirements
- Python 3.9 or higher
- PostgreSQL 12+
- Databricks Workspace Access
- Git

# Supported Operating Systems
- macOS (Intel/Apple Silicon)
- Linux (Ubuntu 20.04+, CentOS 7+)
- Windows 10/11 (via WSL2 recommended)
```

### Installation

**Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/energy-billing-analytics.git
cd energy-billing-analytics
```

**Step 2: Create Python Virtual Environment**

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**Step 3: Install Dependencies**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Key dependencies:**

```
pandas==2.2.3
numpy==1.26.4
pyarrow>=12.0.0
requests==2.31.0
psycopg2-binary==2.9.9
SQLAlchemy==2.0.34
pytest==8.3.3
dbt-core==1.10.13
```

**Step 4: Configure Databricks Connection**

> Create databricks/databricks.cfg:

```ini
[DEFAULT]
workspace_url = https://your-workspace.cloud.databricks.com
token = Your warehouse token (e.g dapixxxxxxxxxxabcdef)
warehouse_id = Your warehouse ID

[POSTGRES]
host = localhost
port = PORT
dbname = DATABASE NAME
user = DATABASE USER
password = your_secure_password
```

**How to get credentials:**

- Databricks workspace URL: From workspace settings
- Token: Generate in Databricks → User Settings → Developer Tools → Personal Access Tokens
- Warehouse ID: From Databricks SQL Endpoints list

**Step 5: Set Up PostgreSQL for Observability**

```bash
# Option A: Local PostgreSQL
brew install postgresql  # macOS
# or
sudo apt-get install postgresql postgresql-contrib  # Linux

# Create database
createdb bronze_control
psql bronze_control -c "CREATE USER pipeline_user WITH PASSWORD 'password';"
psql bronze_control -c "GRANT ALL PRIVILEGES ON DATABASE bronze_control TO pipeline_user;"

# Option B: Docker Container
docker run --name postgres-bronze \
  -e POSTGRES_USER=pipeline_user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=bronze_control \
  -p 5432:5432 \
  -d postgres:13
```

**Step 6: Verify Installation**

```bash
# Test Python imports
python -c "import pandas; import psycopg2; print('✓ Dependencies OK')"

# Test Databricks connection
python -c "from databricks.databricks_client import DatabricksSQLClient; print('✓ Databricks OK')"

# Test PostgreSQL connection
psql -h localhost -U pipeline_user -d bronze_control -c "SELECT version();"
```

**Step 7: Initialize Observability Schema**

_The observability tables are created automatically on first pipeline run via ensure_observability_tables()._

```bash
python -c "
from bronze.observer import ensure_observability_tables
ensure_observability_tables('databricks/databricks.cfg')
print('✓ Observability tables created')
"
```

**Step 8: Environment Variables (Optional)**

- Create .env:

```bash
DATABASE_URL=postgresql://postgres:password@localhost:port/database_name
HUGGINGFACE_DATA_URL=https://huggingface.co/collections/electricsheepafrica/nigeria-energy-sector

# Execution
RUN_MODE=execute  # or 'dry_run'
LOG_LEVEL=INFO
```

**Load in scripts:**

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 4. Pipeline Layers

### Bronze Layer: Raw Data Ingestion

The purpose of the bronze layer is to Ingest raw external datasets into trusted internal Delta tables with minimal transformation. It consist of Key Components such as; the _bronze_orchestrator.py_, which is the main orchestration it creates tables, ingests data, optimizes them. _SQL_generator.py_ generates idempotent SQL from metadata, _partition_strategy.py_ automatically determines optimal partitioning, _schema_mapper.py_ maps contract types to Databricks SQL types _databricks_client.py_ submits SQL to Databricks SQL API

#### Key Features:

- **Idempotent operations:** Safe to re-run without data corruption
- **Automatic partitioning:** Determines best strategy based on data size and columns
- **Schema enforcement:** Prevents schema drift in downstream tables
- **Error handling:** Graceful failures with detailed audit trails

### Silver Layer: Cleaned & Enriched Data

The silver layer transform bronze tables into business-ready datasets through deduplication, type casting, and column derivation. It reads cleaned data from the bronze tables and removes duplicates based on business logic (row number, timestamp), derives new columns while performing joins on multiple bronze tables for enriching customer views as well as partitioning relevant time columns for query efficiency.

#### Example Transformations:

- **billing_payments:** Deduplication + payment gap calculation
- **tariff_reference:** Latest pricing lookup table
- **customers_enriched:** Multi-table join (complaints + billing)

#### Key Features:

- **Dependency management:** Executes transformations in correct order
- **_Deduplication:_** Keeps only latest records per key
- **Derived metrics:** Calculates business-relevant columns
- **Data enrichment:** Combines multiple data sources

### Gold Layer: Business Analytics (Planned)

**Purpose:** Create aggregated, business-ready datasets for dashboards, reports, and ML features.
**Planned Datasets:**

- Customer lifetime value analysis
- Distribution company (DISCO) performance metrics
- Tariff elasticity studies
- Grid health indicators

**Status:** Design phase; will follow Silver layer pattern

## 5. Testing Overview

Comprehensive test suite with 165+ tests covering all pipeline components.

### Test Modules

| Module                  | Tests | Focus                                          | Location                            |
| ----------------------- | ----- | ---------------------------------------------- | ----------------------------------- |
| **SQL Generator**       | 55    | CREATE TABLE, COPY INTO, MERGE, OPTIMIZE logic | `tests/test_sql_generator.py`       |
| **Data Downloader**     | 30    | Download, retry, validation, cleanup           | `tests/test_data_downloader.py `    |
| **Partition Strategy**  | 15    | Heuristics for all dataset sizes               | `tests/test_partition_strategy.py`  |
| **Bronze Orchestrator** | 64    | End-to-end pipeline, idempotency, atomicity    | `tests/test_bronze_orchestrator.py` |
| **Smoke Test**          | 1     | Basic sanity check                             | `tests/test_hello_world.py`         |

### Test Coverage

- **Real Contract Data:** All tests use actual metadata from bronze_ingestion_contract.json (no hardcoded test data)
- **Edge Cases:** Boundary conditions, failure scenarios, concurrency, SQL injection, Unicode handling
- **Mocking Strategy:** Real Python modules; mocked Databricks API and HTTP calls
  ### Running Tests
  ```bash
  pytest tests/ -v # Run all tests
  pytest tests/test_partition_strategy.py -v # Run single module
  pytest tests/ --cov=bronze --cov-report=html # Coverage report
  ```
  For detailed test documentation, see individual test files with docstrings.

## 6. Observability System

**Purpose:** Provide lightweight, metadata-driven monitoring without Prometheus/Grafana/OpenTelemetry.

## Core Components

### 1. Audit Table

**What:** Run-level execution history
**Where:** PostgreSQL bronze_ingestion_audit table
**Tracks:** Status (RUNNING/SUCCESS/FAILED), row count, duration, error messages
**Use:** Root cause analysis, retry decisions

### 2. Metrics Aggregator

**What:** Daily counters per dataset
**Where:** PostgreSQL bronze_ingestion_metrics table
**Tracks:** Success/failure counts, total rows, duration, schema changes
**Use:** Trend detection, performance monitoring

### 3. Structured Logging

**What:** JSON-formatted event logs
**Format:** Single-line JSON to stdout
**Events:** bronze_sql_generated, bronze_sql_executed, bronze_sql_failed
**Use:** Real-time monitoring, debugging

### 4. Trace Correlation

**What:** UUID linking all events from one ingestion run
**Use:** Reconstruct execution flow across audit, metrics, and logs
**Code:** All observer modules

### How It Works

- Orchestrator generates trace ID at ingestion start
- Audit row created with status='RUNNING'
- SQL generated → log event with trace_id
- SQL executed → log event with result
- Audit row updated with status, row_count, duration
- Metrics aggregated to daily counters
- Rules evaluated (optional) → warnings if thresholds exceeded

### Schema Bootstrap

> Observability tables are created automatically on first pipeline run via ensure_observability_tables() in bronze/observer/observability_schema.py.
