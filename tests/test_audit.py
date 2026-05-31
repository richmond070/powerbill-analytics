"""
Integration Tests — AuditWriter
=================================
File location : tests/test_audit.py
Covers        : Audit lifecycle Cases A–E (equivalent of metrics Cases A–E
                but for the bronze_ingestion_audit table)

    Case A — insert_running() creates a RUNNING row immediately with all
              expected fields written correctly. The returned audit_id is
              the real SERIAL PK that can be used to complete the record.

    Case B — Full success lifecycle: RUNNING → SUCCESS.
              insert_running() then update_completed(status="SUCCEEDED")
              produces a single row with status="SUCCESS", statement_id,
              row_count, duration_ms all stored, error_message NULL.

    Case C — Full failure lifecycle: RUNNING → FAILED.
              insert_running() then update_completed(status="FAILED")
              stores error_message and leaves row_count NULL.

    Case D — mark_failed() convenience wrapper produces the same result
              as calling update_completed() directly with FAILED status.
              statement_id must be "N/A" and row_count must be NULL.

    Case E — Multiple datasets each produce their own independent audit rows.
              Rows never overwrite each other — every run is a new INSERT.

Key differences from test_metrics_integration.py
-------------------------------------------------
- audit table has one row PER RUN (no upsert — every insert_running() adds
  a new row). Isolation is straightforward: TRUNCATE before each test.
- The two-step write (INSERT → UPDATE) means we test intermediate state
  (RUNNING) before testing final state (SUCCESS / FAILED).
- Status normalisation must be verified: the SUT receives "SUCCEEDED" from
  the Databricks API but must write "SUCCESS" to satisfy the CHECK constraint.

Prerequisites
-------------
Same Postgres instance used by test_metrics_integration.py.
Credentials via env vars or databricks/databricks.cfg [POSTGRES] section:

    PG_HOST=localhost
    PG_PORT=5432
    PG_DB=xxxxx_db
    PG_USER=postgres
    PG_PASSWORD=<password>

Run:
    docker-compose up -d postgres
    pytest tests/test_audit.py -v
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------
from bronze.observer.audit_writer import AuditWriter
from bronze.observer.db_pool import close_pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ALPHA     = "billing_payments"
DATASET_BETA      = "grid_load"
STRATEGY_TIME     = "time_based"
STRATEGY_HYBRID   = "hybrid"
FAKE_STATEMENT_ID = "01ef-abc123-databricks-stmt"


# ===========================================================================
# Infrastructure fixtures  (mirror test_metrics_integration.py exactly)
# ===========================================================================

def _raw_conn() -> psycopg2.extensions.connection:
    """
    Direct psycopg2 connection for fixture setup/teardown and result reading.
    Never used for SUT calls — kept completely separate from the pool.
    """
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )


@pytest.fixture(scope="session")
def pg_schema():
    """
    Session-scoped: ensure both observability tables exist before any test runs.
    Safe to call repeatedly — all DDL uses IF NOT EXISTS.
    """
    from bronze.observer.observability_schema import ensure_observability_tables
    ensure_observability_tables()
    yield


@pytest.fixture(autouse=True)
def clean_tables(pg_schema):
    """
    Function-scoped (autouse): TRUNCATE audit table before every test so
    each case starts with zero rows.

    AuditWriter.insert_running() commits via its own pg_connection() call,
    so we cannot use a transaction rollback for isolation — TRUNCATE is the
    only reliable approach.

    Also resets the db_pool singleton after every test to prevent connection
    state from leaking between tests.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE bronze_ingestion_audit  RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE bronze_ingestion_metrics RESTART IDENTITY CASCADE;")
        conn.commit()
    finally:
        conn.close()

    yield  # ← test body runs here

    close_pool()   # reset singleton so next test gets a fresh pool


@pytest.fixture
def writer() -> AuditWriter:
    """Return an AuditWriter wired to the test database via env-vars."""
    return AuditWriter()


# ===========================================================================
# Helpers — read audit rows back from Postgres
# ===========================================================================

def _fetch_by_id(audit_id: int) -> Optional[Dict]:
    """
    Fetch a single audit row by its SERIAL PK.
    Returns a dict of all columns, or None if the row doesn't exist.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    trace_id,
                    dataset_name,
                    partition_strategy,
                    statement_id,
                    status,
                    row_count,
                    duration_ms,
                    execution_time,
                    error_message
                FROM bronze_ingestion_audit
                WHERE id = %s;
                """,
                (audit_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "id":                 row[0],
        "trace_id":           row[1],
        "dataset_name":       row[2],
        "partition_strategy": row[3],
        "statement_id":       row[4],
        "status":             row[5],
        "row_count":          row[6],
        "duration_ms":        row[7],
        "execution_time":     row[8],
        "error_message":      row[9],
    }


def _fetch_by_trace(trace_id: uuid.UUID) -> Optional[Dict]:
    """
    Fetch an audit row by trace_id UUID.
    Useful when audit_id is not directly available in the test.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, trace_id, dataset_name, partition_strategy,
                    statement_id, status, row_count, duration_ms,
                    execution_time, error_message
                FROM bronze_ingestion_audit
                WHERE trace_id = %s;
                """,
                (str(trace_id),),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "id":                 row[0],
        "trace_id":           row[1],
        "dataset_name":       row[2],
        "partition_strategy": row[3],
        "statement_id":       row[4],
        "status":             row[5],
        "row_count":          row[6],
        "duration_ms":        row[7],
        "execution_time":     row[8],
        "error_message":      row[9],
    }


def _count_rows_for_dataset(dataset_name: str) -> int:
    """Return total audit rows for a given dataset_name."""
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM bronze_ingestion_audit WHERE dataset_name = %s;",
                (dataset_name,),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


def _count_all_rows() -> int:
    """Return total row count across the entire audit table."""
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bronze_ingestion_audit;")
            return cur.fetchone()[0]
    finally:
        conn.close()


# ===========================================================================
# Case A — insert_running() creates a RUNNING record
# ===========================================================================

class TestCaseA_InsertRunning:
    """
    insert_running() is the first step in every audit lifecycle.
    It must:
      - Insert exactly one row into bronze_ingestion_audit
      - Set status = 'RUNNING'
      - Store trace_id, dataset_name, partition_strategy correctly
      - Return the SERIAL PK (audit_id) that the caller will use for the update
      - Leave statement_id, row_count, duration_ms, error_message all NULL
        because execution has not happened yet
      - Populate execution_time via Postgres DEFAULT (must not be NULL)
    """

    def test_insert_running_returns_integer_audit_id(self, writer):
        """insert_running() must return an int — the SERIAL PK."""
        trace_id  = uuid.uuid4()
        audit_id  = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)

        assert isinstance(audit_id, int)
        assert audit_id > 0

    def test_exactly_one_row_created(self, writer):
        """A single insert_running() call must produce exactly one audit row."""
        writer.insert_running(uuid.uuid4(), DATASET_ALPHA, STRATEGY_TIME)

        assert _count_all_rows() == 1

    def test_status_is_running(self, writer):
        """The initial status must be 'RUNNING', not SUCCESS or FAILED."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["status"] == "RUNNING"

    def test_trace_id_stored_correctly(self, writer):
        """The trace_id written to the DB must equal the UUID passed in."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        # psycopg2 returns UUID objects from UUID columns
        assert str(row["trace_id"]) == str(trace_id)

    def test_dataset_name_stored_correctly(self, writer):
        """dataset_name must be stored exactly as passed."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["dataset_name"] == DATASET_ALPHA

    def test_partition_strategy_stored_correctly(self, writer):
        """partition_strategy must be stored exactly as passed."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_HYBRID)
        row      = _fetch_by_id(audit_id)

        assert row["partition_strategy"] == STRATEGY_HYBRID

    def test_execution_time_is_populated_by_postgres(self, writer):
        """execution_time must be set by Postgres DEFAULT — never NULL."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["execution_time"] is not None

    def test_statement_id_is_null_before_completion(self, writer):
        """statement_id is only known after Databricks executes — must be NULL on insert."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["statement_id"] is None

    def test_row_count_is_null_before_completion(self, writer):
        """row_count is unknown until execution completes — must be NULL on insert."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["row_count"] is None

    def test_duration_ms_is_null_before_completion(self, writer):
        """duration_ms is unknown until execution completes — must be NULL on insert."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["duration_ms"] is None

    def test_error_message_is_null_before_completion(self, writer):
        """error_message is NULL unless a failure occurs."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["error_message"] is None

    def test_returned_audit_id_matches_stored_id(self, writer):
        """The returned audit_id must equal the id column in the stored row."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        row      = _fetch_by_id(audit_id)

        assert row["id"] == audit_id


# ===========================================================================
# Case B — Full success lifecycle: RUNNING → SUCCESS
# ===========================================================================

class TestCaseB_SuccessLifecycle:
    """
    The happy path: insert_running() followed by update_completed() with
    status="SUCCEEDED".

    The SUT normalises "SUCCEEDED" (Databricks API value) → "SUCCESS"
    (audit table CHECK constraint value). This normalisation must be tested.

    After update_completed():
      - status        = "SUCCESS"   (not "SUCCEEDED")
      - statement_id  = stored value
      - row_count     = stored value
      - duration_ms   = stored value
      - error_message = NULL
      - Still exactly ONE row — update never creates a second row
    """

    def _full_success_run(self, writer, trace_id, row_count=10_000, duration_ms=2_500):
        """Helper: run the full RUNNING → SUCCESS lifecycle."""
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=FAKE_STATEMENT_ID,
            status="SUCCEEDED",         # Databricks API value
            row_count=row_count,
            duration_ms=duration_ms,
        )
        return audit_id

    def test_status_is_success_after_update(self, writer):
        """status must be 'SUCCESS' (not 'SUCCEEDED') after update_completed."""
        trace_id = uuid.uuid4()
        audit_id = self._full_success_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["status"] == "SUCCESS"

    def test_databricks_succeeded_normalised_to_success(self, writer):
        """
        Explicit normalisation test: the SUT receives "SUCCEEDED" but must
        write "SUCCESS" to satisfy the CHECK constraint.
        """
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=FAKE_STATEMENT_ID,
            status="SUCCEEDED",         # raw Databricks value
            row_count=1_000,
            duration_ms=1_000,
        )
        row = _fetch_by_id(audit_id)

        # Must NOT store "SUCCEEDED" — that violates the CHECK constraint
        assert row["status"] != "SUCCEEDED"
        assert row["status"] == "SUCCESS"

    def test_statement_id_stored_after_completion(self, writer):
        """statement_id must be written by update_completed."""
        trace_id = uuid.uuid4()
        audit_id = self._full_success_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["statement_id"] == FAKE_STATEMENT_ID

    def test_row_count_stored_after_completion(self, writer):
        """row_count must equal the value passed to update_completed."""
        trace_id = uuid.uuid4()
        audit_id = self._full_success_run(writer, trace_id, row_count=10_000)
        row      = _fetch_by_id(audit_id)

        assert row["row_count"] == 10_000

    def test_duration_ms_stored_after_completion(self, writer):
        """duration_ms must equal the value passed to update_completed."""
        trace_id = uuid.uuid4()
        audit_id = self._full_success_run(writer, trace_id, duration_ms=2_500)
        row      = _fetch_by_id(audit_id)

        assert row["duration_ms"] == 2_500

    def test_error_message_is_null_on_success(self, writer):
        """error_message must remain NULL when the run succeeds."""
        trace_id = uuid.uuid4()
        audit_id = self._full_success_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["error_message"] is None

    def test_still_exactly_one_row_after_update(self, writer):
        """
        update_completed() is an UPDATE not an INSERT.
        The table must still have exactly one row after the full lifecycle.
        """
        trace_id = uuid.uuid4()
        self._full_success_run(writer, trace_id)

        assert _count_all_rows() == 1

    def test_trace_id_unchanged_after_update(self, writer):
        """The trace_id stored on INSERT must not change after UPDATE."""
        trace_id = uuid.uuid4()
        audit_id = self._full_success_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert str(row["trace_id"]) == str(trace_id)

    def test_row_count_none_stored_as_null(self, writer):
        """
        When row_count is None (e.g. DDL statements don't return row counts)
        the column must be NULL, not zero or some other value.
        """
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=FAKE_STATEMENT_ID,
            status="SUCCEEDED",
            row_count=None,             # DDL / no row count available
            duration_ms=1_000,
        )
        row = _fetch_by_id(audit_id)

        assert row["row_count"] is None


# ===========================================================================
# Case C — Full failure lifecycle: RUNNING → FAILED
# ===========================================================================

class TestCaseC_FailureLifecycle:
    """
    The failure path: insert_running() then update_completed() with
    status="FAILED".

    After update_completed():
      - status        = "FAILED"
      - error_message = stored value (never NULL on failure)
      - row_count     = NULL (no rows ingested on failure)
      - statement_id  = stored (the API may still return one even on failure)
      - duration_ms   = stored (wall-clock time still elapsed)
    """

    def _full_failure_run(
        self,
        writer,
        trace_id,
        error_message="Databricks SQL execution failed: timeout",
        duration_ms=1_200,
    ):
        """Helper: run the full RUNNING → FAILED lifecycle."""
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=FAKE_STATEMENT_ID,
            status="FAILED",
            row_count=None,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        return audit_id

    def test_status_is_failed_after_update(self, writer):
        trace_id = uuid.uuid4()
        audit_id = self._full_failure_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["status"] == "FAILED"

    def test_error_message_stored_on_failure(self, writer):
        """error_message must be persisted exactly as passed."""
        error_msg = "Databricks SQL execution failed: timeout"
        trace_id  = uuid.uuid4()
        audit_id  = self._full_failure_run(writer, trace_id, error_message=error_msg)
        row       = _fetch_by_id(audit_id)

        assert row["error_message"] == error_msg

    def test_error_message_is_not_null_on_failure(self, writer):
        """A failed run must always have an error_message — never NULL."""
        trace_id = uuid.uuid4()
        audit_id = self._full_failure_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["error_message"] is not None

    def test_row_count_is_null_on_failure(self, writer):
        """No rows are ingested on failure — row_count must be NULL."""
        trace_id = uuid.uuid4()
        audit_id = self._full_failure_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["row_count"] is None

    def test_duration_ms_stored_on_failure(self, writer):
        """Wall-clock time elapsed even on failure — must be persisted."""
        trace_id = uuid.uuid4()
        audit_id = self._full_failure_run(writer, trace_id, duration_ms=1_200)
        row      = _fetch_by_id(audit_id)

        assert row["duration_ms"] == 1_200

    def test_statement_id_stored_on_failure(self, writer):
        """Databricks may return a statement_id even on failure — must be stored."""
        trace_id = uuid.uuid4()
        audit_id = self._full_failure_run(writer, trace_id)
        row      = _fetch_by_id(audit_id)

        assert row["statement_id"] == FAKE_STATEMENT_ID

    def test_still_exactly_one_row_after_failure_update(self, writer):
        """update_completed() must not create a second row on failure."""
        trace_id = uuid.uuid4()
        self._full_failure_run(writer, trace_id)

        assert _count_all_rows() == 1

    def test_non_succeeded_status_normalised_to_failed(self, writer):
        """
        Any status that is not "SUCCEEDED" must be stored as "FAILED".
        This covers Databricks states: CANCELED, CLOSED, ERROR.
        """
        for databricks_status in ("CANCELED", "CLOSED", "ERROR"):
            trace_id = uuid.uuid4()
            audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
            writer.update_completed(
                audit_id=audit_id,
                trace_id=trace_id,
                statement_id=FAKE_STATEMENT_ID,
                status=databricks_status,
                row_count=None,
                duration_ms=500,
                error_message=f"Run ended with status: {databricks_status}",
            )
            row = _fetch_by_id(audit_id)
            assert row["status"] == "FAILED", (
                f"Expected 'FAILED' for Databricks status '{databricks_status}', "
                f"got '{row['status']}'"
            )


# ===========================================================================
# Case D — mark_failed() convenience wrapper
# ===========================================================================

class TestCaseD_MarkFailed:
    """
    mark_failed() is a convenience wrapper around update_completed() for
    the common case where an exception fires before the Databricks API returns.

    It must produce the same result as calling update_completed() directly
    with status="FAILED", statement_id="N/A", row_count=None.

    Specific contract:
      - status        = "FAILED"
      - statement_id  = "N/A"     (not a real Databricks statement ID)
      - row_count     = NULL
      - error_message = passed-in exception message
      - duration_ms   = passed-in value (or 0 if not provided)
    """

    def test_status_is_failed(self, writer):
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(
            audit_id=audit_id,
            trace_id=trace_id,
            error_message="Connection to Databricks lost",
        )
        row = _fetch_by_id(audit_id)

        assert row["status"] == "FAILED"

    def test_statement_id_is_na(self, writer):
        """
        mark_failed() is called before the Databricks API returns a statement ID.
        The placeholder 'N/A' must be stored.
        """
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(
            audit_id=audit_id,
            trace_id=trace_id,
            error_message="Pre-execution failure",
        )
        row = _fetch_by_id(audit_id)

        assert row["statement_id"] == "N/A"

    def test_row_count_is_null(self, writer):
        """No rows were ingested — row_count must be NULL."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(
            audit_id=audit_id,
            trace_id=trace_id,
            error_message="Pre-execution failure",
        )
        row = _fetch_by_id(audit_id)

        assert row["row_count"] is None

    def test_error_message_stored(self, writer):
        """The error message passed to mark_failed() must be persisted."""
        error_msg = "Connection to Databricks lost before execution"
        trace_id  = uuid.uuid4()
        audit_id  = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(
            audit_id=audit_id,
            trace_id=trace_id,
            error_message=error_msg,
        )
        row = _fetch_by_id(audit_id)

        assert row["error_message"] == error_msg

    def test_duration_ms_stored_when_provided(self, writer):
        """When duration_ms is provided it must be persisted."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(
            audit_id=audit_id,
            trace_id=trace_id,
            error_message="Timeout",
            duration_ms=750,
        )
        row = _fetch_by_id(audit_id)

        assert row["duration_ms"] == 750

    def test_duration_ms_defaults_to_zero(self, writer):
        """When duration_ms is not provided the default (0) must be stored."""
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(
            audit_id=audit_id,
            trace_id=trace_id,
            error_message="Timeout",
            # duration_ms not passed — default is 0
        )
        row = _fetch_by_id(audit_id)

        assert row["duration_ms"] == 0

    def test_mark_failed_produces_identical_result_to_update_completed(self, writer):
        """
        mark_failed() must be exactly equivalent to calling update_completed()
        with status='FAILED', statement_id='N/A', row_count=None.
        Compare two separate runs side-by-side.
        """
        error_msg    = "Same error"
        duration_ms  = 300

        # Run A — use mark_failed()
        trace_a  = uuid.uuid4()
        id_a     = writer.insert_running(trace_a, DATASET_ALPHA, STRATEGY_TIME)
        writer.mark_failed(id_a, trace_a, error_msg, duration_ms)
        row_a    = _fetch_by_id(id_a)

        # Run B — use update_completed() directly with equivalent arguments
        trace_b  = uuid.uuid4()
        id_b     = writer.insert_running(trace_b, DATASET_ALPHA, STRATEGY_TIME)
        writer.update_completed(
            audit_id=id_b,
            trace_id=trace_b,
            statement_id="N/A",
            status="FAILED",
            row_count=None,
            duration_ms=duration_ms,
            error_message=error_msg,
        )
        row_b    = _fetch_by_id(id_b)

        # Both rows must have identical observable state (excluding id & trace_id)
        for field in ("status", "statement_id", "row_count", "duration_ms", "error_message"):
            assert row_a[field] == row_b[field], (
                f"Field '{field}' differs: mark_failed={row_a[field]!r}, "
                f"update_completed={row_b[field]!r}"
            )


# ===========================================================================
# Case E — Multiple datasets produce independent audit rows
# ===========================================================================

class TestCaseE_MultipleDatasets:
    """
    Unlike bronze_ingestion_metrics (which upserts one row per dataset per day),
    the audit table creates a NEW row for every single run.

    This means:
      - N runs across M datasets = N total rows (no merging)
      - Rows for dataset_A are completely independent from rows for dataset_B
      - A failure for dataset_A must not affect dataset_B's audit records
      - The audit_id returned from each insert_running() must be unique
    """

    def test_two_datasets_produce_two_rows(self, writer):
        """One insert_running() per dataset = two rows in the table."""
        writer.insert_running(uuid.uuid4(), DATASET_ALPHA, STRATEGY_TIME)
        writer.insert_running(uuid.uuid4(), DATASET_BETA,  STRATEGY_HYBRID)

        assert _count_all_rows() == 2

    def test_each_dataset_has_its_own_row(self, writer):
        """Each dataset must appear in exactly one row — no mixing of names."""
        writer.insert_running(uuid.uuid4(), DATASET_ALPHA, STRATEGY_TIME)
        writer.insert_running(uuid.uuid4(), DATASET_BETA,  STRATEGY_HYBRID)

        assert _count_rows_for_dataset(DATASET_ALPHA) == 1
        assert _count_rows_for_dataset(DATASET_BETA)  == 1

    def test_three_runs_same_dataset_produce_three_rows(self, writer):
        """
        The audit table never upserts — every run is a new INSERT.
        Three runs for the same dataset must produce three rows.
        """
        for _ in range(3):
            writer.insert_running(uuid.uuid4(), DATASET_ALPHA, STRATEGY_TIME)

        assert _count_rows_for_dataset(DATASET_ALPHA) == 3

    def test_audit_ids_are_unique_across_runs(self, writer):
        """Each call to insert_running() must return a distinct SERIAL PK."""
        ids = [
            writer.insert_running(uuid.uuid4(), DATASET_ALPHA, STRATEGY_TIME)
            for _ in range(5)
        ]
        assert len(ids) == len(set(ids)), "Duplicate audit_ids returned"

    def test_completing_alpha_does_not_affect_beta_row(self, writer):
        """
        update_completed() targets by audit_id (PK).
        Completing run A must leave run B's row untouched (still RUNNING).
        """
        trace_a  = uuid.uuid4()
        trace_b  = uuid.uuid4()

        id_a = writer.insert_running(trace_a, DATASET_ALPHA, STRATEGY_TIME)
        id_b = writer.insert_running(trace_b, DATASET_BETA,  STRATEGY_HYBRID)

        # Complete only run A
        writer.update_completed(
            audit_id=id_a,
            trace_id=trace_a,
            statement_id=FAKE_STATEMENT_ID,
            status="SUCCEEDED",
            row_count=5_000,
            duration_ms=2_000,
        )

        row_a = _fetch_by_id(id_a)
        row_b = _fetch_by_id(id_b)

        assert row_a["status"] == "SUCCESS"   # completed
        assert row_b["status"] == "RUNNING"   # untouched

    def test_failing_alpha_does_not_affect_beta_row(self, writer):
        """
        mark_failed() on run A must leave run B's row in RUNNING state.
        """
        trace_a = uuid.uuid4()
        trace_b = uuid.uuid4()

        id_a = writer.insert_running(trace_a, DATASET_ALPHA, STRATEGY_TIME)
        id_b = writer.insert_running(trace_b, DATASET_BETA,  STRATEGY_HYBRID)

        writer.mark_failed(id_a, trace_a, "Dataset A exploded")

        row_a = _fetch_by_id(id_a)
        row_b = _fetch_by_id(id_b)

        assert row_a["status"] == "FAILED"
        assert row_b["status"] == "RUNNING"

    def test_each_run_stores_its_own_trace_id(self, writer):
        """
        Every row must carry the specific trace_id from its own run.
        No trace_id should appear on a row that belongs to a different run.
        """
        trace_alpha = uuid.uuid4()
        trace_beta  = uuid.uuid4()

        id_alpha = writer.insert_running(trace_alpha, DATASET_ALPHA, STRATEGY_TIME)
        id_beta  = writer.insert_running(trace_beta,  DATASET_BETA,  STRATEGY_HYBRID)

        row_alpha = _fetch_by_id(id_alpha)
        row_beta  = _fetch_by_id(id_beta)

        assert str(row_alpha["trace_id"]) == str(trace_alpha)
        assert str(row_beta["trace_id"])  == str(trace_beta)

        # Cross-check: no row should carry the other's trace_id
        assert str(row_alpha["trace_id"]) != str(trace_beta)
        assert str(row_beta["trace_id"])  != str(trace_alpha)