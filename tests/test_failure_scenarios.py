"""
Failure Simulation, Edge Cases & Trace Correlation Tests
=========================================================
File location : tests/test_failure_scenarios.py
Covers        :
    Case G  — Postgres unavailable (simulated via mock)
    Case H  — Audit / metrics write fails (constraint violation, timeout)
    Case I  — Partial failure: Databricks succeeded, metrics write failed
    EC 1–6  — Edge cases (duplicate run, midnight boundary, large values,
               long duration, schema evolution storm, dataset rename)
    Case J  — Trace correlation: one trace_id flows through audit, metrics,
               and every BronzeLogger event

Test isolation strategy
-----------------------
Cases G & H  — mock-only, no real Postgres required.
               pg_connection is patched at the import site of each SUT module.
Cases I, EC, J — real Postgres required (same setup as other integration tests).
               autouse clean_tables fixture TRUNCATEs both tables before each test.

Mocking convention
------------------
Always patch where the name is USED, not where it is defined:
    "bronze.observer.metrics_aggregator.pg_connection"
    "bronze.observer.audit_writer.pg_connection"
This ensures the mock intercepts the actual call made by the SUT.

Prerequisites (for integration-backed tests)
--------------------------------------------
    PG_HOST=localhost  PG_PORT=5432
    PG_DB=xxxxx_db     PG_USER=postgres  PG_PASSWORD=<password>

    docker-compose up -d postgres
    pytest tests/test_failure_scenarios.py -v
"""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------
from bronze.ingestion.observer.metrics_aggregator import MetricsAggregator
from bronze.ingestion.observer.audit_writer import AuditWriter
from bronze.ingestion.observer.bronze_logger import BronzeLogger
from bronze.ingestion.observer.db_pool import close_pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET      = "billing_payments"
DATASET_B    = "grid_load"
STRATEGY     = "time_based"
STMT_ID      = "stmt-test-001"
TEST_DATE    = date(2024, 6, 15)
DATE_A       = date(2024, 6, 14)   # "23:59:59" side of midnight boundary
DATE_B       = date(2024, 6, 15)   # "00:00:01" side of midnight boundary


# ===========================================================================
# Shared infrastructure (mirrors other integration test files exactly)
# ===========================================================================

def _raw_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )


@pytest.fixture(scope="session")
def pg_schema():
    from bronze.ingestion.observer.observability_schema import ensure_observability_tables
    ensure_observability_tables()
    yield


@pytest.fixture(autouse=True)
def clean_tables(pg_schema):
    """TRUNCATE both tables + reset pool before every test."""
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE bronze_ingestion_metrics RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE bronze_ingestion_audit   RESTART IDENTITY CASCADE;")
        conn.commit()
    finally:
        conn.close()
    yield
    close_pool()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_metrics(dataset_name: str, metric_date: date) -> Optional[Dict]:
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ingestion_success_total, ingestion_failures_total,
                       ingestion_rows_total, ingestion_duration_seconds,
                       schema_evolution_count
                FROM bronze_ingestion_metrics
                WHERE dataset_name = %s AND metric_date = %s;
                """,
                (dataset_name, metric_date),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "ingestion_success_total":    row[0],
        "ingestion_failures_total":   row[1],
        "ingestion_rows_total":       row[2],
        "ingestion_duration_seconds": row[3],
        "schema_evolution_count":     row[4],
    }


def _count_metrics_rows(dataset_name: str, metric_date: date) -> int:
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM bronze_ingestion_metrics "
                "WHERE dataset_name = %s AND metric_date = %s;",
                (dataset_name, metric_date),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


def _fetch_audit_by_trace(trace_id: uuid.UUID) -> Optional[Dict]:
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, trace_id, dataset_name, status,
                       statement_id, row_count, duration_ms, error_message
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
        "id":            row[0],
        "trace_id":      row[1],
        "dataset_name":  row[2],
        "status":        row[3],
        "statement_id":  row[4],
        "row_count":     row[5],
        "duration_ms":   row[6],
        "error_message": row[7],
    }


def _count_audit_rows() -> int:
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bronze_ingestion_audit;")
            return cur.fetchone()[0]
    finally:
        conn.close()


# ===========================================================================
# Case G — Postgres unavailable
# ===========================================================================

class TestCaseG_PostgresDown:
    """
    Simulates Postgres being unreachable by patching pg_connection to raise
    psycopg2.OperationalError — the same error the driver raises when the
    server is down, the port is blocked, or the connection is refused.

    Critical contract:
      - MetricsAggregator must NOT swallow the error silently
      - AuditWriter must NOT swallow the error silently
      - The exception must propagate to the caller (bronze_orchestrator)
        so it can decide whether to retry, alert, or abort
      - Observability failure must never silently corrupt ingestion state
    """

    @pytest.fixture
    def aggregator(self):
        return MetricsAggregator()

    @pytest.fixture
    def writer(self):
        return AuditWriter()

    def test_metrics_record_ingestion_raises_when_postgres_down(self, aggregator):
        """
        MetricsAggregator.record_ingestion() must propagate OperationalError
        when pg_connection raises — not swallow it or return silently.
        """
        db_error = psycopg2.OperationalError("could not connect to server")

        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=db_error,
        ):
            with pytest.raises(psycopg2.OperationalError):
                aggregator.record_ingestion(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    success=True,
                    row_count=1_000,
                    duration_ms=1_000,
                    metric_date=TEST_DATE,
                )

    def test_metrics_error_is_not_swallowed_silently(self, aggregator):
        """
        A silent failure is worse than a raised exception — if record_ingestion()
        returns None without raising, the caller has no signal that the write failed.
        This test confirms the error surface is visible.
        """
        db_error = psycopg2.OperationalError("connection refused")

        raised = False
        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=db_error,
        ):
            try:
                aggregator.record_ingestion(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    success=True,
                    row_count=1_000,
                    duration_ms=1_000,
                    metric_date=TEST_DATE,
                )
            except psycopg2.OperationalError:
                raised = True

        assert raised, (
            "record_ingestion() returned silently when Postgres was down. "
            "Errors must propagate so the caller can handle them."
        )

    def test_audit_insert_running_raises_when_postgres_down(self, writer):
        """
        AuditWriter.insert_running() must propagate OperationalError
        when pg_connection raises.
        """
        db_error = psycopg2.OperationalError("server closed the connection unexpectedly")

        with patch(
            "bronze.observer.audit_writer.pg_connection",
            side_effect=db_error,
        ):
            with pytest.raises(psycopg2.OperationalError):
                writer.insert_running(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    partition_strategy=STRATEGY,
                )

    def test_audit_update_completed_raises_when_postgres_down(self, writer):
        """
        AuditWriter.update_completed() must propagate OperationalError
        when pg_connection raises during the UPDATE call.
        """
        db_error = psycopg2.OperationalError("SSL connection has been closed unexpectedly")

        with patch(
            "bronze.observer.audit_writer.pg_connection",
            side_effect=db_error,
        ):
            with pytest.raises(psycopg2.OperationalError):
                writer.update_completed(
                    audit_id=999,
                    trace_id=uuid.uuid4(),
                    statement_id=STMT_ID,
                    status="SUCCEEDED",
                    row_count=1_000,
                    duration_ms=1_000,
                )

    def test_close_pool_does_not_raise_after_connection_failure(self, aggregator):
        """
        close_pool() must be safe to call even after a broken connection state.
        This is called during cleanup — if it raises, it could hide the
        original error in an exception chain.
        """
        db_error = psycopg2.OperationalError("connection refused")

        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=db_error,
        ):
            try:
                aggregator.record_ingestion(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    success=True,
                    row_count=1_000,
                    duration_ms=1_000,
                    metric_date=TEST_DATE,
                )
            except psycopg2.OperationalError:
                pass

        # Must not raise
        close_pool()


# ===========================================================================
# Case H — Audit / metrics write failure (constraint violation, timeout)
# ===========================================================================

class TestCaseH_WriteFailures:
    """
    Simulates specific write failures that the SUT might encounter:
      - Database error mid-write (InterfaceError, ProgrammingError)
      - Connection timeout during write
      - Verify: exception always propagates — no infinite retry loops
      - Verify: no partial/phantom rows left when INSERT fails
    """

    @pytest.fixture
    def writer(self):
        return AuditWriter()

    @pytest.fixture
    def aggregator(self):
        return MetricsAggregator()

    def test_metrics_write_raises_on_programming_error(self, aggregator):
        """
        A ProgrammingError (e.g. table not found, syntax error in generated SQL)
        must propagate out of record_ingestion().
        """
        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=psycopg2.ProgrammingError("relation does not exist"),
        ):
            with pytest.raises(psycopg2.ProgrammingError):
                aggregator.record_ingestion(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    success=True,
                    row_count=500,
                    duration_ms=500,
                    metric_date=TEST_DATE,
                )

    def test_audit_insert_raises_on_interface_error(self, writer):
        """
        An InterfaceError (e.g. connection dropped mid-write) must propagate
        out of insert_running().
        """
        with patch(
            "bronze.observer.audit_writer.pg_connection",
            side_effect=psycopg2.InterfaceError("cursor already closed"),
        ):
            with pytest.raises(psycopg2.InterfaceError):
                writer.insert_running(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    partition_strategy=STRATEGY,
                )

    def test_failed_insert_running_leaves_no_partial_row(self, writer):
        """
        If insert_running() raises, there must be zero audit rows in the table.
        A partial row (RUNNING record with no corresponding update) left behind
        would make it impossible to distinguish a failed run from a stuck run.
        """
        with patch(
            "bronze.observer.audit_writer.pg_connection",
            side_effect=psycopg2.OperationalError("timeout"),
        ):
            with pytest.raises(psycopg2.OperationalError):
                writer.insert_running(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    partition_strategy=STRATEGY,
                )

        # Real DB check — no row was committed
        assert _count_audit_rows() == 0, (
            "A failed insert_running() must not leave a partial row in the audit table."
        )

    def test_write_failure_does_not_loop_indefinitely(self, aggregator):
        """
        The SUT must not retry the write in a loop when the DB is unavailable.
        Verify by counting how many times pg_connection was called — it must
        be called exactly once per record_ingestion() invocation.
        """
        call_count = 0

        def _erroring_context(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise psycopg2.OperationalError("connection refused")

        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=_erroring_context,
        ):
            with pytest.raises(psycopg2.OperationalError):
                aggregator.record_ingestion(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    success=True,
                    row_count=1_000,
                    duration_ms=1_000,
                    metric_date=TEST_DATE,
                )

        assert call_count == 1, (
            f"pg_connection was called {call_count} times. "
            "record_ingestion() must attempt the write exactly once — "
            "retry logic belongs in the orchestrator, not the SUT."
        )

    def test_audit_write_failure_does_not_loop_indefinitely(self, writer):
        """
        AuditWriter.insert_running() must attempt the write exactly once.
        """
        call_count = 0

        def _erroring_context(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise psycopg2.OperationalError("connection refused")

        with patch(
            "bronze.observer.audit_writer.pg_connection",
            side_effect=_erroring_context,
        ):
            with pytest.raises(psycopg2.OperationalError):
                writer.insert_running(
                    trace_id=uuid.uuid4(),
                    dataset_name=DATASET,
                    partition_strategy=STRATEGY,
                )

        assert call_count == 1, (
            f"pg_connection was called {call_count} times. "
            "insert_running() must not retry internally."
        )


# ===========================================================================
# Case I — Partial failure (Databricks succeeded, metrics write failed)
# ===========================================================================

class TestCaseI_PartialFailure:
    """
    Scenario:
      1. AuditWriter.insert_running()  → real DB write succeeds  (RUNNING row)
      2. Databricks SQL executes       → SUCCEEDED
      3. AuditWriter.update_completed()→ real DB write succeeds  (SUCCESS row)
      4. MetricsAggregator.record_ingestion() → RAISES (metrics DB write failed)

    Expected:
      - Audit row exists and shows SUCCESS  (Databricks did succeed)
      - Metrics table has NO row            (write never committed)
      - Exception propagates from record_ingestion() — no silent success signal
      - trace_id in audit row ties the two layers together
    """

    @pytest.fixture
    def writer(self):
        return AuditWriter()

    @pytest.fixture
    def aggregator(self):
        return MetricsAggregator()

    def test_audit_row_exists_with_success_when_metrics_fails(
        self, writer, aggregator
    ):
        """
        Audit row must reflect the true Databricks outcome (SUCCESS)
        even when the metrics write subsequently fails.
        """
        trace_id = uuid.uuid4()

        # Step 1 & 3: real audit writes
        audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=STMT_ID,
            status="SUCCEEDED",
            row_count=5_000,
            duration_ms=2_000,
        )

        # Step 4: metrics write fails
        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=psycopg2.OperationalError("metrics DB unreachable"),
        ):
            with pytest.raises(psycopg2.OperationalError):
                aggregator.record_ingestion(
                    trace_id=trace_id,
                    dataset_name=DATASET,
                    success=True,
                    row_count=5_000,
                    duration_ms=2_000,
                    metric_date=TEST_DATE,
                )

        # Audit row must show SUCCESS
        audit_row = _fetch_audit_by_trace(trace_id)
        assert audit_row is not None, "Audit row must exist"
        assert audit_row["status"] == "SUCCESS"

    def test_metrics_table_has_no_row_when_write_fails(
        self, writer, aggregator
    ):
        """
        When the metrics write raises, no row must be committed to
        bronze_ingestion_metrics. An absent row is correct — a partial
        row with wrong counters would be worse than nothing.
        """
        trace_id = uuid.uuid4()

        audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=STMT_ID,
            status="SUCCEEDED",
            row_count=5_000,
            duration_ms=2_000,
        )

        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=psycopg2.OperationalError("metrics DB unreachable"),
        ):
            with pytest.raises(psycopg2.OperationalError):
                aggregator.record_ingestion(
                    trace_id=trace_id,
                    dataset_name=DATASET,
                    success=True,
                    row_count=5_000,
                    duration_ms=2_000,
                    metric_date=TEST_DATE,
                )

        assert _count_metrics_rows(DATASET, TEST_DATE) == 0

    def test_trace_id_ties_audit_to_failed_metrics_write(
        self, writer, aggregator
    ):
        """
        Even in a partial failure the trace_id in the audit row must equal
        the trace_id that was passed to record_ingestion().
        This is what lets an operator correlate the audit record with the
        metrics failure log entry.
        """
        trace_id = uuid.uuid4()

        audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=STMT_ID,
            status="SUCCEEDED",
            row_count=5_000,
            duration_ms=2_000,
        )

        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=psycopg2.OperationalError("metrics DB unreachable"),
        ):
            with pytest.raises(psycopg2.OperationalError):
                aggregator.record_ingestion(
                    trace_id=trace_id,
                    dataset_name=DATASET,
                    success=True,
                    row_count=5_000,
                    duration_ms=2_000,
                    metric_date=TEST_DATE,
                )

        audit_row = _fetch_audit_by_trace(trace_id)
        assert str(audit_row["trace_id"]) == str(trace_id)

    def test_partial_failure_exception_propagates(self, writer, aggregator):
        """
        The metrics OperationalError must not be swallowed.
        The caller (orchestrator) must receive it so it can log/alert
        the metrics gap without confusing it with a Databricks failure.
        """
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)
        writer.update_completed(
            audit_id=audit_id, trace_id=trace_id,
            statement_id=STMT_ID, status="SUCCEEDED",
            row_count=1_000, duration_ms=1_000,
        )

        with patch(
            "bronze.observer.metrics_aggregator.pg_connection",
            side_effect=psycopg2.OperationalError("down"),
        ):
            raised = False
            try:
                aggregator.record_ingestion(
                    trace_id=trace_id, dataset_name=DATASET,
                    success=True, row_count=1_000, duration_ms=1_000,
                    metric_date=TEST_DATE,
                )
            except psycopg2.OperationalError:
                raised = True

        assert raised, (
            "MetricsAggregator must not swallow the OperationalError. "
            "The orchestrator needs to see it to handle the partial failure."
        )


# ===========================================================================
# Edge Cases EC1 – EC6
# ===========================================================================

class TestEdgeCases:
    """
    Edge cases that real pipelines encounter. Each is a standalone integration
    test against the real Postgres instance.
    """

    @pytest.fixture
    def aggregator(self):
        return MetricsAggregator()

    @pytest.fixture
    def writer(self):
        return AuditWriter()

    # ------------------------------------------------------------------
    # EC1 — Duplicate ingestion execution (sequential)
    # ------------------------------------------------------------------

    def test_ec1_duplicate_sequential_ingestion_increments_twice(self, aggregator):
        """
        The same dataset triggered twice in sequence (not concurrently).
        ingestion_success_total must be 2, no constraint violation.
        The concurrent version of this test lives in test_concurrency.py.
        """
        for _ in range(2):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(),
                dataset_name=DATASET,
                success=True,
                row_count=1_000,
                duration_ms=1_000,
                metric_date=TEST_DATE,
            )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["ingestion_success_total"] == 2
        assert _count_metrics_rows(DATASET, TEST_DATE) == 1  # still one row

    # ------------------------------------------------------------------
    # EC2 — Midnight boundary
    # ------------------------------------------------------------------

    def test_ec2_runs_on_different_sides_of_midnight_go_to_correct_dates(
        self, aggregator
    ):
        """
        Run at 23:59:59 → DATE_A (2024-06-14)
        Run at 00:00:01 → DATE_B (2024-06-15)

        Using metric_date override to simulate the clock crossing midnight.
        Each date must have exactly one independent row.
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=DATASET,
            success=True, row_count=1_000, duration_ms=500,
            metric_date=DATE_A,    # 23:59:59 side
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=DATASET,
            success=True, row_count=2_000, duration_ms=800,
            metric_date=DATE_B,    # 00:00:01 side
        )

        row_a = _fetch_metrics(DATASET, DATE_A)
        row_b = _fetch_metrics(DATASET, DATE_B)

        # Each date gets its own independent row
        assert row_a is not None, "DATE_A row must exist"
        assert row_b is not None, "DATE_B row must exist"

        # Rows must not overlap or accumulate across dates
        assert row_a["ingestion_rows_total"] == 1_000
        assert row_b["ingestion_rows_total"] == 2_000

    def test_ec2_no_timezone_bleed_between_dates(self, aggregator):
        """
        The two midnight-boundary runs must produce exactly two distinct rows.
        One row means the dates were treated as equal (timezone bug).
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=DATASET,
            success=True, row_count=1_000, duration_ms=500,
            metric_date=DATE_A,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=DATASET,
            success=True, row_count=2_000, duration_ms=800,
            metric_date=DATE_B,
        )

        assert _count_metrics_rows(DATASET, DATE_A) == 1
        assert _count_metrics_rows(DATASET, DATE_B) == 1

    # ------------------------------------------------------------------
    # EC3 — Very large row count
    # ------------------------------------------------------------------

    def test_ec3_very_large_row_count_stored_without_overflow(self, aggregator):
        """
        row_count = 10_000_000 must be stored exactly in BIGINT column.
        BIGINT supports values up to 9_223_372_036_854_775_807 — no overflow.
        """
        large_row_count = 10_000_000

        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=DATASET,
            success=True, row_count=large_row_count, duration_ms=5_000,
            metric_date=TEST_DATE,
        )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["ingestion_rows_total"] == large_row_count

    def test_ec3_accumulated_large_row_counts_correct(self, aggregator):
        """
        Two runs of 10M rows each must accumulate to exactly 20M.
        Catches integer overflow in the SUM path.
        """
        large_row_count = 10_000_000

        for _ in range(2):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(), dataset_name=DATASET,
                success=True, row_count=large_row_count, duration_ms=5_000,
                metric_date=TEST_DATE,
            )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["ingestion_rows_total"] == 2 * large_row_count

    # ------------------------------------------------------------------
    # EC4 — Very long duration (8 hours)
    # ------------------------------------------------------------------

    def test_ec4_long_duration_stored_without_truncation(self, aggregator):
        """
        duration_ms = 8 hours in milliseconds.
        Must be stored and retrieved without floating-point truncation.
        DOUBLE PRECISION has ~15 significant digits — 8 hours in ms = 28_800_000
        which is well within that range.
        """
        eight_hours_ms  = 8 * 60 * 60 * 1_000   # 28_800_000 ms
        expected_sec    = eight_hours_ms / 1000.0  # 28_800.0 seconds

        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=DATASET,
            success=True, row_count=50_000, duration_ms=eight_hours_ms,
            metric_date=TEST_DATE,
        )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(expected_sec)

    def test_ec4_long_durations_accumulate_correctly(self, aggregator):
        """
        Two 8-hour runs must accumulate to exactly 16 hours in seconds.
        """
        eight_hours_ms = 8 * 60 * 60 * 1_000
        expected_sec   = 2 * eight_hours_ms / 1000.0

        for _ in range(2):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(), dataset_name=DATASET,
                success=True, row_count=50_000, duration_ms=eight_hours_ms,
                metric_date=TEST_DATE,
            )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(expected_sec)

    # ------------------------------------------------------------------
    # EC5 — Schema evolution storm (100 runs)
    # ------------------------------------------------------------------

    def test_ec5_schema_evolution_storm_count_is_exact(self, aggregator):
        """
        100 consecutive runs all with schema_evolved=True.
        schema_evolution_count must equal exactly 100 — not 99, not 101.
        Any deviation means increments are being dropped or double-counted.
        """
        n_runs = 100

        for _ in range(n_runs):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(), dataset_name=DATASET,
                success=True, row_count=100, duration_ms=100,
                schema_evolved=True, metric_date=TEST_DATE,
            )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["schema_evolution_count"] == n_runs

    def test_ec5_other_counters_unaffected_by_evolution_storm(self, aggregator):
        """
        During the 100-run evolution storm, success and row counters must
        accumulate correctly alongside schema_evolution_count.
        """
        n_runs    = 100
        rows_each = 100

        for _ in range(n_runs):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(), dataset_name=DATASET,
                success=True, row_count=rows_each, duration_ms=100,
                schema_evolved=True, metric_date=TEST_DATE,
            )

        row = _fetch_metrics(DATASET, TEST_DATE)
        assert row["ingestion_success_total"] == n_runs
        assert row["ingestion_rows_total"]    == n_runs * rows_each

    # ------------------------------------------------------------------
    # EC6 — Dataset rename
    # ------------------------------------------------------------------

    def test_ec6_renamed_dataset_gets_fresh_row(self, aggregator):
        """
        If the dataset_name changes (e.g. "billing_v1" → "billing_v2"),
        the new name must produce an entirely new row.
        The old name's row must not be touched.
        """
        old_name = "billing_v1"
        new_name = "billing_v2"

        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=old_name,
            success=True, row_count=1_000, duration_ms=1_000,
            metric_date=TEST_DATE,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=new_name,
            success=True, row_count=2_000, duration_ms=2_000,
            metric_date=TEST_DATE,
        )

        old_row = _fetch_metrics(old_name, TEST_DATE)
        new_row = _fetch_metrics(new_name, TEST_DATE)

        assert old_row is not None, "Old dataset row must still exist"
        assert new_row is not None, "New dataset row must be created"

    def test_ec6_old_dataset_row_intact_after_rename(self, aggregator):
        """
        The old dataset's counters must be exactly what they were before
        the rename — writing to the new name must not overwrite them.
        """
        old_name = "billing_v1"
        new_name = "billing_v2"

        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=old_name,
            success=True, row_count=1_000, duration_ms=1_000,
            metric_date=TEST_DATE,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(), dataset_name=new_name,
            success=True, row_count=2_000, duration_ms=2_000,
            metric_date=TEST_DATE,
        )

        old_row = _fetch_metrics(old_name, TEST_DATE)
        new_row = _fetch_metrics(new_name, TEST_DATE)

        assert old_row["ingestion_rows_total"] == 1_000
        assert new_row["ingestion_rows_total"] == 2_000

        # Explicit: new name must NOT have overwritten old name's counters
        assert old_row["ingestion_rows_total"] != new_row["ingestion_rows_total"]


# ===========================================================================
# Case J — Trace Correlation
# ===========================================================================

class TestCaseJ_TraceCorrelation:
    """
    A single trace_id is generated at the start of each ingestion run.
    It must flow identically through every layer of the observability stack:

      1. Audit table   — trace_id column in bronze_ingestion_audit
      2. Metrics log   — trace_id field in the structured JSON log entry
                         emitted by MetricsAggregator
      3. BronzeLogger  — trace_id field in each of the three events:
                         bronze_sql_generated, bronze_sql_executed,
                         bronze_sql_failed

    If ANY layer uses a different trace_id — or generates a new one —
    correlation is broken and incident investigation becomes impossible.

    Log capture uses pytest's built-in caplog fixture, targeting the
    "bronze.observability" logger at DEBUG level to capture all events.
    """

    @pytest.fixture
    def aggregator(self):
        return MetricsAggregator()

    @pytest.fixture
    def writer(self):
        return AuditWriter()

    @pytest.fixture
    def bronze_logger(self):
        return BronzeLogger(dataset_name=DATASET)

    def _get_log_records(self, caplog) -> List[logging.LogRecord]:
        """Return all log records captured from bronze.observability."""
        return [r for r in caplog.records if r.name == "bronze.observability"]

    # ------------------------------------------------------------------
    # J.1 — Audit table carries the correct trace_id
    # ------------------------------------------------------------------

    def test_j1_audit_row_stores_exact_trace_id(self, writer, caplog):
        """
        The trace_id inserted by insert_running() must be retrievable from
        the audit table as the exact same UUID that was passed in.
        """
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.DEBUG, logger="bronze.observability"):
            audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)

        audit_row = _fetch_audit_by_trace(trace_id)
        assert audit_row is not None, "Audit row must exist"
        assert str(audit_row["trace_id"]) == str(trace_id)

    def test_j1_update_completed_does_not_change_trace_id(self, writer):
        """
        update_completed() must not overwrite or alter the trace_id stored
        during insert_running().
        """
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)
        writer.update_completed(
            audit_id=audit_id, trace_id=trace_id,
            statement_id=STMT_ID, status="SUCCEEDED",
            row_count=1_000, duration_ms=1_000,
        )

        audit_row = _fetch_audit_by_trace(trace_id)
        assert str(audit_row["trace_id"]) == str(trace_id)

    # ------------------------------------------------------------------
    # J.2 — MetricsAggregator log carries the correct trace_id
    # ------------------------------------------------------------------

    def test_j2_metrics_log_entry_contains_trace_id(self, aggregator, caplog):
        """
        record_ingestion() must emit a log entry that includes the exact
        trace_id passed into the call.
        """
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.INFO, logger="bronze.observability"):
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=DATASET,
                success=True, row_count=1_000, duration_ms=1_000,
                metric_date=TEST_DATE,
            )

        records = self._get_log_records(caplog)
        assert records, "At least one log entry must be emitted by record_ingestion()"

        # Find the metrics_updated event
        metrics_records = [r for r in records if getattr(r, "event", "") == "bronze_metrics_updated"]
        assert metrics_records, "A 'bronze_metrics_updated' log entry must be emitted"

        logged_trace_id = getattr(metrics_records[0], "trace_id", None)
        assert logged_trace_id == str(trace_id), (
            f"Metrics log trace_id '{logged_trace_id}' != "
            f"expected '{trace_id}'"
        )

    def test_j2_metrics_log_contains_dataset_name(self, aggregator, caplog):
        """
        The metrics log entry must also carry dataset_name so operators can
        filter logs by dataset without needing to decode the trace_id.
        """
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.INFO, logger="bronze.observability"):
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=DATASET,
                success=True, row_count=1_000, duration_ms=1_000,
                metric_date=TEST_DATE,
            )

        records = self._get_log_records(caplog)
        metrics_records = [r for r in records if getattr(r, "event", "") == "bronze_metrics_updated"]
        assert metrics_records

        logged_dataset = getattr(metrics_records[0], "dataset_name", None)
        assert logged_dataset == DATASET

    # ------------------------------------------------------------------
    # J.3 — BronzeLogger events all carry the same trace_id
    # ------------------------------------------------------------------

    def test_j3_log_sql_generated_contains_trace_id(self, bronze_logger, caplog):
        """log_sql_generated() must emit a record with the exact trace_id."""
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.INFO, logger="bronze.observability"):
            bronze_logger.log_sql_generated(
                trace_id=trace_id,
                partition_strategy=STRATEGY,
                sql_type="CREATE_TABLE",
            )

        records = self._get_log_records(caplog)
        gen_records = [r for r in records if getattr(r, "event", "") == "bronze_sql_generated"]
        assert gen_records, "bronze_sql_generated event must be logged"

        assert getattr(gen_records[0], "trace_id") == str(trace_id)

    def test_j3_log_sql_executed_contains_trace_id(self, bronze_logger, caplog):
        """log_sql_executed() must emit a record with the exact trace_id."""
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.INFO, logger="bronze.observability"):
            bronze_logger.log_sql_executed(
                trace_id=trace_id,
                statement_id=STMT_ID,
                status="SUCCEEDED",
                row_count=1_000,
                duration_ms=1_000,
            )

        records = self._get_log_records(caplog)
        exec_records = [r for r in records if getattr(r, "event", "") == "bronze_sql_executed"]
        assert exec_records, "bronze_sql_executed event must be logged"

        assert getattr(exec_records[0], "trace_id") == str(trace_id)

    def test_j3_log_sql_failed_contains_trace_id(self, bronze_logger, caplog):
        """log_sql_failed() must emit a record with the exact trace_id."""
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.ERROR, logger="bronze.observability"):
            bronze_logger.log_sql_failed(
                trace_id=trace_id,
                error_message="Databricks execution failed",
            )

        records = self._get_log_records(caplog)
        fail_records = [r for r in records if getattr(r, "event", "") == "bronze_sql_failed"]
        assert fail_records, "bronze_sql_failed event must be logged"

        assert getattr(fail_records[0], "trace_id") == str(trace_id)

    def test_j3_all_three_events_share_same_trace_id(self, bronze_logger, caplog):
        """
        All three BronzeLogger events emitted for ONE run must carry the
        same trace_id.  If any event carries a different or missing trace_id,
        log correlation is broken.
        """
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.DEBUG, logger="bronze.observability"):
            bronze_logger.log_sql_generated(
                trace_id=trace_id, partition_strategy=STRATEGY, sql_type="INGEST",
            )
            bronze_logger.log_sql_executed(
                trace_id=trace_id, statement_id=STMT_ID,
                status="SUCCEEDED", row_count=1_000, duration_ms=1_000,
            )
            # log_sql_failed would not normally fire alongside executed, but
            # we test the field value in isolation — not the calling order.
            bronze_logger.log_sql_failed(
                trace_id=trace_id, error_message="forced for test",
            )

        records = self._get_log_records(caplog)
        assert len(records) == 3, (
            f"Expected 3 log records, got {len(records)}: "
            f"{[getattr(r, 'event', r.message) for r in records]}"
        )

        trace_ids_in_logs = {getattr(r, "trace_id", None) for r in records}
        assert trace_ids_in_logs == {str(trace_id)}, (
            f"Not all log events carry the same trace_id. Found: {trace_ids_in_logs}"
        )

    # ------------------------------------------------------------------
    # J.4 — One run, one trace_id across audit + metrics + logs
    # ------------------------------------------------------------------

    def test_j4_single_run_trace_id_consistent_across_all_layers(
        self, writer, aggregator, bronze_logger, caplog
    ):
        """
        End-to-end trace correlation test.
        One trace_id is generated and passed through:
          - AuditWriter (insert_running + update_completed)
          - BronzeLogger (log_sql_generated + log_sql_executed)
          - MetricsAggregator (record_ingestion)

        All layers must carry the exact same trace_id string.
        Any discrepancy means the trace was broken somewhere in the pipeline.
        """
        trace_id = uuid.uuid4()

        with caplog.at_level(logging.DEBUG, logger="bronze.observability"):
            # 1. Audit: RUNNING
            audit_id = writer.insert_running(trace_id, DATASET, STRATEGY)

            # 2. Log: SQL generated
            bronze_logger.log_sql_generated(
                trace_id=trace_id, partition_strategy=STRATEGY, sql_type="INGEST",
            )

            # 3. Log: SQL executed
            bronze_logger.log_sql_executed(
                trace_id=trace_id, statement_id=STMT_ID,
                status="SUCCEEDED", row_count=5_000, duration_ms=2_000,
            )

            # 4. Audit: SUCCESS
            writer.update_completed(
                audit_id=audit_id, trace_id=trace_id,
                statement_id=STMT_ID, status="SUCCEEDED",
                row_count=5_000, duration_ms=2_000,
            )

            # 5. Metrics: record
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=DATASET,
                success=True, row_count=5_000, duration_ms=2_000,
                metric_date=TEST_DATE,
            )

        # ── Audit layer ──
        audit_row = _fetch_audit_by_trace(trace_id)
        assert audit_row is not None
        assert str(audit_row["trace_id"]) == str(trace_id)

        # ── Log layer ──
        records = self._get_log_records(caplog)
        log_trace_ids = {getattr(r, "trace_id", None) for r in records}

        # Every log record must carry the same trace_id
        # (exclude None — some debug records from other paths may lack it)
        non_null_trace_ids = {t for t in log_trace_ids if t is not None}
        assert non_null_trace_ids == {str(trace_id)}, (
            f"Multiple trace_ids found in logs: {non_null_trace_ids}. "
            "Only one trace_id must exist for a single run."
        )

    def test_j4_two_runs_produce_two_distinct_trace_ids_in_logs(
        self, writer, aggregator, caplog
    ):
        """
        Two separate runs must each emit their own distinct trace_id.
        If both runs share one trace_id, the logs from the second run
        cannot be distinguished from the first.
        """
        trace_id_1 = uuid.uuid4()
        trace_id_2 = uuid.uuid4()

        with caplog.at_level(logging.INFO, logger="bronze.observability"):
            # Run 1
            audit_id_1 = writer.insert_running(trace_id_1, DATASET, STRATEGY)
            writer.update_completed(
                audit_id=audit_id_1, trace_id=trace_id_1,
                statement_id=STMT_ID, status="SUCCEEDED",
                row_count=1_000, duration_ms=1_000,
            )
            aggregator.record_ingestion(
                trace_id=trace_id_1, dataset_name=DATASET,
                success=True, row_count=1_000, duration_ms=1_000,
                metric_date=TEST_DATE,
            )

            # Run 2
            audit_id_2 = writer.insert_running(trace_id_2, DATASET, STRATEGY)
            writer.update_completed(
                audit_id=audit_id_2, trace_id=trace_id_2,
                statement_id=STMT_ID, status="SUCCEEDED",
                row_count=2_000, duration_ms=2_000,
            )
            aggregator.record_ingestion(
                trace_id=trace_id_2, dataset_name=DATASET,
                success=True, row_count=2_000, duration_ms=2_000,
                metric_date=TEST_DATE,
            )

        records  = self._get_log_records(caplog)
        all_tids = {getattr(r, "trace_id", None) for r in records if hasattr(r, "trace_id")}

        assert str(trace_id_1) in all_tids, "trace_id_1 must appear in logs"
        assert str(trace_id_2) in all_tids, "trace_id_2 must appear in logs"
        assert str(trace_id_1) != str(trace_id_2), "Two runs must have distinct trace_ids"