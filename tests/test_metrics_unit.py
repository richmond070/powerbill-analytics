"""
test_metrics_unit.py
====================
Unit tests for the Bronze observability layer.
No real Postgres database required — all DB calls are mocked.

Maps directly to observability_testscript.md:

  Section 3  — Unit Tests (Fast + Deterministic)
    Test 1   — Metric delta logic
    Test 2   — Date handling
    Test 3   — Logging structure
  Section 8  — Edge Cases
    Edge 3   — Very large row counts   (no integer overflow)
    Edge 4   — Long duration           (no float truncation)

Run:
    pytest tests/test_metrics_unit.py -v
"""

import logging
import uuid
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from bronze.ingestion.observer.metrics_aggregator import MetricsAggregator
from bronze.ingestion.observer.bronze_logger import BronzeLogger, configure_logging


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_logging():
    """Ensure JSON logging is active for every test in this module."""
    configure_logging()


@pytest.fixture
def trace_id():
    return uuid.uuid4()


@pytest.fixture
def dataset_name():
    return "billing_payments"


@pytest.fixture
def aggregator():
    return MetricsAggregator(config_path="databricks/databricks.cfg")


# ---------------------------------------------------------------------------
# Helper — intercept pg_connection so no real DB is needed
# ---------------------------------------------------------------------------

def _mock_pg(aggregator):
    """
    Patches pg_connection inside metrics_aggregator.
    Returns (patcher, mock_cursor).

    The cursor's execute() captures exactly what would be sent to Postgres,
    letting tests assert on the SQL parameter tuple without a live connection.
    """
    mock_cursor = MagicMock()
    mock_conn   = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__  = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
    patcher = patch(
        "bronze.observer.metrics_aggregator.pg_connection",
        return_value=mock_conn,
    )
    return patcher, mock_cursor


def _upsert_args(mock_cursor):
    """
    Return the positional parameter tuple from cursor.execute().

    Matches the upsert in metrics_aggregator.py:
      index 0  dataset_name
      index 1  metric_date
      index 2  ingestion_success_total   (delta)
      index 3  ingestion_failures_total  (delta)
      index 4  ingestion_rows_total      (delta)
      index 5  ingestion_duration_seconds (delta)
      index 6  schema_evolution_count    (delta)
    """
    return mock_cursor.execute.call_args[0][1]


# ===========================================================================
# TEST 1 — Metric Delta Logic
# (observability_testscript.md §3 Test 1)
# ===========================================================================

class TestMetricDeltaLogic:
    """
    Validates that MetricsAggregator.record_ingestion() assembles the
    correct delta values before passing them to the Postgres upsert.
    """

    def test_success_increments_success_counter(self, aggregator, trace_id, dataset_name):
        """success=True  →  success_delta=1, failure_delta=0."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=1000, duration_ms=5000,
            )
        a = _upsert_args(cur)
        assert a[2] == 1, "success counter delta should be 1"
        assert a[3] == 0, "failure counter delta should be 0"

    def test_failure_increments_failure_counter(self, aggregator, trace_id, dataset_name):
        """success=False  →  success_delta=0, failure_delta=1."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=False, row_count=None, duration_ms=2000,
            )
        a = _upsert_args(cur)
        assert a[2] == 0, "success counter delta should be 0"
        assert a[3] == 1, "failure counter delta should be 1"

    def test_none_row_count_treated_as_zero(self, aggregator, trace_id, dataset_name):
        """None row_count must not raise and must be stored as 0."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=False, row_count=None, duration_ms=1000,
            )
        assert _upsert_args(cur)[4] == 0

    def test_schema_evolved_true_increments_schema_counter(self, aggregator, trace_id, dataset_name):
        """schema_evolved=True  →  schema_evolution_count delta = 1."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=500, duration_ms=3000,
                schema_evolved=True,
            )
        assert _upsert_args(cur)[6] == 1

    def test_schema_evolved_false_keeps_delta_zero(self, aggregator, trace_id, dataset_name):
        """schema_evolved=False (default)  →  schema_evolution_count delta = 0."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=200, duration_ms=1000,
            )
        assert _upsert_args(cur)[6] == 0

    def test_zero_duration_ms_is_accepted(self, aggregator, trace_id, dataset_name):
        """duration_ms=0 is valid — must store 0.0 seconds without raising."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=100, duration_ms=0,
            )
        assert _upsert_args(cur)[5] == 0.0

    def test_duration_ms_converts_to_seconds_correctly(self, aggregator, trace_id, dataset_name):
        """5 000 ms  →  5.0 seconds exactly."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=100, duration_ms=5000,
            )
        assert _upsert_args(cur)[5] == pytest.approx(5.0)

    def test_row_count_passes_through_unchanged(self, aggregator, trace_id, dataset_name):
        """An explicit row_count must reach the upsert unchanged."""
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=42_000, duration_ms=1000,
            )
        assert _upsert_args(cur)[4] == 42_000


# ===========================================================================
# TEST 2 — Date Handling
# (observability_testscript.md §3 Test 2)
# ===========================================================================

class TestDateHandling:
    """
    Validates UTC date assignment and the metric_date override mechanism.
    Midnight boundary tests cover Edge Case 2 from §8.
    """

    def test_metric_date_override_is_used(self, aggregator, trace_id, dataset_name):
        """Explicit metric_date must reach Postgres — today's date must NOT be used."""
        override = date(2025, 6, 15)
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=100, duration_ms=1000,
                metric_date=override,
            )
        assert _upsert_args(cur)[1] == override

    def test_utc_date_used_by_default(self, aggregator, trace_id, dataset_name):
        """Without an override the date must be today in UTC — not local time."""
        fixed_utc = datetime(2025, 12, 31, 23, 30, 0, tzinfo=timezone.utc)
        expected  = date(2025, 12, 31)
        p, cur = _mock_pg(aggregator)
        with p, patch(
            "bronze.observer.metrics_aggregator.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = fixed_utc
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=10, duration_ms=500,
            )
        assert _upsert_args(cur)[1] == expected

    def test_midnight_boundary_23_59_59(self, aggregator, trace_id, dataset_name):
        """23:59:59 UTC on Mar 1  →  metric_date = Mar 1  (not Mar 2)."""
        p, cur = _mock_pg(aggregator)
        with p, patch(
            "bronze.observer.metrics_aggregator.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 1, 23, 59, 59, tzinfo=timezone.utc)
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=10, duration_ms=100,
            )
        assert _upsert_args(cur)[1] == date(2025, 3, 1)

    def test_midnight_boundary_00_00_01(self, aggregator, trace_id, dataset_name):
        """00:00:01 UTC on Mar 2  →  metric_date = Mar 2  (not Mar 1)."""
        p, cur = _mock_pg(aggregator)
        with p, patch(
            "bronze.observer.metrics_aggregator.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 2, 0, 0, 1, tzinfo=timezone.utc)
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=10, duration_ms=100,
            )
        assert _upsert_args(cur)[1] == date(2025, 3, 2)


# ===========================================================================
# TEST 3 — Logging Structure
# (observability_testscript.md §3 Test 3)
# ===========================================================================

class TestLoggingStructure:
    """
    Validates that every BronzeLogger event carries all mandatory fields.
    Uses pytest caplog to capture LogRecord objects and inspect their extras.
    """

    def test_sql_generated_carries_trace_id_dataset_event(self, caplog, trace_id, dataset_name):
        caplog.set_level(logging.INFO, logger="bronze.observability")
        BronzeLogger(dataset_name).log_sql_generated(
            trace_id=trace_id, partition_strategy="time_based", sql_type="CREATE_TABLE",
        )
        r = caplog.records[-1]
        assert r.trace_id     == str(trace_id)
        assert r.dataset_name == dataset_name
        assert r.event        == "bronze_sql_generated"

    def test_sql_executed_carries_statement_id_status_row_count_duration(self, caplog, trace_id, dataset_name):
        caplog.set_level(logging.INFO, logger="bronze.observability")
        BronzeLogger(dataset_name).log_sql_executed(
            trace_id=trace_id, statement_id="stmt-abc",
            status="SUCCEEDED", row_count=5000, duration_ms=3200,
        )
        r = caplog.records[-1]
        assert r.trace_id     == str(trace_id)
        assert r.dataset_name == dataset_name
        assert r.statement_id == "stmt-abc"
        assert r.status       == "SUCCEEDED"
        assert r.row_count    == 5000
        assert r.duration_ms  == 3200

    def test_sql_failed_carries_error_message(self, caplog, trace_id, dataset_name):
        caplog.set_level(logging.ERROR, logger="bronze.observability")
        BronzeLogger(dataset_name).log_sql_failed(
            trace_id=trace_id, error_message="Connection refused",
        )
        r = caplog.records[-1]
        assert r.trace_id      == str(trace_id)
        assert r.dataset_name  == dataset_name
        assert r.error_message == "Connection refused"
        assert r.event         == "bronze_sql_failed"

    def test_trace_id_is_never_missing(self, caplog, dataset_name):
        """Missing trace_id is a red flag — it must always be present and non-empty."""
        caplog.set_level(logging.INFO, logger="bronze.observability")
        tid = uuid.uuid4()
        BronzeLogger(dataset_name).log_sql_generated(
            trace_id=tid, partition_strategy="none", sql_type="INGEST",
        )
        r = caplog.records[-1]
        assert hasattr(r, "trace_id"), "trace_id must always exist on the log record"
        assert r.trace_id, "trace_id must be a non-empty string"

    def test_each_logger_instance_tags_its_own_dataset_name(self, caplog):
        """Two separate BronzeLogger instances must not bleed dataset names."""
        caplog.set_level(logging.INFO, logger="bronze.observability")
        for ds in ["billing_payments", "grid_load", "retail_tariffs"]:
            BronzeLogger(ds).log_sql_generated(
                trace_id=uuid.uuid4(), partition_strategy="none", sql_type="INGEST",
            )
        emitted = [r.dataset_name for r in caplog.records[-3:]]
        assert emitted == ["billing_payments", "grid_load", "retail_tariffs"]


# ===========================================================================
# Edge Case 3 — Very Large Row Counts
# (observability_testscript.md §8 Edge Case 3)
# ===========================================================================

class TestVeryLargeRowCounts:

    def test_ten_million_rows_passes_through_without_overflow(self, aggregator, trace_id, dataset_name):
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=10_000_000, duration_ms=1000,
            )
        assert _upsert_args(cur)[4] == 10_000_000


# ===========================================================================
# Edge Case 4 — Long Duration
# (observability_testscript.md §8 Edge Case 4)
# ===========================================================================

class TestLongDuration:

    def test_eight_hour_duration_converts_without_truncation(self, aggregator, trace_id, dataset_name):
        """8 hours = 28_800_000 ms  →  28_800.0 seconds, no float precision loss."""
        eight_hours_ms = 8 * 60 * 60 * 1000
        p, cur = _mock_pg(aggregator)
        with p:
            aggregator.record_ingestion(
                trace_id=trace_id, dataset_name=dataset_name,
                success=True, row_count=100, duration_ms=eight_hours_ms,
            )
        assert _upsert_args(cur)[5] == pytest.approx(28_800.0, rel=1e-6)
