"""
Integration Tests — MetricsAggregator
======================================
File location : tests/test_metrics_integration.py
Covers        : Test Cases A–E from the observability test strategy doc

    Case A  — First ingestion of the day creates a fresh row whose
               counters equal exactly the values of that single run.
    Case B  — Three successive runs on the same day accumulate into the
               single (dataset_name, metric_date) row.
    Case C  — A failure run increments the failure counter, leaves
               ingestion_rows_total unchanged, but still accumulates duration.
    Case D  — A run with schema_evolved=True increments
               schema_evolution_count by 1 without disturbing other counters.
    Case E  — Two different datasets on the same day get separate rows
               with no cross-contamination between their counters.

Prerequisites
-------------
A real PostgreSQL instance must be reachable.  Connection details are read
from environment variables (the same priority order as db_pool._build_dsn):

    PG_HOST      localhost  (or the Docker service name in CI)
    PG_PORT      5432
    PG_DB        test_bronze  (or any dedicated test database)
    PG_USER      <user>
    PG_PASSWORD  <password>

The docker-compose.yaml at the project root already exposes a postgres
service — bring it up before running:

    docker-compose up -d postgres
    

Isolation strategy
------------------
- Session-scoped fixture  : creates the observability tables once per session.
- Function-scoped fixture : TRUNCATES both tables before every test so each
  case starts with a clean slate.
  (We cannot use a transaction rollback because MetricsAggregator calls
   pg_connection() which commits inside its own context manager — the test
   connection and the SUT connection are separate.)
- The db_pool singleton is torn down after every test so env-var changes in
  one test cannot bleed into the next.
"""

import os
import uuid
from datetime import date, timedelta
from typing import Dict, Optional

import psycopg2
import pytest

# ---------------------------------------------------------------------------
# The module under test
# ---------------------------------------------------------------------------
from bronze.observer.metrics_aggregator import MetricsAggregator
from bronze.observer.db_pool import close_pool, get_pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DATE     = date(2024, 6, 15)   # Fixed date — tests are time-independent
DATASET_ALPHA = "billing_payments"
DATASET_BETA  = "grid_load"


# ===========================================================================
# Fixtures
# ===========================================================================

def _raw_conn() -> psycopg2.extensions.connection:
    """
    Open a direct psycopg2 connection using the same env-vars as db_pool.
    Used only for fixture setup/teardown and result-reading — never for SUT
    calls, so it never interferes with the pool under test.
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
    Session-scoped: create the observability tables once for the whole run.
    Uses observability_schema.ensure_observability_tables() — idempotent,
    so safe even if the tables already exist from a previous session.
    """
    from bronze.observer.observability_schema import ensure_observability_tables
    ensure_observability_tables()
    yield
    # Nothing to drop — we leave the schema intact so the next run is fast.


@pytest.fixture(autouse=True)
def clean_tables(pg_schema):
    """
    Function-scoped (autouse): TRUNCATE both tables before every test.
    Also resets the db_pool singleton so each test gets a fresh pool —
    important when tests manipulate env-vars or need connection isolation.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE bronze_ingestion_metrics RESTART IDENTITY CASCADE;")
            cur.execute("TRUNCATE TABLE bronze_ingestion_audit   RESTART IDENTITY CASCADE;")
        conn.commit()
    finally:
        conn.close()

    yield  # ← test runs here

    # Tear down the pool singleton after every test so the next test
    # recreates it cleanly (guards against env-var leakage between tests).
    close_pool()


@pytest.fixture
def aggregator() -> MetricsAggregator:
    """Return a MetricsAggregator wired to the test database via env-vars."""
    return MetricsAggregator()


# ===========================================================================
# Helpers
# ===========================================================================

def _fetch_metrics(dataset_name: str, metric_date: date) -> Optional[Dict]:
    """
    Read a single row from bronze_ingestion_metrics and return it as a dict.
    Returns None if no row exists for the (dataset_name, metric_date) pair.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ingestion_success_total,
                    ingestion_failures_total,
                    ingestion_rows_total,
                    ingestion_duration_seconds,
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


def _count_metric_rows(dataset_name: str, metric_date: date) -> int:
    """
    Return the number of rows in bronze_ingestion_metrics for the given
    (dataset_name, metric_date) pair. Should always be 0 or 1 thanks to PK.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM bronze_ingestion_metrics
                WHERE dataset_name = %s AND metric_date = %s;
                """,
                (dataset_name, metric_date),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


# ===========================================================================
# Case A — First ingestion of the day
# ===========================================================================

class TestCaseA_FirstIngestionOfDay:
    """
    The very first call for a (dataset, date) pair must INSERT a fresh row
    and the row's counters must equal exactly the values passed in that call.
    Nothing is accumulated from a previous run because there is no previous run.
    """

    def test_new_row_is_created(self, aggregator):
        """A new (dataset_name, metric_date) row is inserted on first call."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=3_000,
            metric_date=TEST_DATE,
        )
        assert _count_metric_rows(DATASET_ALPHA, TEST_DATE) == 1

    def test_success_counter_equals_one(self, aggregator):
        """ingestion_success_total == 1 after the first successful run."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=3_000,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_success_total"] == 1

    def test_failure_counter_is_zero_on_success(self, aggregator):
        """ingestion_failures_total == 0 when the first run succeeded."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=3_000,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_failures_total"] == 0

    def test_rows_total_equals_run_row_count(self, aggregator):
        """ingestion_rows_total == row_count passed into the call."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=3_000,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_rows_total"] == 5_000

    def test_duration_seconds_equals_ms_divided_by_1000(self, aggregator):
        """ingestion_duration_seconds == duration_ms / 1000."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=3_000,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(3.0)

    def test_schema_evolution_count_zero_when_not_evolved(self, aggregator):
        """schema_evolution_count == 0 when schema_evolved was not passed."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=3_000,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["schema_evolution_count"] == 0


# ===========================================================================
# Case B — Multiple runs on the same day accumulate
# ===========================================================================

class TestCaseB_MultipleRunsSameDay:
    """
    Three successive calls for the same (dataset, date) must accumulate
    into a single row — the upsert must add deltas, not overwrite.
    """

    def _run_three_times(self, aggregator):
        """
        Helper: execute three distinct ingestion runs, all on TEST_DATE.
        Returns the expected aggregated totals so tests can assert against them.
        """
        runs = [
            dict(success=True,  row_count=1_000, duration_ms=1_000),
            dict(success=True,  row_count=2_500, duration_ms=2_000),
            dict(success=True,  row_count=3_000, duration_ms=3_500),
        ]
        for run in runs:
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(),
                dataset_name=DATASET_ALPHA,
                metric_date=TEST_DATE,
                **run,
            )
        return runs

    def test_only_one_row_exists_after_three_runs(self, aggregator):
        """The upsert must not create duplicate rows."""
        self._run_three_times(aggregator)
        assert _count_metric_rows(DATASET_ALPHA, TEST_DATE) == 1

    def test_success_counter_accumulates(self, aggregator):
        """ingestion_success_total == 3 after three successful runs."""
        self._run_three_times(aggregator)
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_success_total"] == 3

    def test_rows_total_accumulates(self, aggregator):
        """ingestion_rows_total == sum of all row_count values."""
        runs = self._run_three_times(aggregator)
        expected = sum(r["row_count"] for r in runs)          # 6_500
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_rows_total"] == expected

    def test_duration_accumulates(self, aggregator):
        """ingestion_duration_seconds == sum of all duration_ms / 1000."""
        runs = self._run_three_times(aggregator)
        expected_sec = sum(r["duration_ms"] for r in runs) / 1000.0  # 6.5
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(expected_sec)

    def test_failure_counter_remains_zero_for_all_successes(self, aggregator):
        """ingestion_failures_total stays at 0 when every run succeeded."""
        self._run_three_times(aggregator)
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_failures_total"] == 0


# ===========================================================================
# Case C — Failure run behaviour
# ===========================================================================

class TestCaseC_FailureRun:
    """
    A failed run (success=False, row_count=None) must:
      - Increment ingestion_failures_total
      - Leave ingestion_rows_total unchanged (0 rows on failure)
      - Still accumulate duration_seconds (even failed runs have wall-clock time)
      - Not increment ingestion_success_total
    """

    def test_failure_counter_increments(self, aggregator):
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=False,
            row_count=None,
            duration_ms=800,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_failures_total"] == 1

    def test_success_counter_unchanged_on_failure(self, aggregator):
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=False,
            row_count=None,
            duration_ms=800,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_success_total"] == 0

    def test_rows_total_unchanged_on_failure(self, aggregator):
        """None row_count is treated as 0 — the total must not change."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=False,
            row_count=None,
            duration_ms=800,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_rows_total"] == 0

    def test_duration_still_accumulates_on_failure(self, aggregator):
        """
        Failed runs still consumed wall-clock time.
        Even a failure run must contribute its duration to the daily total.
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=False,
            row_count=None,
            duration_ms=800,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(0.8)

    def test_success_then_failure_both_counted(self, aggregator):
        """
        Run 1 succeeds, run 2 fails.
        Both counters must reflect their respective increments.
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=1_000,
            duration_ms=1_200,
            metric_date=TEST_DATE,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=False,
            row_count=None,
            duration_ms=400,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)

        assert row["ingestion_success_total"]  == 1
        assert row["ingestion_failures_total"] == 1
        assert row["ingestion_rows_total"]     == 1_000          # only from run 1
        assert row["ingestion_duration_seconds"] == pytest.approx(1.6)  # 1200+400 ms


# ===========================================================================
# Case D — Schema evolution run
# ===========================================================================

class TestCaseD_SchemaEvolution:
    """
    When schema_evolved=True the schema_evolution_count must increment by 1.
    All other counters must behave normally — schema evolution is additive,
    not a replacement for success/failure tracking.
    """

    def test_schema_evolution_count_increments_by_one(self, aggregator):
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=2_000,
            duration_ms=1_500,
            schema_evolved=True,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["schema_evolution_count"] == 1

    def test_success_counter_still_increments_alongside_evolution(self, aggregator):
        """schema_evolved=True must not suppress the success counter."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=2_000,
            duration_ms=1_500,
            schema_evolved=True,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_success_total"] == 1

    def test_rows_and_duration_normal_on_schema_evolution(self, aggregator):
        """Row counts and duration accumulate normally during schema evolution."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=2_000,
            duration_ms=1_500,
            schema_evolved=True,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_rows_total"]     == 2_000
        assert row["ingestion_duration_seconds"] == pytest.approx(1.5)

    def test_two_runs_only_one_evolved(self, aggregator):
        """
        Run 1: schema_evolved=False, Run 2: schema_evolved=True.
        schema_evolution_count must be exactly 1 (not 2).
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=1_000,
            duration_ms=1_000,
            schema_evolved=False,
            metric_date=TEST_DATE,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=1_000,
            duration_ms=1_000,
            schema_evolved=True,
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["schema_evolution_count"] == 1

    def test_no_schema_evolution_flag_defaults_to_zero(self, aggregator):
        """schema_evolved defaults to False — count must stay 0."""
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=1_000,
            duration_ms=500,
            # schema_evolved not passed — defaults to False
            metric_date=TEST_DATE,
        )
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["schema_evolution_count"] == 0


# ===========================================================================
# Case E — Multiple datasets on the same day
# ===========================================================================

class TestCaseE_MultipleDatasets:
    """
    Two datasets ingested on the same day must produce two separate rows
    with no cross-contamination: writing to DATASET_ALPHA must never affect
    the counters of DATASET_BETA, and vice versa.
    """

    def test_separate_rows_exist_for_each_dataset(self, aggregator):
        """One distinct row per (dataset_name, metric_date) pair."""
        for ds in (DATASET_ALPHA, DATASET_BETA):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(),
                dataset_name=ds,
                success=True,
                row_count=1_000,
                duration_ms=1_000,
                metric_date=TEST_DATE,
            )
        assert _count_metric_rows(DATASET_ALPHA, TEST_DATE) == 1
        assert _count_metric_rows(DATASET_BETA,  TEST_DATE) == 1

    def test_alpha_counters_not_affected_by_beta_run(self, aggregator):
        """
        DATASET_ALPHA has 1 run with 5 000 rows.
        DATASET_BETA  has 2 runs with 999 rows each.
        DATASET_ALPHA's ingestion_rows_total must still be 5 000.
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=2_000,
            metric_date=TEST_DATE,
        )
        for _ in range(2):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(),
                dataset_name=DATASET_BETA,
                success=True,
                row_count=999,
                duration_ms=1_000,
                metric_date=TEST_DATE,
            )
        row_alpha = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row_alpha["ingestion_rows_total"]  == 5_000
        assert row_alpha["ingestion_success_total"] == 1

    def test_beta_counters_not_affected_by_alpha_run(self, aggregator):
        """
        Mirror of the above — DATASET_BETA's counters must only reflect
        its own two runs, not DATASET_ALPHA's single run.
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=5_000,
            duration_ms=2_000,
            metric_date=TEST_DATE,
        )
        for _ in range(2):
            aggregator.record_ingestion(
                trace_id=uuid.uuid4(),
                dataset_name=DATASET_BETA,
                success=True,
                row_count=999,
                duration_ms=1_000,
                metric_date=TEST_DATE,
            )
        row_beta = _fetch_metrics(DATASET_BETA, TEST_DATE)
        assert row_beta["ingestion_rows_total"]    == 1_998  # 2 × 999
        assert row_beta["ingestion_success_total"] == 2

    def test_different_dates_for_same_dataset_are_separate_rows(self, aggregator):
        """
        The same dataset on two different calendar dates must produce two rows.
        This is a prerequisite for daily aggregation to work correctly.
        """
        date_today     = TEST_DATE
        date_yesterday = TEST_DATE - timedelta(days=1)

        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=1_000,
            duration_ms=1_000,
            metric_date=date_today,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=True,
            row_count=2_000,
            duration_ms=2_000,
            metric_date=date_yesterday,
        )

        row_today     = _fetch_metrics(DATASET_ALPHA, date_today)
        row_yesterday = _fetch_metrics(DATASET_ALPHA, date_yesterday)

        # Each date gets its own independent row
        assert row_today["ingestion_rows_total"]     == 1_000
        assert row_yesterday["ingestion_rows_total"] == 2_000

    def test_failure_on_one_dataset_does_not_contaminate_other(self, aggregator):
        """
        DATASET_ALPHA fails. DATASET_BETA succeeds.
        DATASET_BETA's failure counter must remain 0.
        """
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_ALPHA,
            success=False,
            row_count=None,
            duration_ms=500,
            metric_date=TEST_DATE,
        )
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=DATASET_BETA,
            success=True,
            row_count=3_000,
            duration_ms=2_000,
            metric_date=TEST_DATE,
        )
        row_beta = _fetch_metrics(DATASET_BETA, TEST_DATE)
        assert row_beta["ingestion_failures_total"] == 0