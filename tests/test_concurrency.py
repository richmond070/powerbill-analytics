"""
Concurrency Tests — MetricsAggregator & AuditWriter
=====================================================
File location : tests/test_concurrency.py
Covers        : Case F from the observability test strategy doc

    Case F — 10 concurrent threads writing metrics for the same dataset.
    Expected:
      - No duplicate rows in bronze_ingestion_metrics
      - All counters equal the exact sum of every thread's contribution
      - No deadlocks
      - No lost increments
      - No PoolError / connection exhaustion crashes

    Extended coverage beyond the base Case F:
      - Mixed concurrent writes (some success, some failure)
      - Concurrent audit INSERT + UPDATE (AuditWriter under load)
      - Multiple datasets written concurrently — no cross-contamination
      - Pool exhaustion safety (10 threads > default maxconn of 5)

Threading approach
------------------
Every test uses:
  1. threading.Barrier   — holds all threads at a start gate so they fire
                           simultaneously rather than sequentially.
  2. ThreadPoolExecutor  — manages the thread pool lifecycle.
  3. threading.Lock      — protects the shared results/errors list.
  4. futures.result()    — re-raises any exception from a worker thread so
                           pytest catches it as a test failure with a clear
                           traceback, not a silent pass.

Pool sizing
-----------
db_pool.get_pool() defaults to maxconn=5.  With 10 concurrent threads that
each hold a connection, 5 threads would be blocked waiting.
psycopg2's ThreadedConnectionPool raises PoolError (not a queue/wait) when
all connections are exhausted.

The `wide_pool` fixture tears down the singleton and recreates it with
maxconn=20 before every concurrency test, then restores it afterwards.
This reflects how production should be sized for concurrent pipelines.

Prerequisites
-------------
Same Postgres instance as the other integration tests.
Credentials via env vars or databricks/databricks.cfg [POSTGRES] section:

    PG_HOST=localhost
    PG_PORT=5432
    PG_DB=xxxxx_db
    PG_USER=postgres
    PG_PASSWORD=<password>

Run:
    docker-compose up -d postgres
    pytest tests/test_concurrency.py -v
"""

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Dict, List, Optional, Tuple

import psycopg2
import pytest

# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------
from bronze.observer.metrics_aggregator import MetricsAggregator
from bronze.observer.audit_writer import AuditWriter
from bronze.observer.db_pool import close_pool, get_pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DATE      = date(2024, 6, 15)
DATASET_ALPHA  = "billing_payments"
DATASET_BETA   = "grid_load"
THREAD_COUNT   = 10       # Case F specifies 10 concurrent threads
POOL_MAXCONN   = 20       # Must be >= THREAD_COUNT to avoid PoolError
ROWS_PER_RUN   = 1_000
DURATION_MS    = 2_000    # 2 seconds per run
STMT_ID        = "stmt-concurrent-test"
STRATEGY       = "time_based"


# ===========================================================================
# Infrastructure fixtures
# ===========================================================================

def _raw_conn() -> psycopg2.extensions.connection:
    """
    Direct psycopg2 connection for fixture setup and result reading.
    Never used for SUT calls.
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
    Session-scoped: ensure both observability tables exist before any test.
    """
    from bronze.observer.observability_schema import ensure_observability_tables
    ensure_observability_tables()
    yield


@pytest.fixture(autouse=True)
def clean_tables(pg_schema):
    """
    Function-scoped (autouse): TRUNCATE both tables before every test.
    MetricsAggregator and AuditWriter each commit inside their own
    pg_connection() calls, so TRUNCATE is the only reliable isolation strategy.
    """
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


@pytest.fixture(autouse=True)
def wide_pool(clean_tables):
    """
    Function-scoped (autouse): recreate the connection pool with maxconn=20
    so THREAD_COUNT=10 concurrent threads never exhaust it.

    Default maxconn is 5 — with 10 threads that would cause 5 threads to
    get PoolError rather than a connection.  Production pipelines should
    size the pool to match their concurrency level; this fixture simulates
    that correctly-configured state.

    Runs AFTER clean_tables (declared as its dependency) so the pool reset
    order is: TRUNCATE → close_pool() → reopen with wide pool.
    """
    close_pool()
    get_pool(maxconn=POOL_MAXCONN)
    yield
    close_pool()


# ===========================================================================
# Helpers — read results back from Postgres
# ===========================================================================

def _fetch_metrics(dataset_name: str, metric_date: date) -> Optional[Dict]:
    """Read one metrics row and return it as a dict, or None if absent."""
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


def _count_metrics_rows(dataset_name: str, metric_date: date) -> int:
    """Return number of metrics rows for (dataset_name, metric_date)."""
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


def _count_audit_rows(dataset_name: str) -> int:
    """Return total audit rows for a dataset."""
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


def _fetch_all_audit_ids(dataset_name: str) -> List[int]:
    """Return all audit row IDs for a dataset, sorted ascending."""
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM bronze_ingestion_audit "
                "WHERE dataset_name = %s ORDER BY id;",
                (dataset_name,),
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# ===========================================================================
# Thread worker factories
# ===========================================================================

def _make_metrics_worker(
    aggregator: MetricsAggregator,
    barrier: threading.Barrier,
    dataset_name: str,
    success: bool,
    row_count: Optional[int],
    duration_ms: int,
    schema_evolved: bool = False,
):
    """
    Return a callable that:
      1. Waits at the barrier until all threads are ready
      2. Calls record_ingestion() once
      3. Returns (thread_id, None) on success or raises on error

    Using a Barrier ensures all THREAD_COUNT threads start their Postgres
    write at the same instant rather than staggering sequentially.
    """
    def worker():
        barrier.wait()          # synchronise — fire all threads together
        aggregator.record_ingestion(
            trace_id=uuid.uuid4(),
            dataset_name=dataset_name,
            success=success,
            row_count=row_count,
            duration_ms=duration_ms,
            schema_evolved=schema_evolved,
            metric_date=TEST_DATE,
        )
    return worker


def _make_audit_worker(
    writer: AuditWriter,
    barrier: threading.Barrier,
    dataset_name: str,
):
    """
    Return a callable that:
      1. Waits at the barrier
      2. Calls insert_running() then update_completed()
      3. Returns the audit_id on success

    Both steps are inside the same thread so each thread owns the full
    INSERT → UPDATE lifecycle for its own run.
    """
    def worker() -> int:
        barrier.wait()          # synchronise all threads
        trace_id = uuid.uuid4()
        audit_id = writer.insert_running(
            trace_id=trace_id,
            dataset_name=dataset_name,
            partition_strategy=STRATEGY,
        )
        writer.update_completed(
            audit_id=audit_id,
            trace_id=trace_id,
            statement_id=STMT_ID,
            status="SUCCEEDED",
            row_count=ROWS_PER_RUN,
            duration_ms=DURATION_MS,
        )
        return audit_id
    return worker


def _run_concurrent(workers: list) -> Tuple[List, List[Exception]]:
    """
    Submit all workers to a ThreadPoolExecutor and collect results.

    Returns:
        (results, errors)
        results : list of return values from successful workers
        errors  : list of exceptions raised by failed workers

    Every exception is captured so the test can report ALL failures,
    not just the first one.
    """
    results = []
    errors  = []

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(w) for w in workers]
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                errors.append(exc)
            else:
                results.append(future.result())

    return results, errors


# ===========================================================================
# Case F — 10 concurrent threads, MetricsAggregator
# ===========================================================================

class TestCaseF_ConcurrentMetrics:
    """
    10 threads simultaneously call record_ingestion() for the same
    (dataset_name, metric_date) pair.

    The ON CONFLICT DO UPDATE upsert in MetricsAggregator is designed to
    handle this — Postgres serialises conflicting upserts at the row level.

    All counter assertions use exact arithmetic:
        expected = THREAD_COUNT * per_thread_value
    If any increment is lost, the assertion fails and the race condition
    is exposed.
    """

    @pytest.fixture
    def aggregator(self) -> MetricsAggregator:
        return MetricsAggregator()

    # ------------------------------------------------------------------
    # F.1 — All threads succeed
    # ------------------------------------------------------------------

    def test_no_threads_raise_exceptions(self, aggregator):
        """
        All 10 threads must complete without raising any exception.
        A PoolError here means the pool was too small (wide_pool fixture
        guards against this, so a failure here is a fixture misconfiguration).
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(THREAD_COUNT)
        ]

        _, errors = _run_concurrent(workers)

        assert errors == [], (
            f"{len(errors)} thread(s) raised exceptions:\n"
            + "\n".join(f"  [{i}] {type(e).__name__}: {e}" for i, e in enumerate(errors))
        )

    def test_exactly_one_metrics_row(self, aggregator):
        """
        The upsert must produce exactly ONE row for (dataset_name, metric_date)
        regardless of how many threads wrote concurrently.
        Any value other than 1 means duplicate rows were inserted.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(THREAD_COUNT)
        ]
        _run_concurrent(workers)

        assert _count_metrics_rows(DATASET_ALPHA, TEST_DATE) == 1

    def test_success_counter_equals_thread_count(self, aggregator):
        """
        Every thread reports success=True.
        ingestion_success_total must equal exactly THREAD_COUNT.
        Any value less than THREAD_COUNT means increments were lost.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(THREAD_COUNT)
        ]
        _run_concurrent(workers)

        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_success_total"] == THREAD_COUNT

    def test_rows_total_equals_thread_count_times_rows_per_run(self, aggregator):
        """
        Each thread contributes ROWS_PER_RUN rows.
        ingestion_rows_total must equal THREAD_COUNT * ROWS_PER_RUN exactly.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(THREAD_COUNT)
        ]
        _run_concurrent(workers)

        expected = THREAD_COUNT * ROWS_PER_RUN
        row      = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_rows_total"] == expected

    def test_duration_equals_thread_count_times_duration_per_run(self, aggregator):
        """
        Each thread contributes DURATION_MS / 1000 seconds.
        ingestion_duration_seconds must equal THREAD_COUNT * DURATION_MS / 1000.
        Uses pytest.approx to tolerate floating-point rounding.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(THREAD_COUNT)
        ]
        _run_concurrent(workers)

        expected = THREAD_COUNT * DURATION_MS / 1000.0
        row      = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(expected)

    def test_failure_counter_is_zero_when_all_threads_succeed(self, aggregator):
        """
        All threads report success — ingestion_failures_total must be 0.
        A non-zero value means the upsert applied a delta to the wrong counter.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(THREAD_COUNT)
        ]
        _run_concurrent(workers)

        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_failures_total"] == 0

    # ------------------------------------------------------------------
    # F.2 — Mixed success and failure threads
    # ------------------------------------------------------------------

    def test_mixed_success_and_failure_counters_are_correct(self, aggregator):
        """
        5 threads succeed, 5 threads fail.
        Both counters must equal exactly 5.

        This catches the specific bug where the upsert applies the wrong
        delta under concurrent writes of different types.
        """
        n_success = THREAD_COUNT // 2   # 5
        n_failure = THREAD_COUNT // 2   # 5

        barrier = threading.Barrier(THREAD_COUNT)

        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(n_success)
        ] + [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=False, row_count=None, duration_ms=DURATION_MS,
            )
            for _ in range(n_failure)
        ]

        _, errors = _run_concurrent(workers)

        assert errors == [], (
            f"{len(errors)} thread(s) raised: "
            + ", ".join(f"{type(e).__name__}: {e}" for e in errors)
        )

        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)

        assert row["ingestion_success_total"]  == n_success
        assert row["ingestion_failures_total"] == n_failure

    def test_mixed_rows_total_counts_only_successful_rows(self, aggregator):
        """
        Only success threads contribute row_count.
        Failure threads pass row_count=None (treated as 0).
        ingestion_rows_total must equal n_success * ROWS_PER_RUN.
        """
        n_success = THREAD_COUNT // 2
        n_failure = THREAD_COUNT // 2

        barrier = threading.Barrier(THREAD_COUNT)

        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(n_success)
        ] + [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=False, row_count=None, duration_ms=DURATION_MS,
            )
            for _ in range(n_failure)
        ]

        _run_concurrent(workers)

        expected_rows = n_success * ROWS_PER_RUN
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_rows_total"] == expected_rows

    def test_mixed_duration_accumulates_from_all_threads(self, aggregator):
        """
        Duration accumulates for BOTH success and failure threads.
        Even failed runs consume wall-clock time.
        Total = THREAD_COUNT * DURATION_MS / 1000
        """
        barrier = threading.Barrier(THREAD_COUNT)

        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=(i % 2 == 0),   # alternating success/failure
                row_count=ROWS_PER_RUN if (i % 2 == 0) else None,
                duration_ms=DURATION_MS,
            )
            for i in range(THREAD_COUNT)
        ]

        _run_concurrent(workers)

        expected = THREAD_COUNT * DURATION_MS / 1000.0
        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["ingestion_duration_seconds"] == pytest.approx(expected)

    # ------------------------------------------------------------------
    # F.3 — Schema evolution under concurrency
    # ------------------------------------------------------------------

    def test_schema_evolution_count_exact_under_concurrency(self, aggregator):
        """
        Half the threads set schema_evolved=True.
        schema_evolution_count must equal exactly n_evolved.
        Any value less than n_evolved means schema increments were lost.
        """
        n_evolved  = THREAD_COUNT // 2   # 5
        n_normal   = THREAD_COUNT - n_evolved

        barrier = threading.Barrier(THREAD_COUNT)

        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
                schema_evolved=True,
            )
            for _ in range(n_evolved)
        ] + [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
                schema_evolved=False,
            )
            for _ in range(n_normal)
        ]

        _run_concurrent(workers)

        row = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        assert row["schema_evolution_count"] == n_evolved

    # ------------------------------------------------------------------
    # F.4 — Multiple datasets written concurrently
    # ------------------------------------------------------------------

    def test_concurrent_writes_to_different_datasets_no_cross_contamination(
        self, aggregator
    ):
        """
        5 threads write to DATASET_ALPHA, 5 threads write to DATASET_BETA,
        all firing simultaneously.

        Each dataset's counters must reflect only its own threads' contributions.
        Any bleed-over means the upsert key is being applied incorrectly.
        """
        n_per_dataset = THREAD_COUNT // 2   # 5 threads each

        barrier = threading.Barrier(THREAD_COUNT)

        workers = [
            _make_metrics_worker(
                aggregator, barrier, DATASET_ALPHA,
                success=True, row_count=ROWS_PER_RUN, duration_ms=DURATION_MS,
            )
            for _ in range(n_per_dataset)
        ] + [
            _make_metrics_worker(
                aggregator, barrier, DATASET_BETA,
                success=True, row_count=ROWS_PER_RUN * 2, duration_ms=DURATION_MS * 2,
            )
            for _ in range(n_per_dataset)
        ]

        _, errors = _run_concurrent(workers)
        assert errors == [], f"Thread errors: {errors}"

        row_alpha = _fetch_metrics(DATASET_ALPHA, TEST_DATE)
        row_beta  = _fetch_metrics(DATASET_BETA,  TEST_DATE)

        # ALPHA: 5 threads × ROWS_PER_RUN
        assert row_alpha["ingestion_success_total"] == n_per_dataset
        assert row_alpha["ingestion_rows_total"]    == n_per_dataset * ROWS_PER_RUN

        # BETA: 5 threads × ROWS_PER_RUN * 2
        assert row_beta["ingestion_success_total"]  == n_per_dataset
        assert row_beta["ingestion_rows_total"]     == n_per_dataset * ROWS_PER_RUN * 2

        # Each dataset has exactly one row — no duplicates under concurrent load
        assert _count_metrics_rows(DATASET_ALPHA, TEST_DATE) == 1
        assert _count_metrics_rows(DATASET_BETA,  TEST_DATE) == 1


# ===========================================================================
# Case F — 10 concurrent threads, AuditWriter
# ===========================================================================

class TestCaseF_ConcurrentAudit:
    """
    10 threads each run the full audit lifecycle (INSERT RUNNING → UPDATE SUCCESS)
    simultaneously.

    Unlike MetricsAggregator (which upserts one row), AuditWriter inserts a
    NEW row per run — so 10 threads must produce exactly 10 rows with 10
    distinct audit_ids.  The SERIAL primary key guarantees uniqueness at the
    DB level; the test verifies the application layer never duplicates or
    loses a row.
    """

    @pytest.fixture
    def writer(self) -> AuditWriter:
        return AuditWriter()

    def test_no_threads_raise_exceptions(self, writer):
        """All 10 audit lifecycle threads must complete without any exception."""
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_audit_worker(writer, barrier, DATASET_ALPHA)
            for _ in range(THREAD_COUNT)
        ]

        _, errors = _run_concurrent(workers)

        assert errors == [], (
            f"{len(errors)} audit thread(s) raised exceptions:\n"
            + "\n".join(f"  [{i}] {type(e).__name__}: {e}" for i, e in enumerate(errors))
        )

    def test_exactly_thread_count_audit_rows_created(self, writer):
        """
        10 concurrent runs must produce exactly 10 audit rows.
        The audit table never upserts — every run is a new INSERT.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_audit_worker(writer, barrier, DATASET_ALPHA)
            for _ in range(THREAD_COUNT)
        ]
        _run_concurrent(workers)

        assert _count_audit_rows(DATASET_ALPHA) == THREAD_COUNT

    def test_all_audit_ids_are_unique(self, writer):
        """
        Each call to insert_running() must return a distinct SERIAL PK.
        Duplicate audit_ids would mean two threads wrote to the same row,
        which could cause one thread's update to overwrite another's.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_audit_worker(writer, barrier, DATASET_ALPHA)
            for _ in range(THREAD_COUNT)
        ]

        audit_ids, errors = _run_concurrent(workers)

        assert errors == [], f"Thread errors: {errors}"
        assert len(audit_ids) == THREAD_COUNT
        assert len(set(audit_ids)) == THREAD_COUNT, (
            f"Duplicate audit_ids found: "
            f"{[x for x in audit_ids if audit_ids.count(x) > 1]}"
        )

    def test_all_audit_rows_have_status_success(self, writer):
        """
        Every thread completes the full lifecycle including update_completed().
        All rows must end in status='SUCCESS' — no row should be stuck at RUNNING.
        A row stuck at RUNNING means update_completed() never fired for that thread.
        """
        barrier = threading.Barrier(THREAD_COUNT)
        workers = [
            _make_audit_worker(writer, barrier, DATASET_ALPHA)
            for _ in range(THREAD_COUNT)
        ]

        audit_ids, errors = _run_concurrent(workers)
        assert errors == [], f"Thread errors: {errors}"

        # Fetch all audit rows and check every one is SUCCESS
        conn = _raw_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM bronze_ingestion_audit "
                    "WHERE dataset_name = %s;",
                    (DATASET_ALPHA,),
                )
                statuses = [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

        assert len(statuses) == THREAD_COUNT
        stuck = [s for s in statuses if s != "SUCCESS"]
        assert stuck == [], (
            f"{len(stuck)} audit row(s) did not reach SUCCESS status: {stuck}"
        )

    def test_concurrent_audit_rows_for_different_datasets_are_independent(
        self, writer
    ):
        """
        5 threads write audit rows for DATASET_ALPHA, 5 for DATASET_BETA.
        Row counts for each dataset must be exactly 5 — no rows mixed across
        datasets and no rows lost under concurrent load.
        """
        n_per_dataset = THREAD_COUNT // 2

        barrier = threading.Barrier(THREAD_COUNT)
        workers = (
            [_make_audit_worker(writer, barrier, DATASET_ALPHA) for _ in range(n_per_dataset)]
            + [_make_audit_worker(writer, barrier, DATASET_BETA)  for _ in range(n_per_dataset)]
        )

        _, errors = _run_concurrent(workers)
        assert errors == [], f"Thread errors: {errors}"

        assert _count_audit_rows(DATASET_ALPHA) == n_per_dataset
        assert _count_audit_rows(DATASET_BETA)  == n_per_dataset