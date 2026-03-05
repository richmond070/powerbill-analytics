"""
Stress Test: partition_strategy.py
===================================
Tests PartitionHeuristics.determine_strategy() and generate_partition_clause()
across 15 dataset scenarios — 6 derived directly from bronze_ingestion_contract.json
metadata and 9 synthetic edge-case/boundary scenarios.

"""

import os
import sys
import json
import pytest
from bronze.partition_strategy import PartitionHeuristics, PartitionStrategy, PartitionConfig

# ---------------------------------------------------------------------------
# Contract loader — bronze_ingestion_contract.json as single source of truth
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BRONZE_METADATA_DIR = os.path.join(PROJECT_ROOT, "bronze_metadata")
_CONTRACT_PATH = os.path.join(BRONZE_METADATA_DIR, "bronze_ingestion_contract.json")

if not os.path.exists(_CONTRACT_PATH):
    raise FileNotFoundError(
        f"bronze_ingestion_contract.json not found at: {_CONTRACT_PATH}\n"
        f"Expected folder: {BRONZE_METADATA_DIR}\n"
        f"Project root:    {PROJECT_ROOT}"
    )

with open(_CONTRACT_PATH, "r") as _f:
    _raw_contract = json.load(_f)

if not _raw_contract.get("datasets"):
    raise ValueError(
        f"Contract at {_CONTRACT_PATH} contains no datasets — "
        f"check the file is not empty or corrupt."
    )

# Keyed lookup: dataset_name → contract entry
_DS = {ds["dataset_name"]: ds for ds in _raw_contract["datasets"]}


def make_columns(*names: str) -> list[dict]:
    """Convert column name strings into the contract column-metadata format."""
    return [{"name": n, "type": "string", "nullable": True} for n in names]


def _contract_columns(dataset_name: str) -> list[dict]:
    """
    Return make_columns()-compatible output for a real contract dataset.
    Strips only the column names from the contract (type/nullable are not
    used by PartitionHeuristics.determine_strategy) so the call signature
    into make_columns() remains identical to the original hardcoded tests.
    """
    names = [col["name"] for col in _DS[dataset_name]["files"][0]["columns"]]
    return make_columns(*names)


def _contract_rows(dataset_name: str) -> int:
    """Return total_rows for a real contract dataset."""
    return _DS[dataset_name]["total_rows"]


# ===========================================================================
# SCENARIO GROUP A — Real datasets from bronze_ingestion_contract.json
# ===========================================================================

class TestRealContractDatasets:
    """
    Scenarios 1-6: Exact metadata sourced from bronze_ingestion_contract.json.
    These act as integration-level regression tests; if the contract changes,
    these tests surface it immediately.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — billing_payments (200,000 rows, 10 cols)
    # Contract: has 'billing_month' (time) + 'disco' (category)
    # Row range: 100k–500k → medium band → TIME_BASED (Rule 5)
    # NOTE: 200k < LARGE_DATASET_THRESHOLD(500k) → falls to RULE 5 (time only)
    # BOUNDARY FINDING: use_append_only condition in Rule 5 is `>= 200_000`,
    # so exactly 200k rows returns True. Verified against partition_strategy.py
    # line: use_append_only=total_rows >= 200_000
    # ------------------------------------------------------------------
    def test_scenario_01_billing_payments(self):
        columns = _contract_columns("billing_payments")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="billing_payments",
            total_rows=_contract_rows("billing_payments"),
            columns=columns,
            file_count=1
        )
        # 200k is between SMALL(100k) and LARGE(500k) → Rule 5: TIME_BASED
        assert config.strategy == PartitionStrategy.TIME_BASED, (
            f"billing_payments (200k rows) should be TIME_BASED, got {config.strategy}"
        )
        assert "billing_month" in config.partition_columns, (
            "'billing_month' must be selected as partition column"
        )
        # BOUNDARY: condition is `>= 200_000`, so exactly 200k → True
        assert config.use_append_only is True
        assert config.reason, "Reason string must not be empty"

    # ------------------------------------------------------------------
    # Scenario 2 — commercial_industries_consumption (220,000 rows, 11 cols)
    # Contract: has 'timestamp' (time) + 'disco', 'site_type' (category)
    # Row range: medium → TIME_BASED, use_append_only=True (220k > 200k)
    # ------------------------------------------------------------------
    def test_scenario_02_commercial_industries_consumption(self):
        columns = _contract_columns("commercial_industries_consumption")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="commercial_industries_consumption",
            total_rows=_contract_rows("commercial_industries_consumption"),
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.TIME_BASED, (
            f"commercial_industries_consumption (220k rows) should be TIME_BASED, got {config.strategy}"
        )
        assert "timestamp" in config.partition_columns, (
            "'timestamp' should be prioritised as the time partition column"
        )
        assert config.use_append_only is True
        assert config.reason

    # ------------------------------------------------------------------
    # Scenario 3 — customers_complaint (100,000 rows, 9 cols)
    # Contract: has 'created_time', 'resolved_time' (time) + 'disco', 'category'
    # Exactly at SMALL_DATASET_THRESHOLD → Rule 1 should NOT fire (< 100k)
    # 100k is NOT < 100k, so we enter medium path → TIME_BASED
    # ------------------------------------------------------------------
    def test_scenario_03_customers_complaint(self):
        columns = _contract_columns("customers_complaint")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="customers_complaint",
            total_rows=_contract_rows("customers_complaint"),
            columns=columns,
            file_count=1
        )
        # 100k is NOT < 100k (SMALL threshold is strict <), so enters medium/large path
        assert config.strategy != PartitionStrategy.NONE, (
            "100k rows is exactly at threshold; NONE strategy should not fire (condition is strict <)"
        )
        assert config.partition_columns, "Partition columns must be selected"
        assert config.reason

    # ------------------------------------------------------------------
    # Scenario 4 — grid_load (200,000 rows, 10 cols)
    # Contract: has 'timestamp' (time) + 'disco' (category) + 'source' (category-adjacent)
    # Medium band → TIME_BASED
    # BOUNDARY: exactly 200k rows → use_append_only=True (condition is >= 200_000)
    # ------------------------------------------------------------------
    def test_scenario_04_grid_load(self):
        columns = _contract_columns("grid_load")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="grid_load",
            total_rows=_contract_rows("grid_load"),
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.TIME_BASED
        assert "timestamp" in config.partition_columns
        # BOUNDARY: condition is `>= 200_000`, so exactly 200k → True
        assert config.use_append_only is True
        assert config.reason

    # ------------------------------------------------------------------
    # Scenario 5 — power_flow (200,000 rows, 10 cols)
    # Contract: has 'timestamp' (time) + 'disco' (category)
    # Medium band → TIME_BASED
    # BOUNDARY: exactly 200k rows → use_append_only=True (condition is >= 200_000)
    # ------------------------------------------------------------------
    def test_scenario_05_power_flow(self):
        columns = _contract_columns("power_flow")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="power_flow",
            total_rows=_contract_rows("power_flow"),
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.TIME_BASED
        assert "timestamp" in config.partition_columns
        # BOUNDARY: condition is `>= 200_000`, so exactly 200k → True
        assert config.use_append_only is True
        assert config.reason

    # ------------------------------------------------------------------
    # Scenario 6 — retail_tariffs (90,000 rows, 6 cols)
    # Contract: has 'as_of_date' (time) + 'disco', 'customer_class' (category)
    # 90k < SMALL_DATASET_THRESHOLD(100k) → Rule 1: NONE, use_append_only=False
    # ------------------------------------------------------------------
    def test_scenario_06_retail_tariffs(self):
        columns = _contract_columns("retail_tariffs")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="retail_tariffs",
            total_rows=_contract_rows("retail_tariffs"),
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.NONE, (
            f"retail_tariffs (90k rows) is below SMALL threshold; expected NONE, got {config.strategy}"
        )
        assert config.partition_columns == [], "No partition columns expected for small dataset"
        assert config.use_append_only is False
        assert config.reason


# ===========================================================================
# SCENARIO GROUP B — Synthetic Edge-Case / Boundary Scenarios
# ===========================================================================

class TestEdgeCasesAndBoundaries:
    """
    Scenarios 7-15: Stress the heuristic rules at boundaries and unusual
    column configurations that would not appear in the current contract
    but could arise as new datasets are onboarded.
    """

    # ------------------------------------------------------------------
    # Scenario 7 — Exact small threshold boundary (99,999 rows)
    # One row below SMALL_DATASET_THRESHOLD → must return NONE
    # ------------------------------------------------------------------
    def test_scenario_07_just_below_small_threshold(self):
        columns = make_columns("timestamp", "disco", "value")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="tiny_dataset",
            total_rows=99_999,
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.NONE, (
            "99,999 rows is strictly below SMALL threshold; strategy must be NONE"
        )
        assert config.use_append_only is False
        assert config.partition_columns == []

    # ------------------------------------------------------------------
    # Scenario 8 — Exactly at LARGE threshold (500,000 rows)
    # Has both time + category → must return HYBRID
    # ------------------------------------------------------------------
    def test_scenario_08_exactly_at_large_threshold(self):
        columns = make_columns("timestamp", "disco", "region", "value")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="threshold_dataset",
            total_rows=500_000,
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.HYBRID, (
            "500k rows meets LARGE threshold; with time+category cols should be HYBRID"
        )
        assert config.use_append_only is True
        assert len(config.partition_columns) == 2

    # ------------------------------------------------------------------
    # Scenario 9 — Large dataset (1M rows), time only, no category column
    # Should return TIME_BASED with use_append_only=True
    # ------------------------------------------------------------------
    def test_scenario_09_large_time_only_no_category(self):
        columns = make_columns(
            "record_id", "timestamp", "sensor_reading", "unit", "batch_id"
        )
        config = PartitionHeuristics.determine_strategy(
            dataset_name="telemetry_raw",
            total_rows=1_000_000,
            columns=columns,
            file_count=5
        )
        assert config.strategy == PartitionStrategy.TIME_BASED
        assert "timestamp" in config.partition_columns
        assert config.use_append_only is True

    # ------------------------------------------------------------------
    # Scenario 10 — Large dataset (600,000 rows), category only, no time column
    # Should return CATEGORY_BASED with use_append_only=True
    # ------------------------------------------------------------------
    def test_scenario_10_large_category_only_no_time(self):
        columns = make_columns(
            "record_id", "region", "state", "amount", "status"
        )
        config = PartitionHeuristics.determine_strategy(
            dataset_name="regional_metrics",
            total_rows=600_000,
            columns=columns,
            file_count=3
        )
        assert config.strategy == PartitionStrategy.CATEGORY_BASED, (
            "Large dataset with only category columns should be CATEGORY_BASED"
        )
        assert config.use_append_only is True
        assert config.partition_columns  # must not be empty

    # ------------------------------------------------------------------
    # Scenario 11 — Large dataset (700,000 rows), NO time, NO category columns
    # Should fall through to NONE with use_append_only=True (>200k fallback)
    # ------------------------------------------------------------------
    def test_scenario_11_large_no_matching_columns(self):
        columns = make_columns(
            "record_id", "hash_key", "value_a", "value_b", "checksum"
        )
        config = PartitionHeuristics.determine_strategy(
            dataset_name="raw_hash_table",
            total_rows=700_000,
            columns=columns,
            file_count=2
        )
        assert config.strategy == PartitionStrategy.NONE, (
            "Large dataset with no time/category columns must still return NONE (Rule 6)"
        )
        assert config.use_append_only is True, (
            "700k > 200k fallback threshold → use_append_only should be True"
        )
        assert config.partition_columns == []

    # ------------------------------------------------------------------
    # Scenario 12 — Zero rows dataset (empty source file)
    # Simulates an upstream feed that delivered an empty parquet file
    # Should return NONE (0 < SMALL threshold)
    # ------------------------------------------------------------------
    def test_scenario_12_zero_rows(self):
        columns = make_columns("timestamp", "disco", "value")
        config = PartitionHeuristics.determine_strategy(
            dataset_name="empty_feed",
            total_rows=0,
            columns=columns,
            file_count=1
        )
        assert config.strategy == PartitionStrategy.NONE
        assert config.use_append_only is False
        assert config.partition_columns == []

    # ------------------------------------------------------------------
    # Scenario 13 — Column names that partially match patterns (false-positive check)
    # "disco_backup", "re_timestamp_raw" — should these trigger matches?
    # The current heuristic uses `in` substring match, so they WILL match.
    # This test documents that behaviour as a known characteristic.
    # ------------------------------------------------------------------
    def test_scenario_13_partial_pattern_column_names(self):
        columns = make_columns(
            "disco_backup", "re_timestamp_raw", "load_value"
        )
        config = PartitionHeuristics.determine_strategy(
            dataset_name="partial_match_dataset",
            total_rows=150_000,
            columns=columns,
            file_count=1
        )
        # Substring match means 'disco_backup' contains 'disco' and
        # 're_timestamp_raw' contains 'timestamp' → TIME_BASED expected
        assert config.strategy == PartitionStrategy.TIME_BASED, (
            "Substring pattern match means partial column names still trigger time detection"
        )
        assert config.partition_columns, "Partition column must be selected via substring match"

    # ------------------------------------------------------------------
    # Scenario 14 — Medium dataset (300,000 rows) mimicking a second
    # billing_payments batch with ALL original contract columns present.
    # Verifies that 'billing_month' is correctly prioritised over 'disco'
    # as the lead partition column in the medium band (Rule 5).
    # ------------------------------------------------------------------
    def test_scenario_14_medium_billing_payments_priority_check(self):
        columns = make_columns(
            "customer_id", "disco", "billing_month", "tariff_band",
            "kwh", "price_ngn_kwh", "amount_billed_ngn",
            "amount_paid_ngn", "paid_on_time", "arrears_ngn"
        )
        config = PartitionHeuristics.determine_strategy(
            dataset_name="billing_payments_batch2",
            total_rows=300_000,
            columns=columns,
            file_count=2
        )
        assert config.strategy == PartitionStrategy.TIME_BASED
        # billing_month should be first in partition_columns (priority ordering)
        assert config.partition_columns[0] == "billing_month", (
            "'billing_month' must be first partition column per priority order"
        )
        assert config.use_append_only is True  # 300k > 200k

    # ------------------------------------------------------------------
    # Scenario 15 — Very large multi-file dataset (5M rows, 10 files)
    # Mimics a future scaled-up version of grid_load with both time + category.
    # Should return HYBRID with use_append_only=True.
    # Also validates generate_partition_clause() output format.
    # ------------------------------------------------------------------
    def test_scenario_15_very_large_multifile_hybrid_and_sql_clause(self):
        columns = make_columns(
            "timestamp", "disco", "substation_id", "voltage_level_kv",
            "load_mw", "forecast_mw", "temp_c", "humidity",
            "frequency_hz", "source"
        )
        config = PartitionHeuristics.determine_strategy(
            dataset_name="grid_load_scaled",
            total_rows=5_000_000,
            columns=columns,
            file_count=10
        )

        # Strategy assertions
        assert config.strategy == PartitionStrategy.HYBRID
        assert len(config.partition_columns) == 2
        assert "timestamp" in config.partition_columns
        assert "disco" in config.partition_columns
        assert config.use_append_only is True
        assert config.reason

        # SQL clause assertions
        clause = PartitionHeuristics.generate_partition_clause(config)
        assert clause.startswith("PARTITIONED BY ("), (
            f"SQL clause must start with 'PARTITIONED BY (', got: '{clause}'"
        )
        assert "timestamp" in clause
        assert "disco" in clause
        assert clause.endswith(")"), "SQL clause must end with ')'"


# ===========================================================================
# SCENARIO GROUP C — generate_partition_clause() unit tests
# ===========================================================================

class TestGeneratePartitionClause:
    """
    Direct unit tests for the SQL clause generator, covering all four
    PartitionStrategy enum values.
    """

    def test_clause_none_strategy_returns_empty_string(self):
        config = PartitionConfig(
            strategy=PartitionStrategy.NONE,
            partition_columns=[],
            reason="Small dataset",
            use_append_only=False
        )
        assert PartitionHeuristics.generate_partition_clause(config) == ""

    def test_clause_time_based_single_column(self):
        config = PartitionConfig(
            strategy=PartitionStrategy.TIME_BASED,
            partition_columns=["billing_month"],
            reason="Time partition",
            use_append_only=True
        )
        clause = PartitionHeuristics.generate_partition_clause(config)
        assert clause == "PARTITIONED BY (billing_month)"

    def test_clause_category_based_single_column(self):
        config = PartitionConfig(
            strategy=PartitionStrategy.CATEGORY_BASED,
            partition_columns=["disco"],
            reason="Category partition",
            use_append_only=True
        )
        clause = PartitionHeuristics.generate_partition_clause(config)
        assert clause == "PARTITIONED BY (disco)"

    def test_clause_hybrid_two_columns(self):
        config = PartitionConfig(
            strategy=PartitionStrategy.HYBRID,
            partition_columns=["timestamp", "disco"],
            reason="Hybrid partition",
            use_append_only=True
        )
        clause = PartitionHeuristics.generate_partition_clause(config)
        assert clause == "PARTITIONED BY (timestamp, disco)"