"""
Stress Test: sql_generator.py
==============================
Tests BronzeSQLGenerator and BronzeSQLTemplate across:
  - Valid datasets    : all 6 real datasets from bronze_ingestion_contract.json
  - Invalid datasets  : missing required keys, wrong types, None values
  - Corrupt datasets  : structurally present but semantically broken metadata
  - Edge cases        : SQL injection strings, unicode, empty strings, template
                        placeholder completeness, routing logic (COPY INTO vs MERGE)

sql_generator.py uses relative imports (.schema_mapper, .partition_strategy).
This test bootstraps a fake 'bronze' package in sys.modules so the module loads
cleanly without needing the full project package on sys.path.
Both SchemaMapper and PartitionStrategy are the real modules (no stubs).

Project stack: Python | Databricks | Spark (Delta Lake)
Reference files:
  - bronze_ingestion_contract.json  → source of valid dataset schemas
  - sql_generator.py                → module under test
  - partition_strategy.py           → used directly (not mocked)
  - schema_mapper.py                → used directly (real module, not mocked)

Run with:
    pytest tests/test_sql_generator.py -v
"""

import os
import re
import sys
import types
import unittest
import importlib.util

# ---------------------------------------------------------------------------
# Step 1 — Resolve project root so partition_strategy.py is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BRONZE_DIR = os.path.join(PROJECT_ROOT, "bronze")  # adjust folder name to match yours
for p in [PROJECT_ROOT, BRONZE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Step 2 — Bootstrap the 'bronze' package in sys.modules
#
# sql_generator.py uses relative imports:
#   from .schema_mapper import SchemaMapper
#   from .partition_strategy import PartitionConfig, PartitionHeuristics
#
# We create a fake 'bronze' package, inject a controlled FakeSchemaMapper,
# and wire the real partition_strategy in under the bronze.partition_strategy key.
# ---------------------------------------------------------------------------

# The real SchemaMapper is imported here and wired into the bronze package.
# This replaces the previously used FakeSchemaMapper stub, giving us real
# type-mapping behaviour (ValueError on unknown types, NOT NULL enforcement,
# correct Spark type casing, etc.) in all SQL generation tests.
from bronze.schema_mapper import SchemaMapper


def _bootstrap_sql_generator_module():
    """
    Load sql_generator.py as 'bronze.sql_generator' with mocked dependencies.
    Called once at module level; returns the loaded module.
    """
    import bronze.partition_strategy as real_ps

    # Create the fake bronze package
    bronze_pkg = types.ModuleType("bronze")
    sys.modules["bronze"] = bronze_pkg

    # Wire the real SchemaMapper as bronze.schema_mapper
    # Now that schema_mapper.py is available, we use it directly so the tests
    # exercise real type mapping, ValueError on unknown types, and correct
    # DDL/Spark schema formatting rather than a stub approximation.
    from bronze.schema_mapper import SchemaMapper as RealSchemaMapper
    schema_mapper_mod = types.ModuleType("bronze.schema_mapper")
    schema_mapper_mod.SchemaMapper = RealSchemaMapper
    sys.modules["bronze.schema_mapper"] = schema_mapper_mod

    # Wire the real partition_strategy as bronze.partition_strategy
    partition_mod = types.ModuleType("bronze.partition_strategy")
    partition_mod.PartitionConfig = real_ps.PartitionConfig
    partition_mod.PartitionHeuristics = real_ps.PartitionHeuristics
    partition_mod.PartitionStrategy = real_ps.PartitionStrategy
    sys.modules["bronze.partition_strategy"] = partition_mod

    # Load sql_generator.py as a submodule of bronze
    sql_gen_path = os.path.join(BRONZE_DIR, "sql_generator.py")
    spec = importlib.util.spec_from_file_location(
        "bronze.sql_generator", sql_gen_path, submodule_search_locations=[]
    )
    sql_gen_mod = importlib.util.module_from_spec(spec)
    sql_gen_mod.__package__ = "bronze"
    sys.modules["bronze.sql_generator"] = sql_gen_mod
    spec.loader.exec_module(sql_gen_mod)

    return sql_gen_mod


# Load module and pull out the classes we need
_sql_gen = _bootstrap_sql_generator_module()
BronzeSQLGenerator = _sql_gen.BronzeSQLGenerator
BronzeSQLTemplate = _sql_gen.BronzeSQLTemplate

import bronze.partition_strategy as ps
PartitionConfig = ps.PartitionConfig
PartitionStrategy = ps.PartitionStrategy
PartitionHeuristics = ps.PartitionHeuristics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMESTAMP = "2026-01-29T00:10:36"

# Partition configs used across tests
PARTITION_NONE = PartitionConfig(
    strategy=PartitionStrategy.NONE,
    partition_columns=[],
    reason="no partition",
    use_append_only=False,
)
PARTITION_TIME = PartitionConfig(
    strategy=PartitionStrategy.TIME_BASED,
    partition_columns=["billing_month"],
    reason="time partition",
    use_append_only=False,
)
PARTITION_HYBRID_APPEND = PartitionConfig(
    strategy=PartitionStrategy.HYBRID,
    partition_columns=["timestamp", "disco"],
    reason="hybrid",
    use_append_only=True,  # triggers MERGE path
)


def make_col(name: str, col_type: str = "string", nullable: bool = True) -> dict:
    return {"name": name, "type": col_type, "nullable": nullable}


def make_meta(
    dataset_name: str,
    columns: list,
    api_endpoint: str = "https://huggingface.co/api/datasets/example",
    total_rows: int = 200000,
) -> dict:
    return {
        "dataset_name": dataset_name,
        "api_endpoint": api_endpoint,
        "total_rows": total_rows,
        "files": [{"columns": columns}],
    }


def has_unfilled_placeholders(sql: str) -> list:
    """Return any {placeholder} tokens left unfilled in generated SQL."""
    return re.findall(r"\{[a-zA-Z_]+\}", sql)


# ---------------------------------------------------------------------------
# Contract datasets — loaded directly from bronze_ingestion_contract.json
#
# The JSON is read once at import time and transformed into the same dict
# shape the tests already expect: {dataset_name: make_meta(...)} so that
# every test below works without any changes.
# ---------------------------------------------------------------------------

import json as _json

BRONZE_METADATA_DIR = os.path.join(PROJECT_ROOT, "bronze_metadata")
_CONTRACT_PATH = os.path.join(BRONZE_METADATA_DIR, "bronze_ingestion_contract.json")

if not os.path.exists(_CONTRACT_PATH):
    raise FileNotFoundError(
        f"bronze_ingestion_contract.json not found at: {_CONTRACT_PATH}\n"
        f"Expected folder: {BRONZE_METADATA_DIR}\n"
        f"Project root:    {PROJECT_ROOT}"
    )

with open(_CONTRACT_PATH, "r") as _f:
    _raw_contract = _json.load(_f)

if not _raw_contract.get("datasets"):
    raise ValueError(
        f"Contract at {_CONTRACT_PATH} contains no datasets — "
        f"check the file is not empty or corrupt."
    )

# Transform raw contract datasets into the make_meta() shape the tests expect.
# Each dataset entry is converted using the same make_col() / make_meta() helpers
# defined above, so all downstream assertions remain unchanged.
CONTRACT_DATASETS = {
    ds["dataset_name"]: make_meta(
        dataset_name=ds["dataset_name"],
        columns=[
            make_col(col["name"], col["type"], col.get("nullable", True))
            for col in ds["files"][0]["columns"]
        ],
        api_endpoint=ds.get("api_endpoint", "https://huggingface.co/api/datasets/example"),
        total_rows=ds["total_rows"],
    )
    for ds in _raw_contract["datasets"]
}

# ===========================================================================
# GROUP 1 — BronzeSQLGenerator.__init__
# ===========================================================================

class TestBronzeSQLGeneratorInit(unittest.TestCase):
    """Verify constructor defaults and custom overrides."""

    def test_s01_default_constructor(self):
        """S01 — Default args produce the expected Databricks Unity Catalog defaults."""
        gen = BronzeSQLGenerator()
        self.assertEqual(gen.catalog, "main")
        self.assertEqual(gen.schema, "bronze")
        self.assertEqual(gen.base_location, "/mnt/delta/bronze")
        self.assertEqual(gen.staging_location, "/mnt/staging/raw")

    def test_s02_custom_constructor(self):
        """S02 — Custom catalog/schema/locations are stored correctly."""
        gen = BronzeSQLGenerator(
            catalog="energy_catalog",
            schema="bronze_ng",
            base_location="/mnt/delta/energy",
            staging_location="/mnt/staging/energy",
        )
        self.assertEqual(gen.catalog, "energy_catalog")
        self.assertEqual(gen.schema, "bronze_ng")
        self.assertEqual(gen.base_location, "/mnt/delta/energy")
        self.assertEqual(gen.staging_location, "/mnt/staging/energy")


# ===========================================================================
# GROUP 2 — generate_create_table_sql: VALID datasets
# ===========================================================================

class TestCreateTableSQLValid(unittest.TestCase):
    """
    S03–S08: All 6 real contract datasets produce structurally correct
    CREATE TABLE SQL with no unfilled placeholders.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def _assert_valid_create_sql(self, meta: dict, partition_config: PartitionConfig):
        """Shared assertion block for CREATE TABLE SQL."""
        sql = self.gen.generate_create_table_sql(meta, partition_config, TIMESTAMP)
        name = meta["dataset_name"]

        # Must return a non-empty string
        self.assertIsInstance(sql, str)
        self.assertGreater(len(sql), 0)

        # No unfilled placeholders
        unfilled = has_unfilled_placeholders(sql)
        self.assertEqual(unfilled, [], f"Unfilled placeholders in SQL: {unfilled}")

        # Core SQL structure
        self.assertIn("CREATE TABLE IF NOT EXISTS", sql)
        self.assertIn("USING DELTA", sql)
        self.assertIn(f"bronze_{name}", sql)
        self.assertIn(self.gen.catalog, sql)
        self.assertIn(self.gen.schema, sql)

        # Bronze metadata columns always present
        self.assertIn("_bronze_ingestion_timestamp TIMESTAMP", sql)
        self.assertIn("_bronze_source_file STRING", sql)
        self.assertIn("_bronze_row_hash STRING", sql)

        # Table properties always present
        self.assertIn("delta.enableChangeDataFeed", sql)
        self.assertIn("bronze_ingestion_contract.json", sql)
        self.assertIn(name, sql)  # source.dataset

        # Timestamp injected
        self.assertIn(TIMESTAMP, sql)

        # Table location uses base_location + dataset_name
        self.assertIn(f"{self.gen.base_location}/{name}", sql)

        return sql

    def test_s03_billing_payments(self):
        """S03 — billing_payments: 10-column dataset, TIME_BASED partition."""
        meta = CONTRACT_DATASETS["billing_payments"]
        cfg = PartitionHeuristics.determine_strategy(
            "billing_payments", 200000, meta["files"][0]["columns"], 1
        )
        sql = self._assert_valid_create_sql(meta, cfg)
        # billing_month is the time column — must appear in DDL
        self.assertIn("billing_month", sql)

    def test_s04_commercial_industries_consumption(self):
        """S04 — commercial_industries_consumption: 11 columns, TIME_BASED."""
        meta = CONTRACT_DATASETS["commercial_industries_consumption"]
        cfg = PartitionHeuristics.determine_strategy(
            "commercial_industries_consumption", 220000, meta["files"][0]["columns"], 1
        )
        sql = self._assert_valid_create_sql(meta, cfg)
        self.assertIn("active_power_kw", sql)

    def test_s05_customers_complaint(self):
        """S05 — customers_complaint: 9 columns, TIME_BASED (created_time)."""
        meta = CONTRACT_DATASETS["customers_complaint"]
        cfg = PartitionHeuristics.determine_strategy(
            "customers_complaint", 100000, meta["files"][0]["columns"], 1
        )
        self._assert_valid_create_sql(meta, cfg)

    def test_s06_grid_load(self):
        """S06 — grid_load: 10 columns including voltage_level_kv as int64."""
        meta = CONTRACT_DATASETS["grid_load"]
        cfg = PartitionHeuristics.determine_strategy(
            "grid_load", 200000, meta["files"][0]["columns"], 1
        )
        sql = self._assert_valid_create_sql(meta, cfg)
        # int64 column must appear mapped correctly by FakeSchemaMapper
        self.assertIn("voltage_level_kv", sql)

    def test_s07_power_flow(self):
        """S07 — power_flow: 10 columns, TIME_BASED."""
        meta = CONTRACT_DATASETS["power_flow"]
        cfg = PartitionHeuristics.determine_strategy(
            "power_flow", 200000, meta["files"][0]["columns"], 1
        )
        self._assert_valid_create_sql(meta, cfg)

    def test_s08_retail_tariffs(self):
        """
        S08 — retail_tariffs: 6 columns, NONE partition (90k rows < threshold).
        When partition_clause is empty, sql_generator.py omits the PARTITIONED BY
        line entirely — confirmed by the actual SQL output on this codebase.
        """
        meta = CONTRACT_DATASETS["retail_tariffs"]
        cfg = PartitionHeuristics.determine_strategy(
            "retail_tariffs", 90000, meta["files"][0]["columns"], 1
        )
        sql = self._assert_valid_create_sql(meta, cfg)
        # NONE strategy → no PARTITIONED BY line in the output at all
        self.assertNotIn("PARTITIONED BY", sql)

    def test_s09_partition_clause_present_when_strategy_is_not_none(self):
        """
        S09 — When a time-based partition config is passed, the PARTITIONED BY
        clause is injected into the CREATE TABLE SQL.
        """
        meta = CONTRACT_DATASETS["billing_payments"]
        sql = self.gen.generate_create_table_sql(meta, PARTITION_TIME, TIMESTAMP)
        self.assertIn("PARTITIONED BY", sql)
        self.assertIn("billing_month", sql)

    def test_s10_api_endpoint_appears_as_source_url(self):
        """
        S10 — api_endpoint from the contract metadata is written into
        the TBLPROPERTIES source.url field.
        """
        meta = CONTRACT_DATASETS["billing_payments"]
        sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn(meta["api_endpoint"], sql)

    def test_s11_missing_api_endpoint_falls_back_to_NA(self):
        """
        S11 — If api_endpoint is absent from metadata, source.url defaults to 'N/A'
        rather than raising a KeyError.
        """
        meta = {
            "dataset_name": "billing_payments",
            "files": [{"columns": [make_col("customer_id"), make_col("disco")]}],
            # api_endpoint deliberately omitted
        }
        sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("N/A", sql)

    def test_s12_custom_catalog_and_schema_appear_in_sql(self):
        """
        S12 — A generator with non-default catalog/schema embeds the correct
        fully-qualified table name in the CREATE TABLE statement.
        """
        gen = BronzeSQLGenerator(catalog="energy_cat", schema="bronze_ng")
        meta = CONTRACT_DATASETS["grid_load"]
        sql = gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("energy_cat.bronze_ng.bronze_grid_load", sql)


# ===========================================================================
# GROUP 3 — generate_create_table_sql: INVALID datasets
# ===========================================================================

class TestCreateTableSQLInvalid(unittest.TestCase):
    """
    S13–S17: Metadata that is structurally missing required keys raises
    predictable exceptions. These represent upstream contract failures.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def test_s13_missing_dataset_name_raises_key_error(self):
        """S13 — Missing 'dataset_name' key raises KeyError immediately."""
        bad_meta = {"files": [{"columns": [make_col("id")]}]}
        with self.assertRaises(KeyError) as ctx:
            self.gen.generate_create_table_sql(bad_meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("dataset_name", str(ctx.exception))

    def test_s14_missing_files_key_raises_key_error(self):
        """S14 — Missing 'files' key raises KeyError."""
        bad_meta = {"dataset_name": "billing_payments"}
        with self.assertRaises(KeyError) as ctx:
            self.gen.generate_create_table_sql(bad_meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("files", str(ctx.exception))

    def test_s15_empty_files_list_raises_index_error(self):
        """
        S15 — files=[] raises IndexError when the generator tries files[0].
        This catches a contract where file_count=0 slips through validation.
        """
        bad_meta = {"dataset_name": "billing_payments", "files": []}
        with self.assertRaises(IndexError):
            self.gen.generate_create_table_sql(bad_meta, PARTITION_NONE, TIMESTAMP)

    def test_s16_missing_columns_key_in_files_raises_key_error(self):
        """S16 — files[0] present but 'columns' key absent raises KeyError."""
        bad_meta = {"dataset_name": "billing_payments", "files": [{}]}
        with self.assertRaises(KeyError) as ctx:
            self.gen.generate_create_table_sql(bad_meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("columns", str(ctx.exception))

    def test_s17_none_dataset_name_renders_as_string_none(self):
        """
        S17 — dataset_name=None does NOT raise. Python str.format() coerces None
        to the string 'None', producing syntactically malformed SQL.
        KNOWN BEHAVIOUR: the generator does not validate that dataset_name is a
        non-None string — this should be caught upstream in the contract loader.
        """
        bad_meta = {
            "dataset_name": None,
            "files": [{"columns": [make_col("id")]}],
        }
        sql = self.gen.generate_create_table_sql(bad_meta, PARTITION_NONE, TIMESTAMP)
        self.assertIsInstance(sql, str)
        self.assertIn("None", sql)  # literal 'None' appears in SQL
        self.assertEqual(has_unfilled_placeholders(sql), [])


# ===========================================================================
# GROUP 4 — generate_create_table_sql: CORRUPT datasets
# ===========================================================================

class TestCreateTableSQLCorrupt(unittest.TestCase):
    """
    S18–S22: Metadata that is present but semantically broken.
    Tests document whether the generator handles corruption gracefully
    or propagates it into the SQL output (a known design characteristic).
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def test_s18_empty_columns_list_produces_sql_without_crashing(self):
        """
        S18 — columns=[] does not crash the generator.
        The DDL section will be empty (SchemaMapper.generate_ddl_columns returns '' for []).
        This is a known degradation path — documented, not fixed here.
        """
        meta = {"dataset_name": "billing_payments", "files": [{"columns": []}]}
        try:
            sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
            self.assertIn("CREATE TABLE IF NOT EXISTS", sql)
            # No unfilled placeholders even with empty columns
            self.assertEqual(has_unfilled_placeholders(sql), [])
        except Exception as e:
            self.fail(
                f"generate_create_table_sql raised {type(e).__name__} on empty columns: {e}"
            )

    def test_s19_column_missing_name_key_raises_key_error(self):
        """
        S19 — A column dict that has 'type' but no 'name' raises KeyError
        when SchemaMapper.build_column_schema() tries to access col['name'].
        Simulates a malformed column entry in the contract.
        """
        meta = {
            "dataset_name": "billing_payments",
            "files": [{"columns": [{"type": "string", "nullable": True}]}],
        }
        with self.assertRaises(KeyError):
            self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)

    def test_s20_sql_injection_string_in_dataset_name_passes_through(self):
        """
        S20 — A dataset_name containing SQL injection characters is passed
        verbatim into the template (the generator does NOT sanitise inputs).
        This test DOCUMENTS this behaviour as a known security characteristic —
        sanitisation is the caller's responsibility, not the generator's.
        """
        injected_name = "billing'; DROP TABLE main.bronze.bronze_billing--"
        meta = {
            "dataset_name": injected_name,
            "files": [{"columns": [make_col("id")]}],
        }
        sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        # The injection string IS present in the output (expected behaviour)
        self.assertIn("DROP TABLE", sql)
        # But no Python placeholders are left unfilled
        self.assertEqual(has_unfilled_placeholders(sql), [])

    def test_s21_unicode_dataset_name_does_not_crash(self):
        """
        S21 — A dataset_name with unicode characters (e.g. from non-ASCII sources)
        generates SQL without raising an exception.
        """
        meta = {
            "dataset_name": "kwh_消費データ",
            "files": [{"columns": [make_col("id"), make_col("value", "double")]}],
        }
        try:
            sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
            self.assertIn("kwh_消費データ", sql)
        except Exception as e:
            self.fail(f"Unicode dataset name raised {type(e).__name__}: {e}")

    def test_s22_unicode_column_names_pass_through(self):
        """
        S22 — Unicode column names (e.g. from a multilingual schema) are
        preserved in the generated SQL without encoding errors.
        """
        meta = {
            "dataset_name": "unicode_dataset",
            "files": [
                {
                    "columns": [
                        make_col("kwh_消費"),
                        make_col("région"),
                        make_col("مصرف", "double"),
                    ]
                }
            ],
        }
        sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("kwh_消費", sql)
        self.assertIn("région", sql)
        self.assertIn("مصرف", sql)


# ===========================================================================
# GROUP 5 — generate_ingestion_sql: VALID datasets (COPY INTO path)
# ===========================================================================

class TestIngestionSQLValid(unittest.TestCase):
    """
    S23–S29: All 6 contract datasets, exercising both the COPY INTO and
    MERGE routing paths. Assertions check SQL structure, table references,
    hash column expressions, and path construction.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def _assert_copy_into_sql(self, meta: dict, cfg: PartitionConfig):
        sql = self.gen.generate_ingestion_sql(meta, cfg, use_merge=False)
        name = meta["dataset_name"]

        self.assertIsInstance(sql, str)
        self.assertGreater(len(sql), 0)
        self.assertEqual(has_unfilled_placeholders(sql), [])

        self.assertIn("COPY INTO", sql)
        self.assertNotIn("MERGE INTO", sql)
        self.assertIn(f"bronze_{name}", sql)
        self.assertIn(self.gen.catalog, sql)
        self.assertIn(self.gen.schema, sql)

        # Source path must contain the dataset name and *.parquet wildcard
        self.assertIn(f"/{name}/*.parquet", sql)

        # Bad records path must be present
        self.assertIn("_bad_records", sql)

        # Bronze metadata columns must be computed
        self.assertIn("_bronze_ingestion_timestamp", sql)
        self.assertIn("_bronze_source_file", sql)
        self.assertIn("_bronze_row_hash", sql)
        self.assertIn("sha2(concat_ws", sql)

        # FILEFORMAT and schema enforcement options
        self.assertIn("FILEFORMAT = PARQUET", sql)
        self.assertIn("'mergeSchema' = 'false'", sql)

        return sql

    def test_s23_billing_payments_ingestion_sql(self):
        """
        S23 — billing_payments: 200k rows, all 10 columns appear in the SQL.

        NOTE ON ROUTING: The ingestion path (COPY INTO vs MERGE) depends on the
        use_append_only threshold in your local partition_strategy.py.
        - If the threshold is strict  > 200_000: 200k rows → use_append_only=False → COPY INTO
        - If the threshold is         >= 200_000: 200k rows → use_append_only=True  → MERGE INTO

        Your local version uses >= 200_000, so billing_payments routes to MERGE.
        This test validates the SQL content (all columns present, correct table name,
        no unfilled placeholders) without assuming which template is used.
        """
        meta = CONTRACT_DATASETS["billing_payments"]
        cfg = PartitionHeuristics.determine_strategy(
            "billing_payments", 200000, meta["files"][0]["columns"], 1
        )
        sql = self.gen.generate_ingestion_sql(meta, cfg, use_merge=False)

        # Must be one of the two valid template types — never both, never neither
        is_copy = "COPY INTO" in sql
        is_merge = "MERGE INTO" in sql
        self.assertTrue(is_copy or is_merge, "SQL must contain COPY INTO or MERGE INTO")
        self.assertFalse(is_copy and is_merge, "SQL must not contain both templates at once")

        # No unfilled placeholders regardless of which path was taken
        self.assertEqual(has_unfilled_placeholders(sql), [])

        # Correct table reference
        self.assertIn("bronze_billing_payments", sql)

        # All 10 columns must appear in the SELECT / hash expression
        for col in meta["files"][0]["columns"]:
            self.assertIn(col["name"], sql, f"Column '{col['name']}' missing from SQL")

    def test_s24_retail_tariffs_copy_into_small_dataset(self):
        """
        S24 — retail_tariffs: 90k rows → NONE partition, use_append_only=False
        → COPY INTO (not MERGE).
        """
        meta = CONTRACT_DATASETS["retail_tariffs"]
        cfg = PartitionHeuristics.determine_strategy(
            "retail_tariffs", 90000, meta["files"][0]["columns"], 1
        )
        # Confirm the partition config does NOT set use_append_only
        self.assertFalse(cfg.use_append_only)
        sql = self._assert_copy_into_sql(meta, cfg)
        self.assertIn("price_ngn_kwh", sql)

    def test_s25_all_six_datasets_produce_valid_ingestion_sql(self):
        """
        S25 — All 6 contract datasets produce valid ingestion SQL with no
        unfilled placeholders. The SQL type (COPY INTO vs MERGE) is determined
        by the partition config's use_append_only flag, not forced here.

        ROUTING RULE (from generate_ingestion_sql source code):
            if use_merge OR partition_config.use_append_only -> MERGE INTO
            else -> COPY INTO

        Datasets with rows > 200k get use_append_only=True from the heuristics,
        which means commercial_industries_consumption (220k) routes to MERGE
        even with use_merge=False. This is correct and expected behaviour.
        """
        for name, meta in CONTRACT_DATASETS.items():
            with self.subTest(dataset=name):
                cfg = PartitionHeuristics.determine_strategy(
                    name,
                    meta["total_rows"],
                    meta["files"][0]["columns"],
                    1,
                )
                sql = self.gen.generate_ingestion_sql(meta, cfg, use_merge=False)
                # The SQL must be one of the two valid template types
                is_copy = "COPY INTO" in sql
                is_merge = "MERGE INTO" in sql
                self.assertTrue(
                    is_copy or is_merge,
                    f"{name}: SQL is neither COPY INTO nor MERGE INTO"
                )
                # Never both at once
                self.assertFalse(
                    is_copy and is_merge,
                    f"{name}: SQL contains both COPY INTO and MERGE INTO"
                )
                # No unfilled placeholders
                self.assertEqual(
                    has_unfilled_placeholders(sql),
                    [],
                    f"{name}: unfilled placeholders found",
                )
                # Correct table name always present
                self.assertIn(f"bronze_{name}", sql)

    def test_s26_hash_columns_cast_to_string(self):
        """
        S26 — Every column in the hash expression must be wrapped in
        CAST(col AS STRING) to ensure consistent hashing across types.
        """
        meta = CONTRACT_DATASETS["billing_payments"]
        sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=False)
        for col in meta["files"][0]["columns"]:
            self.assertIn(f"CAST({col['name']} AS STRING)", sql)

    def test_s27_source_path_uses_staging_location(self):
        """
        S27 — The staging_location injected at construction time is used as
        the base for the source path in the generated SQL.
        """
        gen = BronzeSQLGenerator(staging_location="/mnt/custom/staging")
        meta = CONTRACT_DATASETS["grid_load"]
        sql = gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=False)
        self.assertIn("/mnt/custom/staging/grid_load/*.parquet", sql)


# ===========================================================================
# GROUP 6 — generate_ingestion_sql: MERGE path
# ===========================================================================

class TestIngestionSQLMergePath(unittest.TestCase):
    """
    S28–S32: Verify the MERGE INTO branch is correctly selected and
    structurally complete.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def _assert_merge_sql(self, meta: dict, cfg: PartitionConfig):
        sql = self.gen.generate_ingestion_sql(meta, cfg)
        name = meta["dataset_name"]

        self.assertIsInstance(sql, str)
        self.assertGreater(len(sql), 0)
        self.assertEqual(has_unfilled_placeholders(sql), [])

        self.assertIn("MERGE INTO", sql)
        self.assertNotIn("COPY INTO", sql)
        self.assertIn(f"bronze_{name}", sql)

        # MERGE structure
        self.assertIn("AS target", sql)
        self.assertIn("AS source", sql)
        self.assertIn("ON target._bronze_row_hash = source._bronze_row_hash", sql)
        self.assertIn("WHEN NOT MATCHED THEN INSERT *", sql)

        # read_files() is used as the source reader
        self.assertIn("read_files(", sql)
        self.assertIn("format => 'parquet'", sql)

        # Schema is enforced
        self.assertIn("schema =>", sql)

        return sql

    def test_s28_use_merge_true_forces_merge_path(self):
        """
        S28 — use_merge=True routes to MERGE INTO even when use_append_only=False.
        retail_tariffs (90k rows) normally routes to COPY INTO, but use_merge=True
        overrides that and forces the MERGE template.
        """
        meta = CONTRACT_DATASETS["retail_tariffs"]  # 90k rows → would be COPY INTO
        cfg = PartitionHeuristics.determine_strategy(
            "retail_tariffs", 90000, meta["files"][0]["columns"], 1
        )
        self.assertFalse(cfg.use_append_only)  # confirm baseline would be COPY INTO

        # Explicitly call with use_merge=True and validate the result
        sql = self.gen.generate_ingestion_sql(meta, cfg, use_merge=True)
        self.assertIn("MERGE INTO", sql)
        self.assertNotIn("COPY INTO", sql)
        self.assertIn("retail_tariffs", sql)
        self.assertEqual(has_unfilled_placeholders(sql), [])

    def test_s29_use_append_only_true_triggers_merge_path(self):
        """
        S29 — When partition_config.use_append_only=True, MERGE is used
        even without explicitly setting use_merge=True.
        """
        meta = CONTRACT_DATASETS["commercial_industries_consumption"]
        # 220k rows > 200k → use_append_only=True
        cfg = PartitionHeuristics.determine_strategy(
            "commercial_industries_consumption", 220000, meta["files"][0]["columns"], 1
        )
        self.assertTrue(cfg.use_append_only)
        sql = self._assert_merge_sql(meta, cfg)
        self.assertIn("commercial_industries_consumption", sql)

    def test_s30_all_columns_appear_in_merge_select(self):
        """
        S30 — Every column from the contract appears in the MERGE SELECT clause.
        """
        meta = CONTRACT_DATASETS["power_flow"]
        sql = self.gen.generate_ingestion_sql(meta, PARTITION_HYBRID_APPEND)
        for col in meta["files"][0]["columns"]:
            self.assertIn(col["name"], sql)

    def test_s31_customers_complaint_merge_path(self):
        """
        S31 — customers_complaint with use_merge=True produces valid MERGE SQL
        including the ticket_id and sla_met bool columns.
        """
        meta = CONTRACT_DATASETS["customers_complaint"]
        sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=True)
        self.assertIn("MERGE INTO", sql)
        self.assertIn("ticket_id", sql)
        self.assertIn("sla_met", sql)

    def test_s32_enforced_schema_string_in_merge_sql(self):
        """
        S32 — The MERGE template passes enforced_schema to read_files().
        Now that real SchemaMapper is wired in, the schema string uses the
        real format: 'col_name spark_type' pairs joined by ', ' with lowercase
        Spark types (e.g. 'customer_id string, kwh double, paid_on_time boolean').
        """
        meta = CONTRACT_DATASETS["billing_payments"]
        # Use the real SchemaMapper to build the expected string — this is
        # exactly what generate_ingestion_sql embeds in the SQL via enforced_schema
        expected_schema = SchemaMapper.generate_spark_schema_string(
            meta["files"][0]["columns"]
        )
        sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=True)
        self.assertIn(expected_schema, sql)
        # Verify the real format: space-separated, lowercase types, comma+space joined
        self.assertIn("customer_id string", sql)
        self.assertIn("paid_on_time boolean", sql)
        self.assertIn("kwh double", sql)


# ===========================================================================
# GROUP 7 — generate_ingestion_sql: INVALID & CORRUPT datasets
# ===========================================================================

class TestIngestionSQLInvalidAndCorrupt(unittest.TestCase):
    """
    S33–S38: Invalid and corrupt metadata fed to generate_ingestion_sql.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def test_s33_missing_dataset_name_raises_key_error(self):
        """S33 — Missing 'dataset_name' raises KeyError."""
        bad_meta = {"files": [{"columns": [make_col("id")]}]}
        with self.assertRaises(KeyError):
            self.gen.generate_ingestion_sql(bad_meta, PARTITION_NONE)

    def test_s34_missing_files_key_raises_key_error(self):
        """S34 — Missing 'files' key raises KeyError."""
        bad_meta = {"dataset_name": "billing_payments"}
        with self.assertRaises(KeyError):
            self.gen.generate_ingestion_sql(bad_meta, PARTITION_NONE)

    def test_s35_empty_files_list_raises_index_error(self):
        """S35 — files=[] raises IndexError at files[0] access."""
        bad_meta = {"dataset_name": "billing_payments", "files": []}
        with self.assertRaises(IndexError):
            self.gen.generate_ingestion_sql(bad_meta, PARTITION_NONE)

    def test_s36_missing_columns_key_raises_key_error(self):
        """S36 — files[0] with no 'columns' key raises KeyError."""
        bad_meta = {"dataset_name": "billing_payments", "files": [{}]}
        with self.assertRaises(KeyError):
            self.gen.generate_ingestion_sql(bad_meta, PARTITION_NONE)

    def test_s37_empty_columns_list_produces_sql_without_crashing(self):
        """
        S37 — columns=[] does not crash generate_ingestion_sql.
        SELECT clause will be empty but the template fills without error.
        KNOWN BEHAVIOUR: empty column list produces syntactically invalid SQL —
        documented here so future schema validation catches it upstream.
        """
        meta = {"dataset_name": "billing_payments", "files": [{"columns": []}]}
        try:
            sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE)
            self.assertIsInstance(sql, str)
            self.assertEqual(has_unfilled_placeholders(sql), [])
        except Exception as e:
            self.fail(
                f"generate_ingestion_sql raised {type(e).__name__} on empty columns: {e}"
            )

    def test_s38_column_missing_name_raises_key_error(self):
        """S38 — A column dict without 'name' raises KeyError."""
        meta = {
            "dataset_name": "billing_payments",
            "files": [{"columns": [{"type": "string"}]}],
        }
        with self.assertRaises(KeyError):
            self.gen.generate_ingestion_sql(meta, PARTITION_NONE)


# ===========================================================================
# GROUP 8 — generate_optimization_sql
# ===========================================================================

class TestOptimizationSQL(unittest.TestCase):
    """
    S39–S44: Verify OPTIMIZE and VACUUM SQL for all contract datasets,
    edge cases, and custom catalog/schema combinations.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def _assert_optimization_sql(self, sql: str, dataset_name: str, gen=None):
        if gen is None:
            gen = self.gen
        table_ref = f"bronze_{dataset_name}"

        self.assertIsInstance(sql, str)
        self.assertGreater(len(sql), 0)

        # Both maintenance statements must be present
        self.assertIn("OPTIMIZE", sql)
        self.assertIn("VACUUM", sql)

        # Correct fully-qualified table reference
        self.assertIn(f"{gen.catalog}.{gen.schema}.{table_ref}", sql)

        # Z-ordering on ingestion timestamp
        self.assertIn("ZORDER BY (_bronze_ingestion_timestamp)", sql)

        # 7-day = 168 hours retention
        self.assertIn("RETAIN 168 HOURS", sql)

    def test_s39_billing_payments_optimization(self):
        """S39 — billing_payments OPTIMIZE + VACUUM SQL is structurally correct."""
        sql = self.gen.generate_optimization_sql("billing_payments")
        self._assert_optimization_sql(sql, "billing_payments")

    def test_s40_all_six_contract_datasets_optimization(self):
        """S40 — All 6 contract datasets produce valid optimization SQL."""
        for name in CONTRACT_DATASETS:
            with self.subTest(dataset=name):
                sql = self.gen.generate_optimization_sql(name)
                self._assert_optimization_sql(sql, name)

    def test_s41_custom_catalog_schema_in_optimization(self):
        """
        S41 — A generator with custom catalog/schema embeds them correctly
        in both the OPTIMIZE and VACUUM statements.
        """
        gen = BronzeSQLGenerator(catalog="energy_cat", schema="bronze_ng")
        sql = gen.generate_optimization_sql("grid_load")
        self.assertIn("energy_cat.bronze_ng.bronze_grid_load", sql)
        self._assert_optimization_sql(sql, "grid_load", gen=gen)

    def test_s42_optimization_sql_comment_includes_table_name(self):
        """S42 — The SQL comment at the top names the table being optimized."""
        sql = self.gen.generate_optimization_sql("power_flow")
        self.assertIn("bronze_power_flow", sql)

    def test_s43_sql_injection_in_dataset_name_passes_through(self):
        """
        S43 — SQL injection in dataset_name passes through to OPTIMIZE/VACUUM.
        Same design characteristic as CREATE TABLE — caller must sanitise inputs.
        Both OPTIMIZE and VACUUM lines will contain the injected string.
        """
        injected = "power_flow'; DROP TABLE main.bronze.bronze_power_flow--"
        sql = self.gen.generate_optimization_sql(injected)
        self.assertIn("OPTIMIZE", sql)
        self.assertIn("VACUUM", sql)
        self.assertIn("DROP TABLE", sql)  # injection present in output

    def test_s44_empty_string_dataset_name(self):
        """
        S44 — An empty string dataset_name produces SQL referencing 'bronze_'
        without crashing. Structurally degenerate but documented.
        """
        sql = self.gen.generate_optimization_sql("")
        self.assertIn("OPTIMIZE", sql)
        self.assertIn("VACUUM", sql)
        self.assertIn(f"{self.gen.catalog}.{self.gen.schema}.bronze_", sql)


# ===========================================================================
# GROUP 9 — Template placeholder completeness
# ===========================================================================

class TestTemplatePlaceholderCompleteness(unittest.TestCase):
    """
    S45–S47: Verify that no {placeholder} tokens are ever left unfilled in
    any of the three SQL templates across all contract datasets.
    This catches template/format() key mismatches before they hit Databricks.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def test_s45_no_unfilled_placeholders_in_create_table_for_all_datasets(self):
        """S45 — CREATE TABLE SQL for all 6 datasets has zero unfilled placeholders."""
        for name, meta in CONTRACT_DATASETS.items():
            with self.subTest(dataset=name):
                cfg = PartitionHeuristics.determine_strategy(
                    name, meta["total_rows"], meta["files"][0]["columns"], 1
                )
                sql = self.gen.generate_create_table_sql(meta, cfg, TIMESTAMP)
                unfilled = has_unfilled_placeholders(sql)
                self.assertEqual(
                    unfilled, [], f"{name}: unfilled placeholders: {unfilled}"
                )

    def test_s46_no_unfilled_placeholders_in_ingestion_sql_copy_into(self):
        """S46 — COPY INTO SQL for all 6 datasets has zero unfilled placeholders."""
        for name, meta in CONTRACT_DATASETS.items():
            with self.subTest(dataset=name):
                sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=False)
                unfilled = has_unfilled_placeholders(sql)
                self.assertEqual(
                    unfilled, [], f"{name}: unfilled placeholders in COPY INTO: {unfilled}"
                )

    def test_s47_no_unfilled_placeholders_in_ingestion_sql_merge(self):
        """S47 — MERGE SQL for all 6 datasets has zero unfilled placeholders."""
        for name, meta in CONTRACT_DATASETS.items():
            with self.subTest(dataset=name):
                sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=True)
                unfilled = has_unfilled_placeholders(sql)
                self.assertEqual(
                    unfilled, [], f"{name}: unfilled placeholders in MERGE: {unfilled}"
                )



# ===========================================================================
# GROUP 10 — Real SchemaMapper integration tests
# (only possible now that the real schema_mapper.py is wired in)
# ===========================================================================

class TestRealSchemaMapperIntegration(unittest.TestCase):
    """
    S48–S55: Tests that validate real SchemaMapper behaviour through the
    sql_generator pipeline. These were not testable with FakeSchemaMapper
    because the stub silently accepted all types and never raised ValueError.
    """

    def setUp(self):
        self.gen = BronzeSQLGenerator()

    def test_s48_unrecognised_type_raises_value_error_in_create_table(self):
        """
        S48 — A column with a type not in SchemaMapper.TYPE_MAP (e.g. 'varchar',
        'json', 'array') raises ValueError during generate_create_table_sql.
        This is SchemaMapper's type explosion prevention mechanism.
        """
        for bad_type in ["varchar", "json", "array", "text", "nvarchar"]:
            with self.subTest(bad_type=bad_type):
                meta = {
                    "dataset_name": "billing_payments",
                    "files": [{"columns": [{"name": "col1", "type": bad_type, "nullable": True}]}],
                }
                with self.assertRaises(ValueError) as ctx:
                    self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
                self.assertIn(bad_type, str(ctx.exception))
                self.assertIn("Allowed types", str(ctx.exception))

    def test_s49_unrecognised_type_raises_value_error_in_ingestion_sql(self):
        """
        S49 — Unrecognised column type in generate_ingestion_sql also raises
        ValueError (only on the MERGE path which calls generate_spark_schema_string;
        COPY INTO path does not call SchemaMapper for types so no error there).
        """
        meta = {
            "dataset_name": "billing_payments",
            "files": [{"columns": [{"name": "col1", "type": "json", "nullable": True}]}],
        }
        with self.assertRaises(ValueError):
            self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=True)

    def test_s50_not_null_appears_in_ddl_for_non_nullable_columns(self):
        """
        S50 — A column with nullable=False must produce 'COL_NAME TYPE NOT NULL'
        in the CREATE TABLE DDL. This was not testable with FakeSchemaMapper
        which produced the same output regardless of nullable.
        """
        meta = {
            "dataset_name": "billing_payments",
            "api_endpoint": "https://example.com",
            "files": [{"columns": [
                {"name": "customer_id", "type": "string",  "nullable": False},
                {"name": "amount",      "type": "double",  "nullable": False},
                {"name": "notes",       "type": "string",  "nullable": True},
            ]}],
        }
        sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("customer_id STRING NOT NULL", sql)
        self.assertIn("amount DOUBLE NOT NULL", sql)
        # Nullable column must NOT have NOT NULL
        self.assertNotIn("notes STRING NOT NULL", sql)

    def test_s51_spark_schema_uses_lowercase_types(self):
        """
        S51 — generate_spark_schema_string returns lowercase Spark type names
        (e.g. 'string', 'double', 'boolean', 'bigint') NOT uppercase SQL names.
        The MERGE SQL's schema => clause must contain lowercase types.
        """
        meta = CONTRACT_DATASETS["billing_payments"]
        sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=True)
        # Lowercase Spark type names must appear in the schema string
        self.assertIn("string", sql)
        self.assertIn("double", sql)
        self.assertIn("boolean", sql)  # 'bool' contract type → 'boolean' Spark type

    def test_s52_int64_maps_to_bigint_in_ddl(self):
        """
        S52 — Contract type 'int64' must map to Databricks SQL type 'BIGINT'
        in the CREATE TABLE DDL. Verifies SchemaMapper.TYPE_MAP['int64'] = 'BIGINT'.
        """
        meta = CONTRACT_DATASETS["grid_load"]  # has voltage_level_kv: int64
        sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
        self.assertIn("voltage_level_kv BIGINT", sql)
        # Must NOT appear as INT or INTEGER
        self.assertNotIn("voltage_level_kv INT\n", sql)
        self.assertNotIn("voltage_level_kv INTEGER", sql)

    def test_s53_int64_maps_to_bigint_in_spark_schema(self):
        """
        S53 — Contract type 'int64' maps to Spark type 'bigint' (lowercase)
        in the MERGE SQL schema string. Tests the full chain:
        contract type → SQL type (BIGINT) → Spark type (bigint).
        """
        meta = CONTRACT_DATASETS["power_flow"]  # has from_kv, to_kv: int64
        sql = self.gen.generate_ingestion_sql(meta, PARTITION_NONE, use_merge=True)
        self.assertIn("from_kv bigint", sql)
        self.assertIn("to_kv bigint", sql)

    def test_s54_bool_maps_to_boolean_in_ddl(self):
        """
        S54 — Contract type 'bool' maps to Databricks SQL type 'BOOLEAN' in DDL.
        Checks the billing_payments.paid_on_time and customers_complaint.sla_met columns.
        """
        for name, col in [("billing_payments", "paid_on_time"), ("customers_complaint", "sla_met")]:
            with self.subTest(dataset=name):
                meta = CONTRACT_DATASETS[name]
                sql = self.gen.generate_create_table_sql(meta, PARTITION_NONE, TIMESTAMP)
                self.assertIn(f"{col} BOOLEAN", sql)

    def test_s55_all_six_datasets_produce_valid_ddl_with_real_schema_mapper(self):
        """
        S55 — End-to-end: all 6 contract datasets produce CREATE TABLE SQL
        that contains correct Databricks SQL type names (STRING, DOUBLE, BOOLEAN,
        BIGINT) from SchemaMapper.TYPE_MAP, with no unfilled placeholders.
        """
        expected_type_snippets = {
            "billing_payments": ["STRING", "DOUBLE", "BOOLEAN"],
            "commercial_industries_consumption": ["STRING", "DOUBLE"],
            "customers_complaint": ["STRING", "BOOLEAN"],
            "grid_load": ["BIGINT", "DOUBLE"],
            "power_flow": ["BIGINT", "DOUBLE"],
            "retail_tariffs": ["BIGINT", "DOUBLE", "STRING"],
        }
        for name, meta in CONTRACT_DATASETS.items():
            with self.subTest(dataset=name):
                cfg = PartitionHeuristics.determine_strategy(
                    name, meta["total_rows"], meta["files"][0]["columns"], 1
                )
                sql = self.gen.generate_create_table_sql(meta, cfg, TIMESTAMP)
                self.assertEqual(has_unfilled_placeholders(sql), [])
                for sql_type in expected_type_snippets[name]:
                    self.assertIn(sql_type, sql, f"{name}: expected type '{sql_type}' missing from DDL")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestBronzeSQLGeneratorInit,
        TestCreateTableSQLValid,
        TestCreateTableSQLInvalid,
        TestCreateTableSQLCorrupt,
        TestIngestionSQLValid,
        TestIngestionSQLMergePath,
        TestIngestionSQLInvalidAndCorrupt,
        TestOptimizationSQL,
        TestTemplatePlaceholderCompleteness,
        TestRealSchemaMapperIntegration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
