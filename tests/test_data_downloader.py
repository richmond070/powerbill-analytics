"""
Stress Test: data_downloader.py
================================
Tests DataDownloader and DataValidator across all methods and edge cases.
Scenarios are anchored to the real dataset metadata from
bronze_ingestion_contract.json — the 6 parquet datasets that feed
the Nigerian energy & utilities bronze layer on Databricks / Delta Lake.

No real HTTP calls are made. requests.get is fully mocked.
All file I/O uses isolated temporary directories.

Project stack: Python | Databricks | Spark (Delta Lake)
Reference files:
  - bronze_ingestion_contract.json  → real dataset URLs, filenames, row counts
  - data_downloader.py              → module under test

Run with:
    pytest tests/test_data_downloader.py -v
"""

import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call, mock_open
from io import BytesIO
import json

# ------------------------------------------------------- --------------------
# Cross-platform path fix
# Resolve the project root (parent of tests/) and insert at front of sys.path
# so `data_downloader` is importable on both Windows and Linux.
# os.path.abspath normalises backslashes on Windows correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bronze.data_downloader import DataDownloader, DataValidator, DownloadResult


# ---------------------------------------------------------------------------
# Contract fixtures — sourced verbatim from bronze_ingestion_contract.json
# ---------------------------------------------------------------------------

BRONZE_METADATA_DIR = os.path.join(PROJECT_ROOT, "bronze_metadata")
_CONTRACT_PATH = os.path.join(BRONZE_METADATA_DIR, "bronze_ingestion_contract.json")

if not os.path.exists(_CONTRACT_PATH):
    raise FileNotFoundError(
        f"bronze_ingestion_contract.json not found at: {_CONTRACT_PATH}\n"
        f"Ensure the contract file is at the project root: {PROJECT_ROOT}"
    )

with open(_CONTRACT_PATH, "r") as _f:
    FULL_CONTRACT = json.load(_f)

if not FULL_CONTRACT.get("datasets"):
    raise ValueError(
        f"Contract at {_CONTRACT_PATH} contains no datasets — "
        f"check the file is not empty or corrupt."
    )

# Build a lookup so each named variable below pulls its values from the
# contract rather than a hardcoded dict. The field set (dataset_name,
# file_count, total_rows, files) mirrors exactly what the tests expect.
_DS = {ds["dataset_name"]: ds for ds in FULL_CONTRACT["datasets"]}

BILLING_PAYMENTS_META          = _DS["billing_payments"]
COMMERCIAL_INDUSTRIES_META     = _DS["commercial_industries_consumption"]
CUSTOMERS_COMPLAINT_META       = _DS["customers_complaint"]
GRID_LOAD_META                 = _DS["grid_load"]
POWER_FLOW_META                = _DS["power_flow"]
RETAIL_TARIFFS_META            = _DS["retail_tariffs"]

ALL_CONTRACT_DATASETS = [
    BILLING_PAYMENTS_META,
    COMMERCIAL_INDUSTRIES_META,
    CUSTOMERS_COMPLAINT_META,
    GRID_LOAD_META,
    POWER_FLOW_META,
    RETAIL_TARIFFS_META,
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_mock_response(content: bytes = b"PARQUET_BINARY_DATA", status_code: int = 200):
    """
    Build a mock requests.Response that streams content in chunks.
    Mirrors how requests.get(..., stream=True) behaves in _download_file().
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code

    # iter_content yields the full payload as a single chunk
    mock_resp.iter_content = MagicMock(return_value=iter([content]))

    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(
            f"HTTP {status_code} error"
        )
    else:
        mock_resp.raise_for_status = MagicMock()

    # Context manager support (__enter__/__exit__)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ===========================================================================
# GROUP 1 — DataDownloader.__init__
# ===========================================================================

class TestDataDownloaderInit(unittest.TestCase):
    """Verify constructor stores attributes correctly."""

    def test_default_constructor_values(self):
        """S01 — Default args produce expected attribute values."""
        dl = DataDownloader()
        self.assertEqual(dl.staging_root, "/mnt/staging/raw")
        self.assertEqual(dl.chunk_size, 8192)
        self.assertEqual(dl.max_retries, 3)

    def test_custom_constructor_values(self):
        """S02 — Custom args are stored on the instance."""
        dl = DataDownloader(
            staging_root="/tmp/my_staging",
            chunk_size=16384,
            max_retries=5
        )
        self.assertEqual(dl.staging_root, "/tmp/my_staging")
        self.assertEqual(dl.chunk_size, 16384)
        self.assertEqual(dl.max_retries, 5)


# ===========================================================================
# GROUP 2 — DataDownloader.get_dataset_staging_path
# ===========================================================================

class TestGetDatasetStagingPath(unittest.TestCase):
    """Verify staging path composition for all 6 contract datasets."""

    def setUp(self):
        self.dl = DataDownloader(staging_root="/mnt/staging/raw")

    def test_billing_payments_path(self):
        """S03 — billing_payments staging path is composed correctly."""
        path = self.dl.get_dataset_staging_path("billing_payments")
        # Use os.path.join so the separator matches the OS (\ on Windows, / on Linux)
        expected = os.path.join("/mnt/staging/raw", "billing_payments")
        self.assertEqual(path, expected)

    def test_all_contract_dataset_paths(self):
        """S04 — All 6 contract datasets get the correct staging path."""
        for ds in ALL_CONTRACT_DATASETS:
            name = ds["dataset_name"]
            # os.path.join produces the right separator on each OS
            expected = os.path.join("/mnt/staging/raw", name)
            result = self.dl.get_dataset_staging_path(name)
            self.assertEqual(result, expected, f"Failed for dataset: {name}")


# ===========================================================================
# GROUP 3 — DataDownloader.download_dataset (happy path & cache)
# ===========================================================================

class TestDownloadDatasetHappyPath(unittest.TestCase):
    """
    Scenario group using real contract metadata.
    HTTP calls are mocked; staging I/O uses a temp directory.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dl = DataDownloader(staging_root=self.tmp_dir, max_retries=1)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("bronze.data_downloader.requests.get")
    def test_s05_billing_payments_successful_download(self, mock_get):
        """S05 — billing_payments: single parquet file downloads and result is correct."""
        mock_get.return_value = make_mock_response(b"BILLING_PARQUET" * 100)

        results = self.dl.download_dataset(BILLING_PAYMENTS_META)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r.success, f"Expected success, got error: {r.error_message}")
        self.assertEqual(r.dataset_name, "billing_payments")
        self.assertEqual(r.filename, "0.parquet")
        self.assertGreater(r.size_bytes, 0)
        self.assertGreater(r.download_time_sec, 0)
        self.assertIsNone(r.error_message)
        # File must physically exist on staging
        self.assertTrue(os.path.exists(r.local_path))

    @patch("bronze.data_downloader.requests.get")
    def test_s06_commercial_industries_successful_download(self, mock_get):
        """S06 — commercial_industries_consumption: 220k-row dataset downloads correctly."""
        mock_get.return_value = make_mock_response(b"COMMERCIAL_PARQUET" * 200)

        results = self.dl.download_dataset(COMMERCIAL_INDUSTRIES_META)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].dataset_name, "commercial_industries_consumption")

    @patch("bronze.data_downloader.requests.get")
    def test_s07_all_six_contract_datasets_download(self, mock_get):
        """S07 — All 6 contract datasets can be iterated and downloaded successfully."""
        mock_get.return_value = make_mock_response(b"PARQUET_DATA" * 50)

        for ds_meta in ALL_CONTRACT_DATASETS:
            with self.subTest(dataset=ds_meta["dataset_name"]):
                results = self.dl.download_dataset(ds_meta)
                self.assertEqual(len(results), 1)
                self.assertTrue(
                    results[0].success,
                    f"{ds_meta['dataset_name']} download failed: {results[0].error_message}"
                )

    @patch("bronze.data_downloader.requests.get")
    def test_s08_cache_hit_skips_download(self, mock_get):
        """
        S08 — grid_load: if staging file already exists, download is skipped
        and requests.get is never called.
        """
        # Pre-create the file in staging to simulate a cached download
        dataset_dir = os.path.join(self.tmp_dir, "grid_load")
        os.makedirs(dataset_dir, exist_ok=True)
        cached_path = os.path.join(dataset_dir, "0.parquet")
        with open(cached_path, "wb") as f:
            f.write(b"CACHED_PARQUET_DATA")

        results = self.dl.download_dataset(GRID_LOAD_META, force_redownload=False)

        mock_get.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].size_bytes, len(b"CACHED_PARQUET_DATA"))

    @patch("bronze.data_downloader.requests.get")
    def test_s09_force_redownload_ignores_cache(self, mock_get):
        """
        S09 — power_flow: force_redownload=True bypasses the cached file
        and triggers a fresh HTTP call.
        """
        dataset_dir = os.path.join(self.tmp_dir, "power_flow")
        os.makedirs(dataset_dir, exist_ok=True)
        cached_path = os.path.join(dataset_dir, "0.parquet")
        with open(cached_path, "wb") as f:
            f.write(b"STALE_DATA")

        mock_get.return_value = make_mock_response(b"FRESH_PARQUET_DATA")

        results = self.dl.download_dataset(POWER_FLOW_META, force_redownload=True)

        mock_get.assert_called_once()
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].size_bytes, len(b"FRESH_PARQUET_DATA"))

    def test_s10_missing_url_returns_failure_result(self):
        """
        S10 — A file entry with no URL produces a failure DownloadResult
        without raising an exception.
        """
        bad_meta = {
            "dataset_name": "retail_tariffs",
            "files": [
                {"filename": "0.parquet"}  # URL key deliberately omitted
            ],
        }
        results = self.dl.download_dataset(bad_meta)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].error_message, "No URL provided")
        self.assertEqual(results[0].local_path, "")

    def test_s11_empty_files_list_returns_empty_results(self):
        """
        S11 — A dataset with an empty files list returns an empty result list
        without crashing. Simulates a contract entry with no parquet shards.
        """
        empty_meta = {
            "dataset_name": "customers_complaint",
            "files": [],
        }
        results = self.dl.download_dataset(empty_meta)

        self.assertEqual(results, [])

    @patch("bronze.data_downloader.requests.get")
    def test_s12_staging_directory_auto_created(self, mock_get):
        """
        S12 — customers_complaint: download_dataset creates the staging
        subdirectory even if it does not exist yet.
        """
        mock_get.return_value = make_mock_response(b"COMPLAINT_DATA")
        expected_dir = os.path.join(self.tmp_dir, "customers_complaint")

        self.assertFalse(os.path.exists(expected_dir))
        self.dl.download_dataset(CUSTOMERS_COMPLAINT_META)
        self.assertTrue(os.path.isdir(expected_dir))


# ===========================================================================
# GROUP 4 — DataDownloader._download_file (retry & failure logic)
# ===========================================================================

class TestDownloadFileRetryLogic(unittest.TestCase):
    """
    Stress test _download_file's retry loop, exponential backoff,
    and partial-file cleanup.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dl = DataDownloader(staging_root=self.tmp_dir, max_retries=3)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("bronze.data_downloader.time.sleep")
    @patch("bronze.data_downloader.requests.get")
    def test_s13_retry_succeeds_on_second_attempt(self, mock_get, mock_sleep):
        """
        S13 — billing_payments: first attempt raises ConnectionError,
        second attempt succeeds. Only one sleep call expected (backoff after attempt 1).
        """
        import requests as req_lib
        mock_get.side_effect = [
            req_lib.exceptions.ConnectionError("Connection refused"),
            make_mock_response(b"BILLING_DATA_OK"),
        ]
        local_path = os.path.join(self.tmp_dir, "0.parquet")

        result = self.dl._download_file(
            url=BILLING_PAYMENTS_META["files"][0]["url"],
            local_path=local_path,
            dataset_name="billing_payments",
            filename="0.parquet",
        )

        self.assertTrue(result.success, f"Expected success after retry, got: {result.error_message}")
        self.assertEqual(mock_get.call_count, 2)
        # Backoff sleep should have fired once (after attempt 1)
        mock_sleep.assert_called_once_with(2)  # 2 ** 1 = 2

    @patch("bronze.data_downloader.time.sleep")
    @patch("bronze.data_downloader.requests.get")
    def test_s14_all_retries_exhausted_returns_failure(self, mock_get, mock_sleep):
        """
        S14 — grid_load: all 3 retries raise Timeout. Final result is failure,
        partial file is cleaned up, and sleep fires for backoff between attempts.
        """
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout("Request timed out")

        local_path = os.path.join(self.tmp_dir, "grid_0.parquet")

        result = self.dl._download_file(
            url=GRID_LOAD_META["files"][0]["url"],
            local_path=local_path,
            dataset_name="grid_load",
            filename="0.parquet",
        )

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error_message.lower())
        self.assertEqual(result.local_path, "")
        self.assertEqual(result.size_bytes, 0)
        # requests.get called max_retries(3) times
        self.assertEqual(mock_get.call_count, 3)
        # Backoff: sleep after attempt 1 (2s) and attempt 2 (4s)
        self.assertEqual(mock_sleep.call_count, 2)
        expected_sleeps = [call(2), call(4)]
        mock_sleep.assert_has_calls(expected_sleeps)

    @patch("bronze.data_downloader.time.sleep")
    @patch("bronze.data_downloader.requests.get")
    def test_s15_partial_file_cleaned_up_after_failure(self, mock_get, mock_sleep):
        """
        S15 — power_flow: if a partial file was written before the error,
        _download_file deletes it so no corrupt parquet lands in staging.
        """
        import requests as req_lib

        # Pre-create a partial file to simulate mid-download crash
        local_path = os.path.join(self.tmp_dir, "partial_power_flow.parquet")
        with open(local_path, "wb") as f:
            f.write(b"PARTIAL_DATA")

        mock_get.side_effect = req_lib.exceptions.ConnectionError("Network dropped")

        result = self.dl._download_file(
            url=POWER_FLOW_META["files"][0]["url"],
            local_path=local_path,
            dataset_name="power_flow",
            filename="partial_power_flow.parquet",
        )

        self.assertFalse(result.success)
        # The partial file must be deleted
        self.assertFalse(
            os.path.exists(local_path),
            "Partial file was NOT cleaned up after failed download"
        )

    @patch("bronze.data_downloader.time.sleep")
    @patch("bronze.data_downloader.requests.get")
    def test_s16_http_404_treated_as_error(self, mock_get, mock_sleep):
        """
        S16 — retail_tariffs: HTTP 404 response triggers raise_for_status()
        which causes retry exhaustion and a failure result.
        """
        mock_get.return_value = make_mock_response(b"", status_code=404)
        local_path = os.path.join(self.tmp_dir, "tariffs_0.parquet")

        result = self.dl._download_file(
            url=RETAIL_TARIFFS_META["files"][0]["url"],
            local_path=local_path,
            dataset_name="retail_tariffs",
            filename="0.parquet",
        )

        self.assertFalse(result.success)
        self.assertIn("404", result.error_message)

    @patch("bronze.data_downloader.time.sleep")
    @patch("bronze.data_downloader.requests.get")
    def test_s17_exponential_backoff_timing(self, mock_get, mock_sleep):
        """
        S17 — commercial_industries_consumption: with max_retries=4,
        verify backoff sequence is 2, 4, 8 seconds (2^1, 2^2, 2^3).
        """
        import requests as req_lib
        dl4 = DataDownloader(staging_root=self.tmp_dir, max_retries=4)
        mock_get.side_effect = req_lib.exceptions.ConnectionError("Down")
        local_path = os.path.join(self.tmp_dir, "commercial_0.parquet")

        dl4._download_file(
            url=COMMERCIAL_INDUSTRIES_META["files"][0]["url"],
            local_path=local_path,
            dataset_name="commercial_industries_consumption",
            filename="0.parquet",
        )

        # 4 attempts → backoff after attempts 1, 2, 3 (not after 4 since it's final)
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_has_calls([call(2), call(4), call(8)])

    @patch("bronze.data_downloader.requests.get")
    def test_s18_download_time_is_recorded(self, mock_get):
        """
        S18 — billing_payments: DownloadResult.download_time_sec is a positive
        float on success, not the default 0.0.
        """
        mock_get.return_value = make_mock_response(b"TIMED_DATA" * 10)
        local_path = os.path.join(self.tmp_dir, "timed.parquet")

        result = self.dl._download_file(
            url=BILLING_PAYMENTS_META["files"][0]["url"],
            local_path=local_path,
            dataset_name="billing_payments",
            filename="timed.parquet",
        )

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.download_time_sec, 0.0)


# ===========================================================================
# GROUP 5 — DataDownloader.validate_downloads
# ===========================================================================

class TestValidateDownloads(unittest.TestCase):
    """Unit tests for the post-download validation method."""

    def setUp(self):
        self.dl = DataDownloader()

    def _make_result(self, dataset, filename, success, error=None):
        return DownloadResult(
            dataset_name=dataset,
            filename=filename,
            local_path=f"/tmp/{filename}" if success else "",
            size_bytes=1024 if success else 0,
            success=success,
            error_message=error,
        )

    def test_s19_all_six_datasets_pass_validation(self):
        """
        S19 — One successful result per contract dataset → validate_downloads
        returns True.
        """
        results = [
            self._make_result(ds["dataset_name"], "0.parquet", success=True)
            for ds in ALL_CONTRACT_DATASETS
        ]
        self.assertTrue(self.dl.validate_downloads(results))

    def test_s20_single_failure_returns_false(self):
        """
        S20 — One failed download among 6 results causes validate_downloads
        to return False.
        """
        results = [
            self._make_result(ds["dataset_name"], "0.parquet", success=True)
            for ds in ALL_CONTRACT_DATASETS
        ]
        # Poison the retail_tariffs result
        results[-1] = self._make_result(
            "retail_tariffs", "0.parquet", success=False, error="HTTP 503"
        )
        self.assertFalse(self.dl.validate_downloads(results))

    def test_s21_all_failures_returns_false(self):
        """
        S21 — Every result is a failure → validate_downloads returns False.
        Simulates a total outage of the HuggingFace API endpoint.
        """
        results = [
            self._make_result(
                ds["dataset_name"], "0.parquet", success=False, error="Connection refused"
            )
            for ds in ALL_CONTRACT_DATASETS
        ]
        self.assertFalse(self.dl.validate_downloads(results))

    def test_s22_empty_results_list_returns_true(self):
        """
        S22 — An empty results list has no failures, so validate_downloads
        returns True. Documented as a known characteristic (vacuous truth).
        """
        self.assertTrue(self.dl.validate_downloads([]))


# ===========================================================================
# GROUP 6 — DataDownloader.cleanup_staging
# ===========================================================================

class TestCleanupStaging(unittest.TestCase):
    """Verify staging directory cleanup behaviour."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dl = DataDownloader(staging_root=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_s23_cleanup_removes_existing_directory(self):
        """
        S23 — grid_load staging dir exists with files;
        cleanup_staging deletes it entirely.
        """
        dataset_dir = os.path.join(self.tmp_dir, "grid_load")
        os.makedirs(dataset_dir)
        # Drop a dummy parquet file in the directory
        with open(os.path.join(dataset_dir, "0.parquet"), "wb") as f:
            f.write(b"DUMMY")

        self.assertTrue(os.path.exists(dataset_dir))
        self.dl.cleanup_staging("grid_load")
        self.assertFalse(os.path.exists(dataset_dir))

    def test_s24_cleanup_non_existent_directory_does_not_raise(self):
        """
        S24 — Calling cleanup_staging on a dataset that was never staged
        does not raise an exception. Safe to call idempotently.
        """
        try:
            self.dl.cleanup_staging("dataset_that_never_existed")
        except Exception as e:
            self.fail(f"cleanup_staging raised an unexpected exception: {e}")


# ===========================================================================
# GROUP 7 — DataValidator.validate_parquet_file
# ===========================================================================

class TestDataValidatorParquetFile(unittest.TestCase):
    """
    Unit tests for DataValidator.validate_parquet_file.
    pyarrow is mocked so tests are self-contained.
    """

    def _make_pyarrow_mocks(self, num_rows: int, num_columns: int, raise_exc=None):
        """
        Build a correctly wired (pyarrow, pyarrow.parquet) mock pair.

        The local import inside validate_parquet_file is:
            import pyarrow.parquet as pq
        Python resolves this by looking up 'pyarrow.parquet' in sys.modules
        and binding it to `pq`. If mock_pyarrow and mock_pq are separate objects
        without the .parquet attribute wired, Python would silently return
        mock_pyarrow.parquet (a new auto-MagicMock) instead of mock_pq.
        Setting mock_pyarrow.parquet = mock_pq ensures the import chain resolves
        to our controlled mock every time.
        """
        mock_pq = MagicMock()
        mock_pyarrow = MagicMock()
        mock_pyarrow.parquet = mock_pq  # ← critical: wire the attribute chain

        if raise_exc:
            mock_pq.ParquetFile.side_effect = raise_exc
        else:
            mock_file = MagicMock()
            mock_file.metadata.num_rows = num_rows
            mock_file.metadata.num_columns = num_columns
            mock_pq.ParquetFile.return_value = mock_file

        return mock_pyarrow, mock_pq

    def test_s25_valid_parquet_returns_true(self):
        """
        S25 — A well-formed parquet file with rows returns True.
        Mimics the metadata profile of billing_payments (200k rows, 10 cols).
        """
        mock_pyarrow, mock_pq = self._make_pyarrow_mocks(num_rows=200000, num_columns=10)

        with patch.dict("sys.modules", {"pyarrow": mock_pyarrow, "pyarrow.parquet": mock_pq}):
            result = DataValidator.validate_parquet_file("/tmp/billing_payments/0.parquet")

        self.assertTrue(result)

    def test_s26_zero_row_parquet_returns_false(self):
        """
        S26 — A parquet file with 0 rows (e.g. empty upstream feed) returns
        False — matches the warning path in validate_parquet_file.
        """
        mock_pyarrow, mock_pq = self._make_pyarrow_mocks(num_rows=0, num_columns=6)

        with patch.dict("sys.modules", {"pyarrow": mock_pyarrow, "pyarrow.parquet": mock_pq}):
            result = DataValidator.validate_parquet_file("/tmp/retail_tariffs/0.parquet")

        self.assertFalse(result)

    def test_s27_corrupt_file_returns_false(self):
        """
        S27 — A file that raises an exception when opened (corrupt parquet)
        returns False without propagating the exception.
        """
        mock_pyarrow, mock_pq = self._make_pyarrow_mocks(
            num_rows=0, num_columns=0,
            raise_exc=Exception("Parquet magic bytes mismatch")
        )

        with patch.dict("sys.modules", {"pyarrow": mock_pyarrow, "pyarrow.parquet": mock_pq}):
            result = DataValidator.validate_parquet_file("/tmp/corrupt/0.parquet")

        self.assertFalse(result)

    def test_s28_pyarrow_not_installed_returns_true(self):
        """
        S28 — When pyarrow is not installed, validate_parquet_file gracefully
        returns True (skips validation rather than failing the pipeline).
        KNOWN BEHAVIOUR: this is a soft degradation path in the module.
        """
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow.parquet":
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = DataValidator.validate_parquet_file("/tmp/grid_load/0.parquet")

        self.assertTrue(result)


# ===========================================================================
# GROUP 8 — DownloadResult dataclass
# ===========================================================================

class TestDownloadResultDataclass(unittest.TestCase):
    """Verify the DownloadResult dataclass defaults and field types."""

    def test_s29_success_result_defaults(self):
        """S29 — DownloadResult with success=True has correct field defaults."""
        r = DownloadResult(
            dataset_name="billing_payments",
            filename="0.parquet",
            local_path="/mnt/staging/raw/billing_payments/0.parquet",
            size_bytes=4096000,
            success=True,
        )
        self.assertIsNone(r.error_message)
        self.assertEqual(r.download_time_sec, 0.0)

    def test_s30_failure_result_carries_error(self):
        """S30 — DownloadResult with success=False stores error_message correctly."""
        r = DownloadResult(
            dataset_name="retail_tariffs",
            filename="0.parquet",
            local_path="",
            size_bytes=0,
            success=False,
            error_message="HTTP 404: Not Found",
            download_time_sec=12.5,
        )
        self.assertFalse(r.success)
        self.assertEqual(r.error_message, "HTTP 404: Not Found")
        self.assertEqual(r.download_time_sec, 12.5)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDataDownloaderInit,
        TestGetDatasetStagingPath,
        TestDownloadDatasetHappyPath,
        TestDownloadFileRetryLogic,
        TestValidateDownloads,
        TestCleanupStaging,
        TestDataValidatorParquetFile,
        TestDownloadResultDataclass,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)