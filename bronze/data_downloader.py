"""
Data Downloader
Downloads raw data files from URLs to staging area
Handles retries, validation, and batching for large datasets
"""

import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class DownloadResult:
    """Result of file download"""

    dataset_name: str
    filename: str
    volume_path: str
    success: bool
    size_bytes: int = 0
    error_message: Optional[str] = None
    download_time_sec: float = 0.0


class DataDownloader:
    """
    Downloads raw data files to staging area
    Python orchestrates, does not process data
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize data downloader
        """
        self.max_retries = max_retries

    def resolve_dataset_urls(self, dataset_metadata: Dict) -> List[DownloadResult]:
        """
        Resolves HuggingFace parquet URLs for a dataset.
        No bytes are downloaded locally. Each URL is verified reachable
        via HTTP HEAD. Databricks SQL reads directly from these URLs.
        """
        dataset_name = dataset_metadata["dataset_name"]
        files = dataset_metadata.get("files", [])
        results = []

        print(f"\n>> Resolving URLs: {dataset_name} ({len(files)} file(s))")

        for i, file_meta in enumerate(files, 1):
            url = file_meta.get("url")
            filename = file_meta.get("filename", f"file_{i}.parquet")

            if not url:
                results.append(DownloadResult(
                    dataset_name=dataset_name,
                    filename=filename,
                    volume_path="",
                    size_bytes=0,
                    success=False,
                    error_message="No URL in contract",
                ))
                continue

            result = self._verify_url(url, dataset_name, filename)
            results.append(result)
            status = "OK" if result.success else f"FAILED: {result.error_message}"
            print(f"   [{i}/{len(files)}] {filename} -> {status}")

        success_count = sum(1 for r in results if r.success)
        print(f"   Resolved: {success_count}/{len(files)} URLs")
        return results

    def _verify_url(self, url: str, dataset_name: str, filename: str) -> DownloadResult:
        """
        Verifies a remote parquet URL is reachable via HTTP HEAD.
        No data is transferred. Returns the URL as volume_path on success.
        """
        start_time = time.time()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.head(url, timeout=15, allow_redirects=True)
                response.raise_for_status()

                size_bytes = int(response.headers.get("Content-Length", 0))

                return DownloadResult(
                    dataset_name=dataset_name,
                    filename=filename,
                    volume_path=url,          # SQL reads from this URL directly
                    size_bytes=size_bytes,
                    success=True,
                    download_time_sec=time.time() - start_time,
                )

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return DownloadResult(
                    dataset_name=dataset_name,
                    filename=filename,
                    volume_path="",
                    size_bytes=0,
                    success=False,
                    error_message=str(e),
                    download_time_sec=time.time() - start_time,
                )

    def validate_urls(self, results: List[DownloadResult]) -> bool:
        """
        Validate all downloads completed successfully

        Args:
            results: List of download results

        Returns:
            True if all downloads succeeded
        """
        failed = [r for r in results if not r.success]

        if failed:
            print(f"\n {len(failed)} URL(s) could not be verified::")
            for r in failed:
                print(f"   - {r.filename}: {r.error_message}")
            return False

        print(f"\n All {len(results)} URL(s) verified successfully")
        return True