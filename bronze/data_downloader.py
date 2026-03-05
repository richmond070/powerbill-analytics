"""
Data Downloader
Downloads raw data files from URLs to staging area
Handles retries, validation, and batching for large datasets
"""

import os
import requests
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import time


@dataclass
class DownloadResult:
    """Result of file download"""
    dataset_name: str
    filename: str
    local_path: str
    size_bytes: int
    success: bool
    error_message: Optional[str] = None
    download_time_sec: float = 0.0


class DataDownloader:
    """
    Downloads raw data files to staging area
    Python orchestrates, does not process data
    """
    
    def __init__(
        self,
        staging_root: str = "/mnt/staging/raw",
        chunk_size: int = 8192,
        max_retries: int = 3
    ):
        """
        Initialize data downloader
        
        Args:
            staging_root: Root directory for staging files
            chunk_size: Download chunk size in bytes
            max_retries: Maximum download retry attempts
        """
        self.staging_root = staging_root
        self.chunk_size = chunk_size
        self.max_retries = max_retries
    
    def download_dataset(
        self,
        dataset_metadata: Dict,
        force_redownload: bool = False
    ) -> List[DownloadResult]:
        """
        Download all files for a dataset
        
        Args:
            dataset_metadata: Dataset metadata from contract
            force_redownload: Re-download even if file exists
            
        Returns:
            List of DownloadResult for each file
        """
        dataset_name = dataset_metadata['dataset_name']
        files = dataset_metadata.get('files', [])
        
        # Create dataset staging directory
        dataset_dir = os.path.join(self.staging_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        results = []
        
        print(f"\n Downloading dataset: {dataset_name}")
        print(f"   Files to download: {len(files)}")
        print(f"   Staging directory: {dataset_dir}")
        
        for i, file_meta in enumerate(files, 1):
            url = file_meta.get('url')
            filename = file_meta.get('filename', f"file_{i}.parquet")
            
            if not url:
                results.append(DownloadResult(
                    dataset_name=dataset_name,
                    filename=filename,
                    local_path="",
                    size_bytes=0,
                    success=False,
                    error_message="No URL provided"
                ))
                continue
            
            local_path = os.path.join(dataset_dir, filename)
            
            # Skip if already exists and not forcing redownload
            if os.path.exists(local_path) and not force_redownload:
                file_size = os.path.getsize(local_path)
                print(f"   [{i}/{len(files)}]  {filename} (cached, {file_size:,} bytes)")
                results.append(DownloadResult(
                    dataset_name=dataset_name,
                    filename=filename,
                    local_path=local_path,
                    size_bytes=file_size,
                    success=True
                ))
                continue
            
            # Download file
            print(f"   [{i}/{len(files)}] ⬇ {filename}...", end=" ")
            result = self._download_file(url, local_path, dataset_name, filename)
            results.append(result)
            
            if result.success:
                print(f" ({result.size_bytes:,} bytes, {result.download_time_sec:.1f}s)")
            else:
                print(f"{result.error_message}")
        
        success_count = sum(1 for r in results if r.success)
        print(f"\n   Downloaded: {success_count}/{len(files)} files")
        
        return results
    
    def _download_file(
        self,
        url: str,
        local_path: str,
        dataset_name: str,
        filename: str
    ) -> DownloadResult:
        """
        Download a single file with retry logic
        
        Args:
            url: URL to download from
            local_path: Local path to save file
            dataset_name: Dataset name
            filename: Filename
            
        Returns:
            DownloadResult
        """
        start_time = time.time()
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Stream download to handle large files
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                # Write to file in chunks
                total_size = 0
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                download_time = time.time() - start_time
                
                return DownloadResult(
                    dataset_name=dataset_name,
                    filename=filename,
                    local_path=local_path,
                    size_bytes=total_size,
                    success=True,
                    download_time_sec=download_time
                )
                
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    # All retries failed
                    if os.path.exists(local_path):
                        os.remove(local_path)  # Clean up partial download
                    
                    return DownloadResult(
                        dataset_name=dataset_name,
                        filename=filename,
                        local_path="",
                        size_bytes=0,
                        success=False,
                        error_message=str(e),
                        download_time_sec=time.time() - start_time
                    )
    
    def validate_downloads(self, results: List[DownloadResult]) -> bool:
        """
        Validate all downloads completed successfully
        
        Args:
            results: List of download results
            
        Returns:
            True if all downloads succeeded
        """
        failed = [r for r in results if not r.success]
        
        if failed:
            print(f"\n {len(failed)} download(s) failed:")
            for r in failed:
                print(f"   - {r.filename}: {r.error_message}")
            return False
        
        print(f"\n All {len(results)} file(s) downloaded successfully")
        return True
    
    def get_dataset_staging_path(self, dataset_name: str) -> str:
        """Get staging path for a dataset"""
        return os.path.join(self.staging_root, dataset_name)
    
    def cleanup_staging(self, dataset_name: str):
        """
        Clean up staging files for a dataset
        
        Args:
            dataset_name: Dataset name to clean up
        """
        dataset_dir = os.path.join(self.staging_root, dataset_name)
        
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            print(f" Cleaned up staging directory: {dataset_dir}")


class DataValidator:
    """
    Validates downloaded data files
    Checks file integrity and basic structure
    """
    
    @staticmethod
    def validate_parquet_file(file_path: str) -> bool:
        """
        Validate a parquet file can be read
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            True if valid
        """
        try:
            # Try to read with pyarrow (doesn't load data, just metadata)
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(file_path)
            num_rows = parquet_file.metadata.num_rows
            num_cols = parquet_file.metadata.num_columns
            
            if num_rows == 0:
                print(f"   ⚠ Warning: {file_path} has 0 rows")
                return False
            
            return True
            
        except ImportError:
            # pyarrow not available, skip validation
            print("   Warning: pyarrow not available for validation")
            return True
        except Exception as e:
            print(f"    Validation failed: {e}")
            return False