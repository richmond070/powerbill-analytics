import requests
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HuggingFaceDatasetResolver:
    """
    Resolves Hugging Face dataset API endpoints
    into concrete Parquet file URLs that can be ingested by Databricks SQL.
    
    The HuggingFace dataset URLs are logical directory endpoints that contain
    metadata about the actual parquet files. This resolver queries that metadata
    and extracts the direct download URLs for each parquet file.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def resolve(self, dataset_url: str) -> List[Dict[str, str]]:
        """
        Queries Hugging Face dataset metadata API and extracts
        all parquet file URLs with their metadata.

        Args:
            dataset_url: HuggingFace dataset API endpoint
                        (e.g., https://huggingface.co/api/datasets/.../parquet/default/train)

        Returns:
            List of dicts containing file metadata:
            [
                {
                    "url": "https://huggingface.co/.../train-00000.parquet",
                    "filename": "train-00000.parquet",
                    "size_bytes": 12345
                },
                ...
            ]
            
        Raises:
            ValueError: If the API response format is unexpected or no parquet files found
            requests.HTTPError: If the API request fails
        """
        logger.info(f"Resolving dataset: {dataset_url}")
        
        response = requests.get(dataset_url, timeout=self.timeout)
        response.raise_for_status()

        payload = response.json()

        # Handle different response formats from HuggingFace API
        parquet_files = []
        
        if isinstance(payload, list):
            # Check if it's a list of strings (URLs) or list of dicts (file metadata)
            if payload and isinstance(payload[0], str):
                # Direct list of parquet URLs: ["https://...file1.parquet", "https://...file2.parquet"]
                for url in payload:
                    if url.endswith(".parquet"):
                        parquet_files.append({
                            "url": url,
                            "filename": url.split("/")[-1],
                            "size_bytes": 0  # Size not provided in this format
                        })
            elif payload and isinstance(payload[0], dict):
                # List of file metadata dicts: [{"path": "...", "url": "...", "size": ...}, ...]
                for file_info in payload:
                    if file_info.get("path", "").endswith(".parquet"):
                        parquet_files.append({
                            "url": file_info["url"],
                            "filename": file_info["path"].split("/")[-1],
                            "size_bytes": file_info.get("size", 0)
                        })
        elif isinstance(payload, dict) and "files" in payload:
            # Dictionary with 'files' key: {"files": [{...}, {...}]}
            for file_info in payload["files"]:
                if file_info.get("path", "").endswith(".parquet"):
                    parquet_files.append({
                        "url": file_info["url"],
                        "filename": file_info["path"].split("/")[-1],
                        "size_bytes": file_info.get("size", 0)
                    })
        else:
            raise ValueError(
                f"Unexpected HuggingFace response format. "
                f"Expected list or dict with 'files' key. "
                f"Got type: {type(payload)}"
            )

        if not parquet_files:
            raise ValueError(f"No parquet files found in dataset: {dataset_url}")

        logger.info(f"Found {len(parquet_files)} parquet file(s)")
        return parquet_files