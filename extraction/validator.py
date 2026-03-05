import pyarrow.parquet as pq
import io
import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ParquetValidator:
    """
    Lightweight validation of remote parquet files before Bronze ingestion.
    Validates schema and basic metadata without downloading entire file.
    """

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def validate_remote_parquet(self, url: str, filename: str = None) -> Dict:
        """
        Validates a remote parquet file by downloading and inspecting metadata.
        
        Args:
            url: Direct URL to parquet file
            filename: Optional name of the file for logging
            
        Returns:
            Dictionary containing validation metadata including:
            - filename: Name of the file
            - url: Source URL
            - num_rows: Total number of rows
            - num_columns: Total number of columns
            - num_row_groups: Number of row groups
            - columns: List of column metadata (name, type, nullable)
            - validation_status: 'success' or 'failed'
            - error: Error message if validation failed
        """
        if filename is None:
            filename = url.split('/')[-1]
            
        logger.info(f"Validating: {filename}")
        
        try:
            # Download with streaming to handle large files
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            # Load into memory for PyArrow inspection
            buffer = io.BytesIO(response.content)
            parquet_file = pq.ParquetFile(buffer)

            schema = parquet_file.schema.to_arrow_schema()
            
            # Extract column information
            columns = []
            for field in schema:
                columns.append({
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable
                })

            metadata = {
                "filename": filename,
                "url": url,
                "num_rows": parquet_file.metadata.num_rows,
                "num_columns": parquet_file.metadata.num_columns,
                "num_row_groups": parquet_file.metadata.num_row_groups,
                "columns": columns,
                "validation_status": "success"
            }
            
            logger.info(
                f"✓ Validated {filename}: "
                f"{metadata['num_rows']:,} rows, "
                f"{metadata['num_columns']} columns"
            )
            
            return metadata

        except Exception as e:
            logger.error(f"✗ Validation failed for {filename}: {str(e)}")
            return {
                "filename": filename,
                "url": url,
                "validation_status": "failed",
                "error": str(e)
            }