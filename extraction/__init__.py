"""
Nigerian Energy & Utilities Data Ingestion Package

This package handles the ingestion of energy sector datasets from HuggingFace
into a local bronze metadata contract for Databricks SQL consumption.
"""

from .resolver import HuggingFaceDatasetResolver
from .validator import ParquetValidator

__version__ = "1.0.0"

__all__ = [
    "HuggingFaceDatasetResolver",
    "ParquetValidator",
]