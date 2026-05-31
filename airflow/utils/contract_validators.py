import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Required top-level keys in the contract
_REQUIRED_CONTRACT_KEYS = {"generated_at", "datasets"}

# Required keys on each dataset entry
_REQUIRED_DATASET_KEYS = {
    "dataset_name",
    "api_endpoint",
    "file_count",
    "total_rows",
    "files",
}

# Required keys on each file entry inside a dataset
_REQUIRED_FILE_KEYS = {
    "url",
    "filename",
    "num_rows",
    "num_columns",
    "columns",
    "validation_status",
}

# All 6 expected datasets from bronze_ingestion_contract.json
EXPECTED_DATASETS = {
    "billing_payments",
    "commercial_industries_consumption",
    "customers_complaint",
    "grid_load",
    "power_flow",
    "retail_tariffs",
}


def validate_bronze_contract(
    project_root: str = None,
    max_age_hours: int = 24,
    **kwargs,    # absorbs Airflow context kwargs
) -> dict:
   
    root = _resolve_root(project_root)
    contract_path = root / "bronze_metadata" / "bronze_ingestion_contract.json"

    logger.info(f"Validating contract: {contract_path}")

    # ------------------------------------------------------------------
    # Check 1: File exists
    # ------------------------------------------------------------------
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Bronze contract not found at {contract_path}. "
            "Run extract_metadata task first."
        )

    # ------------------------------------------------------------------
    # Check 2: Valid JSON
    # ------------------------------------------------------------------
    try:
        with open(contract_path, "r") as f:
            contract = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Contract is not valid JSON: {exc}") from exc

    # ------------------------------------------------------------------
    # Check 3: Top-level keys present
    # ------------------------------------------------------------------
    missing_keys = _REQUIRED_CONTRACT_KEYS - set(contract.keys())
    if missing_keys:
        raise ValueError(
            f"Contract missing required top-level keys: {missing_keys}"
        )

    # ------------------------------------------------------------------
    # Check 4: Contract is recent enough
    # ------------------------------------------------------------------
    try:
        generated_at = datetime.fromisoformat(
            contract["generated_at"].replace("Z", "+00:00")
        )
        age = datetime.now(timezone.utc) - generated_at
        if age > timedelta(hours=max_age_hours):
            raise ValueError(
                f"Contract is {age.total_seconds() / 3600:.1f}h old "
                f"(max allowed: {max_age_hours}h). "
                "Re-run extract_metadata to refresh it."
            )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Cannot parse generated_at timestamp: {exc}") from exc

    # ------------------------------------------------------------------
    # Check 5: datasets list is non-empty
    # ------------------------------------------------------------------
    datasets = contract.get("datasets", [])
    if not datasets:
        raise ValueError("Contract contains no datasets.")

    # ------------------------------------------------------------------
    # Check 6: All expected datasets are present
    # ------------------------------------------------------------------
    present = {d.get("dataset_name") for d in datasets}
    missing_datasets = EXPECTED_DATASETS - present
    if missing_datasets:
        raise ValueError(
            f"Contract is missing expected datasets: {missing_datasets}. "
            f"Present: {present}"
        )

    # ------------------------------------------------------------------
    # Check 7: Per-dataset validation
    # ------------------------------------------------------------------
    failed_datasets = []

    for ds in datasets:
        name = ds.get("dataset_name", "<unknown>")

        # Required keys on the dataset entry
        missing_ds_keys = _REQUIRED_DATASET_KEYS - set(ds.keys())
        if missing_ds_keys:
            raise ValueError(
                f"Dataset '{name}' missing required keys: {missing_ds_keys}"
            )

        # Must have at least one file
        files = ds.get("files", [])
        if not files:
            raise ValueError(
                f"Dataset '{name}' has no files. "
                "ParquetValidator may have failed for all files."
            )

        # Per-file validation
        for i, file_entry in enumerate(files):
            missing_file_keys = _REQUIRED_FILE_KEYS - set(file_entry.keys())
            if missing_file_keys:
                raise ValueError(
                    f"Dataset '{name}' file[{i}] missing keys: {missing_file_keys}"
                )

            # Check validation_status set by extraction/validator.py
            if file_entry.get("validation_status") == "failed":
                failed_datasets.append(
                    f"{name}/{file_entry.get('filename', f'file[{i}]')}: "
                    f"{file_entry.get('error', 'unknown error')}"
                )

            # At least one column must be present
            if not file_entry.get("columns"):
                raise ValueError(
                    f"Dataset '{name}' file[{i}] has no columns defined."
                )

            # num_rows must be positive
            if file_entry.get("num_rows", 0) == 0:
                logger.warning(
                    f"Dataset '{name}' file[{i}] reports 0 rows. "
                    "This may indicate an empty source feed."
                )

    # Fail if any file failed ParquetValidator
    if failed_datasets:
        raise ValueError(
            f"Contract contains {len(failed_datasets)} file(s) with "
            f"validation_status='failed':\n" + "\n".join(failed_datasets)
        )

    # ------------------------------------------------------------------
    # All checks passed
    # ------------------------------------------------------------------
    dataset_names = [d["dataset_name"] for d in datasets]
    logger.info(
        f"Contract valid — {len(datasets)} datasets, "
        f"generated at {contract['generated_at']}"
    )

    return {
        "contract_path": str(contract_path),
        "generated_at":  contract["generated_at"],
        "dataset_count": len(datasets),
        "datasets":      dataset_names,
        "status":        "valid",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_root(project_root: Optional[str]) -> Path:
    """Resolve project root using the same priority as run_bronze.py."""
    import os
    if project_root:
        return Path(project_root).resolve()
    if os.getenv("PROJECT_ROOT"):
        return Path(os.getenv("PROJECT_ROOT")).resolve()
    # airflow/utils/contract_validators.py → airflow/ → project root
    return Path(__file__).parent.parent.parent.resolve()