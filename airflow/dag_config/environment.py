import os
from pathlib import Path
from typing import Dict

def _resolve_project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # airflow/config/environment.py is two levels below project root
    return Path(__file__).parent.parent.parent.resolve()


PROJECT_ROOT = _resolve_project_root()

BRONZE_CONTRACT_PATH = str(
    PROJECT_ROOT / "bronze"/"bronze_metadata" / "bronze_ingestion_contract.json"
)

SILVER_CONTRACT_PATH = str(
    PROJECT_ROOT / "silver"/"silver_contract.json"
)

DATABRICKS_CONFIG_PATH = str(
    PROJECT_ROOT / "databricks" / "databricks.cfg"
)

DBT_PROJECT_DIR = str(
    PROJECT_ROOT / "dbt_project"
)

def get_databricks_config() -> Dict[str, str]:
    return {
        "workspace_url": os.getenv("DATABRICKS_WORKSPACE_URL", ""),
        "token":         os.getenv("DATABRICKS_TOKEN", ""),
        "warehouse_id":  os.getenv("DATABRICKS_WAREHOUSE_ID", ""),
        "catalog":       os.getenv("DATABRICKS_CATALOG", "main"),
        "schema":        os.getenv("DATABRICKS_SCHEMA", "bronze"),
    }


def get_postgres_config() -> Dict[str, str]:
    return {
        "host":     os.getenv("PG_HOST", "postgres"),
        "port":     os.getenv("PG_PORT", "5432"),
        "dbname":   os.getenv("PG_DB", "bronze_control"),
        "user":     os.getenv("PG_USER", ""),
        "password": os.getenv("PG_PASSWORD", ""),
    }


def validate_environment() -> bool:
    required_vars = [
        "DATABRICKS_WORKSPACE_URL",
        "DATABRICKS_TOKEN",
        "DATABRICKS_WAREHOUSE_ID",
        "PG_HOST",
        "PG_USER",
        "PG_PASSWORD",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]

    if missing:
        raise ValueError(
            f"Required environment variables not set: {missing}. "
            "Check your .env file and docker-compose environment blocks."
        )

    return True