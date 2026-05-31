import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow.config.default_args import SILVER_DEFAULT_ARGS
from airflow.config.schedules import SILVER_SCHEDULE
from airflow.config.environment import SILVER_CONTRACT_PATH
from airflow.operators.silver_operator import SilverTransformationOperator

# Silver datasets from silver_contract.json — 3 datasets only
SILVER_DATASETS =str(
    PROJECT_ROOT / "silver"/"silver_contract.json"
)

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="silver_transformation_dag",
    default_args=SILVER_DEFAULT_ARGS,
    schedule=SILVER_SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    tags=["silver", "transformation"],
    params={
        "project_root":          str(PROJECT_ROOT),
        "silver_contract_path":  SILVER_CONTRACT_PATH,
    },
    description=(
        "Silver layer: clean, deduplicate, and enrich bronze data; "
        "executes 3 transformer SQL files via Databricks SQL API"
    ),
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Run silver transformations
    # ------------------------------------------------------------------
    run_silver_transformations = SilverTransformationOperator(
        task_id="run_silver_transformations",
        contract_path=SILVER_CONTRACT_PATH,
    )

    # ------------------------------------------------------------------
    # Task 2: Confirm clean completion
    # ------------------------------------------------------------------
    def validate_silver_completion(**kwargs):
        """
        Log completed datasets for operational visibility.
        Reaching this task confirms the orchestrator did not raise.
        """
        print(
            f"Silver layer completed successfully. "
            f"Datasets: {', '.join(SILVER_DATASETS)}"
        )
        return {"status": "success", "datasets": SILVER_DATASETS}

    validate_silver_completion_task = PythonOperator(
        task_id="validate_silver_completion",
        python_callable=validate_silver_completion,
    )

    # ------------------------------------------------------------------
    # Dependency chain
    # ------------------------------------------------------------------
    run_silver_transformations >> validate_silver_completion_task