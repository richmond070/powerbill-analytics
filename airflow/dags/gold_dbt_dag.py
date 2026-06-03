import os
import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..dag_config.default_args import GOLD_DEFAULT_ARGS
from ..dag_config.schedules import GOLD_SCHEDULE
from ..dag_config.environment import validate_environment, DBT_PROJECT_DIR
from ..operators.dbt_operator import DbtRunOperator

with DAG(
    dag_id="gold_dbt_dag",
    default_args=GOLD_DEFAULT_ARGS,
    schedule=GOLD_SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    tags=["gold", "analytics", "dbt"],
    description=(
        "Gold layer: execute 9 dbt models (4 staging views + 5 mart tables) "
        "then run schema and data tests"
    ),
) as dag:
    def export_dbt_environment(**kwargs):
        validate_environment()

        os.environ["DBT_PROFILES_DIR"] = DBT_PROJECT_DIR

        print(f"DBT_PROFILES_DIR         = {DBT_PROJECT_DIR}")
        print(f"DATABRICKS_WORKSPACE_URL = {os.getenv('DATABRICKS_WORKSPACE_URL', '[NOT SET]')}")
        print(f"DATABRICKS_WAREHOUSE_ID  = {os.getenv('DATABRICKS_WAREHOUSE_ID', '[NOT SET]')}")

        return {"dbt_profiles_dir": DBT_PROJECT_DIR}

    export_env_task = PythonOperator(
        task_id="export_dbt_environment",
        python_callable=export_dbt_environment,
    )

    dbt_run_task = DbtRunOperator(
        task_id="dbt_run",
        dbt_command="run",
    )

    dbt_test_task = DbtRunOperator(
        task_id="dbt_test",
        dbt_command="test",
    )

    export_env_task >> dbt_run_task >> dbt_test_task