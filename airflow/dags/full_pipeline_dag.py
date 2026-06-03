import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..dag_config.default_args import DEFAULT_ARGS
from ..dag_config.schedules import PIPELINE_SCHEDULE


with DAG(
    dag_id="full_pipeline_dag",
    default_args=DEFAULT_ARGS,
    schedule=PIPELINE_SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    tags=["orchestration", "pipeline", "master"],
    description=(
        "Master pipeline: waits for Bronze completion then triggers "
        "Silver → Gold in sequence"
    ),
) as dag:
    wait_bronze = ExternalTaskSensor(
        task_id="wait_for_bronze_completion",
        external_dag_id="bronze_ingestion",          # matches dag_id in bronze DAG
        external_task_id="validate_audit_metrics",   # final task in bronze DAG
        timeout=7200,        # 2 hours max wait
        mode="poke",
        poke_interval=60,    # check every 60 seconds
    )

    trigger_silver = TriggerDagRunOperator(
        task_id="trigger_silver_dag",
        trigger_dag_id="silver_transformation_dag",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=30,
    )

    trigger_gold = TriggerDagRunOperator(
        task_id="trigger_gold_dag",
        trigger_dag_id="gold_dbt_dag",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=30,
    )
    wait_bronze >> trigger_silver >> trigger_gold