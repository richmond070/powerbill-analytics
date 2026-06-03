import sys
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..dag_config.default_args import BRONZE_DEFAULT_ARGS
from ..dag_config.schedules import SCHEDULES
from ..dag_config.environment import validate_environment

from ..utils.contract_validators import validate_bronze_contract
from ..utils.quality_checks import bronze_quality_gate
from ..utils.observability_checks import validate_audit_metrics

# Convenience constants passed as op_kwargs
BRONZE_CONTRACT_PATH = str(
    PROJECT_ROOT / "bronze"/"bronze_metadata" / "bronze_ingestion_contract.json"
)

with DAG(
    dag_id="bronze_ingestion",
    schedule=SCHEDULES["bronze_ingestion_dag"],
    start_date=days_ago(1),
    catchup=False,
    default_args=BRONZE_DEFAULT_ARGS,
    tags=["bronze", "ingestion"],
    # params makes project_root and contract_path visible + overrideable
    # in the Airflow UI for manual test runs
    params={
        "project_root":        str(PROJECT_ROOT),
        "bronze_contract_path": BRONZE_CONTRACT_PATH,
    },
    description=(
        "Bronze layer: extract HuggingFace metadata, validate contract, "
        "ingest into Delta tables, quality gate, confirm observability"
    ),
) as dag:

    def _extract_metadata(project_root: str = None, **kwargs):
        from extraction.runner import run_bronze_ingestion
        run_bronze_ingestion()

    extract_metadata = PythonOperator(
        task_id="extract_metadata",
        python_callable=_extract_metadata,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
    )

    validate_contract = PythonOperator(
        task_id="validate_contract",
        python_callable=validate_bronze_contract,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
    )

    def _run_bronze_ingestion(project_root: str = None, **kwargs):
        from runners.run_bronze import main as run_bronze_main
        return run_bronze_main(
            project_root=project_root,
            datasets=None,   # all 6 datasets
            download=True,
            optimize=False,
            dry_run=False,
        )

    run_bronze_ingestion = PythonOperator(
        task_id="run_bronze_ingestion",
        python_callable=_run_bronze_ingestion,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
    )

    quality_gate = PythonOperator(
        task_id="bronze_quality_gate",
        python_callable=bronze_quality_gate,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
    )

    audit_check = PythonOperator(
        task_id="validate_audit_metrics",
        python_callable=validate_audit_metrics,
        op_kwargs={"project_root": str(PROJECT_ROOT)},
        # provide_context=True is the Airflow 1.x way;
        # in Airflow 2.x, **kwargs in the callable receives context automatically
    )

    (
        extract_metadata
        >> validate_contract
        >> run_bronze_ingestion
        >> quality_gate
        >> audit_check
    )