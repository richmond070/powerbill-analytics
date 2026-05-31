import sys
from pathlib import Path

from airflow.models import BaseOperator
from airflow.exceptions import AirflowException

# Add project root so silver package is importable inside Airflow Docker
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from silver.silver_orchestrator import SilverOrchestrator


class SilverTransformationOperator(BaseOperator):
    ui_color = "#C0C0C0"   # Silver — visible in Airflow graph view

    def __init__(
        self,
        contract_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contract_path = contract_path

    def execute(self, context):
        """
        Instantiate SilverOrchestrator and run the transformation pipeline.

        Raises AirflowException on any failure so Airflow retries
        according to the retry policy in default_args.py.
        """
        self.log.info(
            f"[{self.task_id}] Starting silver transformations | "
            f"contract={self.contract_path}"
        )

        try:
            orchestrator = SilverOrchestrator(
                contract_path=self.contract_path
            )
            orchestrator.run()

        except Exception as exc:
            self.log.error(f"[{self.task_id}] Silver transformation failed: {exc}")
            raise AirflowException(f"Silver transformation failed: {exc}") from exc

        self.log.info(
            f"[{self.task_id}] Silver transformations completed successfully"
        )
        return {"status": "success"}