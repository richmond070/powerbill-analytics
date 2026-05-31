import sys
from pathlib import Path

from airflow.models import BaseOperator
from airflow.exceptions import AirflowException

# Resolve project root so airflow/utils is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
AIRFLOW_ROOT  = Path(__file__).parent.parent.resolve()   # airflow/

# Add airflow/ dir to path as 'pipeline_airflow' avoids namespace clash
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AIRFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(AIRFLOW_ROOT))

# Import from utils/ directly using the resolved path
from utils.dbt_helpers import run_dbt_command, validate_dbt_run


class DbtRunOperator(BaseOperator):
    ui_color = "#FF6B35"   # Orange — matches dbt brand colour

    def __init__(
        self,
        dbt_command: str = "run",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dbt_command = dbt_command

    def execute(self, context):
        self.log.info(
            f"[{self.task_id}] Running: dbt {self.dbt_command}"
        )

        try:
            exit_code, stdout, stderr = run_dbt_command(
                command=self.dbt_command,
                task_id=self.task_id,
            )
            validate_dbt_run(exit_code, stdout, stderr)

        except Exception as exc:
            self.log.error(
                f"[{self.task_id}] dbt {self.dbt_command} failed: {exc}"
            )
            raise AirflowException(
                f"dbt {self.dbt_command} failed: {exc}"
            ) from exc

        self.log.info(
            f"[{self.task_id}] dbt {self.dbt_command} completed successfully"
        )
        return {"status": "success", "command": self.dbt_command}