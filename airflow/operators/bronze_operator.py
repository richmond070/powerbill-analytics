import sys
from pathlib import Path
from typing import List, Optional

from airflow.models import BaseOperator
from airflow.exceptions import AirflowException

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bronze.bronze_orchestrator import BronzeLayerOrchestrator


class BronzeIngestionOperator(BaseOperator):
    ui_color = "#73C6FF"   # Light blue

    def __init__(
        self,
        contract_path: str,
        config_path: str,
        catalog: str = "main",
        schema: str = "bronze",
        datasets: Optional[List[str]] = None,
        download: bool = True,
        optimize: bool = False,
        dry_run: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contract_path = contract_path
        self.config_path   = config_path
        self.catalog       = catalog
        self.schema        = schema
        self.datasets      = datasets
        self.download      = download
        self.optimize      = optimize
        self.dry_run       = dry_run

    def execute(self, context):
        self.log.info(
            f"[{self.task_id}] Starting bronze ingestion | "
            f"catalog={self.catalog} schema={self.schema} "
            f"datasets={self.datasets or 'ALL'} dry_run={self.dry_run}"
        )

        try:
            orchestrator = BronzeLayerOrchestrator(
                contract_path=self.contract_path,
                config_path=self.config_path,
                catalog=self.catalog,
                schema=self.schema,
            )
            orchestrator.run_full_pipeline(
                datasets=self.datasets,
                download=self.download,
                optimize=self.optimize,
                dry_run=self.dry_run,
            )

        except Exception as exc:
            self.log.error(f"[{self.task_id}] Bronze ingestion failed: {exc}")
            raise AirflowException(f"Bronze ingestion failed: {exc}") from exc

        self.log.info(f"[{self.task_id}] Bronze ingestion completed")
        return {"status": "success", "dry_run": self.dry_run}