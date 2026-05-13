import json
import os
from datetime import datetime, timezone
from typing import Optional

from databricks.databricks_client import DatabricksSQLClient


class SilverOrchestrator:
    def __init__(
        self,
        contract_path: str,
        config_path: str = "databricks/databricks.cfg",
    ):
        self.contract_path = os.path.abspath(contract_path)
        self.contract_dir  = os.path.dirname(self.contract_path)
        self.state_path    = os.path.join(self.contract_dir, "silver_run_state.json")

        self.contract = self._load_contract()
        self.client   = DatabricksSQLClient(config_path=config_path)

        print(f"\n{'='*70}")
        print("Silver Layer Orchestrator Initialized")
        print(f"{'='*70}")
        print(f"Contract:   {self.contract_path}")
        print(f"Run state:  {self.state_path}")
        print(f"Datasets:   {len(self.contract['datasets'])}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Contract + state helpers
    # ------------------------------------------------------------------

    def _load_contract(self) -> dict:
        with open(self.contract_path, "r") as f:
            return json.load(f)

    def _load_state(self) -> dict:
        """
        Load the run-state checkpoint file.
        Returns an empty dict if the file does not exist yet.
        Shape: { "dataset_name": { "status": "SUCCEEDED", "completed_at": "<iso>" } }
        """
        if not os.path.exists(self.state_path):
            return {}
        with open(self.state_path, "r") as f:
            return json.load(f)

    def _mark_succeeded(self, dataset_name: str) -> None:
        """Append a SUCCEEDED entry for dataset_name to the state file."""
        state = self._load_state()
        state[dataset_name] = {
            "status":       "SUCCEEDED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def reset_state(self) -> None:
        """Delete the run-state file so the next run processes all datasets."""
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
            print(f"Run state cleared: {self.state_path}")
        else:
            print("No run state file found — nothing to clear.")

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------

    def _read_sql(self, relative_path: str) -> str:
        """
        Resolve transformer SQL file path relative to the contract directory.
        e.g. "transformers/billing_payments_silver.sql"
             → "<contract_dir>/transformers/billing_payments_silver.sql"
        """
        full_path = os.path.join(self.contract_dir, relative_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Transformer SQL not found: {full_path}\n"
                f"Check that '{relative_path}' exists inside {self.contract_dir}"
            )
        with open(full_path, "r") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Dependency resolution (topological sort via DFS)
    # ------------------------------------------------------------------

    def _resolve_execution_order(self) -> list:
        """
        Return datasets sorted so every dependency executes before the
        datasets that depend on it.  Uses iterative DFS to avoid Python
        recursion limits on large contracts.
        """
        datasets    = self.contract["datasets"]
        name_to_ds  = {d["name"]: d for d in datasets}
        resolved    = []
        seen        = set()

        def visit(ds: dict) -> None:
            if ds["name"] in seen:
                return
            for dep_name in ds.get("depends_on", []):
                if dep_name not in name_to_ds:
                    raise ValueError(
                        f"Dataset '{ds['name']}' depends on '{dep_name}' "
                        f"which is not defined in the contract."
                    )
                visit(name_to_ds[dep_name])
            seen.add(ds["name"])
            resolved.append(ds)

        for ds in datasets:
            visit(ds)

        return resolved

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, force: bool = False) -> None:
        """
        Execute the Silver pipeline in dependency order.

        Args:
            force: If True, ignore the run-state checkpoint and re-run every
                   dataset regardless of prior completion status.
                   Equivalent to deleting silver_run_state.json before running.
        """
        execution_plan = self._resolve_execution_order()
        state          = {} if force else self._load_state()

        print(f"Execution plan ({len(execution_plan)} datasets):")
        for i, ds in enumerate(execution_plan, 1):
            already_done = ds["name"] in state and state[ds["name"]]["status"] == "SUCCEEDED"
            status_tag   = "  [will skip — already SUCCEEDED]" if already_done and not force else ""
            print(f"  {i}. {ds['name']}{status_tag}")
        print()

        for dataset in execution_plan:
            name = dataset["name"]

            # ── Idempotency check ──────────────────────────────────────
            if not force and name in state and state[name]["status"] == "SUCCEEDED":
                print(f"[SKIP]  {name}  (completed at {state[name]['completed_at']})")
                continue

            print(f"[RUN]   {name}")
            print(f"        target  -> {dataset['target_table']}")
            print(f"        sql     -> {dataset['transformer_sql']}")

            # ── Read SQL ───────────────────────────────────────────────
            try:
                sql = self._read_sql(dataset["transformer_sql"])
            except FileNotFoundError as e:
                raise RuntimeError(f"Cannot run '{name}': {e}") from e

            # ── Execute via Databricks SQL API ─────────────────────────
            result = self.client.execute_sql(sql)

            if result.status != "SUCCEEDED":
                raise RuntimeError(
                    f"Dataset '{name}' FAILED.\n"
                    f"  Statement ID : {result.statement_id}\n"
                    f"  Error        : {result.error_message}"
                )

            # ── Mark succeeded in state file ───────────────────────────
            self._mark_succeeded(name)

            print(
                f"[DONE]  {name}  "
                f"({result.duration_ms:,} ms"
                + (f", {result.row_count:,} rows" if result.row_count else "")
                + ")\n"
            )

        print("=" * 70)
        print("Silver pipeline complete.")
        print("=" * 70)


if __name__ == "__main__":
    orchestrator = SilverOrchestrator(
        contract_path="silver/silver_contract.json",
        config_path="databricks/databricks.cfg",
    )

    # Normal run — skips datasets already marked SUCCEEDED
    orchestrator.run()

    # Force full re-run:
    # orchestrator.run(force=True)

    # Reset state manually (same as deleting silver_run_state.json):
    # orchestrator.reset_state()