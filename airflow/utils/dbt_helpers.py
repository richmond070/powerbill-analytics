import subprocess
import os
from typing import Tuple
from pathlib import Path

# Resolve project root (three levels up from airflow/utils/dbt_helpers.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt_project"
DBT_PROFILES_DIR = PROJECT_ROOT / "dbt_project"


def run_dbt_command(command: str, task_id: str) -> Tuple[int, str, str]:
    
    env = os.environ.copy()
    # Ensure dbt can find the profiles directory
    env["DBT_PROFILES_DIR"] = str(DBT_PROFILES_DIR)

    # Split "docs generate" into two tokens if needed
    cmd_parts = command.split()

    full_command = (
        ["dbt"]
        + cmd_parts
        + ["--profiles-dir", str(DBT_PROFILES_DIR)]
        + ["--project-dir", str(DBT_PROJECT_DIR)]
    )

    print(f"[{task_id}] Executing: {' '.join(full_command)}")

    result = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),   # run from project root, same as manual runs
    )

    return result.returncode, result.stdout, result.stderr


def validate_dbt_run(exit_code: int, stdout: str, stderr: str) -> bool:
    
    if exit_code != 0:
        raise Exception(
            f"dbt command failed (exit code {exit_code}).\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    # dbt writes "1 of 9 ERROR" style lines on model failures even when
    # the process exits 0 in some edge cases — catch those explicitly
    if "ERROR" in stdout and "of" in stdout and "ERROR" in stdout:
        raise Exception(
            f"dbt reported model errors in output.\nstdout:\n{stdout}"
        )

    return True