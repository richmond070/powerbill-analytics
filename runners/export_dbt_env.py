"""
runners/export_dbt_env.py
─────────────────────────────────────────────────────────────────────────────
Reads databricks/databricks.cfg and exports the three environment variables
that dbt_project/profiles.yml reads via env_var():

    DATABRICKS_HOST        ← workspace_url  (https:// stripped)
    DATABRICKS_HTTP_PATH   ← http_path
    DATABRICKS_TOKEN       ← token

TWO USAGE MODES
───────────────
Mode 1 — Export into the CURRENT shell session (most common):
    On Linux / macOS:
        eval $(python runners/export_dbt_env.py --shell)
        dbt debug --profiles-dir dbt_project --project-dir dbt_project

    On Windows (PowerShell):
        python runners/export_dbt_env.py --shell | Invoke-Expression
        dbt debug --profiles-dir dbt_project --project-dir dbt_project

Mode 2 — Run dbt as a subprocess (env vars set for child process only):
    python runners/export_dbt_env.py --run "dbt debug --profiles-dir dbt_project --project-dir dbt_project"
    python runners/export_dbt_env.py --run "dbt run --profiles-dir dbt_project --project-dir dbt_project"
    python runners/export_dbt_env.py --run "dbt test --profiles-dir dbt_project --project-dir dbt_project"

Mode 3 — Print env vars to stdout for inspection (default, no flags):
    python runners/export_dbt_env.py

WHY THIS APPROACH
─────────────────
dbt's env_var() reads from the process environment. Python's os.environ
only affects the current process and its children — it cannot inject into
a parent shell. The --shell flag prints export statements so the calling
shell can evaluate them, and --run spawns dbt as a child process with the
vars already set in os.environ.

Single source of truth: databricks/databricks.cfg
No secrets are duplicated or hardcoded anywhere else.
"""

import os
import sys
import subprocess
import configparser
import argparse
from pathlib import Path


# ── Config path resolution ────────────────────────────────────────────────────

def _find_cfg() -> Path:
    """
    Locate databricks/databricks.cfg relative to the project root.
    Works whether the script is called from the project root or runners/.
    """
    script_dir  = Path(__file__).parent.resolve()
    project_root = script_dir.parent          # runners/../  == project root

    candidates = [
        project_root / "databricks" / "databricks.cfg",
        script_dir   / "databricks" / "databricks.cfg",  # fallback
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "databricks/databricks.cfg not found.\n"
        f"Searched:\n" + "\n".join(f"  {p}" for p in candidates) + "\n"
        "Ensure the file exists and contains [DEFAULT] workspace_url, token, http_path."
    )


# ── Config reader ─────────────────────────────────────────────────────────────

def load_databricks_config(cfg_path: Path) -> dict:
    """
    Parse databricks.cfg and return the three values needed by dbt.

    Returns:
        dict with keys: host, http_path, token

    Raises:
        ValueError: If any required key is missing or still a placeholder.
    """
    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    section = parser["DEFAULT"]

    raw_url   = section.get("workspace_url", "").strip()
    http_path = section.get("http_path",     "").strip()
    token     = section.get("token",         "").strip()

    # Strip https:// — dbt-databricks expects the bare hostname
    host = raw_url.replace("https://", "").replace("http://", "").rstrip("/")

    # Validate — catch placeholder values that were never filled in
    errors = []
    if not host or host.startswith("your-"):
        errors.append("  workspace_url is missing or still a placeholder")
    if not http_path or http_path.startswith("/sql/1.0/warehouses/your"):
        errors.append("  http_path is missing or still a placeholder")
    if not token or token in ("dapi...", ""):
        errors.append("  token is missing or still a placeholder")

    if errors:
        raise ValueError(
            f"databricks.cfg has unfilled values:\n" + "\n".join(errors) + "\n"
            f"Config file: {cfg_path}"
        )

    return {
        "DATABRICKS_HOST":      host,
        "DATABRICKS_HTTP_PATH": http_path,
        "DATABRICKS_TOKEN":     token,
    }


# ── Modes ─────────────────────────────────────────────────────────────────────

def print_env_vars(env: dict) -> None:
    """Mode 3 — Print values (token masked) for inspection."""
    masked_token = env["DATABRICKS_TOKEN"][:8] + "..." if env["DATABRICKS_TOKEN"] else ""
    print(f"DATABRICKS_HOST      = {env['DATABRICKS_HOST']}")
    print(f"DATABRICKS_HTTP_PATH = {env['DATABRICKS_HTTP_PATH']}")
    print(f"DATABRICKS_TOKEN     = {masked_token}  (masked)")


def print_shell_exports(env: dict) -> None:
    """
    Mode 1 — Print export statements for shell evaluation.
    Output is intentionally plain — no extra text — so eval/Invoke-Expression works.
    """
    for key, value in env.items():
        # Wrap value in single quotes to handle special characters safely
        print(f"export {key}='{value}'")


def run_dbt_command(env: dict, command: str) -> int:
    """
    Mode 2 — Run a dbt command as a subprocess with env vars injected.

    Args:
        env:     Dict of env vars to inject.
        command: Full dbt command string, e.g. 'dbt run --profiles-dir dbt_project'

    Returns:
        Exit code of the subprocess.
    """
    child_env = {**os.environ, **env}     # inherit current env, add/override our vars

    print(f"\nRunning: {command}")
    print(f"DATABRICKS_HOST      = {env['DATABRICKS_HOST']}")
    print(f"DATABRICKS_HTTP_PATH = {env['DATABRICKS_HTTP_PATH']}")
    print(f"DATABRICKS_TOKEN     = {env['DATABRICKS_TOKEN'][:8]}...  (masked)\n")

    result = subprocess.run(
        command,
        shell=True,
        env=child_env,
    )
    return result.returncode


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Databricks env vars from databricks.cfg for dbt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect loaded values (default)
  python runners/export_dbt_env.py

  # Export into current shell (Linux/macOS)
  eval $(python runners/export_dbt_env.py --shell)

  # Export into current shell (Windows PowerShell)
  python runners/export_dbt_env.py --shell | Invoke-Expression

  # Run dbt debug as subprocess
  python runners/export_dbt_env.py --run "dbt debug --profiles-dir dbt_project --project-dir dbt_project"

  # Run dbt models as subprocess
  python runners/export_dbt_env.py --run "dbt run --profiles-dir dbt_project --project-dir dbt_project"
        """
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print shell export statements (for eval or Invoke-Expression)"
    )
    parser.add_argument(
        "--run",
        metavar="COMMAND",
        help="Run a dbt command as a subprocess with env vars injected"
    )
    args = parser.parse_args()

    # Load config
    try:
        cfg_path = _find_cfg()
        env = load_databricks_config(cfg_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Dispatch to correct mode
    if args.shell:
        print_shell_exports(env)

    elif args.run:
        exit_code = run_dbt_command(env, args.run)
        sys.exit(exit_code)

    else:
        print_env_vars(env)
        print(
            "\nTo export into your shell:\n"
            "  Linux/macOS:  eval $(python runners/export_dbt_env.py --shell)\n"
            "  Windows PS:   python runners/export_dbt_env.py --shell | Invoke-Expression\n"
            "\nTo run dbt directly:\n"
            '  python runners/export_dbt_env.py --run "dbt debug --profiles-dir dbt_project --project-dir dbt_project"'
        )


if __name__ == "__main__":
    main()