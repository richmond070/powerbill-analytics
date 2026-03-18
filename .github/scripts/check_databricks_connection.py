"""
CI Script: Databricks Connectivity Ping
========================================
Reconstructed config is written by the CI workflow from GitHub Secrets
before this script runs. This script:

  1. Validates databricks/databricks.cfg has the correct structure
  2. Validates [POSTGRES] section is present and well-formed
  3. Fires SELECT 1 via DatabricksSQLClient (real HTTP call)
  4. Asserts the result is SUCCEEDED
  5. Exits 0 on success, 1 on any failure

Location  : .github/scripts/check_databricks_connection.py
Called by : .github/workflows/ci.yml  (databricks-connectivity job)
Triggered : PRs to main + pushes to main only

Usage (local):
    python .github/scripts/check_databricks_connection.py

The script resolves paths relative to the project root so it works
regardless of which directory it is called from.
"""

import configparser
import sys
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution — works whether called from project root or scripts/
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()   # .github/scripts/ → root
CONFIG_PATH  = PROJECT_ROOT / "databricks" / "databricks.cfg"

# ---------------------------------------------------------------------------
# Required config keys
# ---------------------------------------------------------------------------
REQUIRED_DEFAULT_KEYS  = ["token", "workspace_url", "warehouse_id"]
REQUIRED_POSTGRES_KEYS = ["host", "port", "dbname", "user", "password"]


# ---------------------------------------------------------------------------
# Step 1 — Validate config file structure
# ---------------------------------------------------------------------------
def validate_config(parser: configparser.ConfigParser) -> list[str]:
    """
    Validate that all required keys are present and non-empty.
    Returns a list of error strings (empty = all good).
    """
    errors = []

    # [DEFAULT] section
    for key in REQUIRED_DEFAULT_KEYS:
        value = parser["DEFAULT"].get(key, "").strip()
        if not value:
            errors.append(f"[DEFAULT] key '{key}' is missing or empty")
            continue

        if key == "workspace_url":
            if not value.startswith("https://"):
                errors.append(
                    f"[DEFAULT] workspace_url must start with 'https://', got: '{value[:30]}'"
                )
            if "cloud.databricks.com" not in value and "azuredatabricks.net" not in value:
                errors.append(
                    f"[DEFAULT] workspace_url doesn't look like a Databricks URL: '{value[:50]}'"
                )

        if key == "token":
            if not value.startswith("dapi"):
                errors.append(
                    f"[DEFAULT] token must start with 'dapi', got prefix: '{value[:6]}'"
                )

        if key == "warehouse_id":
            if len(value) < 8:
                errors.append(
                    f"[DEFAULT] warehouse_id looks too short (got {len(value)} chars): '{value}'"
                )

    # [POSTGRES] section
    if not parser.has_section("POSTGRES"):
        errors.append("Config is missing the [POSTGRES] section entirely")
    else:
        for key in REQUIRED_POSTGRES_KEYS:
            value = parser["POSTGRES"].get(key, "").strip()
            if not value:
                errors.append(f"[POSTGRES] key '{key}' is missing or empty")

    return errors


# ---------------------------------------------------------------------------
# Step 2 — Live Databricks connectivity ping
# ---------------------------------------------------------------------------
def ping_databricks(config_path: str) -> dict:
    """
    Instantiate DatabricksSQLClient and run SELECT 1.
    Returns a result dict with keys: success, status, duration_ms, error.
    """
    # Add project root to path so bronze module is importable
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from bronze.databricks_client import DatabricksSQLClient
    except ImportError as e:
        return {
            "success": False,
            "status":  "IMPORT_ERROR",
            "duration_ms": 0,
            "error": f"Could not import DatabricksSQLClient: {e}",
        }

    start = time.time()
    try:
        client = DatabricksSQLClient(config_path=config_path)
        result = client.execute_sql("SELECT 1")
        duration_ms = int((time.time() - start) * 1000)

        return {
            "success":     result.status == "SUCCEEDED",
            "status":      result.status,
            "duration_ms": duration_ms,
            "statement_id": result.statement_id,
            "error":       result.error_message,
        }

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return {
            "success":     False,
            "status":      "EXCEPTION",
            "duration_ms": duration_ms,
            "error":       str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Databricks Connectivity Check")
    print("=" * 60)
    print(f"Config path : {CONFIG_PATH}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    # ── Guard: config file must exist ────────────────────────────────────
    if not CONFIG_PATH.exists():
        print("FAIL: databricks/databricks.cfg not found.")
        print(
            "  In CI this file is written by the 'Write databricks.cfg' step.\n"
            "  Locally, ensure databricks/databricks.cfg exists."
        )
        sys.exit(1)

    # ── Step 1: Structure validation ─────────────────────────────────────
    print("Step 1 — Validating config structure...")
    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH)

    errors = validate_config(parser)
    if errors:
        print("FAIL: Config structure validation failed:")
        for err in errors:
            print(f"  ✗ {err}")
        sys.exit(1)

    # Safe to print non-sensitive values for CI log transparency
    print(f"  ✓ workspace_url  : {parser['DEFAULT']['workspace_url']}")
    print(f"  ✓ warehouse_id   : {parser['DEFAULT']['warehouse_id']}")
    print(f"  ✓ token prefix   : {parser['DEFAULT']['token'][:8]}...")
    print(f"  ✓ postgres host  : {parser['POSTGRES']['host']}")
    print(f"  ✓ postgres dbname: {parser['POSTGRES']['dbname']}")
    print()

    # ── Step 2: Live connectivity ping ───────────────────────────────────
    print("Step 2 — Pinging Databricks SQL warehouse (SELECT 1)...")
    ping = ping_databricks(str(CONFIG_PATH))

    print(f"  Status      : {ping['status']}")
    print(f"  Duration    : {ping['duration_ms']}ms")

    if ping.get("statement_id"):
        print(f"  Statement ID: {ping['statement_id']}")

    if not ping["success"]:
        print()
        print(f"FAIL: Databricks ping failed.")
        print(f"  Error: {ping['error']}")
        print()
        print("Common causes:")
        print("  - Token has expired (rotate and update DATABRICKS_TOKEN secret)")
        print("  - Warehouse is stopped and not auto-starting (check Databricks UI)")
        print("  - workspace_url or warehouse_id is incorrect")
        print("  - Network access from GitHub Actions to your workspace is blocked")
        sys.exit(1)

    print()
    print("PASS: Databricks connection is healthy.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()