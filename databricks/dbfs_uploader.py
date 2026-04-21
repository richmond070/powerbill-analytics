"""
DBFS Uploader
=============
Uploads locally staged parquet files to Databricks File System (DBFS)
so that Databricks SQL can read them during COPY INTO / MERGE ingestion.

Folder structure is preserved:

  Local (your machine)                    DBFS (Databricks cloud)
  ────────────────────────────────        ──────────────────────────────────────
  staging/raw/                            dbfs:/FileStore/energy-sector/
    billing_payments/                       staging/raw/
      0.parquet                               billing_payments/
    grid_load/                                  0.parquet
      0.parquet                               grid_load/
    ...                                         0.parquet
                                              ...

Usage:
  # Upload all datasets
  python -m databricks.dbfs_uploader

  # Upload specific datasets only
  python -m databricks.dbfs_uploader --datasets billing_payments grid_load

Config (databricks/databricks.cfg):
  [DEFAULT]
  workspace_url = https://your-workspace.cloud.databricks.com
  token         = dapi...
"""

import argparse
import base64
import configparser
import os
import sys
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Path constants — resolved relative to project root
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

CONFIG_PATH        = PROJECT_ROOT / "databricks" / "databricks.cfg"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "staging" / "raw"
DEFAULT_DBFS_ROOT  = "/Volumes/main/bronze/staging/raw"

# DBFS PUT API has a 1 MB limit per request for the base64 payload.
# Large parquet files must be uploaded in chunks using open/addBlock/close.
# Files under this threshold use the simpler single-PUT approach.
CHUNK_THRESHOLD_BYTES = 900_000        # 900 KB — stay safely under 1 MB limit
CHUNK_SIZE_BYTES      = 900_000        # chunk size for large file uploads


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class DatabricksConfig:
    """
    Reads Databricks connection details from databricks/databricks.cfg.
    Reads workspace_url (not host) to stay consistent with the rest of
    the pipeline (databricks_client.py, bronze_orchestrator.py).
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        parser = configparser.ConfigParser()
        parser.read(str(config_path))

        section = parser["DEFAULT"]

        # Support both 'workspace_url' (new) and 'host' (legacy) keys
        self.workspace_url = (
            section.get("workspace_url", "")
            or section.get("host", "")
        ).rstrip("/")

        self.token = section.get("token", "")

        if not self.workspace_url:
            raise ValueError(
                "workspace_url missing in databricks/databricks.cfg [DEFAULT] section.\n"
                "Add:  workspace_url = https://your-workspace.cloud.databricks.com"
            )
        if not self.token:
            raise ValueError(
                "token missing in databricks/databricks.cfg [DEFAULT] section.\n"
                "Add:  token = dapi..."
            )


# ---------------------------------------------------------------------------
# DBFS Uploader
# ---------------------------------------------------------------------------

class DBFSUploader:
    """
    Uploads local parquet files to DBFS using the Databricks DBFS REST API.

    Small files (< 900 KB) use a single PUT request.
    Large files use the three-step open/addBlock/close chunked upload
    to stay within the API's base64 payload size limit.
    """

    def __init__(self, config: DatabricksConfig):
        self.config  = config
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type":  "application/json",
        }
        self._api_base = f"{config.workspace_url}/api/2.0/fs/files"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_staging(
        self,
        local_root:  Path = DEFAULT_LOCAL_ROOT,
        dbfs_root:   str  = DEFAULT_DBFS_ROOT,
        datasets:    Optional[list] = None,
        skip_existing: bool = True,
    ) -> dict:
        """
        Walk local_root and upload all parquet files to dbfs_root,
        preserving the dataset subfolder structure.

        Args:
            local_root:    Local staging directory (staging/raw/).
            dbfs_root:     DBFS destination root path.
            datasets:      Optional list of dataset names to upload.
                           If None, all subfolders are uploaded.
            skip_existing: Skip files already present in DBFS.

        Returns:
            Summary dict with counts of uploaded, skipped, and failed files.
        """
        if not local_root.exists():
            raise FileNotFoundError(
                f"Local staging directory not found: {local_root}\n"
                "Run the pipeline with download=True first to populate it."
            )

        summary = {"uploaded": 0, "skipped": 0, "failed": 0, "errors": []}

        # Discover dataset subfolders
        dataset_dirs = sorted([
            d for d in local_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not dataset_dirs:
            print(f"[WARN] No dataset folders found in {local_root}")
            return summary

        # Filter to requested datasets if specified
        if datasets:
            dataset_dirs = [d for d in dataset_dirs if d.name in datasets]
            if not dataset_dirs:
                print(f"[WARN] None of the requested datasets found in {local_root}")
                return summary

        print(f"\nUploading {len(dataset_dirs)} dataset(s) to DBFS")
        print(f"  Local root : {local_root}")
        print(f"  DBFS root  : {dbfs_root}\n")

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            parquet_files = sorted(dataset_dir.glob("*.parquet"))

            if not parquet_files:
                print(f"  [{dataset_name}] No parquet files found, skipping")
                continue

            print(f"  [{dataset_name}] {len(parquet_files)} file(s)")

            # Ensure the DBFS subfolder exists
            dbfs_dataset_path = f"{dbfs_root}/{dataset_name}"

            for local_file in parquet_files:
                dbfs_file_path = f"{dbfs_dataset_path}/{local_file.name}"

                # Skip if already uploaded and skip_existing is True
                if skip_existing and self.file_exists(dbfs_file_path):
                    print(f"    [SKIP] {local_file.name} already in DBFS")
                    summary["skipped"] += 1
                    continue

                try:
                    file_size = local_file.stat().st_size
                    self._upload_file(local_file, dbfs_file_path)

                    print(
                        f"    [OK]   {local_file.name} "
                        f"({file_size / 1_000_000:.1f} MB)"
                    )
                    summary["uploaded"] += 1

                except Exception as e:
                    print(f"    [FAIL] {local_file.name} — {e}")
                    summary["failed"]  += 1
                    summary["errors"].append({
                        "file":  str(local_file),
                        "error": str(e),
                    })

        return summary

    def file_exists(self, volume_path: str) -> bool:
        resp = requests.get(
            f"{self._api_base}{volume_path}",
            headers={"Authorization": f"Bearer {self.config.token}"},
        )
        return resp.status_code == 200

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upload_file(self, local_path: Path, dbfs_path: str) -> None:
        """
        Upload a single file to DBFS.

        Chooses between single-PUT (small files) and chunked upload
        (large files) based on CHUNK_THRESHOLD_BYTES.

        Args:
            local_path: Local file to upload.
            dbfs_path:  Destination DBFS path.
        """
        file_size = local_path.stat().st_size

        if file_size <= CHUNK_THRESHOLD_BYTES:
            self._upload_small_file(local_path, dbfs_path)
        else:
            self._upload_large_file(local_path, dbfs_path)

    def _upload_small_file(self, local_path: Path, volume_path: str) -> None:
        with open(local_path, "rb") as f:
            data = f.read()

        resp = requests.put(
            f"{self._api_base}{volume_path}",
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/octet-stream",
            },
            data=data,       # raw bytes, no base64 encoding needed
            params={"overwrite": "true"},
        )

        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Volume upload failed for {volume_path}\n"
                f"Status: {resp.status_code} — {resp.text}"
            )

    def _upload_large_file(self, local_path: Path, volume_path: str) -> None:
        # Files API accepts a raw binary stream — no chunking required
        with open(local_path, "rb") as f:
            resp = requests.put(
                f"{self._api_base}{volume_path}",
                headers={
                    "Authorization": f"Bearer {self.config.token}",
                    "Content-Type": "application/octet-stream",
                },
                data=f,
                params={"overwrite": "true"},
            )

        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Volume upload failed for {volume_path}\n"
                f"Status: {resp.status_code} — {resp.text}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dbfs_uploader",
        description=(
            "Upload locally staged parquet files to DBFS so Databricks SQL "
            "can read them during bronze layer ingestion."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="DATASET_NAME",
        default=None,
        help=(
            "Space-separated dataset names to upload. "
            "If omitted, all datasets in staging/raw/ are uploaded. "
            "Example: --datasets billing_payments grid_load"
        ),
    )
    parser.add_argument(
        "--local-root",
        default=str(DEFAULT_LOCAL_ROOT),
        help=f"Local staging root directory. Default: {DEFAULT_LOCAL_ROOT}",
    )
    parser.add_argument(
        "--dbfs-root",
        default=DEFAULT_DBFS_ROOT,
        help=f"DBFS destination root path. Default: {DEFAULT_DBFS_ROOT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-upload files even if they already exist in DBFS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("DBFS UPLOADER")
    print("=" * 70)

    # Load config
    try:
        config = DatabricksConfig(CONFIG_PATH)
    except (ValueError, KeyError) as e:
        print(f"\n[ERROR] Config problem:\n{e}")
        sys.exit(1)

    uploader   = DBFSUploader(config)
    local_root = Path(args.local_root)
    dbfs_root  = args.dbfs_root

    try:
        summary = uploader.upload_staging(
            local_root    = local_root,
            dbfs_root     = dbfs_root,
            datasets      = args.datasets,
            skip_existing = not args.force,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        sys.exit(1)

    # Print summary
    print(f"\n{'=' * 70}")
    print("UPLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Uploaded : {summary['uploaded']}")
    print(f"  Skipped  : {summary['skipped']}  (already in DBFS)")
    print(f"  Failed   : {summary['failed']}")

    if summary["errors"]:
        print("\nFailed files:")
        for err in summary["errors"]:
            print(f"  {err['file']}")
            print(f"    -> {err['error']}")

    if summary["failed"] > 0:
        sys.exit(1)

    print(f"\n[OK] All files uploaded to {dbfs_root}")
    print(
        "\nNext step: run the bronze pipeline\n"
        "  python -m runners.run_bronze\n"
        f"\nSet DATABRICKS_STAGING={dbfs_root} so Databricks SQL "
        "reads from DBFS."
    )


if __name__ == "__main__":
    main()