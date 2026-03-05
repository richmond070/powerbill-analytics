#!/usr/bin/env python3
"""
Bronze Layer Entry Point
Minimal script to run the bronze layer orchestration pipeline
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from ..bronze.bronze_orchestrator import BronzeLayerOrchestrator


def resolve_paths():
    """
    Resolve all required paths for the bronze layer pipeline
    
    Returns:
        dict: Dictionary of resolved paths
    """
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Contract path - check project mount first, then local
    contract_path = "/mnt/project/bronze_ingestion_contract.json"
    if not os.path.exists(contract_path):
        contract_path = script_dir / "bronze_ingestion_contract.json"
        if not os.path.exists(contract_path):
            raise FileNotFoundError(
                f"Contract file not found at /mnt/project/bronze_ingestion_contract.json "
                f"or {contract_path}"
            )
    
    # Databricks config path
    config_path = script_dir / "databricks" / "databricks.cfg"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Databricks config file not found at {config_path}\n"
            f"Create a file at databricks/databricks.cfg with:\n"
            f"[DEFAULT]\n"
            f"workspace_url = https://xxx.cloud.databricks.com\n"
            f"token = dapi...\n"
            f"warehouse_id = abc123..."
        )
    
    # Storage paths - use defaults or from environment
    staging_root = os.getenv('STAGING_ROOT', '/mnt/staging/raw')
    delta_root = os.getenv('DELTA_ROOT', '/mnt/delta/bronze')
    
    # Catalog and schema - use defaults or from environment
    catalog = os.getenv('DATABRICKS_CATALOG', 'main')
    schema = os.getenv('DATABRICKS_SCHEMA', 'bronze')
    
    paths = {
        'contract_path': str(contract_path),
        'config_path': str(config_path),
        'staging_root': staging_root,
        'delta_root': delta_root,
        'catalog': catalog,
        'schema': schema
    }
    
    return paths


def main():
    """Main entry point"""
    
    print("=" * 80)
    print("BRONZE LAYER INGESTION PIPELINE")
    print("=" * 80)
    print(f"Start Time: {datetime.utcnow().isoformat()}Z\n")
    
    try:
        # Step 1: Resolve all paths
        print("Step 1: Resolving paths...")
        paths = resolve_paths()
        
        print(f"  Contract:      {paths['contract_path']}")
        print(f"  Config:        {paths['config_path']}")
        print(f"  Catalog:       {paths['catalog']}")
        print(f"  Schema:        {paths['schema']}")
        print(f"  Staging Root:  {paths['staging_root']}")
        print(f"  Delta Root:    {paths['delta_root']}")
        print()
        
        # Step 2: Import and instantiate orchestrator
        print("Step 2: Initializing orchestrator...")
        
        # Add current directory to Python path
        sys.path.insert(0, str(Path(__file__).parent))
        
        
        
        orchestrator = BronzeLayerOrchestrator(
            contract_path=paths['contract_path'],
            config_path=paths['config_path'],
            catalog=paths['catalog'],
            schema=paths['schema'],
            staging_root=paths['staging_root'],
            delta_root=paths['delta_root']
        )
        
        print("Orchestrator initialized successfully\n")
        
        # Step 3: Run full pipeline
        print("Step 3: Running full pipeline...\n")
        
        orchestrator.run_full_pipeline(
            datasets=None,      # Process all datasets
            download=True,      # Download raw files
            optimize=False,     # Skip optimization for speed
            dry_run=True       # Execute for real
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"End Time: {datetime.utcnow().isoformat()}Z\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nPIPELINE FAILED")
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()