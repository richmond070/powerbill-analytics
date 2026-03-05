"""
Bronze Layer Orchestrator
Main orchestration script for metadata-driven bronze layer ingestion
Python orchestrates only - Spark/Databricks SQL does all processing
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import tempfile

# Import our modules
from .partition_strategy import PartitionHeuristics
from .sql_generator import BronzeSQLGenerator
from .databricks_client import DatabricksSQLClient, SQLExecutionLogger
from .data_downloader import DataDownloader


class BronzeLayerOrchestrator:
    """
    Orchestrates metadata-driven bronze layer ingestion
    All processing pushed to Spark/Databricks SQL
    """
    
    def __init__(
        self,
        contract_path: str,
        config_path: str,
        catalog: str = "main",
        schema: str = "bronze",
        staging_root: str = "/mnt/staging/raw",
        delta_root: str = "/mnt/delta/bronze"
    ):
        """
        Initialize orchestrator
        
        Args:
            contract_path: Path to bronze_ingestion_contract.json
            config_path: Path to databricks.cfg file
            catalog: Unity Catalog name
            schema: Schema/database name
            staging_root: Root path for staging raw files
            delta_root: Root path for Delta tables
        """
        self.contract_path = contract_path
        self.config_path = config_path
        self.catalog = catalog
        self.schema = schema
        
        # Load contract
        with open(contract_path, 'r') as f:
            self.contract = json.load(f)
        
        # Initialize components
        self.sql_generator = BronzeSQLGenerator(
            catalog=catalog,
            schema=schema,
            base_location=delta_root,
            staging_location=staging_root
        )
        
        self.downloader = DataDownloader(staging_root=staging_root)
        
        self.db_client = DatabricksSQLClient(
            config_path=config_path,
            catalog=catalog,
            schema=schema
        )
        
        self.logger = SQLExecutionLogger()
        
        print(f"\n{'='*80}")
        print(f"Bronze Layer Orchestrator Initialized")
        print(f"{'='*80}")
        print(f"Contract:        {contract_path}")
        print(f"Config:          {config_path}")
        print(f"Catalog:         {catalog}")
        print(f"Schema:          {schema}")
        print(f"Datasets:        {len(self.contract['datasets'])}")
        print(f"{'='*80}\n")
    
    def create_bronze_tables(
        self,
        datasets: Optional[List[str]] = None,
        dry_run: bool = False
    ):
        """
        Create bronze tables based on contract metadata
        
        Args:
            datasets: Specific datasets to process (None = all)
            dry_run: Generate SQL but don't execute
        """
        print(f"\n{'='*80}")
        print("STEP 1: CREATE BRONZE TABLES")
        print(f"{'='*80}\n")
        
        datasets_to_process = self._get_datasets_to_process(datasets)
        timestamp = datetime.utcnow().isoformat()
        
        for i, dataset in enumerate(datasets_to_process, 1):
            dataset_name = dataset['dataset_name']
            print(f"\n[{i}/{len(datasets_to_process)}] Processing: {dataset_name}")
            print(f"   Rows: {dataset['total_rows']:,}")
            
            # Step 1: Determine partition strategy
            partition_config = PartitionHeuristics.determine_strategy(
                dataset_name=dataset_name,
                total_rows=dataset['total_rows'],
                columns=dataset['files'][0]['columns'],
                file_count=dataset['file_count']
            )
            print(f"   Strategy: {partition_config.strategy.value}")
            if partition_config.partition_columns:
                print(f"   Partitions: {', '.join(partition_config.partition_columns)}")
            print(f"   Reason: {partition_config.reason}")
            
            # Step 2: Generate CREATE TABLE SQL
            create_sql = self.sql_generator.generate_create_table_sql(
                dataset_metadata=dataset,
                partition_config=partition_config,
                timestamp=timestamp
            )
            
            # Save SQL to file
            sql_file = os.path.join(tempfile.gettempdir(), f"bronze_{dataset_name}_create.sql")
            with open(sql_file, 'w') as f:
                f.write(create_sql)
            print(f"   SQL saved: {sql_file}")
            
            if dry_run:
                print(f"   [DRY RUN] Skipping execution")
                continue
            
            # Step 3: Execute SQL via Databricks SQL API
            print(f"   Executing CREATE TABLE...")
            result = self.db_client.execute_sql(create_sql)
            
            # Log execution
            self.logger.log_execution(
                dataset_name=dataset_name,
                sql_type="CREATE_TABLE",
                result=result,
                sql_statement=create_sql
            )
            
            if result.status == 'SUCCEEDED':
                print(f" Table created successfully ({result.duration_ms}ms)")
            else:
                print(f" Failed: {result.error_message}")
        
        print(f"\n{'='*80}")
        print("Table Creation Complete")
        print(f"{'='*80}\n")
    
    def ingest_data(
        self,
        datasets: Optional[List[str]] = None,
        download: bool = True,
        dry_run: bool = False
    ):
        """
        Ingest data into bronze tables
        
        Args:
            datasets: Specific datasets to process (None = all)
            download: Download raw data first
            dry_run: Generate SQL but don't execute
        """
        print(f"\n{'='*80}")
        print("STEP 2: INGEST DATA")
        print(f"{'='*80}\n")
        
        datasets_to_process = self._get_datasets_to_process(datasets)
        
        for i, dataset in enumerate(datasets_to_process, 1):
            dataset_name = dataset['dataset_name']
            print(f"\n[{i}/{len(datasets_to_process)}] Ingesting: {dataset_name}")
            
            # Step 1: Download raw data if needed
            if download:
                download_results = self.downloader.download_dataset(dataset)
                if not self.downloader.validate_downloads(download_results):
                    print(f"Download failed, skipping ingestion")
                    continue
            
            # Step 2: Determine partition strategy (same as table creation)
            partition_config = PartitionHeuristics.determine_strategy(
                dataset_name=dataset_name,
                total_rows=dataset['total_rows'],
                columns=dataset['files'][0]['columns'],
                file_count=dataset['file_count']
            )
            
            # Step 3: Generate ingestion SQL
            # Use MERGE for large datasets or append-only tables
            use_merge = partition_config.use_append_only or dataset['total_rows'] > 300_000
            
            ingest_sql = self.sql_generator.generate_ingestion_sql(
                dataset_metadata=dataset,
                partition_config=partition_config,
                use_merge=use_merge
            )
            
            # Save SQL to file
            sql_file = os.path.join(tempfile.gettempdir(), f"bronze_{dataset_name}_ingest.sql")
            with open(sql_file, 'w') as f:
                f.write(ingest_sql)
            print(f"   SQL saved: {sql_file}")
            
            if dry_run:
                print(f"   [DRY RUN] Skipping execution")
                continue
            
            # Step 4: Execute ingestion via Databricks SQL API
            method = "MERGE" if use_merge else "COPY INTO"
            print(f"   Executing {method}...")
            result = self.db_client.execute_sql(ingest_sql)
            
            # Log execution
            self.logger.log_execution(
                dataset_name=dataset_name,
                sql_type="INGEST",
                result=result,
                sql_statement=ingest_sql
            )
            
            if result.status == 'SUCCEEDED':
                row_info = f"{result.row_count:,} rows" if result.row_count else "completed"
                print(f" Ingestion complete: {row_info} ({result.duration_ms}ms)")
            else:
                print(f" Failed: {result.error_message}")
        
        print(f"\n{'='*80}")
        print("Data Ingestion Complete")
        print(f"{'='*80}\n")
    
    def optimize_tables(
        self,
        datasets: Optional[List[str]] = None,
        dry_run: bool = False
    ):
        """
        Optimize bronze tables (OPTIMIZE + VACUUM)
        
        Args:
            datasets: Specific datasets to process (None = all)
            dry_run: Generate SQL but don't execute
        """
        print(f"\n{'='*80}")
        print("STEP 3: OPTIMIZE TABLES")
        print(f"{'='*80}\n")
        
        datasets_to_process = self._get_datasets_to_process(datasets)
        
        for i, dataset in enumerate(datasets_to_process, 1):
            dataset_name = dataset['dataset_name']
            print(f"\n[{i}/{len(datasets_to_process)}] Optimizing: {dataset_name}")
            
            optimize_sql = self.sql_generator.generate_optimization_sql(dataset_name)
            
            if dry_run:
                print(f"   [DRY RUN] Skipping execution")
                continue
            
            print(f"   Executing OPTIMIZE...")
            result = self.db_client.execute_sql(optimize_sql)
            
            if result.status == 'SUCCEEDED':
                print(f"Optimization complete ({result.duration_ms}ms)")
            else:
                print(f" Failed: {result.error_message}")
        
        print(f"\n{'='*80}")
        print("Table Optimization Complete")
        print(f"{'='*80}\n")
    
    def run_full_pipeline(
        self,
        datasets: Optional[List[str]] = None,
        download: bool = True,
        optimize: bool = False,
        dry_run: bool = False
    ):
        """
        Run complete bronze layer pipeline
        
        Args:
            datasets: Specific datasets to process (None = all)
            download: Download raw data
            optimize: Optimize tables after ingestion
            dry_run: Generate SQL but don't execute
        """
        print(f"\n{'#'*80}")
        print("BRONZE LAYER PIPELINE - FULL RUN")
        print(f"{'#'*80}\n")
        print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
        print(f"Datasets: {', '.join(datasets) if datasets else 'ALL'}")
        print(f"Download: {download}")
        print(f"Optimize: {optimize}")
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Create tables
            self.create_bronze_tables(datasets=datasets, dry_run=dry_run)
            
            # Step 2: Ingest data
            self.ingest_data(datasets=datasets, download=download, dry_run=dry_run)
            
            # Step 3: Optimize (optional)
            if optimize:
                self.optimize_tables(datasets=datasets, dry_run=dry_run)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            print(f"\n{'#'*80}")
            print(f"PIPELINE COMPLETE")
            print(f"{'#'*80}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"{'#'*80}\n")
            
        except Exception as e:
            print(f"\n{'#'*80}")
            print(f"PIPELINE FAILED")
            print(f"{'#'*80}")
            print(f"Error: {str(e)}")
            print(f"{'#'*80}\n")
            raise
    
    def _get_datasets_to_process(self, dataset_names: Optional[List[str]]) -> List[Dict]:
        """Get datasets to process based on filter"""
        all_datasets = self.contract['datasets']
        
        if dataset_names is None:
            return all_datasets
        
        return [d for d in all_datasets if d['dataset_name'] in dataset_names]


if __name__ == "__main__":
    # Example usage
    print("""
Bronze Layer Orchestrator
========================

Usage:
    orchestrator = BronzeLayerOrchestrator(
        contract_path='/mnt/project/bronze_ingestion_contract.json',
        config_path='databricks/databricks.cfg',
        catalog='main',
        schema='bronze'
    )
    
    # Run full pipeline
    orchestrator.run_full_pipeline(
        datasets=['billing_payments'],  # Or None for all
        download=True,
        optimize=False,
        dry_run=True  # Set to False to execute
    )
    
    # Or run individual steps
    orchestrator.create_bronze_tables(dry_run=True)
    orchestrator.ingest_data(download=True, dry_run=True)
    orchestrator.optimize_tables(dry_run=True)

Config File (databricks/databricks.cfg):
    [DEFAULT]
    workspace_url = https://xxx.cloud.databricks.com
    token = dapi...
    warehouse_id = abc123...
    """)