"""
SQL Template Generator
Generates schema-safe Databricks SQL from templates
Templates are generic patterns, not hardcoded strings
"""

from typing import Dict, List, Optional
from .schema_mapper import SchemaMapper
from .partition_strategy import PartitionConfig, PartitionHeuristics


class BronzeSQLTemplate:
    """
    Generic SQL templates for bronze layer operations
    Templates are filled with metadata, never hardcoded
    """
    
    # Template for creating bronze tables with schema enforcement
    CREATE_TABLE_TEMPLATE = """
-- Bronze table for {dataset_name}
-- Strategy: {strategy}
-- Generated: {timestamp}

CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{table_name} (
    -- Source columns (schema-enforced from contract)
    {column_definitions},
    
    -- Bronze metadata columns
    _bronze_ingestion_timestamp TIMESTAMP,
    _bronze_source_file STRING,
    _bronze_row_hash STRING
)
USING DELTA
{partition_clause}
LOCATION '{table_location}'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'delta.minReaderVersion' = '2',
    'delta.minWriterVersion' = '5',
    'quality.enforceSchema' = 'true',
    'source.contract' = 'bronze_ingestion_contract.json',
    'source.dataset' = '{dataset_name}',
    'source.url' = '{source_url}'
);
""".strip()
    
    # Template for incremental COPY INTO (Databricks SQL streaming)
    COPY_INTO_TEMPLATE = """
-- Incremental ingestion for {dataset_name}
-- Databricks SQL handles streaming and schema validation

COPY INTO {catalog}.{schema}.{table_name}
FROM (
    SELECT 
        -- Source columns (with schema enforcement)
        {select_columns},
        
        -- Bronze metadata
        current_timestamp() AS _bronze_ingestion_timestamp,
        _metadata.file_path AS _bronze_source_file,
        sha2(concat_ws('||', {hash_columns}), 256) AS _bronze_row_hash
    FROM '{source_path}'
)
FILEFORMAT = PARQUET
FORMAT_OPTIONS (
    'mergeSchema' = 'false',  -- Enforce schema, prevent drift
    'badRecordsPath' = '{bad_records_path}'
)
COPY_OPTIONS (
    'mergeSchema' = 'false',  -- Strict schema enforcement
    'force' = 'false'  -- Idempotent: skip already loaded files
);
""".strip()
    
    # Template for append-only merge (for large datasets)
    MERGE_UPSERT_TEMPLATE = """
-- Idempotent merge for {dataset_name}
-- Append-only with deduplication based on row hash

MERGE INTO {catalog}.{schema}.{table_name} AS target
USING (
    SELECT 
        {select_columns},
        current_timestamp() AS _bronze_ingestion_timestamp,
        _metadata.file_path AS _bronze_source_file,
        sha2(concat_ws('||', {hash_columns}), 256) AS _bronze_row_hash
    FROM read_files(
        '{source_path}',
        format => 'parquet',
        schema => '{enforced_schema}'
    )
) AS source
ON target._bronze_row_hash = source._bronze_row_hash
WHEN NOT MATCHED THEN INSERT *;
""".strip()


class BronzeSQLGenerator:
    """
    Generates schema-safe SQL from templates and metadata
    Python orchestrates, Databricks SQL/Spark processes
    """
    
    def __init__(
        self,
        catalog: str = "main",
        schema: str = "bronze",
        base_location: str = "/mnt/delta/bronze",
        staging_location: str = "/mnt/staging/raw"
    ):
        """
        Initialize SQL generator
        
        Args:
            catalog: Unity Catalog name
            schema: Schema/database name
            base_location: Base path for Delta tables
            staging_location: Path for raw/staging files
        """
        self.catalog = catalog
        self.schema = schema
        self.base_location = base_location
        self.staging_location = staging_location
    
    def generate_create_table_sql(
        self,
        dataset_metadata: Dict,
        partition_config: PartitionConfig,
        timestamp: str
    ) -> str:
        """
        Generate CREATE TABLE SQL from metadata
        
        Args:
            dataset_metadata: Dataset metadata from contract
            partition_config: Partitioning configuration
            timestamp: Generation timestamp
            
        Returns:
            Schema-safe CREATE TABLE SQL
        """
        dataset_name = dataset_metadata['dataset_name']
        columns = dataset_metadata['files'][0]['columns']
        
        # Generate column definitions using schema mapper
        column_ddl = SchemaMapper.generate_ddl_columns(columns)
        
        # Generate partition clause
        partition_clause = PartitionHeuristics.generate_partition_clause(partition_config)
        
        # Table location
        table_location = f"{self.base_location}/{dataset_name}"
        
        # Fill template
        sql = BronzeSQLTemplate.CREATE_TABLE_TEMPLATE.format(
            dataset_name=dataset_name,
            strategy=partition_config.strategy.value,
            timestamp=timestamp,
            catalog=self.catalog,
            schema=self.schema,
            table_name=f"bronze_{dataset_name}",
            column_definitions=column_ddl,
            partition_clause=partition_clause,
            table_location=table_location,
            source_url=dataset_metadata.get('api_endpoint', 'N/A')
        )
        
        return sql
    
    def generate_ingestion_sql(
        self,
        dataset_metadata: Dict,
        partition_config: PartitionConfig,
        use_merge: bool = False
    ) -> str:
        """
        Generate ingestion SQL (COPY INTO or MERGE)
        
        Args:
            dataset_metadata: Dataset metadata from contract
            partition_config: Partitioning configuration
            use_merge: Use MERGE instead of COPY INTO for deduplication
            
        Returns:
            Schema-safe ingestion SQL
        """
        dataset_name = dataset_metadata['dataset_name']
        columns = dataset_metadata['files'][0]['columns']
        
        # Generate SELECT columns
        column_names = [col['name'] for col in columns]
        select_columns = ",\n        ".join(column_names)
        
        # Generate hash columns (for deduplication)
        hash_columns = ", ".join([f"CAST({col} AS STRING)" for col in column_names])
        
        # Source and target paths
        source_path = f"{self.staging_location}/{dataset_name}/*.parquet"
        bad_records_path = f"{self.staging_location}/{dataset_name}/_bad_records"
        
        table_name = f"bronze_{dataset_name}"
        
        if use_merge or partition_config.use_append_only:
            # Use MERGE for large datasets or when deduplication needed
            enforced_schema = SchemaMapper.generate_spark_schema_string(columns)
            
            sql = BronzeSQLTemplate.MERGE_UPSERT_TEMPLATE.format(
                dataset_name=dataset_name,
                catalog=self.catalog,
                schema=self.schema,
                table_name=table_name,
                select_columns=select_columns,
                hash_columns=hash_columns,
                source_path=source_path,
                enforced_schema=enforced_schema
            )
        else:
            # Use COPY INTO for simple incremental loads
            sql = BronzeSQLTemplate.COPY_INTO_TEMPLATE.format(
                dataset_name=dataset_name,
                catalog=self.catalog,
                schema=self.schema,
                table_name=table_name,
                select_columns=select_columns,
                hash_columns=hash_columns,
                source_path=source_path,
                bad_records_path=bad_records_path
            )
        
        return sql
    
    def generate_optimization_sql(self, dataset_name: str) -> str:
        """
        Generate OPTIMIZE SQL for table maintenance
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            OPTIMIZE SQL
        """
        table_name = f"bronze_{dataset_name}"
        
        return f"""
-- Optimize {table_name}
OPTIMIZE {self.catalog}.{self.schema}.{table_name}
ZORDER BY (_bronze_ingestion_timestamp);

-- Vacuum old files (7 day retention)
VACUUM {self.catalog}.{self.schema}.{table_name} RETAIN 168 HOURS;
""".strip()


if __name__ == "__main__":
    # Test SQL generation
    from datetime import datetime
    
    test_metadata = {
        'dataset_name': 'billing_payments',
        'api_endpoint': 'https://example.com/data',
        'total_rows': 200000,
        'files': [{
            'columns': [
                {'name': 'customer_id', 'type': 'string', 'nullable': True},
                {'name': 'disco', 'type': 'string', 'nullable': True},
                {'name': 'amount_paid_ngn', 'type': 'double', 'nullable': True}
            ]
        }]
    }
    
    # Determine partition strategy
    partition_config = PartitionHeuristics.determine_strategy(
        test_metadata['dataset_name'],
        test_metadata['total_rows'],
        test_metadata['files'][0]['columns'],
        1
    )
    
    # Generate SQL
    generator = BronzeSQLGenerator()
    
    print("="*80)
    print("CREATE TABLE SQL:")
    print("="*80)
    create_sql = generator.generate_create_table_sql(
        test_metadata, 
        partition_config,
        datetime.utcnow().isoformat()
    )
    print(create_sql)
    
    print("\n" + "="*80)
    print("INGESTION SQL:")
    print("="*80)
    ingest_sql = generator.generate_ingestion_sql(test_metadata, partition_config)
    print(ingest_sql)