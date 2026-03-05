"""
Schema Type Mapper
Converts bronze_ingestion_contract.json types to Databricks SQL types
Prevents type explosion and enforces strict schema
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ColumnSchema:
    """Represents a column schema definition"""
    name: str
    sql_type: str
    nullable: bool
    

class SchemaMapper:
    """Maps contract types to Databricks SQL types with strict enforcement"""
    
    # Type mapping from contract to Databricks SQL
    TYPE_MAP = {
        'string': 'STRING',
        'double': 'DOUBLE',
        'bool': 'BOOLEAN',
        'int64': 'BIGINT',
        'int32': 'INT',
        'float': 'FLOAT',
        'timestamp': 'TIMESTAMP',
        'date': 'DATE',
        'binary': 'BINARY'
    }
    
    @classmethod
    def map_type(cls, contract_type: str) -> str:
        """
        Map contract type to Databricks SQL type
        
        Args:
            contract_type: Type from bronze_ingestion_contract.json
            
        Returns:
            Databricks SQL type string
            
        Raises:
            ValueError: If type is not recognized (prevents type explosion)
        """
        sql_type = cls.TYPE_MAP.get(contract_type.lower())
        if sql_type is None:
            raise ValueError(
                f"Unrecognized type '{contract_type}'. "
                f"Allowed types: {list(cls.TYPE_MAP.keys())}"
            )
        return sql_type
    
    @classmethod
    def build_column_schema(cls, column_metadata: Dict) -> ColumnSchema:
        """
        Build column schema from contract metadata
        
        Args:
            column_metadata: Column definition from contract
            
        Returns:
            ColumnSchema object
        """
        return ColumnSchema(
            name=column_metadata['name'],
            sql_type=cls.map_type(column_metadata['type']),
            nullable=column_metadata.get('nullable', True)
        )
    
    @classmethod
    def generate_ddl_columns(cls, columns: List[Dict]) -> str:
        """
        Generate DDL column definitions from contract columns
        
        Args:
            columns: List of column metadata from contract
            
        Returns:
            DDL string for column definitions
            
        Example:
            >>> columns = [{'name': 'id', 'type': 'string', 'nullable': False}]
            >>> SchemaMapper.generate_ddl_columns(columns)
            'id STRING NOT NULL'
        """
        ddl_parts = []
        
        for col_meta in columns:
            col_schema = cls.build_column_schema(col_meta)
            nullable_clause = "" if col_schema.nullable else " NOT NULL"
            ddl_parts.append(f"{col_schema.name} {col_schema.sql_type}{nullable_clause}")
        
        return ",\n    ".join(ddl_parts)
    
    @classmethod
    def generate_spark_schema_string(cls, columns: List[Dict]) -> str:
        """
        Generate Spark schema string for enforced reads
        Prevents corrupt files from crashing ingestion
        
        Args:
            columns: List of column metadata from contract
            
        Returns:
            Spark schema definition string
        """
        schema_parts = []
        
        for col_meta in columns:
            col_schema = cls.build_column_schema(col_meta)
            # Map SQL types to Spark types
            spark_type = cls._sql_to_spark_type(col_schema.sql_type)
            schema_parts.append(f"{col_schema.name} {spark_type}")
        
        return ", ".join(schema_parts)
    
    @classmethod
    def _sql_to_spark_type(cls, sql_type: str) -> str:
        """Convert SQL type to Spark type string"""
        spark_map = {
            'STRING': 'string',
            'DOUBLE': 'double',
            'BOOLEAN': 'boolean',
            'BIGINT': 'bigint',
            'INT': 'int',
            'FLOAT': 'float',
            'TIMESTAMP': 'timestamp',
            'DATE': 'date',
            'BINARY': 'binary'
        }
        return spark_map.get(sql_type, 'string')


if __name__ == "__main__":
    # Test the mapper
    test_columns = [
        {'name': 'customer_id', 'type': 'string', 'nullable': True},
        {'name': 'amount', 'type': 'double', 'nullable': False},
        {'name': 'is_active', 'type': 'bool', 'nullable': True},
    ]
    
    print("DDL Columns:")
    print(SchemaMapper.generate_ddl_columns(test_columns))
    print("\nSpark Schema:")
    print(SchemaMapper.generate_spark_schema_string(test_columns))