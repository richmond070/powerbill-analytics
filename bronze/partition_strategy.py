"""
Partition Strategy Module
Automatically determines optimal partitioning based on dataset characteristics
Uses heuristics to choose partition columns and strategies
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PartitionStrategy(Enum):
    """Partition strategies based on dataset size"""
    NONE = "none"  # Small datasets, no partitioning
    TIME_BASED = "time_based"  # Partition by date/timestamp
    CATEGORY_BASED = "category_based"  # Partition by categorical column
    HYBRID = "hybrid"  # Combination of time + category


@dataclass
class PartitionConfig:
    """Configuration for table partitioning"""
    strategy: PartitionStrategy
    partition_columns: List[str]
    reason: str
    use_append_only: bool


class PartitionHeuristics:
    """
    Determines optimal partitioning strategy using heuristics
    Follows best practices for Delta Lake partitioning
    """
    
    # Thresholds for partitioning decisions
    SMALL_DATASET_THRESHOLD = 100_000  # Rows - don't partition
    LARGE_DATASET_THRESHOLD = 500_000  # Rows - definitely partition
    PARTITION_SIZE_TARGET = 1_000_000_000  # 1GB target per partition
    
    # Common time column patterns
    TIME_COLUMN_PATTERNS = [
        'timestamp', 'created_time', 'resolved_time', 
        'billing_month', 'as_of_date', 'date', 'datetime'
    ]
    
    # Common categorical column patterns for partitioning
    CATEGORY_COLUMN_PATTERNS = [
        'disco', 'region', 'state', 'country', 'department',
        'category', 'status', 'type', 'site_type'
    ]
    
    @classmethod
    def determine_strategy(
        cls, 
        dataset_name: str,
        total_rows: int,
        columns: List[Dict],
        file_count: int
    ) -> PartitionConfig:
        """
        Determine optimal partition strategy based on dataset characteristics
        
        Args:
            dataset_name: Name of the dataset
            total_rows: Total number of rows
            columns: Column metadata from contract
            file_count: Number of source files
            
        Returns:
            PartitionConfig with strategy and columns
        """
        column_names = [col['name'].lower() for col in columns]
        
        # RULE 1: Small datasets don't need partitioning
        if total_rows < cls.SMALL_DATASET_THRESHOLD:
            return PartitionConfig(
                strategy=PartitionStrategy.NONE,
                partition_columns=[],
                reason=f"Dataset is small ({total_rows:,} rows). No partitioning needed.",
                use_append_only=False
            )
        
        # RULE 2: Find time-based columns
        time_columns = cls._find_time_columns(column_names)
        
        # RULE 3: Find high-cardinality categorical columns
        category_columns = cls._find_category_columns(column_names)
        
        # RULE 4: Determine strategy based on available columns
        if total_rows >= cls.LARGE_DATASET_THRESHOLD:
            # Large dataset - use append-only + partitioning
            if time_columns and category_columns:
                # Best case: time + category partitioning
                return PartitionConfig(
                    strategy=PartitionStrategy.HYBRID,
                    partition_columns=[time_columns[0], category_columns[0]],
                    reason=f"Large dataset ({total_rows:,} rows) with time and category columns. "
                           f"Hybrid partitioning for optimal query performance.",
                    use_append_only=True
                )
            elif time_columns:
                # Time-based partitioning
                return PartitionConfig(
                    strategy=PartitionStrategy.TIME_BASED,
                    partition_columns=[time_columns[0]],
                    reason=f"Large dataset ({total_rows:,} rows) with time column. "
                           f"Time-based partitioning for incremental loads.",
                    use_append_only=True
                )
            elif category_columns:
                # Category-based partitioning
                return PartitionConfig(
                    strategy=PartitionStrategy.CATEGORY_BASED,
                    partition_columns=[category_columns[0]],
                    reason=f"Large dataset ({total_rows:,} rows) with category column. "
                           f"Category-based partitioning for filtered queries.",
                    use_append_only=True
                )
        
        # RULE 5: Medium datasets - simpler partitioning
        if time_columns:
            return PartitionConfig(
                strategy=PartitionStrategy.TIME_BASED,
                partition_columns=[time_columns[0]],
                reason=f"Medium dataset ({total_rows:,} rows). Time-based partitioning.",
                use_append_only=total_rows >= 200_000
            )
        
        # RULE 6: No obvious partition column - no partitioning
        return PartitionConfig(
            strategy=PartitionStrategy.NONE,
            partition_columns=[],
            reason=f"No suitable partition columns found for {total_rows:,} rows.",
            use_append_only=total_rows > 200_000
        )
    
    @classmethod
    def _find_time_columns(cls, column_names: List[str]) -> List[str]:
        """Find time-based columns suitable for partitioning"""
        time_cols = []
        for col in column_names:
            if any(pattern in col for pattern in cls.TIME_COLUMN_PATTERNS):
                time_cols.append(col)
        
        # Prioritize certain patterns
        priority_order = ['timestamp', 'created_time', 'billing_month', 'as_of_date', 'date']
        for priority in priority_order:
            if priority in time_cols:
                time_cols.insert(0, time_cols.pop(time_cols.index(priority)))
        
        return time_cols
    
    @classmethod
    def _find_category_columns(cls, column_names: List[str]) -> List[str]:
        """Find categorical columns suitable for partitioning"""
        cat_cols = []
        for col in column_names:
            if any(pattern in col for pattern in cls.CATEGORY_COLUMN_PATTERNS):
                cat_cols.append(col)
        
        # Prioritize 'disco' (distribution company) if present
        if 'disco' in cat_cols:
            cat_cols.insert(0, cat_cols.pop(cat_cols.index('disco')))
         
        return cat_cols
    
    @classmethod
    def generate_partition_clause(cls, partition_config: PartitionConfig) -> str:
        """
        Generate PARTITIONED BY clause for SQL
        
        Args:
            partition_config: Partition configuration
            
        Returns:
            SQL PARTITIONED BY clause or empty string
        """
        if partition_config.strategy == PartitionStrategy.NONE:
            return ""
        
        partition_cols = ", ".join(partition_config.partition_columns)
        return f"PARTITIONED BY ({partition_cols})"


