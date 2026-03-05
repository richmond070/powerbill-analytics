"""
Databricks SQL API Client
Executes SQL via Databricks SQL API
Handles streaming, batching, and error recovery
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
import configparser


@dataclass
class SQLExecutionResult:
    """Result of SQL execution"""
    statement_id: str
    status: str
    row_count: Optional[int]
    duration_ms: int
    error_message: Optional[str] = None
    
CONFIG_PATH = "databricks/databricks.cfg"

class DatabricksSQLClient:
    """
    Client for Databricks SQL API
    Executes SQL statements and monitors execution
    """
    
    def __init__(
        self,
        config_path = CONFIG_PATH,
        catalog: str = "main",
        schema: str = "bronze"
    ):
        """
        Initialize Databricks SQL client
        
        Args:
            catalog: Unity Catalog name
            schema: Schema/database name
        """
        parser = configparser.ConfigParser()
        parser.read(config_path)


        self.workspace_url = parser["DEFAULT"]["workspace_url"].rstrip("/")
        self.token = parser["DEFAULT"]["token"]
        self.warehouse_id = parser["DEFAULT"]["warehouse_id"]
        self.catalog = catalog
        self.schema = schema
        
        if not self.workspace_url or not self.token :
            raise ValueError("Databricks workspace URL or token missing in config file")

        if not self.warehouse_id:
            raise ValueError("Databricks warehouse_id missing in config file")

        
        self.api_base = f"{self.workspace_url}/api/2.0"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    

    def _load_config(self, config_path: str) -> dict:
        """
        Load Databricks config file if it exists
        """
        config = configparser.ConfigParser()

        if not os.path.exists(config_path):
            return {}

        config.read(config_path)
        return config["DEFAULT"]
    

    def execute_sql(
        self,
        sql_statement: str,
        wait_timeout: int = 300,
    ) -> SQLExecutionResult:
        
        """
        Execute SQL statement via SQL API
        
        Args:
            sql_statement: SQL to execute
            wait_timeout: Max wait time in seconds
            warehouse_id: Override default warehouse
            
        Returns:
            SQLExecutionResult with execution details
        """
        start_time = time.time()
        
        # Submit statement
        statement_id = self._submit_statement(sql_statement)
        
        # Poll for completion
        result = self._wait_for_completion(statement_id, wait_timeout)
        
        duration_ms = int((time.time() - start_time) * 1000)
        result.duration_ms = duration_ms
        
        return result
    
    def _submit_statement(
        self,
        sql_statement: str,
    ) -> str:
        """
        Submit SQL statement for execution
        
        Args:
            sql_statement: SQL to execute
            warehouse_id: SQL warehouse ID
            
        Returns:
            Statement ID for tracking
        """
        endpoint = f"{self.api_base}/sql/statements"
        
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": sql_statement,
            "catalog": self.catalog,
            "schema": self.schema,
            "wait_timeout": "0s"  # Return immediately
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Failed to submit SQL: {response.text}")
        
        result = response.json()
        return result['statement_id']
    
    def _wait_for_completion(
        self,
        statement_id: str,
        wait_timeout: int
    ) -> SQLExecutionResult:
        """
        Wait for statement execution to complete
        
        Args:
            statement_id: Statement ID to monitor
            wait_timeout: Max wait time in seconds
            
        Returns:
            SQLExecutionResult
        """
        endpoint = f"{self.api_base}/sql/statements/{statement_id}"
        
        start_time = time.time()
        poll_interval = 1  # Start with 1 second
        
        while True:
            if time.time() - start_time > wait_timeout:
                raise TimeoutError(f"SQL execution timed out after {wait_timeout}s")
            
            response = requests.get(endpoint, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to get statement status: {response.text}")
            
            result = response.json()
            status = result['status']['state']
            
            if status == 'SUCCEEDED':
                return SQLExecutionResult(
                    statement_id=statement_id,
                    status='SUCCEEDED',
                    row_count=self._extract_row_count(result),
                    duration_ms=0  # Will be set by caller
                )
            
            elif status in ['FAILED', 'CANCELED', 'CLOSED']:
                error_msg = result.get('status', {}).get('error', {}).get('message', 'Unknown error')
                return SQLExecutionResult(
                    statement_id=statement_id,
                    status=status,
                    row_count=None,
                    duration_ms=0,
                    error_message=error_msg
                )
            
            # Still running - wait and retry
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 10)  # Exponential backoff, max 10s
    
    def _extract_row_count(self, result: Dict) -> Optional[int]:
        """Extract row count from result"""
        try:
            manifest = result.get('manifest', {})
            # For DML operations
            if 'row_count' in manifest:
                return manifest['row_count']
            # For SELECT operations
            if 'total_row_count' in manifest:
                return manifest['total_row_count']
        except Exception:
            pass
        return None
    
    def execute_batch(
        self,
        sql_statements: List[str],
        continue_on_error: bool = False
    ) -> List[SQLExecutionResult]:
        """
        Execute multiple SQL statements in sequence
        
        Args:
            sql_statements: List of SQL statements
            continue_on_error: Continue if a statement fails
            
        Returns:
            List of SQLExecutionResult
        """
        results = []
        
        for i, sql in enumerate(sql_statements, 1):
            print(f"Executing statement {i}/{len(sql_statements)}...")
            
            try:
                result = self.execute_sql(sql)
                results.append(result)
                
                if result.status != 'SUCCEEDED':
                    print(f"  ❌ Failed: {result.error_message}")
                    if not continue_on_error:
                        break
                else:
                    row_info = f"{result.row_count} rows" if result.row_count else "completed"
                    print(f"  ✓ Success: {row_info} ({result.duration_ms}ms)")
                    
            except Exception as e:
                error_result = SQLExecutionResult(
                    statement_id="N/A",
                    status="ERROR",
                    row_count=None,
                    duration_ms=0,
                    error_message=str(e)
                )
                results.append(error_result)
                print(f" Error: {str(e)}")
                
                if not continue_on_error:
                    break
        
        return results
    
    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """
        Get information about a table
        
        Args:
            table_name: Name of table
            
        Returns:
            Table information or None
        """
        sql = f"DESCRIBE TABLE EXTENDED {self.catalog}.{self.schema}.{table_name}"
        
        try:
            result = self.execute_sql(sql)
            if result.status == 'SUCCEEDED':
                return {'exists': True, 'statement_id': result.statement_id}
        except Exception:
            pass
        
        return None


class SQLExecutionLogger:
    """Logs SQL execution for audit trail"""
    
    def __init__(self, log_file: str = "bronze_ingestion_log.json"):
        self.log_file = log_file
    
    def log_execution(
        self,
        dataset_name: str,
        sql_type: str,
        result: SQLExecutionResult,
        sql_statement: str
    ):
        """Log SQL execution"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'dataset': dataset_name,
            'sql_type': sql_type,
            'statement_id': result.statement_id,
            'status': result.status,
            'row_count': result.row_count,
            'duration_ms': result.duration_ms,
            'error': result.error_message,
            'sql_preview': sql_statement[:200] + '...' if len(sql_statement) > 200 else sql_statement
        }
        
        # Append to log file
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write log: {e}")


if __name__ == "__main__":
    print("Testing Databricks SQL connection...")

    try:
        client = DatabricksSQLClient()

        # Cheapest possible query
        result = client.execute_sql("SELECT 1")

        print("Connection successful")
        print(result)

    except Exception as e:
        print("Connection failed")
        raise
