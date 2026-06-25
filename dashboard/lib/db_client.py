import configparser
import os
from typing import Optional

import pandas as pd
import streamlit as st
from databricks import sql as databricks_sql

DEFAULT_CONFIG_PATH = os.getenv(
    "DATABRICKS_CFG_PATH",
    "databricks/databricks.cfg",
)


class DatabricksReadClient:
    """
    Thin wrapper around databricks-sql-connector for read-only dashboard queries.

    Args:
        config_path: Path to databricks.cfg. Defaults to the DATABRICKS_CFG_PATH
                     env var (already set for the airflow containers in
                     docker-compose.yaml) or 'databricks/databricks.cfg' locally.
        catalog:     Unity Catalog name.
        schema:      Default schema. Queries in queries.py always fully-qualify
                     as 'main.gold.<table>' so this is mostly a connector default.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        catalog: str = "main",
        schema: str = "gold",
    ) -> None:
        parser = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Databricks config not found at: {config_path}\n"
                f"Expected a [DEFAULT] section with workspace_url, token, warehouse_id."
            )
        parser.read(config_path)

        workspace_url = parser["DEFAULT"]["workspace_url"].rstrip("/")
        token = parser["DEFAULT"]["token"]
        warehouse_id = parser["DEFAULT"]["warehouse_id"]

        if not workspace_url or not token:
            raise ValueError("Databricks workspace_url or token missing in config file")
        if not warehouse_id:
            raise ValueError("Databricks warehouse_id missing in config file")

        # databricks-sql-connector wants a bare hostname, not a full https:// URL
        self.server_hostname = workspace_url.replace("https://", "").replace("http://", "")
        self.http_path = f"/sql/1.0/warehouses/{warehouse_id}"
        self.access_token = token
        self.catalog = catalog
        self.schema = schema

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SELECT statement and return the results as a DataFrame.

        Args:
            sql:    SQL SELECT statement. Use %(name)s placeholders for params.
            params: Optional dict of bind parameters — passed straight to the
                    connector, never f-string'd into the SQL by hand.

        Returns:
            pandas.DataFrame of the result set. Empty DataFrame on zero rows.
        """
        with databricks_sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
            catalog=self.catalog,
            schema=self.schema,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or {})
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return pd.DataFrame(rows, columns=columns)


@st.cache_resource(show_spinner=False)
def get_client() -> DatabricksReadClient:
    """
    Streamlit-cached singleton. cache_resource (not cache_data) is correct here
    because this object holds connection *credentials*, not query results —
    create once per session/process, reuse across reruns.
    """
    return DatabricksReadClient()


@st.cache_data(ttl=300, show_spinner="Querying Databricks…")
def run_query(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Cached query execution. ttl=300s means a given (sql, params) pair is only
    re-sent to the warehouse once every 5 minutes — Gold tables refresh on the
    Airflow schedule, not real-time, so this avoids hammering the warehouse on
    every filter tweak or page rerun.

    NOTE: params must be a hashable, JSON-safe dict (str/int/float/bool/None
    values only) since Streamlit hashes function args to key the cache.
    """
    client = get_client()
    return client.query(sql, params)