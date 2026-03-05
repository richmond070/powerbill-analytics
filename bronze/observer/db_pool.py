"""
PostgreSQL Connection Pool
Shared connection pool for the Bronze observability layer.
Config is read from databricks.cfg [POSTGRES] section or environment variables.

Usage:
    pool = get_pool()
    with pool.getconn() as conn:
        with conn.cursor() as cur:
            cur.execute(...)
        conn.commit()
    pool.putconn(conn)

Or use the context-manager helper:
    with pg_connection() as conn:
        ...
"""

import os
import configparser
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool


# Module-level singleton — created once, shared across the process
_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def _build_dsn(config_path: str = "databricks/databricks.cfg") -> str:
    """
    Build a PostgreSQL DSN string.

    Priority order:
      1. Environment variables  (PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD)
      2. [POSTGRES] section in databricks.cfg

    Args:
        config_path: Path to the shared config file.

    Returns:
        DSN string understood by psycopg2.connect().
    """
    # --- Try environment variables first (CI / cloud-friendly) ---
    host     = os.getenv("PG_HOST")
    port     = os.getenv("PG_PORT", "5432")
    dbname   = os.getenv("PG_DB")
    user     = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")

    if not all([host, dbname, user, password]):
        # Fall back to config file
        parser = configparser.ConfigParser()
        if os.path.exists(config_path):
            parser.read(config_path)
            section = "POSTGRES"
            if parser.has_section(section):
                host     = host     or parser.get(section, "host",     fallback=None)
                port     = port     or parser.get(section, "port",     fallback="5432")
                dbname   = dbname   or parser.get(section, "dbname",   fallback=None)
                user     = user     or parser.get(section, "user",     fallback=None)
                password = password or parser.get(section, "password", fallback=None)

    missing = [k for k, v in
               {"PG_HOST": host, "PG_DB": dbname, "PG_USER": user, "PG_PASSWORD": password}.items()
               if not v]
    if missing:
        raise EnvironmentError(
            f"PostgreSQL connection config incomplete. Missing: {missing}. "
            "Set env vars PG_HOST / PG_DB / PG_USER / PG_PASSWORD, "
            "or add a [POSTGRES] section to databricks/databricks.cfg."
        )

    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def get_pool(
    config_path: str = "databricks/databricks.cfg",
    minconn: int = 1,
    maxconn: int = 5,
) -> pg_pool.ThreadedConnectionPool:
    """
    Return the module-level connection pool, creating it on first call.

    Args:
        config_path: Path to shared config file.
        minconn:     Minimum idle connections kept open.
        maxconn:     Maximum concurrent connections allowed.

    Returns:
        A ThreadedConnectionPool instance safe for multi-threaded use.
    """
    global _pool
    if _pool is None:
        dsn = _build_dsn(config_path)
        _pool = pg_pool.ThreadedConnectionPool(minconn, maxconn, dsn=dsn)
    return _pool


@contextmanager
def pg_connection(config_path: str = "databricks/databricks.cfg"):
    """
    Context manager that yields a psycopg2 connection from the pool.
    Commits on clean exit, rolls back on exception, always returns connection.

    Usage:
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    Args:
        config_path: Path to shared config file (only used if pool not yet created).
    """
    pool = get_pool(config_path)
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def close_pool():
    """Gracefully close all connections in the pool (call at process shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
