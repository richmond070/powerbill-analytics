from typing import Optional
import pandas as pd

from .db_client import run_query

CATALOG_SCHEMA = "main.gold"


def _disco_clause(disco: Optional[str]) -> str:
    return "AND disco = %(disco)s" if disco and disco != "All" else ""


def _date_clause(column: str, start_date, end_date) -> str:
    if start_date and end_date:
        return f"AND {column} BETWEEN %(start_date)s AND %(end_date)s"
    return ""


def _build_params(disco, start_date=None, end_date=None) -> dict:
    params = {}
    if disco and disco != "All":
        params["disco"] = disco
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    return params


# ---------------------------------------------------------------------------
# Grid Reliability  (signature dashboard)
# ---------------------------------------------------------------------------

def get_grid_reliability(
    disco: Optional[str] = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    mart_grid_reliability filtered by disco and/or complaint_month range.
    Powers: stress heatmap, SLA breach rate, instability trend, stress
    complaint share, grid_reliability_status RAG (critical/degraded/
    moderate/stable — pre-computed in dbt).
    """
    sql = f"""
        SELECT *
        FROM {CATALOG_SCHEMA}.mart_grid_reliability
        WHERE 1=1
            {_disco_clause(disco)}
            {_date_clause('complaint_month', start_date, end_date)}
        ORDER BY disco, complaint_month
    """
    return run_query(sql, _build_params(disco, start_date, end_date))


# ---------------------------------------------------------------------------
# Executive Revenue
# ---------------------------------------------------------------------------

def get_revenue_trends(
    disco: Optional[str] = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    mart_revenue_trends filtered by disco and/or billing_month range.
    Powers: monthly revenue trend, collection efficiency, tariff-band
    profitability, MoM revenue change.
    """
    sql = f"""
        SELECT *
        FROM {CATALOG_SCHEMA}.mart_revenue_trends
        WHERE 1=1
            {_disco_clause(disco)}
            {_date_clause('billing_month', start_date, end_date)}
        ORDER BY billing_month, disco, tariff_band
    """
    return run_query(sql, _build_params(disco, start_date, end_date))


def get_payment_behavior(disco: Optional[str] = None) -> pd.DataFrame:
    """
    mart_payment_behavior filtered by disco only (no date column — lifetime
    per-customer payment profile).
    Powers: chronic late payer ratio, payment_segment / collection_tier mix.
    """
    sql = f"""
        SELECT *
        FROM {CATALOG_SCHEMA}.mart_payment_behavior
        WHERE 1=1
            {_disco_clause(disco)}
        ORDER BY total_arrears DESC
    """
    return run_query(sql, _build_params(disco))


# ---------------------------------------------------------------------------
# Customer Risk
# ---------------------------------------------------------------------------

def get_customer_lifetime_value(disco: Optional[str] = None) -> pd.DataFrame:
    """
    mart_customer_lifetime_value filtered by disco only (no date column).
    Powers: CLV segmentation (platinum/gold/silver/bronze tiers), complaint-
    risk correlation via pct_complaints_during_stress vs clv_score.
    """
    sql = f"""
        SELECT *
        FROM {CATALOG_SCHEMA}.mart_customer_lifetime_value
        WHERE 1=1
            {_disco_clause(disco)}
        ORDER BY clv_score DESC
    """
    return run_query(sql, _build_params(disco))


def get_outstanding_balances(
    disco: Optional[str] = None,
    start_date=None,
    end_date=None,
    high_risk_only: bool = False,
) -> pd.DataFrame:
    """
    mart_outstanding_balances filtered by disco and/or billing_month range.
    Powers: high-risk customer table, arrears growth trend.

    Args:
        high_risk_only: adds WHERE arrears_growing_flag = true OR
            high_outstanding_flag = true — both flags are pre-computed in
            dbt, so this is a plain filter, no client-side derivation needed.
    """
    risk_clause = (
        "AND (arrears_growing_flag = true OR high_outstanding_flag = true)"
        if high_risk_only else ""
    )
    sql = f"""
        SELECT *
        FROM {CATALOG_SCHEMA}.mart_outstanding_balances
        WHERE 1=1
            {_disco_clause(disco)}
            {_date_clause('billing_month', start_date, end_date)}
            {risk_clause}
        ORDER BY customer_id, billing_month
    """
    return run_query(sql, _build_params(disco, start_date, end_date))


def get_available_discos() -> list:
    """
    Distinct disco list for the global filter dropdown in app.py.
    Sourced from mart_grid_reliability — broadest complaint coverage of the
    five marts, so any active disco should appear here.
    """
    sql = f"SELECT DISTINCT disco FROM {CATALOG_SCHEMA}.mart_grid_reliability ORDER BY disco"
    df = run_query(sql)
    return df["disco"].tolist() if not df.empty else []