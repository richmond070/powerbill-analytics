"""
Executive Revenue Dashboard — Page 2
======================================
Data sources:
  main.gold.mart_revenue_trends
    → defined in dbt_project/models/marts/finance/mart_revenue_trends.sql
    → grain: (billing_month, disco, tariff_band)
    → query: dashboard/lib/queries.get_revenue_trends()
    → date filter: YES (billing_month column present)

  main.gold.mart_payment_behavior
    → defined in dbt_project/models/marts/finance/mart_payment_behavior.sql
    → grain: (customer_id) — lifetime summary, no date column
    → query: dashboard/lib/queries.get_payment_behavior()
    → date filter: NO — only disco filter applies

Global filters read from st.session_state, written by dashboard/app.py.
All *_pct columns from both marts are already on a 0-100 scale (dbt
multiplies by 100 before writing). formatting.py functions expect 0-100 —
do not multiply again here.

Layout:
  Row 1  KPI cards: total billed, total collected, avg collection rate,
                    chronic late payer ratio
  Row 2  Monthly revenue trend — billed vs collected line chart
  Row 3  MoM revenue change % — bar chart, positive/negative coloured
  Row 4  Tariff-band profitability — billed revenue + collection rate
  Row 5  Payment segment distribution — chronic/occasional/reliable split
  Row 6  Collection tier breakdown — high/medium/low by total arrears
"""

import os
import sys

# Make dashboard/lib/ importable from dashboard/pages/
# os.path.abspath(__file__) → full path to this file (dashboard/pages/2_Executive_Revenue.py)
# os.path.dirname(...)       → dashboard/pages/
# os.path.join(..., "..")    → dashboard/  ← where lib/ lives
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import queries
from lib.formatting import format_delta_pct, format_ngn, format_number, format_pct

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Revenue | Energy Ops",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Read global filters from st.session_state
# Written by dashboard/app.py sidebar on every rerun.
# .get() defaults make the page work when accessed directly by URL.
# ---------------------------------------------------------------------------
disco      = st.session_state.get("global_disco", "All")
start_date = st.session_state.get("global_start_date", None)
end_date   = st.session_state.get("global_end_date", None)

st.title("💰 Executive Revenue")
st.caption(
    f"Sources: `main.gold.mart_revenue_trends` · "
    f"`main.gold.mart_payment_behavior` · "
    f"Disco: **{disco}** · "
    f"Period: **{start_date or 'all time'} → {end_date or 'latest'}**"
)

st.caption(
    "ℹ️ Date range applies to revenue trends only. "
    "Payment behavior is a lifetime customer summary with no date grain."
)

# ---------------------------------------------------------------------------
# Load data — mart_revenue_trends (date filter applies)
# Query function lives in dashboard/lib/queries.py → get_revenue_trends()
# Caching: dashboard/lib/db_client.run_query() caches at ttl=300s
# ---------------------------------------------------------------------------
with st.spinner("Loading mart_revenue_trends…"):
    try:
        df_rev = queries.get_revenue_trends(
            disco=disco,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        st.error(f"Failed to load mart_revenue_trends: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Load data — mart_payment_behavior (disco filter only, no date column)
# ---------------------------------------------------------------------------
with st.spinner("Loading mart_payment_behavior…"):
    try:
        df_pay = queries.get_payment_behavior(disco=disco)
    except Exception as e:
        st.error(f"Failed to load mart_payment_behavior: {e}")
        st.stop()

if df_rev.empty and df_pay.empty:
    st.warning(
        "No data for the selected filters. "
        "Try 'All' discos or a wider date range."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Type normalisation
# billing_month comes back as string from the connector — cast to datetime
# so Plotly and pandas time-series operations work correctly.
# mart_payment_behavior has no date columns — nothing to cast there.
# ---------------------------------------------------------------------------
if not df_rev.empty:
    df_rev["billing_month"] = pd.to_datetime(df_rev["billing_month"])
    df_rev = df_rev.sort_values(["billing_month", "disco", "tariff_band"])

# ---------------------------------------------------------------------------
# Row 1 — KPI Cards
# 
# Card 1: Total Revenue Billed      → mart_revenue_trends.total_revenue_billed
# Card 2: Total Revenue Collected   → mart_revenue_trends.total_revenue_collected
# Card 3: Avg Collection Rate       → mart_revenue_trends.collection_rate_pct
#           already 0-100 in dbt (ROUND(avg_collection_rate * 100, 2))
# Card 4: Chronic Late Payer Ratio  → mart_payment_behavior.payment_segment
#           count where payment_segment = 'chronic_late' / total customers
# ---------------------------------------------------------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Cards 1–3 from mart_revenue_trends
if not df_rev.empty:
    total_billed    = df_rev["total_revenue_billed"].sum()
    total_collected = df_rev["total_revenue_collected"].sum()
    avg_collection  = df_rev["collection_rate_pct"].mean()

    # MoM delta for collection rate — latest month vs previous month
    latest_month = df_rev["billing_month"].max()
    prev_month   = df_rev[df_rev["billing_month"] < latest_month]["billing_month"].max()

    if pd.notna(prev_month):
        collection_latest = df_rev[df_rev["billing_month"] == latest_month]["collection_rate_pct"].mean()
        collection_prev   = df_rev[df_rev["billing_month"] == prev_month]["collection_rate_pct"].mean()
        collection_delta  = collection_latest - collection_prev
    else:
        collection_delta = None
else:
    total_billed    = None
    total_collected = None
    avg_collection  = None
    collection_delta = None

col1.metric(
    label="Total Revenue Billed",
    value=format_ngn(total_billed),
    help="Sum of amount_billed_ngn across all customers, months, and tariff bands in the selected period.",
)

col2.metric(
    label="Total Revenue Collected",
    value=format_ngn(total_collected),
    help="Sum of amount_paid_ngn. Gap vs billed = total uncollected revenue.",
)

col3.metric(
    label="Avg Collection Rate",
    value=format_pct(avg_collection),
    delta=format_delta_pct(collection_delta),
    help=(
        "Average collection_rate_pct from mart_revenue_trends. "
        "Already 0-100 scale in dbt. Delta = latest month vs previous month."
    ),
)

# Card 4 from mart_payment_behavior
if not df_pay.empty:
    total_customers  = len(df_pay)
    chronic_count    = (df_pay["payment_segment"] == "chronic_late").sum()
    chronic_ratio    = (chronic_count / total_customers * 100) if total_customers > 0 else None
else:
    chronic_ratio   = None
    chronic_count   = 0
    total_customers = 0

col4.metric(
    label="Chronic Late Payer Ratio",
    value=format_pct(chronic_ratio),
    delta_color="inverse",
    help=(
        f"Customers with payment_segment = 'chronic_late' "
        f"(on_time_rate < 50%) as % of all customers. "
        f"{format_number(chronic_count)} of {format_number(total_customers)} customers."
    ),
)

st.divider()

# ---------------------------------------------------------------------------
# Row 2 — Monthly Revenue Trend
# Line chart: billed and collected revenue over time, one line per metric.
# Aggregated across all discos and tariff bands in the filtered set so the
# trend is visible at the portfolio level.
# Source columns: billing_month, total_revenue_billed, total_revenue_collected
# ---------------------------------------------------------------------------
st.subheader("Monthly Revenue Trend — Billed vs Collected")
st.caption(
    "Aggregated across all discos and tariff bands in the current filter. "
    "Gap between lines = uncollected revenue (total_revenue_uncollected)."
)

if not df_rev.empty:
    monthly_agg = (
        df_rev.groupby("billing_month")[
            ["total_revenue_billed", "total_revenue_collected", "total_revenue_uncollected"]
        ]
        .sum()
        .reset_index()
    )

    fig_trend = go.Figure()
    fig_trend.add_scatter(
        x=monthly_agg["billing_month"],
        y=monthly_agg["total_revenue_billed"],
        mode="lines+markers",
        name="Billed",
        line=dict(color="#3b82f6", width=2),
    )
    fig_trend.add_scatter(
        x=monthly_agg["billing_month"],
        y=monthly_agg["total_revenue_collected"],
        mode="lines+markers",
        name="Collected",
        line=dict(color="#22c55e", width=2),
    )
    fig_trend.add_scatter(
        x=monthly_agg["billing_month"],
        y=monthly_agg["total_revenue_uncollected"],
        mode="lines+markers",
        name="Uncollected",
        line=dict(color="#ef4444", width=2, dash="dot"),
    )
    fig_trend.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Billing Month"),
        yaxis=dict(title="NGN"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No revenue trend data for the selected filters.")

st.divider()

# ---------------------------------------------------------------------------
# Row 3 — MoM Revenue Change %
# Bar chart coloured green (positive) / red (negative).
# mom_revenue_change_pct is pre-computed in mart_revenue_trends.sql using a
# LAG window function partitioned by (disco, tariff_band).
# Aggregated here to a single portfolio-level MoM signal.
# ---------------------------------------------------------------------------
st.subheader("Month-over-Month Revenue Change (%)")
st.caption(
    "mom_revenue_change_pct from mart_revenue_trends — LAG window function "
    "per (disco, tariff_band). Shown here as portfolio average per month."
)

if not df_rev.empty:
    mom_agg = (
        df_rev.groupby("billing_month")["mom_revenue_change_pct"]
        .mean()
        .reset_index()
        .dropna(subset=["mom_revenue_change_pct"])
    )

    if not mom_agg.empty:
        mom_agg["colour"] = mom_agg["mom_revenue_change_pct"].apply(
            lambda v: "#22c55e" if v >= 0 else "#ef4444"
        )
        fig_mom = go.Figure(
            go.Bar(
                x=mom_agg["billing_month"],
                y=mom_agg["mom_revenue_change_pct"],
                marker_color=mom_agg["colour"],
                hovertemplate="Month: %{x}<br>MoM Change: %{y:.1f}%<extra></extra>",
            )
        )
        fig_mom.add_hline(y=0, line_dash="dash", line_color="#6b7280")
        fig_mom.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Billing Month"),
            yaxis=dict(title="MoM Change (%)"),
        )
        st.plotly_chart(fig_mom, use_container_width=True)
    else:
        st.info("Not enough months in the selected period to compute MoM change.")
else:
    st.info("No revenue data for the selected filters.")

st.divider()

# ---------------------------------------------------------------------------
# Row 4 — Tariff-Band Profitability
# Horizontal bar chart: total_revenue_billed per tariff_band, with the
# average collection_rate_pct per band displayed as a secondary annotation.
# Both columns come directly from mart_revenue_trends — no derivation needed.
# ---------------------------------------------------------------------------
st.subheader("Tariff-Band Profitability")
st.caption(
    "Total revenue billed and average collection rate per tariff band. "
    "Aggregated across all discos and months in the selected filters."
)

if not df_rev.empty:
    band_agg = (
        df_rev.groupby("tariff_band")
        .agg(
            total_revenue_billed=("total_revenue_billed", "sum"),
            avg_collection_rate_pct=("collection_rate_pct", "mean"),
            unique_customers=("unique_customers", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue_billed", ascending=True)
    )

    rev_col, rate_col = st.columns([2, 1])

    with rev_col:
        st.markdown("**Billed Revenue by Tariff Band**")
        fig_band = go.Figure(
            go.Bar(
                x=band_agg["total_revenue_billed"],
                y=band_agg["tariff_band"],
                orientation="h",
                marker_color="#3b82f6",
                text=band_agg["total_revenue_billed"].apply(
                    lambda v: format_ngn(v, compact=True)
                ),
                textposition="outside",
                hovertemplate=(
                    "Band: %{y}<br>"
                    "Billed: ₦%{x:,.0f}<extra></extra>"
                ),
            )
        )
        fig_band.update_layout(
            height=max(300, len(band_agg) * 52 + 80),
            margin=dict(l=0, r=80, t=10, b=0),
            xaxis=dict(title="Total Revenue Billed (NGN)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_band, use_container_width=True)

    with rate_col:
        st.markdown("**Avg Collection Rate by Band**")
        band_display = pd.DataFrame(
            {
                "Tariff Band":      band_agg["tariff_band"],
                "Collection Rate":  band_agg["avg_collection_rate_pct"].apply(format_pct),
                "Customers":        band_agg["unique_customers"].apply(
                    lambda v: format_number(v, compact=True)
                ),
            }
        )
        st.dataframe(band_display, use_container_width=True, hide_index=True)
else:
    st.info("No tariff-band data for the selected filters.")

st.divider()

# ---------------------------------------------------------------------------
# Row 5 — Payment Segment Distribution
# Source: mart_payment_behavior.payment_segment
# Three values defined in mart_payment_behavior.sql:
#   'reliable'        → on_time_rate >= 80%
#   'occasional_late' → 50% <= on_time_rate < 80%
#   'chronic_late'    → on_time_rate < 50%
# No date column on this mart — global date range is intentionally not applied.
# ---------------------------------------------------------------------------
st.subheader("Payment Segment Distribution")
st.caption(
    "Source: `mart_payment_behavior.payment_segment`. Segments are defined in "
    "`dbt_project/models/marts/finance/mart_payment_behavior.sql` based on "
    "each customer's lifetime on_time_rate. Date range filter does not apply "
    "— this mart is a lifetime per-customer summary."
)

if not df_pay.empty:
    seg_col, seg_detail_col = st.columns([1, 1])

    segment_counts = (
        df_pay.groupby("payment_segment")
        .size()
        .reset_index(name="customer_count")
    )

    SEGMENT_COLOURS = {
        "reliable":        "#22c55e",
        "occasional_late": "#f59e0b",
        "chronic_late":    "#ef4444",
    }
    SEGMENT_LABELS = {
        "reliable":        "Reliable (on-time ≥ 80%)",
        "occasional_late": "Occasional Late (50–80%)",
        "chronic_late":    "Chronic Late (< 50%)",
    }

    segment_counts["label"]  = segment_counts["payment_segment"].map(SEGMENT_LABELS)
    segment_counts["colour"] = segment_counts["payment_segment"].map(SEGMENT_COLOURS)

    with seg_col:
        fig_seg = go.Figure(
            go.Pie(
                labels=segment_counts["label"],
                values=segment_counts["customer_count"],
                marker=dict(colors=segment_counts["colour"].tolist()),
                hole=0.45,
                hovertemplate="%{label}<br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        )
        fig_seg.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with seg_detail_col:
        st.markdown("**Segment Summary**")
        seg_summary = df_pay.groupby("payment_segment").agg(
            customers=("customer_id", "count"),
            avg_on_time_rate=("on_time_rate_pct", "mean"),
            avg_collection_rate=("avg_collection_rate_pct", "mean"),
            total_arrears=("total_arrears", "sum"),
        ).reset_index()

        seg_display = pd.DataFrame(
            {
                "Segment":         seg_summary["payment_segment"].map(SEGMENT_LABELS).fillna(seg_summary["payment_segment"]),
                "Customers":       seg_summary["customers"].apply(lambda v: format_number(v, compact=False)),
                "Avg On-Time":     seg_summary["avg_on_time_rate"].apply(format_pct),
                "Avg Collection":  seg_summary["avg_collection_rate"].apply(format_pct),
                "Total Arrears":   seg_summary["total_arrears"].apply(format_ngn),
            }
        )
        st.dataframe(seg_display, use_container_width=True, hide_index=True)
else:
    st.info("No payment behavior data for the selected disco.")

st.divider()

# ---------------------------------------------------------------------------
# Row 6 — Collection Tier Breakdown
# Source: mart_payment_behavior.collection_tier
# Three tiers defined in mart_payment_behavior.sql:
#   'high'   → avg_collection_rate >= 95%
#   'medium' → avg_collection_rate >= 75%
#   'low'    → avg_collection_rate < 75%
# Grouped bar: customer count and total arrears per tier.
# No date filter — same reason as Row 5 (lifetime mart, no date column).
# ---------------------------------------------------------------------------
st.subheader("Collection Tier Breakdown")
st.caption(
    "collection_tier from `mart_payment_behavior`. "
    "High: avg collection ≥ 95% · Medium: ≥ 75% · Low: < 75%. "
    "Shows whether arrears are concentrated in the low tier."
)

if not df_pay.empty:
    tier_agg = (
        df_pay.groupby("collection_tier")
        .agg(
            customer_count=("customer_id", "count"),
            total_arrears=("total_arrears", "sum"),
            avg_collection_rate=("avg_collection_rate_pct", "mean"),
        )
        .reset_index()
    )

    TIER_ORDER  = ["high", "medium", "low"]
    TIER_COLOURS = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}
    tier_agg = tier_agg[tier_agg["collection_tier"].isin(TIER_ORDER)]
    tier_agg["colour"] = tier_agg["collection_tier"].map(TIER_COLOURS)

    tier_col1, tier_col2 = st.columns(2)

    with tier_col1:
        st.markdown("**Customer Count per Tier**")
        fig_tier_count = go.Figure(
            go.Bar(
                x=tier_agg["collection_tier"],
                y=tier_agg["customer_count"],
                marker_color=tier_agg["colour"].tolist(),
                text=tier_agg["customer_count"].apply(
                    lambda v: format_number(v, compact=True)
                ),
                textposition="outside",
                hovertemplate="Tier: %{x}<br>Customers: %{y:,}<extra></extra>",
            )
        )
        fig_tier_count.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Collection Tier", categoryorder="array", categoryarray=TIER_ORDER),
            yaxis=dict(title="Customer Count"),
        )
        st.plotly_chart(fig_tier_count, use_container_width=True)

    with tier_col2:
        st.markdown("**Total Arrears by Tier**")
        fig_tier_arrears = go.Figure(
            go.Bar(
                x=tier_agg["collection_tier"],
                y=tier_agg["total_arrears"],
                marker_color=tier_agg["colour"].tolist(),
                text=tier_agg["total_arrears"].apply(
                    lambda v: format_ngn(v, compact=True)
                ),
                textposition="outside",
                hovertemplate="Tier: %{x}<br>Arrears: ₦%{y:,.0f}<extra></extra>",
            )
        )
        fig_tier_arrears.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Collection Tier", categoryorder="array", categoryarray=TIER_ORDER),
            yaxis=dict(title="Total Arrears (NGN)"),
        )
        st.plotly_chart(fig_tier_arrears, use_container_width=True)
else:
    st.info("No collection tier data for the selected disco.")