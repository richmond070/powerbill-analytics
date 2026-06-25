"""
Customer Risk Dashboard — Page 3
==================================
Data sources:
  main.gold.mart_customer_lifetime_value
    → defined in dbt_project/models/marts/customers/mart_customer_lifetime_value.sql
    → grain: (customer_id) — lifetime summary, NO date column
    → query: dashboard/lib/queries.get_customer_lifetime_value()
    → date filter: NO — only disco filter applies

  main.gold.mart_outstanding_balances
    → defined in dbt_project/models/marts/customers/mart_outstanding_balances.sql
    → grain: (customer_id, billing_month) — time series
    → query: dashboard/lib/queries.get_outstanding_balances()
    → date filter: YES (billing_month column present)

Global filters read from st.session_state, written by dashboard/app.py.
Date range is passed to get_outstanding_balances() but NOT to
get_customer_lifetime_value() — that mart has no date grain.

All *_pct columns from both marts are already on a 0-100 scale (dbt
multiplies by 100 before writing). formatting.py expects 0-100 values —
do not multiply again here.

Layout:
  Row 1  KPI cards: total customers, high-risk count, avg CLV score,
                    platinum tier count
  Row 2  High-risk customer table (arrears_growing_flag OR high_outstanding_flag)
  Row 3  Arrears growth trend — cumulative_arrears over billing_month
  Row 4  CLV tier distribution — donut + summary table
  Row 5  CLV score component breakdown — stacked bar per tier
  Row 6  Complaint-risk correlation — scatter (pct_complaints_during_stress
         vs clv_score, sized by total_arrears)
"""

import os
import sys

# Make dashboard/lib/ importable from dashboard/pages/
# os.path.abspath(__file__) → full path to 3_Customer_Risk.py
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
    page_title="Customer Risk | Energy Ops",
    page_icon="⚠️",
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

st.title("⚠️ Customer Risk")
st.caption(
    f"Sources: `main.gold.mart_customer_lifetime_value` · "
    f"`main.gold.mart_outstanding_balances` · "
    f"Disco: **{disco}** · "
    f"Period: **{start_date or 'all time'} → {end_date or 'latest'}**"
)
st.caption(
    "ℹ️ Date range applies to outstanding balances only. "
    "CLV and payment risk scores are lifetime customer summaries with no date grain."
)

# ---------------------------------------------------------------------------
# Load data — mart_customer_lifetime_value
# No date param — this is a lifetime per-customer summary.
# Query: dashboard/lib/queries.get_customer_lifetime_value()
# ---------------------------------------------------------------------------
with st.spinner("Loading mart_customer_lifetime_value…"):
    try:
        df_clv = queries.get_customer_lifetime_value(disco=disco)
    except Exception as e:
        st.error(f"Failed to load mart_customer_lifetime_value: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Load data — mart_outstanding_balances (full dataset, no high_risk_only filter)
# Used for arrears trend aggregation across all customers.
# Date param applies — billing_month is the time column in this mart.
# ---------------------------------------------------------------------------
with st.spinner("Loading mart_outstanding_balances…"):
    try:
        df_bal = queries.get_outstanding_balances(
            disco=disco,
            start_date=start_date,
            end_date=end_date,
            high_risk_only=False,
        )
    except Exception as e:
        st.error(f"Failed to load mart_outstanding_balances: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Load data — mart_outstanding_balances (high-risk only)
# Separate call using high_risk_only=True which adds:
#   WHERE arrears_growing_flag = true OR high_outstanding_flag = true
# Both flags are pre-computed in mart_outstanding_balances.sql — plain filter.
# ---------------------------------------------------------------------------
with st.spinner("Loading high-risk customers…"):
    try:
        df_risk = queries.get_outstanding_balances(
            disco=disco,
            start_date=start_date,
            end_date=end_date,
            high_risk_only=True,
        )
    except Exception as e:
        st.error(f"Failed to load high-risk balances: {e}")
        st.stop()

if df_clv.empty and df_bal.empty:
    st.warning(
        "No data for the selected filters. "
        "Try 'All' discos or a wider date range."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Type normalisation
# billing_month comes back as string from the connector — cast to datetime
# for mart_outstanding_balances (both df_bal and df_risk).
# mart_customer_lifetime_value has no date columns — nothing to cast there.
# ---------------------------------------------------------------------------
if not df_bal.empty:
    df_bal["billing_month"] = pd.to_datetime(df_bal["billing_month"])
    df_bal = df_bal.sort_values(["customer_id", "billing_month"])

if not df_risk.empty:
    df_risk["billing_month"] = pd.to_datetime(df_risk["billing_month"])
    df_risk = df_risk.sort_values(["customer_id", "billing_month"])

# ---------------------------------------------------------------------------
# Row 1 — KPI Cards
#
# Card 1: Total unique customers       → mart_customer_lifetime_value (count)
# Card 2: High-risk customer count     → mart_outstanding_balances
#           distinct customer_ids where arrears_growing_flag OR
#           high_outstanding_flag — both pre-computed in dbt
# Card 3: Avg CLV score                → mart_customer_lifetime_value.clv_score
#           0-100 composite index: 50% revenue + 30% reliability + 20% friction
# Card 4: Platinum tier customers      → mart_customer_lifetime_value.clv_tier
#           clv_tier = 'platinum' means clv_score >= 75
# ---------------------------------------------------------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Card 1 — total customers from CLV mart
total_customers = df_clv["customer_id"].nunique() if not df_clv.empty else 0

col1.metric(
    label="Total Customers",
    value=format_number(total_customers, compact=False),
    help="Distinct customer_ids in mart_customer_lifetime_value for the selected disco.",
)

# Card 2 — high-risk: distinct customers appearing in df_risk
# (arrears_growing_flag OR high_outstanding_flag on at least one billing_month)
if not df_risk.empty:
    high_risk_customers = df_risk["customer_id"].nunique()
    high_risk_pct       = (high_risk_customers / total_customers * 100) if total_customers > 0 else None
else:
    high_risk_customers = 0
    high_risk_pct       = None

col2.metric(
    label="High-Risk Customers",
    value=format_number(high_risk_customers, compact=False),
    delta=format_pct(high_risk_pct) + " of total" if high_risk_pct is not None else None,
    delta_color="inverse",
    help=(
        "Customers with arrears_growing_flag = true OR high_outstanding_flag = true "
        "on at least one billing_month. Both flags pre-computed in "
        "mart_outstanding_balances.sql."
    ),
)

# Card 3 — avg CLV score
avg_clv = df_clv["clv_score"].mean() if not df_clv.empty else None

col3.metric(
    label="Avg CLV Score",
    value=f"{avg_clv:.1f} / 100" if avg_clv is not None else "—",
    help=(
        "Composite 0-100 index from mart_customer_lifetime_value. "
        "50% revenue component + 30% reliability + 20% low-friction. "
        "Defined in dbt_project/models/marts/customers/mart_customer_lifetime_value.sql."
    ),
)

# Card 4 — platinum tier count (clv_score >= 75)
if not df_clv.empty:
    platinum_count = (df_clv["clv_tier"] == "platinum").sum()
    platinum_pct   = (platinum_count / total_customers * 100) if total_customers > 0 else None
else:
    platinum_count = 0
    platinum_pct   = None

col4.metric(
    label="Platinum Tier Customers",
    value=format_number(platinum_count, compact=False),
    delta=format_pct(platinum_pct) + " of total" if platinum_pct is not None else None,
    help="Customers with clv_tier = 'platinum' (clv_score ≥ 75).",
)

st.divider()

# ---------------------------------------------------------------------------
# Row 2 — High-Risk Customer Table
# Source: mart_outstanding_balances WHERE arrears_growing_flag OR
#         high_outstanding_flag (pre-computed in dbt, plain filter in query)
#
# Shows the latest billing_month snapshot per customer — most recent
# cumulative position rather than every historical row.
#
# Columns shown:
#   customer_id, disco, tariff_band, billing_month (latest),
#   cumulative_arrears, arrears_mom_change,
#   arrears_growing_flag, high_outstanding_flag,
#   outstanding_pct_of_total_billed
# ---------------------------------------------------------------------------
st.subheader("High-Risk Customers")
st.caption(
    "Customers where `arrears_growing_flag = true` (arrears grew for 2+ consecutive "
    "months) or `high_outstanding_flag = true` (cumulative outstanding > 3× average "
    "monthly bill). Both flags pre-computed in `mart_outstanding_balances.sql`. "
    "Showing latest billing_month snapshot per customer."
)

if not df_risk.empty:
    # Latest billing_month snapshot per customer
    latest_risk = (
        df_risk.sort_values("billing_month")
        .groupby("customer_id", as_index=False)
        .last()
    )

    # Optional: local search filter for customer lookups
    search = st.text_input(
        "Search by customer_id", value="", placeholder="e.g. CUST-00123"
    )
    if search:
        latest_risk = latest_risk[
            latest_risk["customer_id"].str.contains(search, case=False, na=False)
        ]

    risk_display = pd.DataFrame(
        {
            "Customer ID":        latest_risk["customer_id"],
            "Disco":              latest_risk["disco"],
            "Tariff Band":        latest_risk["tariff_band"],
            "As Of":              latest_risk["billing_month"].dt.strftime("%b %Y"),
            "Cumulative Arrears": latest_risk["cumulative_arrears"].apply(format_ngn),
            "Arrears MoM Chg":    latest_risk["arrears_mom_change"].apply(
                lambda v: format_ngn(v, compact=True)
            ),
            "Outstanding % Billed": latest_risk["outstanding_pct_of_total_billed"].apply(format_pct),
            "Arrears Growing":    latest_risk["arrears_growing_flag"].map(
                {True: "🔴 Yes", False: "🟢 No"}
            ),
            "High Outstanding":   latest_risk["high_outstanding_flag"].map(
                {True: "🔴 Yes", False: "🟢 No"}
            ),
        }
    )

    st.dataframe(
        risk_display,
        use_container_width=True,
        hide_index=True,
        height=min(400, (len(risk_display) + 1) * 36),
    )
    st.caption(
        f"{format_number(len(latest_risk), compact=False)} high-risk customers shown. "
        f"Use the global filters or search box to narrow down."
    )
else:
    st.success(
        "No high-risk customers for the selected filters. "
        "Either no customers have growing arrears, or all are within "
        "the outstanding threshold defined in mart_outstanding_balances.sql."
    )

st.divider()

# ---------------------------------------------------------------------------
# Row 3 — Arrears Growth Trend
# Source: mart_outstanding_balances (full df_bal, not high-risk only)
# Aggregates cumulative_arrears across all customers per billing_month
# to show the portfolio-level arrears trajectory.
#
# Two lines:
#   Total cumulative arrears — portfolio-level view
#   Total payment gap        — gap between billed and paid each month
# ---------------------------------------------------------------------------
st.subheader("Arrears Growth Trend")
st.caption(
    "Portfolio-level aggregate of `cumulative_arrears` and `payment_gap_ngn` "
    "from `mart_outstanding_balances`, grouped by billing_month. "
    "Rising cumulative arrears = the portfolio's debt position is worsening."
)

if not df_bal.empty:
    arrears_trend = (
        df_bal.groupby("billing_month")[
            ["cumulative_arrears", "payment_gap_ngn", "cumulative_outstanding"]
        ]
        .sum()
        .reset_index()
        .sort_values("billing_month")
    )

    fig_arrears = go.Figure()
    fig_arrears.add_scatter(
        x=arrears_trend["billing_month"],
        y=arrears_trend["cumulative_arrears"],
        mode="lines+markers",
        name="Cumulative Arrears",
        line=dict(color="#ef4444", width=2),
        hovertemplate="Month: %{x}<br>Cumulative Arrears: ₦%{y:,.0f}<extra></extra>",
    )
    fig_arrears.add_scatter(
        x=arrears_trend["billing_month"],
        y=arrears_trend["payment_gap_ngn"],
        mode="lines+markers",
        name="Monthly Payment Gap",
        line=dict(color="#f97316", width=2, dash="dot"),
        hovertemplate="Month: %{x}<br>Payment Gap: ₦%{y:,.0f}<extra></extra>",
    )
    fig_arrears.add_scatter(
        x=arrears_trend["billing_month"],
        y=arrears_trend["cumulative_outstanding"],
        mode="lines+markers",
        name="Cumulative Outstanding",
        line=dict(color="#eab308", width=2, dash="dash"),
        hovertemplate="Month: %{x}<br>Cumulative Outstanding: ₦%{y:,.0f}<extra></extra>",
    )
    fig_arrears.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Billing Month"),
        yaxis=dict(title="NGN"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_arrears, use_container_width=True)
else:
    st.info("No arrears data for the selected filters.")

st.divider()

# ---------------------------------------------------------------------------
# Row 4 — CLV Tier Distribution
# Source: mart_customer_lifetime_value.clv_tier
# Four tiers defined in mart_customer_lifetime_value.sql:
#   'platinum' → clv_score >= 75
#   'gold'     → clv_score >= 50
#   'silver'   → clv_score >= 25
#   'bronze'   → clv_score < 25
# No date filter — lifetime mart.
# ---------------------------------------------------------------------------
st.subheader("CLV Tier Distribution")
st.caption(
    "CLV tier from `mart_customer_lifetime_value`. "
    "Platinum ≥ 75 · Gold ≥ 50 · Silver ≥ 25 · Bronze < 25. "
    "Score methodology: 50% revenue + 30% reliability + 20% low-friction. "
    "Date range does not apply — this is a lifetime customer summary."
)

if not df_clv.empty:
    tier_col, tier_summary_col = st.columns([1, 1])

    TIER_ORDER   = ["platinum", "gold", "silver", "bronze"]
    TIER_COLOURS = {
        "platinum": "#a78bfa",
        "gold":     "#fbbf24",
        "silver":   "#94a3b8",
        "bronze":   "#b45309",
    }
    TIER_LABELS = {
        "platinum": "Platinum (≥ 75)",
        "gold":     "Gold (≥ 50)",
        "silver":   "Silver (≥ 25)",
        "bronze":   "Bronze (< 25)",
    }

    tier_counts = (
        df_clv.groupby("clv_tier")
        .agg(
            customer_count=("customer_id", "count"),
            avg_clv_score=("clv_score", "mean"),
            avg_collection_rate=("avg_collection_rate_pct", "mean"),
            avg_on_time_rate=("on_time_rate_pct", "mean"),
            total_arrears=("total_arrears", "sum"),
        )
        .reindex(TIER_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    with tier_col:
        fig_tier = go.Figure(
            go.Pie(
                labels=tier_counts["clv_tier"].map(TIER_LABELS),
                values=tier_counts["customer_count"],
                marker=dict(
                    colors=[
                        TIER_COLOURS.get(t, "#6b7280")
                        for t in tier_counts["clv_tier"]
                    ]
                ),
                hole=0.45,
                hovertemplate=(
                    "%{label}<br>"
                    "Customers: %{value:,}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )
        )
        fig_tier.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with tier_summary_col:
        st.markdown("**Tier Summary**")
        tier_display = pd.DataFrame(
            {
                "Tier": tier_counts["clv_tier"].map(TIER_LABELS).fillna(
                    tier_counts["clv_tier"]
                ),
                "Customers":      tier_counts["customer_count"].apply(
                    lambda v: format_number(v, compact=False)
                ),
                "Avg CLV Score":  tier_counts["avg_clv_score"].apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                ),
                "Avg Collection": tier_counts["avg_collection_rate"].apply(format_pct),
                "Avg On-Time":    tier_counts["avg_on_time_rate"].apply(format_pct),
                "Total Arrears":  tier_counts["total_arrears"].apply(format_ngn),
            }
        )
        st.dataframe(tier_display, use_container_width=True, hide_index=True)
else:
    st.info("No CLV data for the selected disco.")

st.divider()

# ---------------------------------------------------------------------------
# Row 5 — CLV Score Component Breakdown
# Source: mart_customer_lifetime_value
# Three pre-computed component scores from mart_customer_lifetime_value.sql:
#   clv_revenue_score     → 0-50: total_paid / max(total_paid) * 50
#   clv_reliability_score → 0-30: on_time_rate * 30
#   clv_friction_score    → 0-20: 20 * (1 - complaints / max_complaints)
# Shows average of each component per tier — reveals which component is
# pulling each tier's overall score up or down.
# No date filter — lifetime mart.
# ---------------------------------------------------------------------------
st.subheader("CLV Score Component Breakdown by Tier")
st.caption(
    "Average of each CLV component score per tier. "
    "Revenue (max 50pts) · Reliability (max 30pts) · Low-friction (max 20pts). "
    "Components defined in `mart_customer_lifetime_value.sql`."
)

if not df_clv.empty:
    component_agg = (
        df_clv.groupby("clv_tier")[
            ["clv_revenue_score", "clv_reliability_score", "clv_friction_score"]
        ]
        .mean()
        .reindex(TIER_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    fig_components = go.Figure()
    fig_components.add_bar(
        x=component_agg["clv_tier"],
        y=component_agg["clv_revenue_score"],
        name="Revenue (max 50)",
        marker_color="#3b82f6",
    )
    fig_components.add_bar(
        x=component_agg["clv_tier"],
        y=component_agg["clv_reliability_score"],
        name="Reliability (max 30)",
        marker_color="#22c55e",
    )
    fig_components.add_bar(
        x=component_agg["clv_tier"],
        y=component_agg["clv_friction_score"],
        name="Low-Friction (max 20)",
        marker_color="#a78bfa",
    )
    fig_components.update_layout(
        barmode="stack",
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            title="CLV Tier",
            categoryorder="array",
            categoryarray=TIER_ORDER,
        ),
        yaxis=dict(title="Avg Component Score", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_components, use_container_width=True)
else:
    st.info("No CLV component data for the selected disco.")

st.divider()

# ---------------------------------------------------------------------------
# Row 6 — Complaint-Risk Correlation Scatter
# Source: mart_customer_lifetime_value
#
# X axis: pct_complaints_during_stress
#   — share of this customer's complaints that coincided with a grid stress
#     event. Already 0-100 in dbt:
#     ROUND(100.0 * any_stress_complaints / NULLIF(total_complaints, 0), 2)
#   — High value = customer's issues are grid-driven, not payment-driven
#   — Low value  = customer's issues are billing/service driven
#
# Y axis: clv_score (0-100 composite)
#
# Point size: total_arrears — highlights high-arrears customers within
#   each quadrant without needing a separate chart
#
# Colour: clv_tier — so the segment colours from Row 4 carry through
#
# Interpretation guide:
#   Top-left  (low stress share, high CLV)  → reliable customers, rare complaints
#   Top-right (high stress share, high CLV) → good customers hurt by grid problems
#   Bottom-left (low stress share, low CLV) → chronic non-payers, billing issues
#   Bottom-right (high stress share, low CLV) → highest intervention priority
#
# No date filter — lifetime mart.
# ---------------------------------------------------------------------------
st.subheader("Complaint-Risk Correlation")
st.caption(
    "Each point = one customer. X = share of complaints during grid stress "
    "(`pct_complaints_during_stress`). Y = CLV score. "
    "Size = total arrears. Colour = CLV tier. "
    "Bottom-right quadrant (high stress share, low CLV) = highest intervention priority."
)

if not df_clv.empty:
    # Filter to customers with at least one complaint for a meaningful scatter
    scatter_df = df_clv[df_clv["total_complaints"] > 0].copy()

    if not scatter_df.empty:
        # Clamp bubble size so very large arrears don't crush small ones
        max_arrears = scatter_df["total_arrears"].max()
        scatter_df["bubble_size"] = (
            scatter_df["total_arrears"]
            .clip(lower=0)
            .apply(lambda v: 5 + 40 * (v / max_arrears) if max_arrears > 0 else 5)
        )

        scatter_df["tier_label"] = scatter_df["clv_tier"].map(TIER_LABELS).fillna(
            scatter_df["clv_tier"]
        )

        fig_scatter = px.scatter(
            scatter_df,
            x="pct_complaints_during_stress",
            y="clv_score",
            color="clv_tier",
            color_discrete_map=TIER_COLOURS,
            size="bubble_size",
            size_max=30,
            hover_data={
                "customer_id":                   True,
                "disco":                         True,
                "total_complaints":              True,
                "pct_complaints_during_stress":  ":.1f",
                "clv_score":                     ":.1f",
                "total_arrears":                 ":,.0f",
                "bubble_size":                   False,
                "clv_tier":                      False,
            },
            labels={
                "pct_complaints_during_stress": "% Complaints During Grid Stress",
                "clv_score":                    "CLV Score (0-100)",
                "clv_tier":                     "CLV Tier",
            },
            category_orders={"clv_tier": TIER_ORDER},
        )

        # Quadrant reference lines
        fig_scatter.add_vline(
            x=50,
            line_dash="dot",
            line_color="#6b7280",
            annotation_text="50% stress threshold",
            annotation_position="top right",
        )
        fig_scatter.add_hline(
            y=50,
            line_dash="dot",
            line_color="#6b7280",
            annotation_text="CLV midpoint",
            annotation_position="right",
        )

        # Quadrant labels
        for x_pos, y_pos, label in [
            (2,  95, "Grid-hurt good customers"),
            (75, 95, "Highest intervention priority"),
            (2,  5,  "Billing-driven low-value"),
            (75, 5,  "Grid-driven low-value"),
        ]:
            fig_scatter.add_annotation(
                x=x_pos, y=y_pos,
                text=label,
                showarrow=False,
                font=dict(size=10, color="#9ca3af"),
                xanchor="left" if x_pos < 50 else "right",
            )

        fig_scatter.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="% Complaints During Grid Stress", range=[0, 100]),
            yaxis=dict(title="CLV Score (0-100)",              range=[0, 100]),
            legend=dict(
                title="CLV Tier",
                orientation="h",
                yanchor="bottom",
                y=1.02,
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Quadrant summary counts — gives a quick read without needing to
        # inspect individual points
        st.markdown("**Quadrant Summary**")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)

        top_left  = scatter_df[(scatter_df["pct_complaints_during_stress"] <= 50) & (scatter_df["clv_score"] > 50)]
        top_right = scatter_df[(scatter_df["pct_complaints_during_stress"] >  50) & (scatter_df["clv_score"] > 50)]
        bot_left  = scatter_df[(scatter_df["pct_complaints_during_stress"] <= 50) & (scatter_df["clv_score"] <= 50)]
        bot_right = scatter_df[(scatter_df["pct_complaints_during_stress"] >  50) & (scatter_df["clv_score"] <= 50)]

        q_col1.metric(
            "Grid-hurt good customers",
            format_number(len(top_left), compact=False),
            help="High CLV, low stress share — reliable customers with rare billing complaints.",
        )
        q_col2.metric(
            "Grid-hurt, high CLV",
            format_number(len(top_right), compact=False),
            help="High CLV, high stress share — valuable customers whose complaints are grid-driven.",
        )
        q_col3.metric(
            "Low CLV, billing-driven",
            format_number(len(bot_left), compact=False),
            help="Low CLV, low stress share — chronic non-payers with billing/service issues.",
        )
        q_col4.metric(
            "Highest priority (low CLV + grid stress)",
            format_number(len(bot_right), compact=False),
            delta_color="inverse",
            help=(
                "Low CLV, high stress share — grid-driven complaints AND poor payment. "
                "Highest intervention priority."
            ),
        )
    else:
        st.info(
            "No customers with recorded complaints in the selected filters. "
            "Try 'All' discos."
        )
else:
    st.info("No CLV data for the selected disco.")