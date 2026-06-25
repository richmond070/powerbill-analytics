"""
Grid Reliability Dashboard — Page 1
=====================================
Signature page. Data source: main.gold.mart_grid_reliability
  Written by:  dbt_project/models/marts/operations/mart_grid_reliability.sql
  Grain:       one row per (disco, complaint_month)
  Read via:    dashboard/lib/queries.get_grid_reliability()

Global filters (disco, date range) are read from st.session_state,
set by the sidebar in dashboard/app.py and shared across all pages.
mart_grid_reliability has a date column (complaint_month) so both
filters apply here.

Layout:
  Row 1  KPI cards: stress share, SLA met rate, SLA stress gap, RAG status
  Row 2  Latest RAG status table (one row per disco)
  Row 3  Disco x month heatmap coloured by stress_complaint_share_pct
  Row 4  Two trend lines: stress share 3M avg / frequency deviation
  Row 5  Stress type stacked bar: overload / instability / line stress
  Row 6  SLA split: overall vs during-stress vs no-stress
"""

import sys
import os

# Make dashboard/lib/ importable from dashboard/pages/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import queries
from lib.formatting import format_delta_pct, format_number, format_pct

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Grid Reliability | Energy Ops",
    page_icon="🔌",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Read global filters from session_state (set by dashboard/app.py sidebar)
# Defaults make the page work even when accessed directly by URL.
# ---------------------------------------------------------------------------
disco      = st.session_state.get("global_disco", "All")
start_date = st.session_state.get("global_start_date", None)
end_date   = st.session_state.get("global_end_date", None)

st.title("🔌 Grid Reliability")
st.caption(
    f"Source: `main.gold.mart_grid_reliability` · "
    f"Disco: **{disco}** · "
    f"Period: **{start_date or 'all time'} → {end_date or 'latest'}**"
)

# ---------------------------------------------------------------------------
# Load data
# Calls dashboard/lib/queries.get_grid_reliability() which in turn calls
# dashboard/lib/db_client.run_query() (cached, ttl=300s).
# Any Databricks SQL connector error surfaces here rather than mid-render.
# ---------------------------------------------------------------------------
with st.spinner("Loading mart_grid_reliability…"):
    try:
        df = queries.get_grid_reliability(
            disco=disco,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        st.error(f"Failed to load mart_grid_reliability: {e}")
        st.stop()

if df.empty:
    st.warning(
        "No data for the selected filters. "
        "Try 'All' discos or a wider date range."
    )
    st.stop()

# Normalise types — complaint_month comes back as string from the connector
df["complaint_month"] = pd.to_datetime(df["complaint_month"])
df = df.sort_values(["disco", "complaint_month"])

# ---------------------------------------------------------------------------
# Row 1 — KPI Cards
# Snapshot from the latest month in the filtered dataset.
# ---------------------------------------------------------------------------
st.subheader("Key Metrics")

latest_month = df["complaint_month"].max()
df_latest    = df[df["complaint_month"] == latest_month]

# Period averages for the KPI values
avg_stress_share  = df["stress_complaint_share_pct"].mean()
avg_sla_overall   = df["overall_sla_met_rate_pct"].mean()
avg_sla_gap       = df["sla_stress_gap_ppt"].mean()

# MoM delta for stress share (latest month avg vs previous month avg)
prev_month = df[df["complaint_month"] < latest_month]["complaint_month"].max()
if pd.notna(prev_month):
    avg_stress_prev = (
        df[df["complaint_month"] == prev_month]["stress_complaint_share_pct"].mean()
    )
    stress_delta = avg_stress_share - avg_stress_prev
else:
    stress_delta = None

col1, col2, col3, col4 = st.columns(4)

# delta_color="inverse": rising stress share is bad (turns red not green)
col1.metric(
    label="Avg Stress Complaint Share",
    value=format_pct(avg_stress_share),
    delta=format_delta_pct(stress_delta),
    delta_color="inverse",
    help="% of complaints in the period that coincided with a grid stress event.",
)

col2.metric(
    label="Avg Overall SLA Met Rate",
    value=format_pct(avg_sla_overall),
    help="Share of complaint tickets where SLA was met, across all discos and months.",
)

# delta_color="inverse": a larger SLA drop during stress is bad
col3.metric(
    label="SLA Gap During Stress",
    value=format_delta_pct(avg_sla_gap) or "—",
    delta_color="inverse",
    help=(
        "Average ppt drop in SLA met rate during stress vs non-stress periods. "
        "Higher gap = SLA degrades more when the grid is under pressure."
    ),
)

# Most common grid_reliability_status in the latest month
STATUS_ICON = {
    "critical": "🔴",
    "degraded":  "🟠",
    "moderate":  "🟡",
    "stable":    "🟢",
}
if not df_latest.empty:
    mode_series = df_latest["grid_reliability_status"].mode()
    status_val  = mode_series.iloc[0] if not mode_series.empty else "unknown"
else:
    status_val = "unknown"

col4.metric(
    label="Grid Status (Latest Month)",
    value=f"{STATUS_ICON.get(status_val, '⚪')} {status_val.capitalize()}",
    help="Most common grid_reliability_status across discos in the latest month.",
)

st.divider()

# ---------------------------------------------------------------------------
# Row 2 — Latest RAG status per Disco
# One row per disco showing its most recent complaint_month snapshot.
# All five columns come directly from mart_grid_reliability — no derivation.
# ---------------------------------------------------------------------------
st.subheader("Latest Grid Reliability Status per Disco")

rag_df = (
    df.sort_values("complaint_month")
    .groupby("disco", as_index=False)
    .last()[
        [
            "disco",
            "complaint_month",
            "grid_reliability_status",
            "stress_complaint_share_pct",
            "overall_sla_met_rate_pct",
            "sla_stress_gap_ppt",
        ]
    ]
)

STATUS_BADGE = {
    "critical": "🔴 Critical",
    "degraded":  "🟠 Degraded",
    "moderate":  "🟡 Moderate",
    "stable":    "🟢 Stable",
}

rag_display = pd.DataFrame(
    {
        "Disco":        rag_df["disco"],
        "As Of":        rag_df["complaint_month"].dt.strftime("%b %Y"),
        "Status":       rag_df["grid_reliability_status"].map(STATUS_BADGE).fillna("⚪ Unknown"),
        "Stress Share": rag_df["stress_complaint_share_pct"].apply(format_pct),
        "SLA Met":      rag_df["overall_sla_met_rate_pct"].apply(format_pct),
        "SLA Stress Gap": rag_df["sla_stress_gap_ppt"].apply(
            lambda v: format_delta_pct(v) or "—"
        ),
    }
)

st.dataframe(rag_display, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 3 — Heatmap: Disco × Month coloured by stress_complaint_share_pct
# Pivot to disco (rows) × formatted month string (cols).
# go.Heatmap used for precise colorscale control matching the RAG thresholds
# defined in mart_grid_reliability.sql:
#   >= 60 → critical, >= 40 → degraded, >= 20 → moderate, else → stable
# ---------------------------------------------------------------------------
st.subheader("Stress Complaint Share — Disco × Month Heatmap")
st.caption(
    "Each cell = % of that disco's complaints that month which coincided with "
    "a grid stress event. Colour scale mirrors the RAG thresholds in "
    "`mart_grid_reliability.sql` (green=stable, red=critical)."
)

pivot = df.pivot_table(
    index="disco",
    columns=df["complaint_month"].dt.strftime("%b %Y"),
    values="stress_complaint_share_pct",
    aggfunc="mean",
)

# Preserve chronological column order
month_order = (
    df[["complaint_month"]]
    .drop_duplicates()
    .sort_values("complaint_month")["complaint_month"]
    .dt.strftime("%b %Y")
    .tolist()
)
pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

fig_heatmap = go.Figure(
    go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.00, "#22c55e"],  # 0%  → stable (green)
            [0.20, "#86efac"],  # 20% → moderate threshold
            [0.40, "#fde68a"],  # 40% → degraded threshold
            [0.60, "#fb923c"],  # 60% → critical threshold
            [1.00, "#ef4444"],  # 100%
        ],
        zmin=0,
        zmax=100,
        texttemplate="%{z:.1f}%",
        hovertemplate=(
            "Disco: %{y}<br>Month: %{x}<br>"
            "Stress Share: %{z:.1f}%<extra></extra>"
        ),
        colorbar=dict(title="Stress<br>Share %"),
    )
)
fig_heatmap.update_layout(
    height=max(300, len(pivot.index) * 52 + 120),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(tickangle=-45),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 4 — Trend lines (two charts side by side)
# Left:  stress_share_3m_avg_pct  — rolling smoothed stress share per disco
#         (pre-computed in mart_grid_reliability.sql, 3-row window per disco)
# Right: avg_freq_deviation_hz    — how far grid frequency strayed from 50 Hz
#         at the times customers were calling in complaints
# ---------------------------------------------------------------------------
st.subheader("Instability Trends Over Time")

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    st.markdown("**Stress Share — 3-Month Rolling Average (%)**")
    fig_trend = px.line(
        df,
        x="complaint_month",
        y="stress_share_3m_avg_pct",
        color="disco",
        markers=True,
        labels={
            "complaint_month":      "Month",
            "stress_share_3m_avg_pct": "Stress Share 3M Avg (%)",
            "disco":                "Disco",
        },
    )
    # RAG threshold reference lines
    for threshold, label, colour in [
        (60, "Critical ≥ 60%", "#ef4444"),
        (40, "Degraded ≥ 40%", "#fb923c"),
        (20, "Moderate ≥ 20%", "#eab308"),
    ]:
        fig_trend.add_hline(
            y=threshold, line_dash="dot", line_color=colour,
            annotation_text=label, annotation_position="right",
        )
    fig_trend.update_layout(
        height=330, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with trend_col2:
    st.markdown("**Avg Frequency Deviation at Complaint Time (Hz)**")
    st.caption(
        "Grid target is 50 Hz. This column (avg_freq_deviation_hz) is the "
        "average Hz deviation from 50 at the hour each complaint was raised. "
        "Sustained non-zero deviation = frequency instability."
    )
    fig_freq = px.line(
        df,
        x="complaint_month",
        y="avg_freq_deviation_hz",
        color="disco",
        markers=True,
        labels={
            "complaint_month":     "Month",
            "avg_freq_deviation_hz": "Avg Freq Deviation (Hz)",
            "disco":               "Disco",
        },
    )
    fig_freq.add_hline(
        y=0, line_dash="dash", line_color="#6b7280",
        annotation_text="Target (0 Hz deviation)",
    )
    fig_freq.update_layout(
        height=330, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_freq, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 5 — Stress Type Breakdown (stacked bar)
# Aggregates across all discos in the filtered set.
# Three stress type columns from mart_grid_reliability.sql:
#   overload_complaints    → load_mw > 105% of forecast_mw
#   instability_complaints → freq_deviation_hz > 0.5 Hz
#   line_stress_complaints → avg power factor on lines < 0.85
# ---------------------------------------------------------------------------
st.subheader("Stress Type Breakdown — What Is Driving Stress Complaints?")
st.caption(
    "Overload: load > 105% of forecast. "
    "Instability: freq deviation > 0.5 Hz. "
    "Line stress: avg power factor < 0.85. "
    "Definitions from `silver/grid_stress_complaints_silver.sql`."
)

stress_agg = (
    df.groupby("complaint_month")[
        ["overload_complaints", "instability_complaints", "line_stress_complaints"]
    ]
    .sum()
    .reset_index()
)

fig_types = go.Figure()
fig_types.add_bar(
    x=stress_agg["complaint_month"],
    y=stress_agg["overload_complaints"],
    name="Overload",
    marker_color="#ef4444",
)
fig_types.add_bar(
    x=stress_agg["complaint_month"],
    y=stress_agg["instability_complaints"],
    name="Instability",
    marker_color="#f97316",
)
fig_types.add_bar(
    x=stress_agg["complaint_month"],
    y=stress_agg["line_stress_complaints"],
    name="Line Stress",
    marker_color="#eab308",
)
fig_types.update_layout(
    barmode="stack",
    height=340,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(title="Month"),
    yaxis=dict(title="Complaint Count"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_types, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 6 — SLA Performance Split
# Three SLA columns from mart_grid_reliability.sql:
#   overall_sla_met_rate_pct        → all complaints, both periods
#   sla_met_rate_during_stress_pct  → complaints during stress events only
#   sla_met_rate_no_stress_pct      → complaints outside stress events
#
# A large gap between "No Stress" and "During Stress" lines means the DISCO's
# support team is overwhelmed when the grid is under pressure.
# ---------------------------------------------------------------------------
st.subheader("SLA Performance Split — Overall vs During Stress vs No Stress")
st.caption(
    "A large gap between the green (no stress) and red (during stress) lines "
    "indicates that SLA compliance degrades significantly when the grid is "
    "under pressure — pointing to capacity or process gaps in the DISCO's "
    "customer service operation."
)

sla_agg = (
    df.groupby("complaint_month")[
        [
            "overall_sla_met_rate_pct",
            "sla_met_rate_during_stress_pct",
            "sla_met_rate_no_stress_pct",
        ]
    ]
    .mean()
    .reset_index()
)

fig_sla = go.Figure()
fig_sla.add_scatter(
    x=sla_agg["complaint_month"],
    y=sla_agg["sla_met_rate_no_stress_pct"],
    mode="lines+markers",
    name="No Stress",
    line=dict(color="#22c55e", width=2),
)
fig_sla.add_scatter(
    x=sla_agg["complaint_month"],
    y=sla_agg["overall_sla_met_rate_pct"],
    mode="lines+markers",
    name="Overall",
    line=dict(color="#3b82f6", dash="dash", width=2),
)
fig_sla.add_scatter(
    x=sla_agg["complaint_month"],
    y=sla_agg["sla_met_rate_during_stress_pct"],
    mode="lines+markers",
    name="During Stress",
    line=dict(color="#ef4444", width=2),
)
fig_sla.update_layout(
    height=340,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(title="Month"),
    yaxis=dict(title="SLA Met Rate (%)", range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_sla, use_container_width=True)