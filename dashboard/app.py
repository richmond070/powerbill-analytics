import streamlit as st

from lib import queries
from lib.formatting import format_number, format_pct
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Energy Ops Dashboard", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# Global filters — shared across all pages via st.session_state.
# mart_payment_behavior and mart_customer_lifetime_value have no date column
# (lifetime summaries) — Customer Risk page must not apply the date range
# to those two.
# ---------------------------------------------------------------------------

st.sidebar.header("Global Filters")

if "discos" not in st.session_state:
    try:
        st.session_state["discos"] = ["All"] + queries.get_available_discos()
    except Exception as e:
        st.session_state["discos"] = ["All"]
        st.sidebar.error(f"Could not load disco list: {e}")

selected_disco = st.sidebar.selectbox(
    "Disco", options=st.session_state["discos"], index=0, key="global_disco"
)

date_range = st.sidebar.date_input(
    "Date range (time-series marts only)", value=(), key="global_date_range"
)
if len(date_range) == 2:
    st.session_state["global_start_date"], st.session_state["global_end_date"] = date_range
else:
    st.session_state["global_start_date"], st.session_state["global_end_date"] = None, None

st.sidebar.caption(
    "Date range does not apply to Customer Risk's CLV and payment behavior "
    "tables — those marts are lifetime summaries with no date grain."
)

# ---------------------------------------------------------------------------
# Landing content
# ---------------------------------------------------------------------------

st.title("⚡Power Bill")
st.markdown(
    """
    - **Grid Reliability** — disco stress heatmaps, SLA breach rates, instability trends
    - **Executive Revenue** — collection efficiency, revenue trends, tariff-band profitability
    - **Customer Risk** — high-risk customers, arrears growth, CLV segmentation
    """
)

st.divider()
st.subheader("Connectivity Check")
#st.caption(
  #  "Smoke test against main.gold.mart_grid_reliability — confirms the "
 #   "Databricks SQL connector before the full pages are built."
#)

try:
    df = queries.get_grid_reliability(
        disco=selected_disco,
        start_date=st.session_state["global_start_date"],
        end_date=st.session_state["global_end_date"],
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows returned", format_number(len(df), compact=False))
    if not df.empty:
        col2.metric("Avg stress complaint share", format_pct(df["stress_complaint_share_pct"].mean()))
        col3.metric("Avg SLA met rate", format_pct(df["overall_sla_met_rate_pct"].mean()))
    else:
        col2.metric("Avg stress complaint share", "—")
        col3.metric("Avg SLA met rate", "—")
    #st.success("Connected to Databricks — mart_grid_reliability is reachable.")
except Exception as e:
    st.error(f"Connection failed: {e}")