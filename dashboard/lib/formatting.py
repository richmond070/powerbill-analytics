from typing import Optional, Union

Number = Union[int, float]


def format_ngn(value: Optional[Number], compact: bool = True) -> str:
    """
    Format a Naira amount.

    Args:
        value:   Raw NGN amount (None or NaN renders as '—').
        compact: If True, abbreviate large values (₦1.2M, ₦340.5K).
                 If False, render full value with thousands separators.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "—"

    sign = "-" if value < 0 else ""
    value = abs(value)

    if not compact:
        return f"{sign}₦{value:,.2f}"
    if value >= 1_000_000_000:
        return f"{sign}₦{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{sign}₦{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}₦{value / 1_000:.1f}K"
    return f"{sign}₦{value:,.0f}"


def format_pct(value: Optional[Number], decimals: int = 1) -> str:
    """
    Format a percentage. Every *_pct column in the Gold marts is already
    multiplied by 100 in dbt (collection_rate_pct, on_time_rate_pct,
    stress_complaint_share_pct, etc.) — do NOT pass a 0-1 fraction here.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{decimals}f}%"


def format_delta_pct(value: Optional[Number], decimals: int = 1) -> Optional[str]:
    """
    Format a percentage-point delta for st.metric()'s `delta` argument.
    Returns None (not '—') on missing data, since st.metric hides the delta
    row entirely when delta=None — '—' would render as a confusing literal.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f} ppt"


def format_number(value: Optional[Number], compact: bool = True) -> str:
    """Format a plain count (customers, complaints, bills) with optional K/M abbreviation."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if not compact:
        return f"{value:,.0f}"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"