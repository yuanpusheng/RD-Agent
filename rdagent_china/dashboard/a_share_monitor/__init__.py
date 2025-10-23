"""Utilities and Streamlit entry point for the A-share monitor dashboard."""

from .components import (
    build_watchlist_view,
    compute_market_breadth,
    make_return_heatmap,
    summarize_signals,
)
from .trace import collect_trace_rows, discover_sessions

__all__ = [
    "build_watchlist_view",
    "collect_trace_rows",
    "compute_market_breadth",
    "discover_sessions",
    "make_return_heatmap",
    "summarize_signals",
]
