from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rdagent_china.dashboard.a_share_monitor.components import (
    SignalSummary,
    build_watchlist_view,
    compute_market_breadth,
    filter_by_rules,
    filter_by_view_mode,
    make_return_heatmap,
    summarize_signals,
)
from rdagent_china.dashboard.a_share_monitor.trace import collect_trace_rows, discover_sessions
from rdagent_china.data.provider import UnifiedDataProvider
from rdagent_china.data.universe import resolve_universe
from rdagent_china.db import Database, get_db


@dataclass(frozen=True)
class RuntimeArgs:
    log_dir: Path | None = None
    session: str | None = None


def parse_runtime_args() -> RuntimeArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-dir", dest="log_dir", type=str, default="", help="Path to RD-Agent log folder")
    parser.add_argument(
        "--session",
        dest="session",
        type=str,
        default="",
        help="Preselect a specific session folder within the log directory",
    )
    parser.add_argument(
        "--universe",
        dest="universe",
        type=str,
        default="CSI300",
        help="Default universe to load when the dashboard boots",
    )
    args, _ = parser.parse_known_args()
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else None
    if log_dir is not None:
        try:
            log_dir = log_dir.resolve()
        except FileNotFoundError:
            # resolve fails when intermediate directories are missing; keep expanded version instead
            log_dir = log_dir
    session = args.session or None
    default_universe = args.universe or "CSI300"
    return RuntimeArgs(log_dir=log_dir, session=session), default_universe


@st.cache_resource
def _get_context() -> tuple[Database, UnifiedDataProvider]:
    db = get_db()
    db.init()
    provider = UnifiedDataProvider(db=db)
    return db, provider


@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(
    symbols: Sequence[str] | tuple[str, ...],
    start: str | None,
    end: str | None,
    *,
    intraday: bool,
) -> pd.DataFrame:
    db, provider = _get_context()
    sym_list = list(symbols)
    frame = db.read_prices(symbols=sym_list, start=start, end=end) if intraday else db.read_price_daily(symbols=sym_list, start=start, end=end)
    source = "duckdb"
    if frame.empty and sym_list:
        try:
            fetched = provider.get_price_daily(sym_list, start=start, end=end)
            frame = fetched
            source = "provider"
        except Exception:
            frame = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
            source = "unavailable"
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame.attrs["source"] = source
    frame.attrs["frequency"] = "intraday" if intraday else "eod"
    return frame


@st.cache_data(ttl=120, show_spinner=False)
def load_signals(
    *,
    universe: str | None,
    symbols: Sequence[str] | tuple[str, ...] | None,
    start: str | None,
    end: str | None,
    limit: int = 1000,
) -> pd.DataFrame:
    db, _ = _get_context()
    params: dict[str, object] = {}
    if universe:
        params["universe"] = universe
    if symbols:
        params["symbols"] = list(symbols)
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    frame = db.read_signals(**params)
    if not frame.empty:
        frame = frame.sort_values("timestamp", ascending=False)
        if limit and limit > 0:
            frame = frame.head(limit)
    frame.attrs["source"] = "duckdb"
    return frame


def _coerce_symbol_list(raw: Iterable[str] | None) -> list[str]:
    if not raw:
        return []
    cleaned = [sym.strip() for sym in raw if sym and sym.strip()]
    return sorted(set(cleaned))


def _default_date_window(days: int = 30) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days)
    return start, end


def _format_summary(summary: SignalSummary) -> tuple[str, str]:
    if not summary.total:
        return "", ""
    earliest = summary.earliest.tz_convert("Asia/Shanghai") if summary.earliest is not None else None
    latest = summary.latest.tz_convert("Asia/Shanghai") if summary.latest is not None else None
    if earliest and latest:
        return earliest.strftime("%Y-%m-%d %H:%M"), latest.strftime("%Y-%m-%d %H:%M")
    return "", ""


def _normalize_dates(selection: Sequence[date] | date | None) -> tuple[str | None, str | None]:
    if selection is None:
        return None, None
    if isinstance(selection, date):
        return selection.isoformat(), selection.isoformat()
    if len(selection) == 0:
        return None, None
    if len(selection) == 1:
        return selection[0].isoformat(), selection[0].isoformat()
    start, end = selection
    if start > end:
        start, end = end, start
    return start.isoformat(), end.isoformat()


def _resolve_symbols(primary: str, custom_text: str) -> list[str]:
    if primary == "Manual":
        manual = [part.strip() for part in custom_text.split(",") if part.strip()]
        return _coerce_symbol_list(manual)
    return _coerce_symbol_list(resolve_universe(primary))


def _render_market_breadth_chart(breadth_df: pd.DataFrame):
    if breadth_df.empty:
        st.info("No market breadth data available for the current filter window.")
        return
    breadth_df = breadth_df.sort_values("date")
    column_a, column_b = st.columns([3, 2])
    with column_a:
        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=breadth_df["date"],
                y=breadth_df["advancing"],
                name="Advancing",
                marker_color="#2ca02c",
            )
        )
        bar_fig.add_trace(
            go.Bar(
                x=breadth_df["date"],
                y=-breadth_df["declining"],
                name="Declining",
                marker_color="#d62728",
            )
        )
        bar_fig.update_layout(
            barmode="relative",
            title="Advancers vs Decliners",
            xaxis_title="Date",
            yaxis_title="Count (decliners shown as negative)",
            legend_orientation="h",
            legend_y=-0.2,
            bargap=0.1,
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displaylogo": False})
    with column_b:
        line_fig = px.line(
            breadth_df,
            x="date",
            y="breadth",
            title="Net Breadth",
            markers=True,
        )
        line_fig.update_layout(yaxis_title="(Adv - Dec) / Total", xaxis_title="Date")
        st.plotly_chart(line_fig, use_container_width=True, config={"displaylogo": False})


def _render_heatmap(heatmap_df: pd.DataFrame):
    if heatmap_df.empty:
        st.info("No return heatmap data available.")
        return
    display = heatmap_df.copy()
    display.index = display.index.strftime("%Y-%m-%d")
    display = display.transpose() * 100
    fig = px.imshow(
        display,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(x="Date", y="Symbol", color="% Change"),
    )
    fig.update_layout(title="Rolling % Change Heatmap", xaxis_title="Date", yaxis_title="Symbol")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_watchlist_table(df: pd.DataFrame, highlight_rules: Iterable[str] | None):
    if df.empty:
        st.info("No persisted signals matched the current filters.")
        return
    highlight_rules = {rule for rule in (highlight_rules or []) if rule}
    display = df.copy()
    display["change_pct"] = display["change_pct"].astype(float).fillna(0.0) * 100
    display["triggered"] = display["triggered"].astype(bool)
    display["highlight"] = display["rule"].isin(highlight_rules)
    column_config = {
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "change_pct": st.column_config.NumberColumn("Δ%", format="%.2f"),
        "timestamp": st.column_config.DatetimeColumn("Signal Time"),
        "price_as_of": st.column_config.DatetimeColumn("Price Time"),
        "triggered": st.column_config.CheckboxColumn("Triggered"),
        "highlight": st.column_config.CheckboxColumn("Highlighted"),
    }
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def _render_signal_feed(df: pd.DataFrame):
    if df.empty:
        st.info("No signals available for the selected filters.")
        return
    display = df.copy()
    display["timestamp"] = pd.to_datetime(display["timestamp"], utc=True, errors="coerce")
    display["timestamp"] = display["timestamp"].dt.tz_convert("Asia/Shanghai")
    display["as_of_date"] = pd.to_datetime(display["as_of_date"], errors="coerce")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_price_panel(prices: pd.DataFrame, symbol: str, *, intraday: bool):
    series = prices[prices["symbol"] == symbol].sort_values("date")
    if series.empty:
        st.info(f"No price data available for {symbol} in the selected window.")
        return
    series = series.dropna(subset=["open", "high", "low", "close"])
    if series.empty:
        st.info(f"Price data for {symbol} is incomplete.")
        return

    series["ma5"] = series["close"].rolling(5).mean()
    series["ma20"] = series["close"].rolling(20).mean()

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Candlestick(
            x=series["date"],
            open=series["open"],
            high=series["high"],
            low=series["low"],
            close=series["close"],
            name="Price",
        )
    )
    price_fig.add_trace(
        go.Scatter(x=series["date"], y=series["ma5"], name="MA5", line=dict(color="#1f77b4", width=1.5))
    )
    price_fig.add_trace(
        go.Scatter(x=series["date"], y=series["ma20"], name="MA20", line=dict(color="#ff7f0e", width=1.5))
    )
    price_fig.update_layout(
        title=f"{symbol} Price Chart ({'Intraday' if intraday else 'EOD'})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=-0.2,
    )
    st.plotly_chart(price_fig, use_container_width=True, config={"displaylogo": False})

    vol_fig = px.bar(series, x="date", y="volume", labels={"date": "Date", "volume": "Volume"})
    vol_fig.update_layout(title=f"{symbol} Volume", bargap=0.2)
    st.plotly_chart(vol_fig, use_container_width=True, config={"displaylogo": False})

    show_mpl = st.checkbox("Show mplfinance rendering", value=False, key=f"mpl_{symbol}")
    if show_mpl:
        try:
            import mplfinance as mpf  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            st.warning("mplfinance is not installed in this environment.")
        else:
            mpf_frame = series.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
            mpf_frame.index.name = "Date"
            fig, _ = mpf.plot(
                mpf_frame,
                type="candle",
                mav=(5, 20),
                volume=True,
                show_nontrading=True,
                returnfig=True,
                style="yahoo",
            )
            st.pyplot(fig, clear_figure=True)


def _render_symbol_signal_history(signals: pd.DataFrame, symbol: str):
    subset = signals[signals["symbol"] == symbol]
    if subset.empty:
        st.info(f"No signal history for {symbol} in this window.")
        return
    subset = subset.copy()
    subset["timestamp"] = pd.to_datetime(subset["timestamp"], utc=True, errors="coerce")
    subset["timestamp"] = subset["timestamp"].dt.tz_convert("Asia/Shanghai")
    subset = subset.dropna(subset=["timestamp"])
    fig = px.scatter(
        subset,
        x="timestamp",
        y="rule",
        color="severity",
        size=subset["triggered"].astype(int) + 0.1,
        hover_data=["label", "value"],
        title=f"Signal history for {symbol}",
    )
    fig.update_layout(xaxis_title="Timestamp", yaxis_title="Rule")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def main() -> None:
    runtime_args, default_universe = parse_runtime_args()

    st.set_page_config(
        page_title="RD-Agent A-share Monitor",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded",
    )

    st.title("A-share Monitoring Dashboard")
    st.caption(
        "Interactive visualization of persisted RD-Agent monitoring signals with integrated log trace inspection."
    )

    start_default, end_default = _default_date_window()

    with st.sidebar:
        st.header("Filters")
        default_index = 0
        if default_universe.lower() == "manual":
            default_index = 2
        elif default_universe.upper() in {"ZZ500", "000905"}:
            default_index = 1
        universe_option = st.selectbox(
            "Universe",
            options=["CSI300", "ZZ500", "Manual"],
            index=default_index,
        )
        manual_symbols_text = ""
        if universe_option == "Manual":
            manual_symbols_text = st.text_area(
                "Symbols (comma separated)",
                value="",
                placeholder="600519, 000333, ...",
            )
        resolved_symbols = _resolve_symbols(universe_option, manual_symbols_text)
        if not resolved_symbols:
            st.info("No symbols resolved. Configure the watchlist manually or ingest data first.")

        default_watchlist = resolved_symbols[: min(len(resolved_symbols), 15)]
        watchlist = st.multiselect(
            "Watchlist",
            options=resolved_symbols,
            default=default_watchlist,
        )
        active_symbols = watchlist or resolved_symbols

        date_selection = st.date_input(
            "Analysis window",
            value=(start_default, end_default),
            help="Controls both price and signal windows.",
        )

        cadence = st.radio(
            "Signal cadence",
            options=["EOD", "Intraday", "Both"],
            index=0,
            horizontal=True,
        )
        triggered_only = st.checkbox("Triggered only", value=False)

        log_dir_text = st.text_input(
            "Log directory",
            value=str(runtime_args.log_dir) if runtime_args.log_dir else "",
            help="Point to the RD-Agent run folder to enable trace inspection.",
        )
        sessions = discover_sessions(log_dir_text) if log_dir_text else []
        selected_session = None
        if sessions:
            default_idx = 0
            if runtime_args.session:
                for idx, session in enumerate(sessions):
                    if runtime_args.session in {session.name, str(session)}:
                        default_idx = idx
                        break
            selected_session = st.selectbox(
                "Trace session",
                options=sessions,
                index=default_idx,
                format_func=lambda p: p.name,
            )
        elif log_dir_text:
            st.info("No RD-Agent trace files detected under the provided log directory.")

        severity_placeholder = st.empty()
        rules_placeholder = st.empty()

    if not active_symbols:
        st.warning("Select at least one symbol to load the dashboard.")
        return

    start_iso, end_iso = _normalize_dates(date_selection)

    signals_raw = load_signals(
        universe=None if universe_option == "Manual" else universe_option,
        symbols=tuple(active_symbols),
        start=start_iso,
        end=end_iso,
    )

    all_severities = sorted(set(signals_raw.get("severity", pd.Series(dtype=str)).dropna().tolist()))
    all_rules = sorted(set(signals_raw.get("rule", pd.Series(dtype=str)).dropna().tolist()))

    severity_filter = severity_placeholder.multiselect(
        "Severity",
        options=all_severities,
        default=[],
    )
    highlight_rules = rules_placeholder.multiselect(
        "Highlight rules",
        options=all_rules,
        default=[],
    )

    signals_filtered = filter_by_view_mode(signals_raw, cadence.lower())
    if severity_filter:
        signals_filtered = signals_filtered[signals_filtered["severity"].isin(severity_filter)]
    if triggered_only and not signals_filtered.empty and "triggered" in signals_filtered.columns:
        signals_filtered = signals_filtered[signals_filtered["triggered"].astype(bool)]
    if watchlist:
        signals_filtered = signals_filtered[signals_filtered["symbol"].isin(watchlist)]

    summary = summarize_signals(signals_filtered)

    price_df = load_price_data(tuple(active_symbols), start_iso, end_iso, intraday=cadence.lower() == "intraday")
    watchlist_table = build_watchlist_view(signals_filtered, price_df)

    overview_tab, watchlist_tab, feed_tab, detail_tab = st.tabs(
        ["Overview", "Watchlist & Filters", "Signal Feed", "Stock Detail"]
    )

    with overview_tab:
        st.subheader("Summary metrics")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Signals", f"{summary.total}")
        col_b.metric("Triggered", f"{summary.triggered}", f"{summary.triggered_pct * 100:.1f}%")
        col_c.metric("Symbols", str(summary.unique_symbols))
        col_d.metric("Rules", str(summary.unique_rules))
        start_ts, end_ts = _format_summary(summary)
        if start_ts and end_ts:
            st.caption(f"Window: {start_ts} → {end_ts} (Asia/Shanghai)")
        data_source = price_df.attrs.get("source", "duckdb")
        st.caption(f"Market data source: {data_source} cache")

        st.markdown("### Market breadth")
        breadth_df = compute_market_breadth(price_df)
        _render_market_breadth_chart(breadth_df)

        st.markdown("### Return heatmap")
        heatmap_df = make_return_heatmap(price_df)
        _render_heatmap(heatmap_df)

    with watchlist_tab:
        st.subheader("Watchlist snapshot")
        _render_watchlist_table(watchlist_table, highlight_rules)

    with feed_tab:
        st.subheader("Signal feed")
        _render_signal_feed(signals_filtered)
        if not signals_filtered.empty:
            csv_data = signals_filtered.to_csv(index=False).encode("utf-8")
            st.download_button("Download as CSV", data=csv_data, file_name="a_share_signals.csv")

        st.markdown("### Trace inspector")
        focus_symbol = watchlist[0] if watchlist else (active_symbols[0] if active_symbols else None)
        trace_symbol = st.selectbox(
            "Focus symbol for logs",
            options=active_symbols,
            index=active_symbols.index(focus_symbol) if focus_symbol in active_symbols else 0,
        )
        if selected_session:
            trace_df = collect_trace_rows(
                selected_session,
                symbol_filter=trace_symbol,
                rule_filter=highlight_rules,
                limit=200,
            )
            if trace_df.empty:
                st.info("No trace entries matched the current filters.")
            else:
                st.dataframe(trace_df, use_container_width=True, hide_index=True)
        else:
            st.info("Provide a log directory to inspect RD-Agent traces for matching signals.")

    with detail_tab:
        st.subheader("Stock detail")
        focus_symbol = st.selectbox(
            "Symbol",
            options=active_symbols,
            index=active_symbols.index(watchlist[0]) if watchlist and watchlist[0] in active_symbols else 0,
        )
        _render_price_panel(price_df, focus_symbol, intraday=cadence.lower() == "intraday")
        _render_symbol_signal_history(signals_filtered, focus_symbol)


if __name__ == "__main__":
    main()
