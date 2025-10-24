from __future__ import annotations

import numbers
from datetime import date, timedelta
from typing import Callable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rdagent_china.config import settings
from rdagent_china.db import Database, get_db


DEFAULT_LOOKBACK_DAYS = 30


@st.cache_resource
def get_database() -> Database:
    db = get_db()
    db.init()
    return db


def _default_date_window(days: int = DEFAULT_LOOKBACK_DAYS) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days)
    return start, end


def _normalize_date_range(selection: object) -> tuple[str | None, str | None]:
    if selection is None:
        return None, None
    if isinstance(selection, (list, tuple)):
        if not selection:
            return None, None
        if len(selection) == 1:
            start_obj = selection[0]
            end_obj = selection[0]
        else:
            start_obj, end_obj = selection[:2]
    elif isinstance(selection, date):
        start_obj = selection
        end_obj = selection
    else:
        return None, None
    if isinstance(start_obj, date) and isinstance(end_obj, date) and start_obj > end_obj:
        start_obj, end_obj = end_obj, start_obj
    start = start_obj.isoformat() if isinstance(start_obj, date) else None
    end = end_obj.isoformat() if isinstance(end_obj, date) else None
    return start, end


@st.cache_data(ttl=120, show_spinner=False)
def _load_universes() -> list[str]:
    db = get_database()
    try:
        rows = db.conn.execute(
            "SELECT DISTINCT universe FROM computed_signals WHERE universe IS NOT NULL ORDER BY universe"
        ).fetchall()
    except Exception:
        return []
    return [str(row[0]) for row in rows if row and row[0]]


@st.cache_data(ttl=120, show_spinner=False)
def _load_symbol_universe(universe: str | None) -> list[str]:
    db = get_database()
    params: list[object] = []
    query = "SELECT DISTINCT symbol FROM computed_signals"
    if universe:
        query += " WHERE universe = ?"
        params.append(universe)
    query += " ORDER BY symbol"
    try:
        rows = db.conn.execute(query, params).fetchall()
    except Exception:
        rows = []
    symbols = [str(row[0]) for row in rows if row and row[0]]
    if symbols:
        return symbols
    try:
        rows = db.conn.execute("SELECT DISTINCT symbol FROM price_daily ORDER BY symbol").fetchall()
    except Exception:
        return []
    return [str(row[0]) for row in rows if row and row[0]]


@st.cache_data(ttl=300, show_spinner=False)
def _load_price_symbols() -> list[str]:
    db = get_database()
    try:
        rows = db.conn.execute("SELECT DISTINCT symbol FROM price_daily ORDER BY symbol").fetchall()
    except Exception:
        return []
    return [str(row[0]) for row in rows if row and row[0]]


@st.cache_data(ttl=300, show_spinner=False)
def _load_signals(
    universe: str | None,
    symbols: tuple[str, ...] | None,
    start: str | None,
    end: str | None,
    limit: int,
) -> pd.DataFrame:
    db = get_database()
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
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp", ascending=False)
    if limit:
        frame = frame.head(limit)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.date
    if "triggered" in frame.columns:
        frame["triggered"] = frame["triggered"].astype(bool)
    return frame.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _load_price_history(
    symbols: tuple[str, ...] | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    db = get_database()
    params: dict[str, object] = {}
    if symbols:
        params["symbols"] = list(symbols)
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    frame = db.read_price_daily(**params)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _sidebar_filters(key_prefix: str, *, default_days: int) -> tuple[str | None, tuple[str, ...] | None, str | None, str | None]:
    label_suffix = "" if key_prefix == "signals" else f" ({key_prefix})"
    universes = _load_universes()
    universe_options = ["All"] + universes if universes else ["All"]
    universe_choice = st.sidebar.selectbox(
        f"Universe{label_suffix}",
        options=universe_options,
        index=0,
        key=f"{key_prefix}_universe",
    )
    symbol_options = _load_symbol_universe(None if universe_choice == "All" else universe_choice)
    default_symbols = tuple(symbol_options[: min(len(symbol_options), 5)])
    selected_symbols: Sequence[str] = []
    if symbol_options:
        selected_symbols = st.sidebar.multiselect(
            f"Symbols{label_suffix}",
            options=symbol_options,
            default=list(default_symbols),
            key=f"{key_prefix}_symbols",
        )
    start_default, end_default = _default_date_window(default_days)
    date_selection = st.sidebar.date_input(
        f"Date range{label_suffix}",
        value=(start_default, end_default),
        key=f"{key_prefix}_dates",
    )
    start, end = _normalize_date_range(date_selection)
    universe_filter = None if universe_choice == "All" else universe_choice
    symbol_filter = tuple(selected_symbols) if selected_symbols else None
    return universe_filter, symbol_filter, start, end


def compute_forward_returns(signals: pd.DataFrame, prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if signals.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "rule", "label", "entry_date", "exit_date", "return", "severity"])
    working = signals.copy()
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce").dt.normalize()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    grouped = {symbol: group.sort_values("date").reset_index(drop=True) for symbol, group in prices.groupby("symbol")}
    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        symbol = row.get("symbol")
        if not symbol:
            continue
        price_frame = grouped.get(symbol)
        if price_frame is None or price_frame.empty:
            continue
        as_of = row.get("as_of_date")
        if pd.isna(as_of):
            continue
        dates = price_frame["date"].tolist()
        start_idx = None
        for idx, current in enumerate(dates):
            if current >= as_of:
                start_idx = idx
                break
        if start_idx is None:
            continue
        exit_idx = start_idx + horizon
        if exit_idx >= len(dates):
            continue
        entry_price = price_frame.loc[start_idx, "close"]
        exit_price = price_frame.loc[exit_idx, "close"]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue
        forward_return = (exit_price - entry_price) / entry_price
        rows.append(
            {
                "symbol": symbol,
                "rule": row.get("rule", ""),
                "label": row.get("label", ""),
                "entry_date": dates[start_idx],
                "exit_date": dates[exit_idx],
                "return": forward_return,
                "severity": row.get("severity"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "rule", "label", "entry_date", "exit_date", "return", "severity"])
    return pd.DataFrame(rows)


def extract_factor_importance(signals: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in signals.iterrows():
        payload = row.get("signals")
        if isinstance(payload, dict):
            for factor, detail in payload.items():
                if isinstance(detail, dict):
                    for metric, value in detail.items():
                        if isinstance(value, numbers.Number):
                            records.append({
                                "factor": f"{factor}.{metric}",
                                "value": float(value),
                            })
    if not records:
        return pd.DataFrame(columns=["factor", "occurrences", "avg_value", "avg_abs"])
    frame = pd.DataFrame(records)
    frame["abs"] = frame["value"].abs()
    summary = (
        frame.groupby("factor")
        .agg(
            occurrences=("value", "count"),
            avg_value=("value", "mean"),
            avg_abs=("abs", "mean"),
        )
        .reset_index()
        .sort_values("avg_abs", ascending=False)
    )
    return summary


def render_signals_page(db: Database | None = None) -> None:  # pragma: no cover - exercised via smoke test
    del db
    universe, symbols, start, end = _sidebar_filters("signals", default_days=DEFAULT_LOOKBACK_DAYS)
    limit = int(
        st.sidebar.number_input(
            "Max rows",
            min_value=200,
            max_value=5000,
            value=1000,
            step=100,
            key="signals_limit",
        )
    )
    frame = _load_signals(universe, symbols, start, end, limit)
    st.header("Signals overview")
    if frame.empty:
        st.info("No signals available for the selected filters.")
        return
    st.metric("Signals", f"{len(frame):,}")
    if "triggered" in frame.columns:
        st.metric("Triggered", f"{int(frame['triggered'].sum()):,}")
    if "symbol" in frame.columns:
        st.metric("Symbols", f"{frame['symbol'].nunique():,}")
    timeline = frame.copy()
    if "timestamp" in timeline.columns:
        timeline["date"] = pd.to_datetime(timeline["timestamp"], errors="coerce").dt.date
        timeline = timeline.dropna(subset=["date"])
        if not timeline.empty:
            summary = (
                timeline.groupby(["date", "severity"], dropna=False)
                .size()
                .reset_index(name="signals")
                .sort_values("date")
            )
            if not summary.empty:
                summary["severity"] = summary["severity"].fillna("Unknown")
                fig = px.bar(
                    summary,
                    x="date",
                    y="signals",
                    color="severity",
                    title="Signals per day",
                    labels={"signals": "Signals", "date": "Date", "severity": "Severity"},
                )
                fig.update_layout(legend_title_text="Severity")
                st.plotly_chart(fig, use_container_width=True)
    display = frame.copy()
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"], errors="coerce")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_backtest_page(db: Database | None = None) -> None:  # pragma: no cover - exercised via smoke test
    del db
    universe, symbols, start, end = _sidebar_filters("backtest", default_days=60)
    horizon = int(
        st.sidebar.number_input(
            "Forward horizon (days)",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="backtest_horizon",
        )
    )
    limit = int(
        st.sidebar.number_input(
            "Max signals evaluated",
            min_value=200,
            max_value=5000,
            value=2000,
            step=100,
            key="backtest_limit",
        )
    )
    frame = _load_signals(universe, symbols, start, end, limit)
    st.header("Backtest results")
    if frame.empty:
        st.info("No signals available for the selected filters.")
        return
    source = frame
    if "triggered" in frame.columns:
        source = frame[frame["triggered"].astype(bool)]
    if source.empty:
        st.info("No triggered signals to evaluate.")
        return
    symbol_scope = symbols or tuple(sorted(source["symbol"].dropna().unique().tolist()))
    prices = _load_price_history(symbol_scope, start, end)
    results = compute_forward_returns(source, prices, horizon)
    if results.empty:
        st.info("Insufficient price history to compute forward returns.")
        return
    avg_return = results["return"].mean()
    hit_rate = (results["return"] > 0).mean()
    st.metric("Average forward return", f"{avg_return * 100:.2f}%")
    st.metric("Hit rate", f"{hit_rate * 100:.1f}%")
    per_rule = (
        results.groupby("rule")
        .agg(samples=("return", "count"), mean_return=("return", "mean"), median_return=("return", "median"))
        .reset_index()
        .sort_values("mean_return", ascending=False)
    )
    if not per_rule.empty:
        fig = px.bar(
            per_rule,
            x="rule",
            y="mean_return",
            text="samples",
            title=f"Average {horizon}-day returns by rule",
            labels={"rule": "Rule", "mean_return": "Mean return"},
        )
        fig.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)
    detailed = results.copy()
    detailed["entry_date"] = pd.to_datetime(detailed["entry_date"], errors="coerce")
    detailed["exit_date"] = pd.to_datetime(detailed["exit_date"], errors="coerce")
    detailed["return"] = detailed["return"] * 100
    st.dataframe(detailed, use_container_width=True, hide_index=True)


def render_factor_page(db: Database | None = None) -> None:  # pragma: no cover - exercised via smoke test
    del db
    universe, symbols, start, end = _sidebar_filters("factor", default_days=90)
    limit = int(
        st.sidebar.number_input(
            "Max signals analyzed",
            min_value=200,
            max_value=5000,
            value=3000,
            step=100,
            key="factor_limit",
        )
    )
    frame = _load_signals(universe, symbols, start, end, limit)
    st.header("Factor importance")
    if frame.empty:
        st.info("No signals available for the selected filters.")
        return
    summary = extract_factor_importance(frame)
    if summary.empty:
        st.info("Signal payloads do not contain numeric factor values.")
        return
    st.dataframe(summary, use_container_width=True, hide_index=True)
    top = summary.head(15).sort_values("avg_abs", ascending=True)
    fig = px.bar(
        top,
        x="avg_abs",
        y="factor",
        orientation="h",
        title="Top factor contributions",
        labels={"avg_abs": "Avg |value|", "factor": "Factor"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_kline_page(db: Database | None = None) -> None:  # pragma: no cover - exercised via smoke test
    del db
    universe, symbols, start, end = _sidebar_filters("kline", default_days=120)
    candidates = _load_price_symbols()
    if symbols:
        filtered = [sym for sym in symbols if sym in candidates]
        if filtered:
            candidates = filtered
    if not candidates:
        st.header("K-line explorer")
        st.info("No price data available. Ingest prices to enable K-line charts.")
        return
    selected_symbol = st.sidebar.selectbox(
        "Symbol (K-line)",
        options=candidates,
        index=0,
        key="kline_symbol",
    )
    ma_windows = st.sidebar.multiselect(
        "Moving averages",
        options=[5, 10, 20, 30, 60],
        default=[5, 20],
        key="kline_ma",
    )
    st.header("K-line explorer")
    prices = _load_price_history((selected_symbol,), start, end)
    if prices.empty:
        st.info(f"No price data found for {selected_symbol}.")
        return
    series = prices.sort_values("date").dropna(subset=["open", "high", "low", "close"])
    if series.empty:
        st.info(f"Incomplete price data for {selected_symbol}.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=series["date"],
            open=series["open"],
            high=series["high"],
            low=series["low"],
            close=series["close"],
            name="Price",
        )
    )
    for window in ma_windows:
        column = f"ma_{window}"
        series[column] = series["close"].rolling(int(window)).mean()
        fig.add_trace(go.Scatter(x=series["date"], y=series[column], name=f"MA{window}"))
    fig.update_layout(
        title=f"{selected_symbol} K-line",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    signal_frame = _load_signals(universe, (selected_symbol,), start, end, limit=1000)
    if signal_frame.empty:
        return
    triggered = signal_frame
    if "triggered" in triggered.columns:
        triggered = triggered[triggered["triggered"].astype(bool)]
    triggered = triggered.copy()
    triggered["timestamp"] = pd.to_datetime(triggered["timestamp"], errors="coerce")
    triggered = triggered.dropna(subset=["timestamp"])
    if triggered.empty:
        return
    triggered["timestamp"] = triggered["timestamp"].dt.tz_localize(None)
    scatter = px.scatter(
        triggered,
        x="timestamp",
        y="rule",
        color="severity",
        hover_data=["symbol", "label", "value"],
        title="Signal timeline",
    )
    scatter.update_layout(xaxis_title="Timestamp", yaxis_title="Rule")
    st.plotly_chart(scatter, use_container_width=True)


PAGE_HANDLERS: dict[str, Callable[[Database | None], None]] = {
    "Signals": render_signals_page,
    "Backtests": render_backtest_page,
    "Factor Importance": render_factor_page,
    "K-Line": render_kline_page,
}


def main() -> None:  # pragma: no cover - exercised via smoke test
    st.set_page_config(
        page_title="RD-Agent China Dashboard",
        layout="wide",
        page_icon="📊",
    )
    st.title("RD-Agent China Dashboard")
    st.caption("Monitor signals, evaluate performance, inspect factor importances, and explore K-line charts.")
    st.sidebar.header("Navigation")
    st.sidebar.write(f"DuckDB path: `{settings.duckdb_path}`")
    page = st.sidebar.radio("View", options=list(PAGE_HANDLERS.keys()), index=0, key="page_selector")
    handler = PAGE_HANDLERS.get(page, render_signals_page)
    handler()


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    main()
