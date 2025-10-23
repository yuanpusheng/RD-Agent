from __future__ import annotations

from pathlib import Path

import pandas as pd

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
from rdagent.log.storage import FileStorage


def test_summarize_signals_basic():
    frame = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "timestamp": pd.Timestamp("2024-01-01T01:00:00Z"),
                "rule": "gap",
                "severity": "high",
                "triggered": True,
            },
            {
                "symbol": "000001",
                "timestamp": pd.Timestamp("2024-01-01T02:00:00Z"),
                "rule": "gap",
                "severity": "high",
                "triggered": False,
            },
            {
                "symbol": "000333",
                "timestamp": pd.Timestamp("2024-01-02T09:30:00Z"),
                "rule": "sma",
                "severity": "medium",
                "triggered": True,
            },
        ]
    )

    summary = summarize_signals(frame)

    assert isinstance(summary, SignalSummary)
    assert summary.total == 3
    assert summary.triggered == 2
    assert summary.unique_symbols == 2
    assert summary.unique_rules == 2
    assert summary.earliest <= summary.latest


def test_compute_market_breadth_and_heatmap():
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-01", "close": 10.0},
            {"symbol": "BBB", "date": "2024-01-01", "close": 20.0},
            {"symbol": "AAA", "date": "2024-01-02", "close": 11.0},
            {"symbol": "BBB", "date": "2024-01-02", "close": 19.0},
            {"symbol": "AAA", "date": "2024-01-03", "close": 11.5},
            {"symbol": "BBB", "date": "2024-01-03", "close": 19.0},
        ]
    )

    breadth = compute_market_breadth(prices)
    assert breadth.shape[0] == 2
    latest = breadth.iloc[-1]
    assert latest["advancing"] == 1
    assert latest["declining"] == 0

    heatmap = make_return_heatmap(prices, periods=2)
    assert heatmap.shape == (2, 2)
    assert "AAA" in heatmap.columns
    assert heatmap.index[-1] == pd.Timestamp("2024-01-03")


def test_build_watchlist_view_and_filters():
    signals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": "2024-01-01T00:00:00Z",
                "rule": "gap",
                "label": "Gap open",
                "severity": "high",
                "triggered": True,
                "value": {"delta": 1.0},
                "universe": "CSI300",
                "run_version": "eod-20240101",
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-02T00:00:00Z",
                "rule": "sma",
                "label": "MA crossover",
                "severity": "low",
                "triggered": False,
                "value": {"delta": 0.1},
                "universe": "CSI300",
                "run_version": "intraday-202401020930",
            },
            {
                "symbol": "BBB",
                "timestamp": "2024-01-02T00:15:00Z",
                "rule": "sma",
                "label": "MA crossover",
                "severity": "low",
                "triggered": True,
                "value": {"delta": -0.2},
                "universe": "CSI300",
                "run_version": "intraday-202401020930",
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2024-01-01", "close": 10.0, "open": 9.8, "high": 10.2, "low": 9.6, "volume": 1000},
            {"symbol": "AAA", "date": "2024-01-02", "close": 11.0, "open": 10.5, "high": 11.2, "low": 10.3, "volume": 1200},
            {"symbol": "BBB", "date": "2024-01-02", "close": 20.0, "open": 19.7, "high": 20.2, "low": 19.5, "volume": 1500},
        ]
    )

    watchlist = build_watchlist_view(signals, prices)
    assert watchlist.loc[watchlist["symbol"] == "AAA", "rule"].iloc[0] == "sma"
    assert "change_pct" in watchlist.columns

    intraday_only = filter_by_view_mode(signals, "intraday")
    assert intraday_only.shape[0] == 2
    eod_only = filter_by_view_mode(signals, "eod")
    assert eod_only.shape[0] == 1

    filtered = filter_by_rules(signals, {"gap"})
    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["rule"] == "gap"


def test_trace_discovery_and_collection(tmp_path: Path):
    session = tmp_path / "run-001"
    session.mkdir()
    storage = FileStorage(session)
    storage.log({"symbol": "AAA", "rule": "gap"}, tag="a_share_monitor.loop")
    storage.log("other", tag="other_scenario")

    sessions = discover_sessions(tmp_path)
    assert session in sessions

    rows = collect_trace_rows(session, symbol_filter="AAA", rule_filter=["gap"], limit=10)
    assert rows.shape[0] == 1
    assert "gap" in rows.iloc[0]["summary"]

    empty_rows = collect_trace_rows(session, symbol_filter="ZZZ", rule_filter=[], limit=5)
    assert empty_rows.empty
