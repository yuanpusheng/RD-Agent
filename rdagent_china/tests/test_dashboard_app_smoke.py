from __future__ import annotations

import importlib
import sys
import types
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest


def _install_streamlit_stub(monkeypatch: pytest.MonkeyPatch):
    import types as _types
    from datetime import date as _date

    module = _types.ModuleType("streamlit")
    module._calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(name: str):
        def _inner(*args, **kwargs):
            module._calls.append((name, args, kwargs))
            if "selectbox" in name or "radio" in name:
                options = kwargs.get("options")
                if options is None and len(args) > 1:
                    options = args[1]
                if not options:
                    return kwargs.get("value")
                index = kwargs.get("index", 0)
                if isinstance(options, dict):
                    options = list(options.values())
                try:
                    return options[index]
                except Exception:
                    if isinstance(options, (list, tuple)) and options:
                        return options[0]
                    return kwargs.get("value")
            if "multiselect" in name:
                default = kwargs.get("default")
                if default is None:
                    options = kwargs.get("options") or []
                    return list(options[:1]) if isinstance(options, (list, tuple)) else []
                if isinstance(default, (list, tuple, set)):
                    return list(default)
                return [default]
            if "date_input" in name:
                value = kwargs.get("value")
                if value is None and len(args) > 1:
                    value = args[1]
                if value is None:
                    today = _date.today()
                    return (today, today)
                return value
            if "number_input" in name:
                return kwargs.get("value", kwargs.get("min_value", 0))
            if "checkbox" in name:
                return kwargs.get("value", False)
            return kwargs.get("value")

        return _inner

    module.set_page_config = _record("set_page_config")
    module.title = _record("title")
    module.caption = _record("caption")
    module.header = _record("header")
    module.subheader = _record("subheader")
    module.markdown = _record("markdown")
    module.write = _record("write")
    module.info = _record("info")
    module.warning = _record("warning")
    module.success = _record("success")
    module.error = _record("error")
    module.metric = _record("metric")
    module.dataframe = _record("dataframe")
    module.plotly_chart = _record("plotly_chart")
    module.selectbox = _record("selectbox")
    module.multiselect = _record("multiselect")
    module.date_input = _record("date_input")
    module.number_input = _record("number_input")
    module.radio = _record("radio")
    sidebar = _types.SimpleNamespace()
    for attr in ("header", "selectbox", "multiselect", "date_input", "number_input", "radio", "write"):
        setattr(sidebar, attr, _record(f"sidebar.{attr}"))
    module.sidebar = sidebar
    module.cache_data = lambda **_: (lambda func: func)
    module.cache_resource = lambda **_: (lambda func: func)
    monkeypatch.setitem(sys.modules, "streamlit", module)
    return module


def _install_plotly_stub(monkeypatch: pytest.MonkeyPatch):
    class _FakeFigure:
        def __init__(self, kind: str):
            self.kind = kind
            self.data: list[dict[str, object]] = []
            self.layout: dict[str, object] = {}
            self.args: tuple[object, ...] = ()
            self.kwargs: dict[str, object] = {}

        def add_trace(self, trace: object) -> None:
            self.data.append({"trace": trace})

        def update_layout(self, **kwargs) -> None:
            self.layout.update(kwargs)

        def update_traces(self, **kwargs) -> None:
            self.layout.setdefault("trace_updates", []).append(kwargs)

    def _figure_factory(kind: str):
        def _inner(*args, **kwargs):
            fig = _FakeFigure(kind)
            fig.args = args
            fig.kwargs = kwargs
            return fig

        return _inner

    px_module = types.ModuleType("plotly.express")
    for name in ("line", "bar", "imshow", "scatter", "area"):
        setattr(px_module, name, _figure_factory(name))

    go_module = types.ModuleType("plotly.graph_objects")

    class _Figure(_FakeFigure):
        def __init__(self):
            super().__init__("figure")

    def _trace(kind: str):
        def _inner(*args, **kwargs):
            return {"type": kind, "args": args, "kwargs": kwargs}

        return _inner

    go_module.Figure = _Figure
    go_module.Scatter = _trace("scatter")
    go_module.Candlestick = _trace("candlestick")
    go_module.Bar = _trace("bar")

    plotly_module = types.ModuleType("plotly")
    plotly_module.express = px_module
    plotly_module.graph_objects = go_module

    monkeypatch.setitem(sys.modules, "plotly", plotly_module)
    monkeypatch.setitem(sys.modules, "plotly.express", px_module)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_module)


def _sample_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows: list[dict[str, object]] = []
    for symbol, base in (("AAA", 10.0), ("BBB", 20.0)):
        for idx, dt in enumerate(dates):
            price = base + idx
            rows.append(
                {
                    "symbol": symbol,
                    "date": dt,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000000 + idx * 5000,
                }
            )
    return pd.DataFrame(rows)


def _sample_signals_frame() -> pd.DataFrame:
    base_ts = pd.Timestamp("2024-01-02T09:30:00Z")
    return pd.DataFrame(
        [
            {
                "universe": "TEST",
                "symbol": "AAA",
                "timestamp": base_ts,
                "as_of_date": base_ts.normalize(),
                "rule": "rule_alpha",
                "label": "ALPHA",
                "severity": "medium",
                "triggered": True,
                "value": 0.12,
                "signals": {"volume": {"ratio": 1.4, "score": 0.7}},
                "config_version": "v1",
                "run_version": "rv1",
            },
            {
                "universe": "TEST",
                "symbol": "AAA",
                "timestamp": base_ts + timedelta(days=1),
                "as_of_date": (base_ts + timedelta(days=1)).normalize(),
                "rule": "rule_beta",
                "label": "BETA",
                "severity": "high",
                "triggered": True,
                "value": -0.05,
                "signals": {"trend": {"signal": 1.0, "slope": 0.25}},
                "config_version": "v1",
                "run_version": "rv1",
            },
            {
                "universe": "TEST",
                "symbol": "BBB",
                "timestamp": base_ts,
                "as_of_date": base_ts.normalize(),
                "rule": "rule_alpha",
                "label": "ALPHA",
                "severity": "low",
                "triggered": True,
                "value": 0.08,
                "signals": {"volume": {"ratio": 0.9, "score": 0.4}},
                "config_version": "v1",
                "run_version": "rv1",
            },
        ]
    )


def test_dashboard_app_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_streamlit = _install_streamlit_stub(monkeypatch)
    _install_plotly_stub(monkeypatch)

    duckdb_path = tmp_path / "dashboard.duckdb"
    monkeypatch.setenv("RDC_DUCKDB_PATH", str(duckdb_path))

    import rdagent_china.config as config_module

    importlib.reload(config_module)
    import rdagent_china.db as db_module

    importlib.reload(db_module)
    database = db_module.Database(duckdb_path)
    database.init()
    database.write_price_daily(_sample_price_frame())
    database.write_signals(_sample_signals_frame())

    module_name = "rdagent_china.dashboard.app"
    if module_name in sys.modules:
        del sys.modules[module_name]
    app_module = importlib.import_module(module_name)

    fake_streamlit._calls.clear()
    app_module.render_signals_page()
    app_module.render_backtest_page()
    app_module.render_factor_page()
    app_module.render_kline_page()

    plot_calls = [name for name, _, _ in fake_streamlit._calls if name.endswith("plotly_chart")]
    table_calls = [name for name, _, _ in fake_streamlit._calls if name.endswith("dataframe")]
    assert plot_calls, "Expected plotly charts to be rendered"
    assert table_calls, "Expected tabular views to be rendered"
