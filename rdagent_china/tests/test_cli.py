from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from rdagent_china.cli import app


runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "backtest" in result.stdout
    assert "monitor" in result.stdout


def test_cli_ingest_uses_csi300(monkeypatch):
    calls: dict[str, object] = {"csi": 0, "all": 0}

    class DummyDB:
        def init(self) -> None:
            calls["db_init"] = True

    def fake_get_csi300_symbols():
        calls["csi"] += 1
        return ["000001.SZ", "600519.SH"]

    def fake_get_all(*_args, **_kwargs):
        calls["all"] += 1
        return []

    def fake_ingest_prices(*, symbols, start, end, db):
        calls["ingest"] = (list(symbols), start, end, db)

    monkeypatch.setattr("rdagent_china.cli.get_db", lambda: DummyDB())
    monkeypatch.setattr("rdagent_china.cli.get_csi300_symbols", fake_get_csi300_symbols)
    monkeypatch.setattr("rdagent_china.cli.get_all_a_stock_symbols", fake_get_all)
    monkeypatch.setattr("rdagent_china.cli.ingest_prices", fake_ingest_prices)

    result = runner.invoke(app, [
        "ingest",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
    ])

    assert result.exit_code == 0
    assert calls["csi"] == 1
    assert calls["all"] == 0
    symbols, start, end, db = calls["ingest"]
    assert symbols == ["000001.SZ", "600519.SH"]
    assert start == "2024-01-01" and end == "2024-01-31"
    assert hasattr(db, "init")


def test_cli_ingest_supports_all_universe(monkeypatch):
    calls: dict[str, object] = {"all": 0}

    class DummyDB:
        def init(self) -> None:
            calls["db_init"] = True

    def fake_get_all(limit: int):
        calls["all"] += 1
        calls["limit"] = limit
        return ["000001.SZ"]

    def fake_ingest_prices(*, symbols, start, end, db):
        calls["ingest"] = (list(symbols), start, end, db)

    monkeypatch.setattr("rdagent_china.cli.get_db", lambda: DummyDB())
    monkeypatch.setattr("rdagent_china.cli.get_all_a_stock_symbols", fake_get_all)
    monkeypatch.setattr("rdagent_china.cli.ingest_prices", fake_ingest_prices)

    result = runner.invoke(app, [
        "ingest",
        "--universe",
        "ALL",
        "--limit",
        "5",
    ])

    assert result.exit_code == 0
    assert calls["all"] == 1
    assert calls["limit"] == 5
    symbols, _, _, _ = calls["ingest"]
    assert symbols == ["000001.SZ"]


def test_cli_sync_price_daily_groups_symbols(monkeypatch):
    class DummyDB:
        def __init__(self) -> None:
            self.initialised = False

        def init(self) -> None:
            self.initialised = True

    class DummyProvider:
        def __init__(self, db: DummyDB) -> None:
            self.db = db
            self.latest_requests: list[tuple[str, str]] = []
            self.fetch_requests: list[tuple[tuple[str, ...], str | None, str | None]] = []

        def latest_synced_date(self, symbol: str, dataset: str) -> str | None:
            self.latest_requests.append((symbol, dataset))
            return "2024-01-05" if symbol == "000002.SZ" else None

        def get_price_daily(self, symbols, start=None, end=None):
            self.fetch_requests.append((tuple(symbols), start, end))
            return pd.DataFrame()

    db = DummyDB()
    provider_holder: dict[str, DummyProvider] = {}

    def provider_factory(db_arg: DummyDB) -> DummyProvider:
        provider = DummyProvider(db_arg)
        provider_holder["instance"] = provider
        return provider

    monkeypatch.setattr("rdagent_china.cli.get_db", lambda: db)
    monkeypatch.setattr("rdagent_china.cli.get_csi300_symbols", lambda: ["000001.SZ", "000002.SZ"])
    monkeypatch.setattr("rdagent_china.cli.UnifiedDataProvider", provider_factory)

    result = runner.invoke(app, ["sync-price-daily"])

    assert result.exit_code == 0
    assert db.initialised is True
    provider = provider_holder["instance"]
    assert provider.latest_requests == [
        ("000001.SZ", "price_daily"),
        ("000002.SZ", "price_daily"),
    ]
    assert provider.fetch_requests == [
        (("000001.SZ",), None, None),
        (("000002.SZ",), "2024-01-06", None),
    ]


def test_cli_monitor_run_once_invokes_loop(monkeypatch):
    class DummyLoop:
        def __init__(self) -> None:
            self.contexts = []

        def run_cycle(self, context):
            self.contexts.append(context)
            return pd.DataFrame(
                {
                    "universe": ["CSI300"],
                    "symbol": ["000001.SZ"],
                    "rule": ["volume_breakout_combo"],
                    "timestamp": [pd.Timestamp("2024-01-08T09:35:00Z")],
                    "triggered": [True],
                    "value": [2.4],
                }
            )

    dummy_loop = DummyLoop()
    monkeypatch.setattr("rdagent_china.cli.MonitorLoop", lambda: dummy_loop)

    result = runner.invoke(
        app,
        [
            "monitor",
            "--run-once",
            "--watchlist",
            "000001.SZ,600519.SH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ],
    )

    assert result.exit_code == 0
    assert len(dummy_loop.contexts) == 1
    context = dummy_loop.contexts[0]
    assert context.watchlist == "000001.SZ,600519.SH"
    assert context.intraday is False
    assert context.start == "2024-01-01" and context.end == "2024-01-31"
