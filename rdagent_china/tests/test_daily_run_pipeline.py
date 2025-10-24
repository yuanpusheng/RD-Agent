from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from rdagent_china.daily_run import app as daily_app
from rdagent_china.db import Database


def test_daily_signal_pipeline_persists_signals(monkeypatch, tmp_path):
    runner = CliRunner()

    db_path = tmp_path / "market.duckdb"
    db = Database(db_path)

    provider_holder: dict[str, object] = {}

    class DummyProvider:
        def __init__(self, db: Database, **_: object) -> None:
            self.db = db
            self.calls: list[tuple[tuple[str, ...], str | None, str | None]] = []

        def get_price_daily(self, symbols, start=None, end=None):
            self.calls.append((tuple(symbols), start, end))
            dates = pd.date_range("2024-01-01", periods=6, freq="D")
            rows = []
            for sym in symbols:
                if sym == "AAA":
                    closes = [10, 11, 12, 13, 14, 15]
                else:
                    closes = [20, 19, 18, 17, 16, 15]
                for idx, dt in enumerate(dates):
                    rows.append(
                        {
                            "symbol": sym,
                            "date": dt,
                            "open": closes[idx] - 0.5,
                            "high": closes[idx] + 0.5,
                            "low": closes[idx] - 1.0,
                            "close": closes[idx],
                            "volume": 1_000 + idx,
                        }
                    )
            return pd.DataFrame(rows)

    def provider_factory(*args, **kwargs):
        provider = DummyProvider(*args, **kwargs)
        provider_holder["instance"] = provider
        return provider

    monkeypatch.setattr("rdagent_china.daily_run.get_db", lambda: db)
    monkeypatch.setattr("rdagent_china.daily_run.UnifiedDataProvider", provider_factory)

    result = runner.invoke(
        daily_app,
        [
            "run",
            "--universe",
            "AAA,BBB",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-06",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Generated 2 signals" in result.stdout

    assert "instance" in provider_holder
    provider = provider_holder["instance"]
    assert provider.calls == [(("AAA", "BBB"), "2024-01-01", "2024-01-06")]

    signals = db.read_strategy_signals()
    assert len(signals) == 2
    assert set(signals["symbol"]) == {"AAA", "BBB"}

    aaa_row = signals.loc[signals["symbol"] == "AAA"].iloc[0]
    bbb_row = signals.loc[signals["symbol"] == "BBB"].iloc[0]

    assert int(aaa_row["signal"]) == 1
    assert int(bbb_row["signal"]) == -1
    assert aaa_row["strategy_id"] == "sma"
    assert aaa_row["strategy_version"] == "1.0"
    assert 0.5 < float(aaa_row["confidence"]) <= 0.95
    assert 0.5 < float(bbb_row["confidence"]) <= 0.95
    assert "bullish" in aaa_row["explanation"].lower() or "crossover" in aaa_row["explanation"].lower()
    assert "bearish" in bbb_row["explanation"].lower() or "crossover" in bbb_row["explanation"].lower()

    # ensure export not requested so result export_path remains None
    assert "Signals exported" not in result.stdout
