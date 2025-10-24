import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rdagent_china.cli import app
from rdagent_china.data.eda import MPF_AVAILABLE, generate_symbol_eda


runner = CliRunner()


def _sample_price_frame(symbol: str, periods: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    for idx, dt in enumerate(dates):
        base = 10 + idx
        rows.append(
            {
                "symbol": symbol,
                "date": dt,
                "open": base,
                "high": base + 0.8,
                "low": base - 0.6,
                "close": base + 0.4,
                "volume": 1000 + idx * 100,
            }
        )
    return pd.DataFrame(rows)


def test_generate_symbol_eda_artifacts(tmp_path: Path):
    df = _sample_price_frame("000001.SZ", periods=12)

    result = generate_symbol_eda("000001.SZ", df, tmp_path)

    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text())
    assert summary["rows"] == 12
    assert summary["missing_counts"]["volume"] == 0
    assert summary["indicator_status"]["sma_5"] == "generated"

    if MPF_AVAILABLE:
        assert "kline" in result.plots
        assert result.plots["kline"].exists()
    assert "volume" in result.plots
    assert result.plots["volume"].exists()
    assert "indicators" in result.plots
    assert result.plots["indicators"].exists()


def test_generate_symbol_eda_handles_missing_data(tmp_path: Path):
    empty_df = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

    result = generate_symbol_eda("000002.SZ", empty_df, tmp_path)

    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text())
    assert summary["rows"] == 0
    assert summary["plots"] == {}
    assert any("No data available" in note for note in summary["notes"])


def test_cli_eda_generates_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class DummyDB:
        def __init__(self) -> None:
            self.initialised = False

        def init(self) -> None:
            self.initialised = True

    class DummyProvider:
        def __init__(self, _db: DummyDB) -> None:
            self.db = _db

        def get_price_daily(self, symbols, start=None, end=None):
            frames = [_sample_price_frame(symbol, periods=5) for symbol in symbols]
            return pd.concat(frames, ignore_index=True)

    monkeypatch.setattr("rdagent_china.cli.get_db", lambda: DummyDB())
    monkeypatch.setattr("rdagent_china.cli.get_csi300_symbols", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr("rdagent_china.cli.UnifiedDataProvider", lambda db: DummyProvider(db))

    output_dir = tmp_path / "eda_output"
    result = runner.invoke(
        app,
        [
            "eda",
            "--symbol",
            "AAA",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    symbol_dir = output_dir / "AAA"
    assert symbol_dir.exists()
    summary_path = symbol_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["rows"] == 5
    assert "volume" in summary["plots"]
