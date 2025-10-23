from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rdagent.app.cli import app


def test_a_monitor_backtest_cli_triggers_evaluation(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    calls: dict[str, object] = {}

    class DummyDB:
        def init(self) -> None:
            calls["db_init"] = True

    monkeypatch.setattr("rdagent.app.cli.get_db", lambda: DummyDB())

    def fake_launch(settings, **kwargs) -> None:
        calls["launch"] = {
            "symbols": list(settings.symbols),
            "mode": settings.mode,
        }

    monkeypatch.setattr("rdagent.app.cli.launch_a_share_monitor", fake_launch)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation_windows: [2]\n", encoding="utf-8")
    output_dir = tmp_path / "reports"

    class DummyResult:
        def save(self, path: Path) -> None:
            calls["saved_path"] = Path(path)

        def is_empty(self) -> bool:
            return False

    class DummyBacktester:
        def __init__(self, db, config) -> None:
            calls["backtester_symbols"] = config.symbols
            calls["report_dir"] = config.report.output_dir
            self._result = DummyResult()

        def run(self) -> DummyResult:
            calls["run_called"] = True
            return self._result

    monkeypatch.setattr("rdagent.app.cli.SignalBacktester", DummyBacktester)

    result = runner.invoke(
        app,
        [
            "a-monitor",
            "backtest",
            "--symbol",
            "AAA",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert calls["db_init"] is True
    assert "launch" in calls
    assert calls["backtester_symbols"] == ["AAA"]
    assert calls["run_called"] is True
    assert calls["saved_path"] == output_dir
    assert Path(calls["report_dir"]) == output_dir
