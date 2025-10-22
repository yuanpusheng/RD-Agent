from typer.testing import CliRunner

from rdagent_china.cli import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "backtest" in result.stdout
    assert "monitor" in result.stdout
