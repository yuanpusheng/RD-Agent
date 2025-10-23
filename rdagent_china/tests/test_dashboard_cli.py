from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from rdagent.app.cli import app


def test_a_monitor_ui_command_invokes_streamlit(tmp_path: Path):
    runner = CliRunner()
    with patch("rdagent.app.cli.subprocess.run") as mock_run:
        result = runner.invoke(
            app,
            [
                "a-monitor",
                "ui",
                "--port",
                "12345",
                "--log-dir",
                str(tmp_path),
                "--session",
                "run-1",
                "--universe",
                "ZZ500",
            ],
        )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[:2] == ["streamlit", "run"]
    assert cmd[2].endswith("rdagent_china/dashboard/a_share_monitor/app.py")
    assert f"--server.port=12345" in cmd
    assert "--" in cmd
    assert f"--log-dir={tmp_path}" in cmd
    assert "--session=run-1" in cmd
    assert "--universe=ZZ500" in cmd
