"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath

import typer

from rdagent.app.a_share_monitor.conf import ASHARE_MONITOR_PROP_SETTING, AshareMonitorPropSetting
from rdagent.app.a_share_monitor.loop import launch as launch_a_share_monitor
from rdagent.app.data_science.loop import main as data_science
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.app.qlib_rd_loop.quant import main as fin_quant
from rdagent.app.utils.health_check import health_check
from rdagent.app.utils.info import collect_info
from rdagent.log.mle_summary import grade_summary as grade_summary

app = typer.Typer()


def ui(port=19899, log_dir="", debug: bool = False, data_science: bool = False):
    """
    start web app to show the log traces.
    """
    if data_science:
        with rpath("rdagent.log.ui", "dsapp.py") as app_path:
            cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
            subprocess.run(cmds)
        return
    with rpath("rdagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds)


def server_ui(port=19899):
    """
    start web app to show the log traces in real time
    """
    subprocess.run(["python", "rdagent/log/server/app.py", f"--port={port}"])


def ds_user_interact(port=19900):
    """
    start web app to show the log traces in real time
    """
    commands = ["streamlit", "run", "rdagent/log/ui/ds_user_interact.py", f"--server.port={port}"]
    subprocess.run(commands)


def _make_a_monitor_settings(update: dict[str, object] | None = None) -> AshareMonitorPropSetting:
    settings = ASHARE_MONITOR_PROP_SETTING.model_copy()
    if not update:
        return settings
    filtered = {k: v for k, v in update.items() if v is not None}
    if "symbols" in filtered and isinstance(filtered["symbols"], tuple):
        filtered["symbols"] = list(filtered["symbols"])
    return settings.model_copy(update=filtered)


a_monitor_app = typer.Typer(help="Run the A-share monitoring scenario.")


@a_monitor_app.command("run")
def a_monitor_run(
    symbol: tuple[str, ...] = typer.Option(
        (),
        "--symbol",
        "-s",
        help="Override the watchlist (repeat for multiple symbols).",
    ),
    lookback_days: int | None = typer.Option(None, "--lookback-days", help="Lookback window in days."),
    refresh_minutes: int | None = typer.Option(None, "--refresh-minutes", help="Refresh cadence in minutes."),
    step_n: int | None = typer.Option(None, help="Limit the total number of executed steps."),
    loop_n: int | None = typer.Option(None, help="Limit the number of loops to execute."),
    all_duration: str | None = typer.Option(None, "--duration", help="Overall duration limit, e.g. '30m'."),
    resume: Path | None = typer.Option(None, "--resume", help="Resume from an existing session path."),
    checkout: bool = typer.Option(
        True,
        "--checkout/--no-checkout",
        help="When resuming, reuse the existing session folder by default.",
    ),
) -> None:
    updates: dict[str, object] = {"mode": "live"}
    if symbol:
        updates["symbols"] = list(symbol)
    if lookback_days is not None:
        updates["lookback_days"] = lookback_days
    if refresh_minutes is not None:
        updates["refresh_minutes"] = refresh_minutes

    settings = _make_a_monitor_settings(updates)

    launch_a_share_monitor(
        settings,
        resume_path=resume,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
    )


@a_monitor_app.command("backtest")
def a_monitor_backtest(
    symbol: tuple[str, ...] = typer.Option(
        (),
        "--symbol",
        "-s",
        help="Override the watchlist for the backtest run.",
    ),
    start: str | None = typer.Option(None, "--start", help="Backtest window start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="Backtest window end date YYYY-MM-DD."),
    lookback_days: int | None = typer.Option(None, "--lookback-days", help="Lookback window in days."),
    step_n: int | None = typer.Option(None, help="Limit the total number of executed steps."),
    loop_n: int | None = typer.Option(None, help="Limit the number of loops to execute."),
    all_duration: str | None = typer.Option(None, "--duration", help="Overall duration limit, e.g. '2h'."),
    resume: Path | None = typer.Option(None, "--resume", help="Resume from an existing session path."),
    checkout: bool = typer.Option(
        True,
        "--checkout/--no-checkout",
        help="When resuming, reuse the existing session folder by default.",
    ),
) -> None:
    updates: dict[str, object] = {
        "mode": "backtest",
        "backtest_start": start,
        "backtest_end": end,
    }
    if symbol:
        updates["symbols"] = list(symbol)
    if lookback_days is not None:
        updates["lookback_days"] = lookback_days

    settings = _make_a_monitor_settings(updates)

    launch_a_share_monitor(
        settings,
        resume_path=resume,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
    )


app.command(name="fin_factor")(fin_factor)
app.command(name="fin_model")(fin_model)
app.command(name="fin_quant")(fin_quant)
app.command(name="fin_factor_report")(fin_factor_report)
app.command(name="general_model")(general_model)
app.command(name="data_science")(data_science)
app.command(name="grade_summary")(grade_summary)
app.command(name="ui")(ui)
app.command(name="server_ui")(server_ui)
app.command(name="health_check")(health_check)
app.command(name="collect_info")(collect_info)
app.command(name="ds_user_interact")(ds_user_interact)
app.add_typer(a_monitor_app, name="a-monitor")


if __name__ == "__main__":
    app()
