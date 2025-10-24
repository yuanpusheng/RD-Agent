"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import html
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath

import typer
from loguru import logger

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


class _NoOpAlertDispatcher:
    """Stub dispatcher used when alerting is disabled."""

    def dispatch(self, _records) -> None:  # pragma: no cover - simple stub
        return None


def _run_monitor_eod(
    *,
    universe: str | None,
    watchlist: list[str] | None,
    lookback_days: int | None,
    once: bool,
    alerts: bool,
    artifacts_dir: Path | None,
    limit: int | None,
) -> None:
    from pathlib import Path

    from rdagent_china.config import settings as monitor_settings
    from rdagent_china.data.universe import resolve_universe
    from rdagent_china.monitor import MonitorLoop, MonitorLoopRunner, MonitorRunContext

    artifact_path: Path | None = None
    log_sink_id: int | None = None
    if artifacts_dir is not None:
        artifact_path = artifacts_dir.expanduser().resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        log_sink_id = logger.add(str(artifact_path / "monitor.log"), enqueue=True, level="INFO")

    try:
        base_universe = (universe or monitor_settings.monitor_default_universe or "CSI300").strip()
        if not base_universe:
            base_universe = monitor_settings.monitor_default_universe
        resolved_watchlist: list[str] | None = None
        if watchlist:
            resolved_watchlist = [item for item in dict.fromkeys(watchlist) if item]
        elif limit is not None:
            resolved_watchlist = resolve_universe(base_universe)[:limit]

        monitor_kwargs: dict[str, object] = {}
        if lookback_days is not None:
            monitor_kwargs["lookback_days"] = lookback_days
        if not alerts:
            monitor_kwargs["alert_dispatcher"] = _NoOpAlertDispatcher()

        monitor_loop = MonitorLoop(**monitor_kwargs)
        context = MonitorRunContext(
            universe=base_universe,
            watchlist=resolved_watchlist,
            intraday=False,
        )

        if once:
            persisted = monitor_loop.run_cycle(context=context)
            if artifact_path is not None:
                _write_eod_artifacts(
                    artifact_path=artifact_path,
                    universe=base_universe,
                    persisted=persisted,
                    db=monitor_loop.db,
                )
                typer.echo(f"Smoke artifacts written to {artifact_path}")
            typer.secho(
                f"EOD monitoring run completed for {base_universe} with {len(persisted)} persisted signal(s).",
                fg=typer.colors.GREEN,
            )
            return

        try:
            runner = MonitorLoopRunner(monitor_loop)
        except RuntimeError as exc:  # pragma: no cover - optional dependency missing
            raise typer.BadParameter(str(exc)) from exc

        typer.echo("Starting monitoring scheduler (press Ctrl+C to exit).")
        try:
            import asyncio

            asyncio.run(
                runner.start(
                    intraday=False,
                    universe=base_universe,
                    watchlist=resolved_watchlist,
                )
            )
        except KeyboardInterrupt:  # pragma: no cover - interactive runtime
            runner.stop()
            typer.echo("Monitoring scheduler stopped.")
    finally:
        if log_sink_id is not None:
            logger.remove(log_sink_id)


def _write_eod_artifacts(*, artifact_path: Path, universe: str, persisted, db) -> None:
    import json

    import pandas as pd

    artifact_path.mkdir(parents=True, exist_ok=True)

    persisted_path = artifact_path / "signals_latest.csv"
    persisted.to_csv(persisted_path, index=False)

    history_path: Path | None = None
    all_signals = db.read_signals(universe=universe)
    if not all_signals.empty:
        history_path = artifact_path / "signals_history.csv"
        all_signals.to_csv(history_path, index=False)

    preview_source = all_signals if not all_signals.empty else persisted
    preview_path = _render_dashboard_preview(preview_source, artifact_path)

    summary = {
        "universe": universe,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "persisted_count": int(len(persisted)),
        "artifact_dir": str(artifact_path),
        "artifacts": {
            "signals_latest": persisted_path.name,
            "signals_history": history_path.name if history_path else None,
            "dashboard_preview": preview_path.name if preview_path else None,
            "log": "monitor.log",
        },
    }
    summary_path = artifact_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _render_dashboard_preview(signals, artifact_path: Path) -> Path:
    if signals is None or getattr(signals, "empty", True):
        empty_path = artifact_path / "dashboard_preview.txt"
        empty_path.write_text(
            "No monitoring alerts were generated during this run.\n",
            encoding="utf-8",
        )
        return empty_path

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd
    except Exception:
        svg_path = artifact_path / "dashboard_preview.svg"
        svg_path.write_text(_build_svg_preview(signals), encoding="utf-8")
        return svg_path

    import pandas as pd

    data = signals.copy()
    data["timestamp"] = pd.to_datetime(data.get("timestamp"), errors="coerce")

    counts = (
        data.groupby(
            data.get("symbol", pd.Series(index=data.index, dtype=object)).astype(str)
        )["rule"]
        .count()
        .sort_values()
        .tail(6)
    )
    if counts.empty:
        counts = (
            data.groupby(
                data.get("severity", pd.Series(index=data.index, dtype=object)).fillna("unknown").astype(str)
            )["rule"]
            .count()
            .sort_values()
            .tail(6)
        )

    preview_path = artifact_path / "dashboard_preview.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = counts.index.tolist()
    values = [int(v) for v in counts.values]
    if not categories:
        fallback_series = data.get("symbol", pd.Series(index=data.index, dtype=object))
        categories = fallback_series.astype(str).head(6).tolist()
        values = [1 for _ in categories]
    ax.barh(categories, values, color="#4E79A7")
    ax.set_xlabel("Alert count")
    ax.set_ylabel("Symbol" if "symbol" in data.columns else "Category")
    latest_ts = data["timestamp"].dropna().max()
    if pd.notna(latest_ts):
        ax.set_title(f"Latest alerts · {latest_ts:%Y-%m-%d %H:%M UTC}")
    else:
        ax.set_title("Latest alerts")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(preview_path, dpi=160)
    plt.close(fig)
    return preview_path


def _build_svg_preview(signals) -> str:
    import pandas as pd

    columns = ["timestamp", "symbol", "rule", "severity", "value"]
    frame = signals.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[columns].head(6).copy()
    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
    )

    rows: list[str] = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)
        rows.append(f"<tr>{cells}</tr>")
    if not rows:
        rows.append("<tr><td colspan=\"5\">No signals generated.</td></tr>")

    header = "".join(f"<th>{html.escape(col.title())}</th>" for col in columns)
    table_html = (
        "<table>"
        "<thead><tr>" + header + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )
    height = 180 + 32 * len(rows)
    svg = (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"{height}\">"
        "<style>"
        "  table {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; border-collapse: collapse; width: 700px; }}"
        "  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; background-color: #ffffff; }}"
        "  th {{ background-color: #f6f8fa; }}"
        "  h3 {{ font-family: 'Segoe UI', Arial, sans-serif; margin-bottom: 8px; }}"
        "</style>"
        "<rect width=\"100%\" height=\"100%\" fill=\"#f8f9fb\" />"
        f"<foreignObject x=\"10\" y=\"10\" width=\"700\" height=\"{height - 20}\">"
        "  <body xmlns=\"http://www.w3.org/1999/xhtml\">"
        "    <h3>RD-Agent Monitor Preview</h3>"
        f"    {table_html}"
        "  </body>"
        "</foreignObject>"
        "</svg>"
    )
    return svg


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
    mode: str = typer.Option(
        "live",
        "--mode",
        "-m",
        help="Run mode: 'live', 'backtest', or 'eod'.",
    ),
    universe: str | None = typer.Option(
        None,
        "--universe",
        help="Universe identifier or label for monitoring runs (used in eod/backtest modes).",
    ),
    once: bool = typer.Option(
        True,
        "--once/--loop",
        help="Execute a single monitoring cycle when running in eod mode.",
    ),
    alerts: bool = typer.Option(
        True,
        "--alerts/--no-alerts",
        help="Enable alert dispatch when running in eod mode.",
    ),
    artifacts_dir: Path | None = typer.Option(
        None,
        "--artifacts-dir",
        help="Optional directory where eod mode stores smoke-test artifacts.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Limit the number of symbols to evaluate during eod runs.",
    ),
) -> None:
    mode_normalized = mode.lower()
    if mode_normalized not in {"live", "backtest", "eod"}:
        raise typer.BadParameter("mode must be one of 'live', 'backtest', or 'eod'.")

    if mode_normalized == "eod":
        if limit is not None and limit < 1:
            raise typer.BadParameter("limit must be positive when provided.")
        watchlist = list(dict.fromkeys(symbol)) if symbol else None
        _run_monitor_eod(
            universe=universe,
            watchlist=watchlist,
            lookback_days=lookback_days,
            once=once,
            alerts=alerts,
            artifacts_dir=artifacts_dir,
            limit=limit,
        )
        return

    updates: dict[str, object] = {"mode": mode_normalized}
    if symbol:
        updates["symbols"] = list(symbol)
    elif universe:
        try:
            from rdagent_china.data.universe import resolve_universe

            updates["symbols"] = resolve_universe(universe)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise typer.BadParameter(f"Failed to resolve universe '{universe}': {exc}") from exc
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
    config: Path | None = typer.Option(None, "--config", help="Path to the evaluation YAML/JSON configuration."),
    output_dir: Path = typer.Option(
        Path("rdagent_china/reports/a_monitor_backtest"),
        "--output-dir",
        help="Directory where evaluation reports will be written.",
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

    db = get_db()
    db.init()

    launch_a_share_monitor(
        settings,
        resume_path=resume,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
    )

    from rdagent_china.backtest.config import SignalBacktestConfig
    from rdagent_china.backtest.signals import SignalBacktester

    if config is not None:
        eval_config = SignalBacktestConfig.from_file(config)
    else:
        eval_config = SignalBacktestConfig()

    override_kwargs: dict[str, object] = {}
    base_symbols = list(settings.symbols) if getattr(settings, "symbols", None) else []
    selected_symbols = list(symbol) if symbol else base_symbols
    if selected_symbols:
        override_kwargs["symbols"] = selected_symbols
    if start is not None:
        override_kwargs["start"] = start
    if end is not None:
        override_kwargs["end"] = end
    if override_kwargs:
        eval_config = eval_config.with_overrides(**override_kwargs)

    report_output = eval_config.report.output_dir or output_dir
    report_path = Path(report_output)
    if eval_config.report.output_dir != report_path:
        report_cfg = eval_config.report.model_copy(update={"output_dir": report_path})
        eval_config = eval_config.model_copy(update={"report": report_cfg})

    backtester = SignalBacktester(db=db, config=eval_config)
    result = backtester.run()
    result.save(report_path)
    final_report = report_path / "report.html"
    if result.is_empty():
        logger.warning("Signal evaluation produced no trades; report saved to %s", final_report.resolve())
    else:
        logger.info("Signal evaluation report saved to %s", final_report.resolve())


@a_monitor_app.command("ui")
def a_monitor_ui(
    port: int = typer.Option(19560, help="Streamlit server port for the dashboard."),
    log_dir: Path | None = typer.Option(None, "--log-dir", help="Optional RD-Agent log directory."),
    session: str | None = typer.Option(None, "--session", help="Preselect a session folder inside the log directory."),
    universe: str = typer.Option("CSI300", "--universe", help="Default universe to display on launch."),
) -> None:
    """Launch the A-share monitoring Streamlit dashboard."""

    with rpath("rdagent_china.dashboard.a_share_monitor", "app.py") as app_path:
        commands = ["streamlit", "run", str(app_path), f"--server.port={port}"]
        extra_args: list[str] = []
        if log_dir:
            extra_args.append(f"--log-dir={str(log_dir)}")
        if session:
            extra_args.append(f"--session={session}")
        if universe:
            extra_args.append(f"--universe={universe}")
        if extra_args:
            commands.append("--")
            commands.extend(extra_args)
        try:
            subprocess.run(commands)
        except FileNotFoundError as exc:
            typer.secho("Streamlit command not found. Install Streamlit or launch directly via:", fg=typer.colors.RED)
            typer.secho(
                f"  streamlit run {app_path}",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1) from exc


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
