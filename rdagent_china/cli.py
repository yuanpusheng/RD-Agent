import asyncio
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer
from loguru import logger

from rdagent_china.config import settings
from rdagent_china.data.ingest import ingest_prices
from rdagent_china.data.universe import get_csi300_symbols, get_all_a_stock_symbols
from rdagent_china.data.akshare_client import AkshareClient
from rdagent_china.data.provider import UnifiedDataProvider
from rdagent_china.data.eda import run_eda
from rdagent_china.db import get_db
from rdagent_china.monitor import MonitorLoop, MonitorLoopRunner, MonitorRunContext
from rdagent_china import daily_run as daily_pipeline

app = typer.Typer(help="RD-Agent China CLI (rdc): ingest/backtest/daily-run/report/dashboard")


@app.command()
def ingest(
    universe: str = typer.Option("CSI300", help="Universe: CSI300 or ALL"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    limit: int = typer.Option(100, help="Max number of symbols to ingest (for ALL)"),
):
    db = get_db()
    db.init()

    if universe.upper() == "CSI300":
        symbols = get_csi300_symbols()
    elif universe.upper() == "ALL":
        symbols = get_all_a_stock_symbols(limit=limit)
    else:
        raise typer.BadParameter("universe must be CSI300 or ALL")

    logger.info(f"Ingesting {len(symbols)} symbols into {settings.db_url}")
    ingest_prices(symbols=symbols, start=start, end=end, db=db)
    logger.info("Ingest finished")


@app.command()
def ingest_price_daily(
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
):
    """Fetch CSI300 daily prices via Akshare and persist to DuckDB price_daily with upsert semantics."""
    db = get_db()
    db.init()

    symbols = get_csi300_symbols()
    logger.info(f"Ingesting daily prices for {len(symbols)} CSI300 symbols into {settings.db_url}")
    client = AkshareClient()
    frames = []
    for sym in symbols:
        try:
            df = client.price_daily(sym, start=start, end=end)
        except Exception as e:  # pragma: no cover - defensive fallback
            logger.warning(f"price_daily fetch failed for {sym}: {e}")
            df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])  \
                .assign(symbol=sym)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not full.empty:
        full["date"] = pd.to_datetime(full["date"])  # ensure datetime
        db.write_price_daily(full)
    logger.info("Ingest price_daily finished")


@app.command()
def sync_price_daily(
    start: Optional[str] = typer.Option(None, help="Override start date YYYY-MM-DD; defaults to last synced + 1"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
):
    """Incrementally sync CSI300 daily prices using the unified provider with source fallbacks.
    Determines per-symbol start date from sync metadata when not overridden.
    """
    db = get_db()
    db.init()
    provider = UnifiedDataProvider(db=db)

    symbols = get_csi300_symbols()
    if not symbols:
        logger.info("No symbols to sync")
        return

    # Figure out start dates per symbol
    groups: Dict[str, List[str]] = defaultdict(list)
    for sym in symbols:
        s = start
        if s is None:
            last_dt = provider.latest_synced_date(sym, dataset="price_daily")
            if last_dt is not None:
                s = (pd.to_datetime(last_dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        groups[s or ""]  # ensure key exists
        groups[s or ""].append(sym)

    # Fetch by groups and persist via provider (which also upserts sync_meta)
    for s_key, syms in groups.items():
        s_val = s_key or None
        logger.info(f"Syncing {len(syms)} symbols from {s_val or 'beginning'}")
        _ = provider.get_price_daily(syms, start=s_val, end=end)
    logger.info("Sync price_daily finished")


@app.command()
def eda(
    symbols: Optional[List[str]] = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Symbol(s) to analyse. Defaults to sampling from the CSI300 universe.",
    ),
    sample_size: int = typer.Option(
        3,
        "--sample-size",
        min=1,
        help="Number of CSI300 symbols to analyse when --symbol is not provided.",
    ),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD."),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD."),
    output_dir: Path = typer.Option(
        Path("rdagent_china/reports/eda"), help="Directory where EDA plots and summaries will be written."
    ),
):
    """Generate candlestick, volume, and indicator plots for manual QA."""
    db = get_db()
    db.init()
    provider = UnifiedDataProvider(db=db)

    if symbols:
        target_symbols = list(dict.fromkeys(symbols))
    else:
        universe = get_csi300_symbols()
        if not universe:
            typer.secho("No symbols available to sample from CSI300 universe.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if sample_size < len(universe):
            target_symbols = universe[:sample_size]
        else:
            target_symbols = universe

    typer.echo(f"Generating EDA artifacts for {len(target_symbols)} symbol(s): {', '.join(target_symbols)}")
    results = run_eda(
        symbols=target_symbols,
        output_dir=output_dir,
        provider=provider,
        start=start,
        end=end,
    )

    for sym, res in results.items():
        plot_paths = [str(path) for path in res.plots.values()]
        plot_msg = ", ".join(plot_paths) if plot_paths else "none"
        typer.echo(f"{sym}: summary -> {res.summary_path} | plots -> {plot_msg}")
        if res.notes:
            for note in res.notes:
                typer.echo(f"  note: {note}")


@app.command()
def backtest(
    symbols: Optional[List[str]] = typer.Option(None, help="Symbols to backtest, default to CSI300 top 10"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    strategy: str = typer.Option("sma", help="Strategy id: sma"),
    report_path: Path = typer.Option(Path("rdagent_china/reports/report.html"), help="Output report path"),
):
    db = get_db()
    db.init()

    if not symbols:
        symbols = get_csi300_symbols()[:10]

    # Lazy import to avoid heavy deps at CLI import time
    from rdagent_china.backtest.runner import run_backtest

    result = run_backtest(db=db, symbols=symbols, start=start, end=end, strategy=strategy)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result.html_report)
    logger.info(f"Backtest finished. Report saved to {report_path}")


@app.command(name="daily-run")
def daily_run(
    universe: str = typer.Option(
        daily_pipeline.DEFAULT_UNIVERSE, help="Universe identifier or comma-separated symbol list."
    ),
    start_date: Optional[str] = typer.Option(None, help="Override start date (YYYY-MM-DD)."),
    end_date: Optional[str] = typer.Option(None, help="Override end date (YYYY-MM-DD)."),
    lookback_days: int = typer.Option(
        daily_pipeline.DEFAULT_LOOKBACK_DAYS, help="Lookback window when start date not provided."
    ),
    limit: Optional[int] = typer.Option(None, help="Limit number of symbols to process."),
    strategy: Optional[str] = typer.Option(None, help="Force a specific strategy id (e.g. 'sma')."),
    export_path: Optional[Path] = typer.Option(None, help="Optional file path to export the signals."),
    export_format: Optional[str] = typer.Option(
        None, help="Export format (csv or parquet). Defaults to suffix derived from export_path or csv."
    ),
):
    try:
        result = daily_pipeline.run_pipeline(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            limit=limit,
            strategy=strategy,
            export_path=export_path,
            export_format=export_format,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        typer.secho(f"Daily signal pipeline failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    daily_pipeline.display_summary(result)


@app.command()
def report(report_path: Path = typer.Option(Path("rdagent_china/reports/report.html"))):
    if report_path.exists():
        typer.echo(str(report_path.resolve()))
    else:
        typer.echo("Report not found. Run rdc backtest first.")


@app.command()
def dashboard(port: int = 19555):
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path), f"--server.port={port}"])


@app.command()
def monitor(
    intraday: bool = typer.Option(
        False, "--intraday", help="Enable intraday polling alongside the daily run"
    ),
    universe: Optional[str] = typer.Option(
        None, "--universe", help="Universe identifier or comma-separated symbol list"
    ),
    watchlist: Optional[str] = typer.Option(
        None, "--watchlist", help="Watchlist file path or comma-separated symbols"
    ),
    run_once: bool = typer.Option(
        False, "--run-once", help="Execute a single monitoring cycle and exit"
    ),
    start: Optional[str] = typer.Option(None, help="Override data start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="Override data end date YYYY-MM-DD"),
):
    """Launch the monitoring loop, optionally with scheduling."""

    monitor_loop = MonitorLoop()
    context = MonitorRunContext(
        universe=universe,
        watchlist=watchlist,
        intraday=intraday,
        start=start,
        end=end,
    )

    if run_once:
        persisted = monitor_loop.run_cycle(context=context)
        if not persisted.empty:
            logger.info("Persisted {} alerts during one-off run", len(persisted))
        return

    try:
        runner = MonitorLoopRunner(monitor_loop)
    except RuntimeError as exc:  # pragma: no cover - APScheduler optional installation
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info("Starting monitoring scheduler; press Ctrl+C to exit")
    try:
        asyncio.run(
            runner.start(
                intraday=intraday,
                universe=universe,
                watchlist=watchlist,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive runtime guard
        logger.info("Monitoring scheduler interrupted")


if __name__ == "__main__":
    app()
