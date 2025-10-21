import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from loguru import logger

from rdagent_china.config import settings
from rdagent_china.data.ingest import ingest_prices
from rdagent_china.data.universe import get_csi300_symbols, get_all_a_stock_symbols
from rdagent_china.data.akshare_client import AkshareClient
from rdagent_china.db import get_db

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
def daily_run():
    # simple daily workflow: ingest latest and backtest recent 3 months
    from datetime import date, timedelta

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=120)).isoformat()
    ingest(universe="CSI300", start=start, end=end)
    backtest(symbols=None, start=start, end=end)


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


if __name__ == "__main__":
    app()
