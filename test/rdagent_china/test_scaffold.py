from __future__ import annotations

from datetime import date, datetime

from rdagent_china.agent import ChinaAgent
from rdagent_china.data import (
    FactorRawRecord,
    FundamentalRecord,
    PriceDailyRecord,
    SignalRecord,
    initialize_duckdb,
    iter_schema_ddls,
)


def _price(symbol: str, day: int, *, open_: float, close: float) -> PriceDailyRecord:
    current_date = datetime(2024, 1, day, 15, 0)
    return PriceDailyRecord(
        symbol=symbol,
        date=current_date,
        open=open_,
        high=open_ + 1,
        low=open_ - 1,
        close=close,
        volume=1_000_000,
        turnover=1_500_000,
    )


def test_initialize_duckdb_creates_expected_tables(tmp_path) -> None:
    db_path = tmp_path / "china.duckdb"
    conn = initialize_duckdb(db_path)
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        tables = {str(row[0]).lower() for row in rows}
    finally:
        conn.close()
    expected = {"price_daily", "fundamental", "factor_raw", "signals"}
    assert expected <= tables

    ddls = list(iter_schema_ddls(schema="analytics"))
    assert all(ddl.startswith("CREATE TABLE IF NOT EXISTS") for ddl in ddls)

    scoped = initialize_duckdb(schema="analytics")
    try:
        scoped_rows = scoped.execute(
            "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = 'analytics'"
        ).fetchall()
        scoped_tables = {str(row[1]).lower() for row in scoped_rows}
    finally:
        scoped.close()
    assert expected <= scoped_tables

    price = PriceDailyRecord(
        symbol="000001.SZ",
        date=datetime(2024, 1, 1, 15, 0),
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.2,
        volume=1_200_000,
    )
    fundamental = FundamentalRecord(
        symbol="000001.SZ",
        report_date=date(2023, 12, 31),
        revenue=1.0e8,
        net_income=2.5e7,
    )
    factor = FactorRawRecord(
        symbol="000001.SZ",
        date=datetime(2024, 1, 1, 15, 0),
        factor="momentum",
        value=1.23,
    )
    signal = SignalRecord(
        universe="demo",
        symbol="000001.SZ",
        as_of_date=date(2024, 1, 1),
        timestamp=datetime(2024, 1, 1, 15, 0),
        signal=1,
        strategy_id="demo",
        strategy_version="v0",
    )

    assert price.table_name == "price_daily"
    assert fundamental.table_name == "fundamental"
    assert factor.table_name == "factor_raw"
    assert signal.table_name == "signals"
    assert price.model_dump()["symbol"] == "000001.SZ"


def test_agent_pipeline_smoke() -> None:
    prices = [
        _price("000001.SZ", 1, open_=10.0, close=10.5),
        _price("000001.SZ", 2, open_=10.6, close=10.2),
        _price("000001.SZ", 3, open_=10.3, close=10.8),
        _price("000002.SZ", 1, open_=20.0, close=20.1),
        _price("000002.SZ", 2, open_=19.8, close=19.5),
    ]

    agent = ChinaAgent()
    context = agent.run(prices)

    assert context["dashboard"] == "rdagent-china-dashboard"
    summary = context["summary"]
    assert summary["total"] == len(prices)
    directional_total = summary.get("buy", 0) + summary.get("sell", 0) + summary.get("hold", 0)
    assert directional_total == len(prices)
    report = context["report"]
    assert len(report.metadata["signals"]) == len(prices)
    first_signal = agent.generate_signals(prices)[0]
    assert isinstance(first_signal, SignalRecord)
