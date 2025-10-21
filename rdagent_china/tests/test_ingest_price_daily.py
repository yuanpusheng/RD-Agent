from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from rdagent_china.cli import app
from rdagent_china.config import settings


runner = CliRunner()


def _make_fake_df(rows):
    return pd.DataFrame(rows)


def test_ingest_price_daily_duckdb(monkeypatch, tmp_path):
    # Route DuckDB to a temp file
    db_path = tmp_path / "market.duckdb"
    monkeypatch.setattr(settings, "duckdb_path", db_path, raising=False)

    # Universe: two symbols
    monkeypatch.setattr(
        "rdagent_china.data.universe.get_csi300_symbols",
        lambda: ["000001", "000002"],
    )

    # Fake Akshare responses: Chinese columns like real Akshare
    def fake_fetch_price_daily_raw(self, symbol: str, adjust: str = "qfq"):
        if symbol == "000001":
            return _make_fake_df(
                [
                    {"日期": "2024-01-02", "开盘": 10.0, "最高": 11.0, "最低": 9.5, "收盘": 10.5, "成交量": 12345},
                    {"日期": "2024-01-03", "开盘": 10.5, "最高": 12.0, "最低": 10.0, "收盘": 11.0, "成交量": 23456},
                ]
            )
        elif symbol == "000002":
            return _make_fake_df(
                [
                    {"日期": "2024-01-02", "开盘": 20.0, "最高": 21.0, "最低": 19.5, "收盘": 20.5, "成交量": 54321},
                    {"日期": "2024-01-03", "开盘": 20.5, "最高": 22.0, "最低": 20.0, "收盘": 21.0, "成交量": 65432},
                ]
            )
        raise AssertionError(f"unexpected symbol {symbol}")

    monkeypatch.setattr(
        "rdagent_china.data.akshare_client.AkshareClient._fetch_price_daily_raw",
        fake_fetch_price_daily_raw,
        raising=True,
    )

    # First ingest
    res = runner.invoke(app, ["ingest-price-daily", "--start", "2024-01-01", "--end", "2024-01-31"])
    assert res.exit_code == 0, res.stdout

    # Validate persisted data
    import duckdb

    conn = duckdb.connect(str(db_path))
    # Table exists with expected columns
    cols = conn.execute("PRAGMA table_info('price_daily')").fetch_df()["name"].tolist()
    assert cols == ["symbol", "date", "open", "high", "low", "close", "volume"]

    df = conn.execute(
        "SELECT symbol, date, open, high, low, close, volume FROM price_daily ORDER BY symbol, date"
    ).fetch_df()

    # 4 rows: 2 symbols * 2 days
    assert len(df) == 4

    # Check normalization: symbols and dates exist and are correct
    assert df.iloc[0]["symbol"] == "000001"
    assert pd.to_datetime(df.iloc[0]["date"]).strftime("%Y-%m-%d") == "2024-01-02"
    assert df.iloc[0]["open"] == 10.0
    assert df.iloc[1]["close"] == 11.0

    # Idempotent upsert: run ingest again should not duplicate rows
    res2 = runner.invoke(app, ["ingest-price-daily", "--start", "2024-01-01", "--end", "2024-01-31"])
    assert res2.exit_code == 0, res2.stdout
    df2 = conn.execute("SELECT COUNT(*) AS n FROM price_daily").fetch_df()
    assert int(df2.loc[0, "n"]) == 4

    conn.close()
