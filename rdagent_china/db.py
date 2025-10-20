from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Any

import pandas as pd
from loguru import logger

from rdagent_china.config import settings


PRICE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT,
    date TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE
);
"""


@dataclass
class BacktestResult:
    stats: dict
    equity_curve: pd.DataFrame
    html_report: str


class Database:
    def __init__(self, duckdb_path: Path):
        self.duckdb_path = Path(duckdb_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[Any] = None

    @property
    def conn(self):
        if self._conn is None:
            try:
                import duckdb  # type: ignore
            except Exception as e:  # pragma: no cover - optional runtime dep
                raise ImportError(
                    "DuckDB is required but not installed. Please install 'duckdb' and 'duckdb-engine'."
                ) from e
            logger.debug(f"Connecting to DuckDB at {self.duckdb_path}")
            self._conn = duckdb.connect(str(self.duckdb_path))
        return self._conn

    def init(self):
        self.conn.execute(PRICE_TABLE_SQL)

    def write_prices(self, df: pd.DataFrame):
        if df.empty:
            return
        # Ensure schema
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
        # delete duplicates
        min_dt = df["date"].min()
        max_dt = df["date"].max()
        syms = df["symbol"].unique().tolist()
        syms_list = ",".join([f"'{s}'" for s in syms])
        self.conn.execute(
            f"DELETE FROM prices WHERE symbol IN ({syms_list}) AND date BETWEEN ? AND ?",
            [min_dt.to_pydatetime(), max_dt.to_pydatetime()],
        )
        self.conn.register("prices_df", df)
        self.conn.execute("INSERT INTO prices SELECT * FROM prices_df")
        self.conn.unregister("prices_df")

    def read_prices(
        self, symbols: Optional[Iterable[str]] = None, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        clauses = []
        params: list = []
        if symbols:
            placeholders = ",".join(["?"] * len(list(symbols)))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(list(symbols))
        if start:
            clauses.append("date >= ?")
            params.append(pd.to_datetime(start))
        if end:
            clauses.append("date <= ?")
            params.append(pd.to_datetime(end))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM prices {where} ORDER BY symbol, date"
        return self.conn.execute(query, params).fetch_df()


def get_db() -> Database:
    if settings.clickhouse_url:
        logger.warning("ClickHouse configured but not implemented; falling back to DuckDB")
    return Database(settings.duckdb_path)
