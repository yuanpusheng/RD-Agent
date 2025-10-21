from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


# ------------------ DuckDB helpers ------------------

@dataclass
class DuckDBConfig:
    path: Union[str, Path] = ":memory:"
    read_only: bool = False


class DuckDBIO:
    @staticmethod
    def connect(config: Optional[DuckDBConfig] = None):
        try:
            import duckdb  # type: ignore
        except Exception as e:  # pragma: no cover - optional runtime dep
            raise ImportError("duckdb is required for DuckDBIO") from e
        cfg = config or DuckDBConfig()
        return duckdb.connect(str(cfg.path), read_only=cfg.read_only)

    @staticmethod
    def write_df(conn: Any, df: pd.DataFrame, table: str, mode: str = "overwrite"):
        if df.empty:
            return
        # register and insert
        conn.register("_tmp_df", df)
        if mode == "overwrite":
            conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _tmp_df")
        elif mode == "append":
            # create table if not exists with schema from df
            cols = ", ".join([f"{c} {DuckDBIO._duckdb_type(df[c])}" for c in df.columns])
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
            conn.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
        else:
            raise ValueError("mode must be 'overwrite' or 'append'")
        conn.unregister("_tmp_df")

    @staticmethod
    def read_df(conn: Any, source: str) -> pd.DataFrame:
        if any(ch.isspace() for ch in source):
            return conn.execute(source).fetch_df()
        return conn.execute(f"SELECT * FROM {source}").fetch_df()

    @staticmethod
    def _duckdb_type(s: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(s):
            return "BIGINT"
        if pd.api.types.is_float_dtype(s):
            return "DOUBLE"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "TIMESTAMP"
        if pd.api.types.is_bool_dtype(s):
            return "BOOLEAN"
        return "TEXT"


# ------------------ ClickHouse helpers ------------------

@dataclass
class ClickHouseConfig:
    url: str
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


class ClickHouseIO:
    def __init__(self, config: ClickHouseConfig):
        self.config = config
        try:
            import clickhouse_connect  # type: ignore
        except Exception as e:  # pragma: no cover - optional runtime dep
            raise ImportError("clickhouse-connect is required for ClickHouseIO") from e
        self._client = clickhouse_connect.get_client(
            host=self._host_from_url(config.url),
            username=config.user,
            password=config.password,
            database=config.database,
        )

    def query_df(self, sql: str) -> pd.DataFrame:
        result = self._client.query(sql)
        return result.result_df

    def insert_df(self, table: str, df: pd.DataFrame):
        # clickhouse-connect supports insert_df
        self._client.insert_df(table, df)

    @staticmethod
    def _host_from_url(url: str) -> str:
        # naive parse e.g., http://host:8123
        return url.split("//")[-1].split("/")[0]
