from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Any, Sequence

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

PRICE_DAILY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS price_daily (
    symbol TEXT,
    date TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE
);
"""

BARS_DAILY_RAW_SQL = """
CREATE TABLE IF NOT EXISTS bars_daily_raw (
    symbol TEXT,
    date TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    source TEXT
);
"""

BARS_MINUTE_SQL = """
CREATE TABLE IF NOT EXISTS bars_minute (
    symbol TEXT,
    dt TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE
);
"""

ADJ_FACTORS_SQL = """
CREATE TABLE IF NOT EXISTS adj_factors (
    symbol TEXT,
    date TIMESTAMP,
    adj_factor DOUBLE
);
"""

SYNC_META_SQL = """
CREATE TABLE IF NOT EXISTS sync_meta (
    dataset TEXT,
    symbol TEXT,
    last_date TIMESTAMP,
    source TEXT,
    updated_at TIMESTAMP
);
"""

COMPUTED_SIGNALS_SQL = """
CREATE TABLE IF NOT EXISTS computed_signals (
    universe TEXT,
    symbol TEXT,
    timestamp TIMESTAMP,
    as_of_date DATE,
    rule TEXT,
    label TEXT,
    severity TEXT,
    triggered BOOLEAN,
    value TEXT,
    signals TEXT,
    config_version TEXT,
    run_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

MONITOR_ALERT_STATE_SQL = """
CREATE TABLE IF NOT EXISTS monitor_alert_state (
    universe TEXT,
    symbol TEXT,
    rule TEXT,
    last_triggered TIMESTAMP,
    last_value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        # Initialize legacy and extended tables
        self.conn.execute(PRICE_TABLE_SQL)
        self.conn.execute(PRICE_DAILY_TABLE_SQL)
        self.conn.execute(BARS_DAILY_RAW_SQL)
        self.conn.execute(BARS_MINUTE_SQL)
        self.conn.execute(ADJ_FACTORS_SQL)
        self.conn.execute(SYNC_META_SQL)
        self.conn.execute(COMPUTED_SIGNALS_SQL)
        self.conn.execute(MONITOR_ALERT_STATE_SQL)

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

    def write_price_daily(self, df: pd.DataFrame):
        if df.empty:
            return
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
        min_dt = df["date"].min()
        max_dt = df["date"].max()
        syms = df["symbol"].unique().tolist()
        syms_list = ",".join([f"'{s}'" for s in syms])
        self.conn.execute(
            f"DELETE FROM price_daily WHERE symbol IN ({syms_list}) AND date BETWEEN ? AND ?",
            [min_dt.to_pydatetime(), max_dt.to_pydatetime()],
        )
        self.conn.register("price_daily_df", df)
        self.conn.execute("INSERT INTO price_daily SELECT * FROM price_daily_df")
        self.conn.unregister("price_daily_df")

    def write_bars_daily_raw(self, df: pd.DataFrame):
        if df.empty:
            return
        cols = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
        min_dt = df["date"].min()
        max_dt = df["date"].max()
        syms = df["symbol"].unique().tolist()
        syms_list = ",".join([f"'{s}'" for s in syms])
        self.conn.execute(
            f"DELETE FROM bars_daily_raw WHERE symbol IN ({syms_list}) AND date BETWEEN ? AND ?",
            [min_dt.to_pydatetime(), max_dt.to_pydatetime()],
        )
        self.conn.register("bars_daily_raw_df", df)
        self.conn.execute("INSERT INTO bars_daily_raw SELECT * FROM bars_daily_raw_df")
        self.conn.unregister("bars_daily_raw_df")

    def write_bars_minute(self, df: pd.DataFrame):
        if df.empty:
            return
        cols = ["symbol", "dt", "open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        df["dt"] = pd.to_datetime(df["dt"])  # ensure datetime
        min_dt = df["dt"].min()
        max_dt = df["dt"].max()
        syms = df["symbol"].unique().tolist()
        syms_list = ",".join([f"'{s}'" for s in syms])
        self.conn.execute(
            f"DELETE FROM bars_minute WHERE symbol IN ({syms_list}) AND dt BETWEEN ? AND ?",
            [min_dt.to_pydatetime(), max_dt.to_pydatetime()],
        )
        self.conn.register("bars_minute_df", df)
        self.conn.execute("INSERT INTO bars_minute SELECT * FROM bars_minute_df")
        self.conn.unregister("bars_minute_df")

    def write_adj_factors(self, df: pd.DataFrame):
        if df.empty:
            return
        cols = ["symbol", "date", "adj_factor"]
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
        min_dt = df["date"].min()
        max_dt = df["date"].max()
        syms = df["symbol"].unique().tolist()
        syms_list = ",".join([f"'{s}'" for s in syms])
        self.conn.execute(
            f"DELETE FROM adj_factors WHERE symbol IN ({syms_list}) AND date BETWEEN ? AND ?",
            [min_dt.to_pydatetime(), max_dt.to_pydatetime()],
        )
        self.conn.register("adj_factors_df", df)
        self.conn.execute("INSERT INTO adj_factors SELECT * FROM adj_factors_df")
        self.conn.unregister("adj_factors_df")

    def write_signals(self, df: pd.DataFrame):
        if df.empty:
            return
        required = [
            "universe",
            "symbol",
            "timestamp",
            "as_of_date",
            "rule",
            "label",
            "severity",
            "triggered",
            "value",
            "signals",
            "config_version",
            "run_version",
        ]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for signals persistence: {sorted(missing)}")

        data = df[required].copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["as_of_date"] = pd.to_datetime(data["as_of_date"]).dt.normalize()
        data["triggered"] = data["triggered"].astype(bool)

        def _serialize_payload(payload: Any) -> str:
            if isinstance(payload, str):
                return payload
            return json.dumps(payload, default=str)

        def _serialize_value(value: Any):
            if pd.isna(value):
                return None
            if isinstance(value, str):
                return value
            return json.dumps(value, default=str)

        data["signals"] = data["signals"].apply(_serialize_payload)
        data["value"] = data["value"].apply(_serialize_value)
        data["run_version"] = data["run_version"].apply(lambda value: None if pd.isna(value) else str(value))

        key_frame = data[["symbol", "timestamp", "rule", "run_version"]].drop_duplicates()
        for _, row in key_frame.iterrows():
            symbol = row["symbol"]
            ts = pd.to_datetime(row["timestamp"]).to_pydatetime()
            rule = row["rule"]
            run_version = row["run_version"]
            if pd.isna(run_version):
                self.conn.execute(
                    "DELETE FROM computed_signals WHERE symbol = ? AND timestamp = ? AND rule = ? AND run_version IS NULL",
                    [symbol, ts, rule],
                )
            else:
                self.conn.execute(
                    "DELETE FROM computed_signals WHERE symbol = ? AND timestamp = ? AND rule = ? AND run_version = ?",
                    [symbol, ts, rule, str(run_version)],
                )

        self.conn.register("computed_signals_df", data)
        self.conn.execute(
            """
            INSERT INTO computed_signals
            SELECT universe, symbol, timestamp, as_of_date, rule, label, severity, triggered,
                   value, signals, config_version, run_version
            FROM computed_signals_df
            """
        )
        self.conn.unregister("computed_signals_df")

    def write_monitor_state(self, df: pd.DataFrame):
        if df.empty:
            return
        cols = ["universe", "symbol", "rule", "last_triggered", "last_value"]
        data = df[cols].copy()
        data["last_triggered"] = pd.to_datetime(data["last_triggered"])

        def _serialize_state(value: Any):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, str):
                return value
            return json.dumps(value, default=str)

        data["last_value"] = data["last_value"].apply(_serialize_state)

        key_frame = data[["universe", "symbol", "rule"]].drop_duplicates()
        for _, row in key_frame.iterrows():
            universe = row["universe"]
            symbol = row["symbol"]
            rule = row["rule"]
            if pd.isna(universe):
                self.conn.execute(
                    "DELETE FROM monitor_alert_state WHERE universe IS NULL AND symbol = ? AND rule = ?",
                    [symbol, rule],
                )
            else:
                self.conn.execute(
                    "DELETE FROM monitor_alert_state WHERE universe = ? AND symbol = ? AND rule = ?",
                    [universe, symbol, rule],
                )

        self.conn.register("monitor_alert_state_df", data)
        self.conn.execute(
            """
            INSERT INTO monitor_alert_state (universe, symbol, rule, last_triggered, last_value)
            SELECT universe, symbol, rule, last_triggered, last_value FROM monitor_alert_state_df
            """
        )
        self.conn.unregister("monitor_alert_state_df")

    def read_monitor_state(
        self,
        universe: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        rules: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if universe is not None:
            if universe == "":
                clauses.append("universe IS NULL")
            else:
                clauses.append("universe = ?")
                params.append(universe)
        if symbols:
            placeholders = ",".join(["?"] * len(list(symbols)))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(list(symbols))
        if rules:
            placeholders = ",".join(["?"] * len(list(rules)))
            clauses.append(f"rule IN ({placeholders})")
            params.extend(list(rules))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT universe, symbol, rule, last_triggered, last_value, updated_at FROM monitor_alert_state {where} ORDER BY symbol, rule"
        result = self.conn.execute(query, params).fetch_df()
        if result.empty:
            return result
        result["last_triggered"] = pd.to_datetime(result["last_triggered"])
        result["updated_at"] = pd.to_datetime(result["updated_at"])

        def _deserialize_state(value: Any):
            if value is None or value == "":
                return None
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        result["last_value"] = result["last_value"].apply(_deserialize_state)
        return result

    def upsert_sync_meta(self, df: pd.DataFrame):
        if df.empty:
            return
        # Normalize and ensure required columns
        cols = ["dataset", "symbol", "last_date", "source", "updated_at"]
        df = df[cols].copy()
        df["last_date"] = pd.to_datetime(df["last_date"])  # ensure datetime
        df["updated_at"] = pd.to_datetime(df["updated_at"])  # ensure datetime
        # delete existing rows for keys (dataset,symbol)
        keys = df[["dataset", "symbol"]].drop_duplicates()
        # Build delete with IN list for tuples
        params: list = []
        clauses: list[str] = []
        for _, r in keys.iterrows():
            clauses.append("(dataset = ? AND symbol = ?)")
            params.extend([r["dataset"], r["symbol"]])
        if clauses:
            sql = f"DELETE FROM sync_meta WHERE {' OR '.join(clauses)}"
            self.conn.execute(sql, params)
        # insert
        self.conn.register("sync_meta_df", df)
        self.conn.execute("INSERT INTO sync_meta SELECT * FROM sync_meta_df")
        self.conn.unregister("sync_meta_df")

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

    def read_price_daily(
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
        query = f"SELECT * FROM price_daily {where} ORDER BY symbol, date"
        return self.conn.execute(query, params).fetch_df()

    def read_sync_meta(self, dataset: str, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        clauses = ["dataset = ?"]
        params: list = [dataset]
        if symbols:
            placeholders = ",".join(["?"] * len(list(symbols)))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(list(symbols))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM sync_meta {where} ORDER BY symbol"
        return self.conn.execute(query, params).fetch_df()

    def read_signals(
        self,
        universe: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        rules: Optional[Sequence[str]] = None,
        run_version: Optional[str] = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if universe:
            clauses.append("universe = ?")
            params.append(universe)
        if symbols:
            placeholders = ",".join(["?"] * len(list(symbols)))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(list(symbols))
        if rules:
            placeholders = ",".join(["?"] * len(list(rules)))
            clauses.append(f"rule IN ({placeholders})")
            params.extend(list(rules))
        if run_version:
            clauses.append("run_version = ?")
            params.append(run_version)
        if start:
            clauses.append("timestamp >= ?")
            params.append(pd.to_datetime(start))
        if end:
            clauses.append("timestamp <= ?")
            params.append(pd.to_datetime(end))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM computed_signals {where} ORDER BY symbol, timestamp"
        result = self.conn.execute(query, params).fetch_df()
        if result.empty:
            return result
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result["as_of_date"] = pd.to_datetime(result["as_of_date"]).dt.normalize()

        def _deserialize_payload(payload: Any):
            if isinstance(payload, str):
                if not payload:
                    return {}
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return {}
            if payload is None:
                return {}
            return payload

        def _deserialize_value(value: Any):
            if value is None:
                return None
            if isinstance(value, str) and value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        result["signals"] = result["signals"].apply(_deserialize_payload)
        result["value"] = result["value"].apply(_deserialize_value)
        return result


def get_db() -> Database:
    if settings.clickhouse_url:
        logger.warning("ClickHouse configured but not implemented; falling back to DuckDB")
    return Database(settings.duckdb_path)
