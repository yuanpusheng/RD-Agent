from __future__ import annotations

import types
from collections.abc import Iterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, ClassVar, TYPE_CHECKING, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - Python < 3.11 compatibility guard
    from types import NoneType
except ImportError:  # pragma: no cover - fallback for older runtimes
    NoneType = type(None)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

_TYPE_MAPPING: dict[type[Any], str] = {
    str: "TEXT",
    int: "BIGINT",
    float: "DOUBLE",
    Decimal: "DOUBLE",
    bool: "BOOLEAN",
    date: "DATE",
    datetime: "TIMESTAMP",
}


def _unwrap_annotation(annotation: Any) -> Any:
    """Resolve the underlying concrete type for optional or annotated types."""

    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is Annotated:
        args = get_args(annotation)
        return _unwrap_annotation(args[0]) if args else annotation
    if origin in (types.UnionType, Union):
        args = [arg for arg in get_args(annotation) if arg is not NoneType]
        if len(args) == 1:
            return _unwrap_annotation(args[0])
        return annotation
    return annotation


def _sanitize_identifier(value: str) -> str:
    cleaned = value.strip().replace('"', "")
    if not cleaned:
        raise ValueError("Identifier cannot be empty")
    safe = [char if (char.isalnum() or char == "_") else "_" for char in cleaned]
    return "".join(safe)


def _duckdb_type(annotation: Any) -> str:
    resolved = _unwrap_annotation(annotation)
    if isinstance(resolved, str):
        return resolved
    if isinstance(resolved, type) and resolved in _TYPE_MAPPING:
        return _TYPE_MAPPING[resolved]
    if isinstance(resolved, type) and issubclass(resolved, BaseModel):
        return "JSON"
    return "TEXT"


class DuckDBSchema(BaseModel):
    """Base model providing DuckDB DDL generation facilities."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    table_name: ClassVar[str]

    @classmethod
    def duckdb_columns(cls) -> list[tuple[str, str]]:
        columns: list[tuple[str, str]] = []
        for name, field_info in cls.model_fields.items():
            columns.append((name, _duckdb_type(field_info.annotation)))
        return columns

    @classmethod
    def duckdb_ddl(cls, *, schema: str | None = None) -> str:
        table = _sanitize_identifier(cls.table_name)
        if schema is None:
            qualified = table
        else:
            qualified = f"{_sanitize_identifier(schema)}.{table}"
        column_sql = ",\n".join(
            f"    {_sanitize_identifier(column)} {dtype}" for column, dtype in cls.duckdb_columns()
        )
        return f"CREATE TABLE IF NOT EXISTS {qualified} (\n{column_sql}\n);"


class PriceDailyRecord(DuckDBSchema):
    """Schema for daily bar data."""

    table_name: ClassVar[str] = "price_daily"

    symbol: str = Field(description="Ticker symbol for the security")
    date: datetime = Field(description="Timestamp representing the trading session close")
    open: float = Field(description="Opening price")
    high: float = Field(description="Highest traded price")
    low: float = Field(description="Lowest traded price")
    close: float = Field(description="Closing price")
    volume: float | None = Field(default=None, description="Trading volume for the session")
    turnover: float | None = Field(default=None, description="Turnover for the session")


class FundamentalRecord(DuckDBSchema):
    """Schema for fundamental financial statement snapshots."""

    table_name: ClassVar[str] = "fundamental"

    symbol: str = Field(description="Ticker symbol for the security")
    report_date: date = Field(description="Reference date for the reported fundamentals")
    revenue: float | None = Field(default=None)
    net_income: float | None = Field(default=None)
    eps: float | None = Field(default=None, description="Earnings per share")
    total_assets: float | None = Field(default=None)
    total_liabilities: float | None = Field(default=None)
    operating_cash_flow: float | None = Field(default=None)


class FactorRawRecord(DuckDBSchema):
    """Schema for intermediate factor computations."""

    table_name: ClassVar[str] = "factor_raw"

    symbol: str
    date: datetime
    factor: str = Field(description="Identifier for the factor")
    value: float = Field(description="Factor value")
    provider: str | None = Field(default=None, description="Upstream data provider")
    updated_at: datetime | None = Field(default=None)


class SignalRecord(DuckDBSchema):
    """Schema for generated trading signals."""

    table_name: ClassVar[str] = "signals"

    universe: str | None = Field(default=None, description="Universe identifier")
    symbol: str = Field(description="Ticker symbol associated with the signal")
    as_of_date: date = Field(description="Date the signal applies to")
    timestamp: datetime = Field(description="Exact timestamp when the signal was produced")
    signal: int = Field(description="Discrete trading instruction")
    strategy_id: str | None = Field(default=None)
    strategy_version: str | None = Field(default=None)
    confidence: float | None = Field(default=None)
    explanation: str | None = Field(default=None)


SCHEMA_MODELS: tuple[type[DuckDBSchema], ...] = (
    PriceDailyRecord,
    FundamentalRecord,
    FactorRawRecord,
    SignalRecord,
)


def iter_schema_models() -> Iterator[type[DuckDBSchema]]:
    yield from SCHEMA_MODELS


def iter_schema_ddls(*, schema: str | None = None) -> Iterator[str]:
    for model in SCHEMA_MODELS:
        yield model.duckdb_ddl(schema=schema)


def initialize_duckdb(path: str | Path | None = None, *, schema: str | None = None) -> "duckdb.DuckDBPyConnection":
    """Create the DuckDB database and ensure base tables exist."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("The 'duckdb' package is required to initialize the RD-Agent China database") from exc

    database = ":memory:" if path is None else str(Path(path))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(database)
    if schema:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_sanitize_identifier(schema)}")
    for ddl in iter_schema_ddls(schema=schema):
        conn.execute(ddl)
    return conn


__all__ = [
    "DuckDBSchema",
    "PriceDailyRecord",
    "FundamentalRecord",
    "FactorRawRecord",
    "SignalRecord",
    "iter_schema_models",
    "iter_schema_ddls",
    "initialize_duckdb",
]
