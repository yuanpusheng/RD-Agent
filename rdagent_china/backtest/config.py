from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DataSourceConfig(BaseModel):
    source: Literal["auto", "duckdb", "qlib"] = "auto"
    table: Literal["price_daily", "prices"] = "price_daily"
    fields: list[str] = Field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    qlib_provider_uri: str | None = None
    qlib_region: str = "cn"
    freq: str = "day"

    @field_validator("fields", mode="before")
    @classmethod
    def _ensure_fields(cls, value: Any) -> list[str]:
        if value is None:
            return ["open", "high", "low", "close", "volume"]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
        raise TypeError("fields must be a sequence of field names")


class LabelingConfig(BaseModel):
    method: Literal["forward_return"] = "forward_return"
    horizon: int = 5
    threshold: float = 0.02
    direction: Literal["long", "short"] = "long"
    price_field: str = "close"
    aggregator: Literal["last"] = "last"

    @model_validator(mode="after")
    def _validate(self) -> "LabelingConfig":
        if self.horizon <= 0:
            raise ValueError("labeling.horizon must be positive")
        if self.threshold < 0:
            raise ValueError("labeling.threshold must be non-negative")
        return self


class BenchmarkConfig(BaseModel):
    symbol: str
    price_field: str = "close"


class ReportConfig(BaseModel):
    output_dir: Path | None = None
    filename_prefix: str = "signal_backtest"

    @model_validator(mode="after")
    def _normalize(self) -> "ReportConfig":
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        return self


class SignalBacktestConfig(BaseModel):
    symbols: list[str] | None = None
    universe: str | None = None
    rules: list[str] | None = None
    run_version: str | None = None
    start: str | None = None
    end: str | None = None
    evaluation_windows: list[int] | None = None
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    data: DataSourceConfig = Field(default_factory=DataSourceConfig)
    benchmark: BenchmarkConfig | None = None
    report: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [item.strip() for item in str(value).split(",") if item.strip()]
        if isinstance(value, Sequence):
            return [str(item) for item in value]
        raise TypeError("symbols must be provided as a list or comma-separated string")

    @field_validator("rules", mode="before")
    @classmethod
    def _normalize_rules(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [item.strip() for item in str(value).split(",") if item.strip()]
        if isinstance(value, Sequence):
            return [str(item) for item in value]
        raise TypeError("rules must be provided as a list or comma-separated string")

    @model_validator(mode="after")
    def _ensure_windows(self) -> "SignalBacktestConfig":
        windows = self.evaluation_windows
        if not windows:
            windows = [self.labeling.horizon]
        normalized: list[int] = []
        for item in windows:
            value = int(item)
            if value <= 0:
                raise ValueError("evaluation_windows entries must be positive integers")
            normalized.append(value)
        self.evaluation_windows = sorted({*normalized})
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "SignalBacktestConfig":
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) if raw else {}
        if data is None:
            data = {}
        return cls.model_validate(data)

    def with_overrides(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        rules: Sequence[str] | None = None,
    ) -> "SignalBacktestConfig":
        update: dict[str, Any] = {}
        if symbols is not None:
            update["symbols"] = [str(item) for item in symbols]
        if start is not None:
            update["start"] = str(start)
        if end is not None:
            update["end"] = str(end)
        if rules is not None:
            update["rules"] = [str(item) for item in rules]
        return self.model_copy(update=update)

    def resolved_windows(self) -> list[int]:
        return list(self.evaluation_windows or [self.labeling.horizon])


__all__ = [
    "BenchmarkConfig",
    "DataSourceConfig",
    "LabelingConfig",
    "ReportConfig",
    "SignalBacktestConfig",
]
