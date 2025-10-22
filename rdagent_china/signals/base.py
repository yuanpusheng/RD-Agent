from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import pandas as pd


@dataclass
class IndicatorOutput:
    """Container for indicator or event outputs.

    The :attr:`frame` attribute should contain aligned series indexed by timestamp,
    where each column represents a derived measure (e.g. ``signal`` for trading
    direction, ``upper``/``lower`` bands, etc.).
    """

    name: str
    frame: pd.DataFrame
    metadata: dict[str, Any] | None = None

    def get_series(self, field: str | None = None) -> pd.Series:
        """Return a pandas Series representing the desired output field.

        If *field* is ``None`` the method will prefer a column named ``signal``
        and otherwise fall back to the first column.
        """

        if field:
            if field not in self.frame.columns:
                raise KeyError(f"Field '{field}' not found in indicator '{self.name}' output")
            return self.frame[field]
        if "signal" in self.frame.columns:
            return self.frame["signal"]
        if self.frame.shape[1] == 1:
            return self.frame.iloc[:, 0]
        raise ValueError(
            f"Indicator '{self.name}' output does not expose a default 'signal' column; specify the field explicitly"
        )

    def point_values(self, timestamp: pd.Timestamp) -> dict[str, Any]:
        """Return a dictionary of indicator values at a specific timestamp."""

        if timestamp not in self.frame.index:
            raise KeyError(f"Timestamp {timestamp} not present in indicator '{self.name}' output")
        values = self.frame.loc[timestamp]
        if isinstance(values, pd.Series):
            return values.to_dict()
        return values


@dataclass
class SignalRecord:
    """Normalized structure representing an evaluated rule outcome."""

    universe: str | None
    symbol: str
    timestamp: pd.Timestamp
    rule: str
    label: str
    severity: str
    triggered: bool
    value: float | bool | None = None
    config_version: str | None = None
    run_version: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def as_of_date(self) -> pd.Timestamp:
        return self.timestamp.normalize()

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a persistence-friendly dictionary."""

        return {
            "universe": self.universe,
            "symbol": self.symbol,
            "timestamp": pd.to_datetime(self.timestamp),
            "as_of_date": pd.to_datetime(self.as_of_date),
            "rule": self.rule,
            "label": self.label,
            "severity": self.severity,
            "triggered": bool(self.triggered),
            "value": self.value,
            "config_version": self.config_version,
            "run_version": self.run_version,
            "signals": self.signals,
        }

    @classmethod
    def to_frame(cls, records: Sequence[SignalRecord]) -> pd.DataFrame:
        """Convert a sequence of records into a pandas DataFrame."""

        if not records:
            return pd.DataFrame(
                columns=[
                    "universe",
                    "symbol",
                    "timestamp",
                    "as_of_date",
                    "rule",
                    "label",
                    "severity",
                    "triggered",
                    "value",
                    "config_version",
                    "run_version",
                    "signals",
                ]
            )
        return pd.DataFrame([record.to_dict() for record in records])


class BaseSignalSource(Protocol):
    """Protocol describing a pluggable signal source."""

    def compute(self, data: pd.DataFrame) -> IndicatorOutput:  # pragma: no cover - interface specification
        ...


class FundamentalSignalSource:
    """Placeholder interface for fundamentals-driven signal providers."""

    def compute(self, data: pd.DataFrame) -> IndicatorOutput:  # pragma: no cover - extension hook
        raise NotImplementedError("Fundamental signal computation is not yet implemented")


class NewsSignalSource:
    """Placeholder interface for news/NLP-derived signal providers."""

    def compute(self, data: pd.DataFrame) -> IndicatorOutput:  # pragma: no cover - extension hook
        raise NotImplementedError("News-based signal computation is not yet implemented")
