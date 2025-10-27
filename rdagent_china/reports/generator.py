from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdagent_china.backtest.engine import BacktestSummary


@dataclass
class ReportPayload:
    """Structured payload representing backtest outcomes."""

    title: str
    summary: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Render structured summaries for dashboards and notebooks."""

    def __init__(self, title: str = "China Agent Signal Summary") -> None:
        self.title = title

    def generate(self, result: BacktestSummary) -> ReportPayload:
        summary = dict(result.counts)
        metadata = {"signals": [signal.model_dump() for signal in result.signals]}
        return ReportPayload(title=self.title, summary=summary, metadata=metadata)


__all__ = ["ReportGenerator", "ReportPayload"]
