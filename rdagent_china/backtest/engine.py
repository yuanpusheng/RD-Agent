from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from rdagent_china.data.schemas import SignalRecord


@dataclass
class BacktestSummary:
    """Minimal backtest aggregation summarising signal counts."""

    signals: list[SignalRecord] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "counts": dict(self.counts),
            "signals": [signal.model_dump() for signal in self.signals],
        }


class SimpleBacktestEngine:
    """Placeholder backtest engine that tallies directional signals."""

    def run(self, signals: Sequence[SignalRecord]) -> BacktestSummary:
        tally: Counter[str] = Counter()
        materialized = [SignalRecord.model_validate(signal) if not isinstance(signal, SignalRecord) else signal for signal in signals]
        for record in materialized:
            if record.signal > 0:
                tally["buy"] += 1
            elif record.signal < 0:
                tally["sell"] += 1
            else:
                tally["hold"] += 1
        tally["total"] = len(materialized)
        return BacktestSummary(signals=list(materialized), counts=dict(tally))


__all__ = ["BacktestSummary", "SimpleBacktestEngine"]
