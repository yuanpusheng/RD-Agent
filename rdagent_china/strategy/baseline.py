from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from rdagent_china.data.schemas import PriceDailyRecord, SignalRecord


@dataclass
class MovingAverageStrategy:
    """Simple moving-average cross-over strategy used for scaffolding tests."""

    short_window: int = 5
    long_window: int = 20
    universe: str | None = None
    strategy_id: str = "moving_average"
    strategy_version: str = "v0"

    def generate_signals(self, prices: Sequence[PriceDailyRecord]) -> list[SignalRecord]:
        if not prices:
            return []

        frame = pd.DataFrame([record.model_dump() for record in prices])
        if frame.empty:
            return []

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
        frame["short_ma"] = (
            frame.groupby("symbol")["close"].transform(lambda series: series.rolling(self.short_window, min_periods=1).mean())
        )
        frame["long_ma"] = (
            frame.groupby("symbol")["close"].transform(lambda series: series.rolling(self.long_window, min_periods=1).mean())
        )

        signals: list[SignalRecord] = []
        for row in frame.to_dict(orient="records"):
            short_value = row.get("short_ma")
            long_value = row.get("long_ma")
            signal_value = 0
            if pd.notna(short_value) and pd.notna(long_value):
                if short_value > long_value:
                    signal_value = 1
                elif short_value < long_value:
                    signal_value = -1
            timestamp = pd.Timestamp(row["date"]).to_pydatetime()
            confidence = None
            if pd.notna(short_value) and pd.notna(long_value):
                confidence = float(abs(short_value - long_value)) if signal_value != 0 else 0.0
            signals.append(
                SignalRecord(
                    universe=self.universe,
                    symbol=str(row.get("symbol", "")),
                    as_of_date=timestamp.date(),
                    timestamp=timestamp,
                    signal=int(signal_value),
                    strategy_id=self.strategy_id,
                    strategy_version=self.strategy_version,
                    confidence=confidence,
                    explanation="moving-average crossover" if signal_value else "moving-average neutral",
                )
            )
        return signals


__all__ = ["MovingAverageStrategy"]
