from __future__ import annotations

from .base import (
    BaseSignalSource,
    FundamentalSignalSource,
    IndicatorOutput,
    NewsSignalSource,
    SignalRecord,
)
from .events import EVENT_REGISTRY, gap_open, limit_moves, volatility_breakout
from .indicators import (
    INDICATOR_REGISTRY,
    bollinger_bands,
    exponential_moving_average_cross,
    macd,
    moving_average_cross,
    rsi,
    volume_spike,
)
from .persistence import persist_signal_records, records_to_dataframe
from .rules import RulesEngine, load_config

__all__ = [
    "BaseSignalSource",
    "FundamentalSignalSource",
    "IndicatorOutput",
    "NewsSignalSource",
    "SignalRecord",
    "moving_average_cross",
    "exponential_moving_average_cross",
    "macd",
    "rsi",
    "bollinger_bands",
    "volume_spike",
    "limit_moves",
    "gap_open",
    "volatility_breakout",
    "INDICATOR_REGISTRY",
    "EVENT_REGISTRY",
    "RulesEngine",
    "load_config",
    "persist_signal_records",
    "records_to_dataframe",
]
