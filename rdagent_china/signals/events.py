from __future__ import annotations

from typing import Any

import pandas as pd

from .base import IndicatorOutput


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "date" in frame.columns:
            frame = frame.set_index(pd.to_datetime(frame["date"]))
        else:
            frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def limit_moves(
    frame: pd.DataFrame,
    price_column: str = "close",
    upper_limit: float = 0.1,
    lower_limit: float = 0.1,
) -> IndicatorOutput:
    """Detect limit-up and limit-down events based on percentage moves."""

    prepared = _ensure_datetime_index(frame.copy())
    if price_column not in prepared.columns:
        raise KeyError(f"Column '{price_column}' not found in input frame")

    prices = prepared[price_column].astype(float)
    pct_change = prices.pct_change()
    limit_up = pct_change >= upper_limit
    limit_down = pct_change <= -lower_limit

    signal = pd.Series(0, index=prepared.index, dtype=float)
    signal[limit_up] = 1.0
    signal[limit_down] = -1.0

    output = pd.DataFrame(
        {
            "price": prices,
            "pct_change": pct_change,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "signal": signal,
        }
    )
    return IndicatorOutput(name="limit_moves", frame=output)


def gap_open(
    frame: pd.DataFrame,
    open_column: str = "open",
    close_column: str = "close",
    threshold: float = 0.03,
) -> IndicatorOutput:
    """Detect gap opens relative to the prior close."""

    prepared = _ensure_datetime_index(frame.copy())
    for column in (open_column, close_column):
        if column not in prepared.columns:
            raise KeyError(f"Column '{column}' not found in input frame")

    opens = prepared[open_column].astype(float)
    closes = prepared[close_column].astype(float)
    prev_close = closes.shift(1)
    gap_pct = (opens - prev_close) / prev_close
    gap_up = gap_pct >= threshold
    gap_down = gap_pct <= -threshold

    signal = pd.Series(0, index=prepared.index, dtype=float)
    signal[gap_up] = 1.0
    signal[gap_down] = -1.0

    output = pd.DataFrame(
        {
            "open": opens,
            "prev_close": prev_close,
            "gap_pct": gap_pct,
            "gap_up": gap_up,
            "gap_down": gap_down,
            "signal": signal,
        }
    )
    return IndicatorOutput(name="gap_open", frame=output)


def volatility_breakout(
    frame: pd.DataFrame,
    price_column: str = "close",
    window: int = 20,
    multiplier: float = 1.5,
) -> IndicatorOutput:
    """Detect volatility breakouts when returns exceed a rolling volatility band."""

    prepared = _ensure_datetime_index(frame.copy())
    if price_column not in prepared.columns:
        raise KeyError(f"Column '{price_column}' not found in input frame")

    prices = prepared[price_column].astype(float)
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window=window, min_periods=1).std(ddof=0)
    threshold = rolling_vol * multiplier
    breakout = returns.abs() >= threshold

    signal = breakout.astype(float)

    output = pd.DataFrame(
        {
            "returns": returns,
            "rolling_vol": rolling_vol,
            "threshold": threshold,
            "breakout": breakout,
            "signal": signal,
        }
    )
    return IndicatorOutput(name="volatility_breakout", frame=output)


EVENT_REGISTRY: dict[str, Any] = {
    "limit_moves": limit_moves,
    "gap_open": gap_open,
    "volatility_breakout": volatility_breakout,
}

__all__ = [
    "IndicatorOutput",
    "limit_moves",
    "gap_open",
    "volatility_breakout",
    "EVENT_REGISTRY",
]
