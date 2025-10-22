from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import IndicatorOutput


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "date" in frame.columns:
            frame = frame.set_index(pd.to_datetime(frame["date"]))
        else:
            frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _get_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    prepared = _ensure_datetime_index(frame.copy())
    if column not in prepared.columns:
        raise KeyError(f"Column '{column}' not found in input frame")
    return prepared[column].astype(float)


def moving_average_cross(
    frame: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    price_column: str = "close",
) -> IndicatorOutput:
    """Compute simple moving average cross-over signals."""

    prices = _get_numeric_series(frame, price_column)
    short_ma = prices.rolling(window=short_window, min_periods=1).mean()
    long_ma = prices.rolling(window=long_window, min_periods=1).mean()

    signal = pd.Series(0, index=prices.index, dtype=float)
    signal[short_ma > long_ma] = 1.0
    signal[short_ma < long_ma] = -1.0

    output = pd.DataFrame({
        "ma_short": short_ma,
        "ma_long": long_ma,
        "signal": signal,
    })
    return IndicatorOutput(name="ma_cross", frame=output)


def exponential_moving_average_cross(
    frame: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    price_column: str = "close",
) -> IndicatorOutput:
    """Compute exponential moving average cross-over signals."""

    prices = _get_numeric_series(frame, price_column)
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()

    signal = pd.Series(0, index=prices.index, dtype=float)
    signal[short_ema > long_ema] = 1.0
    signal[short_ema < long_ema] = -1.0

    output = pd.DataFrame({
        "ema_short": short_ema,
        "ema_long": long_ema,
        "signal": signal,
    })
    return IndicatorOutput(name="ema_cross", frame=output)


def macd(
    frame: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_column: str = "close",
) -> IndicatorOutput:
    """Compute the Moving Average Convergence Divergence indicator."""

    prices = _get_numeric_series(frame, price_column)
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    output = pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })
    return IndicatorOutput(name="macd", frame=output)


def rsi(
    frame: pd.DataFrame,
    period: int = 14,
    price_column: str = "close",
) -> IndicatorOutput:
    """Compute the Relative Strength Index."""

    prices = _get_numeric_series(frame, price_column)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    rsi_series = rsi_series.fillna(50.0)
    rsi_series = rsi_series.mask(avg_loss == 0, 100.0)
    rsi_series = rsi_series.mask(avg_gain == 0, 0.0)
    rsi_series = rsi_series.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    rsi_series = rsi_series.replace([np.inf, -np.inf], 100.0)

    output = pd.DataFrame({"rsi": rsi_series.clip(lower=0, upper=100)})
    return IndicatorOutput(name="rsi", frame=output)


def bollinger_bands(
    frame: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_column: str = "close",
) -> IndicatorOutput:
    """Compute Bollinger Bands."""

    prices = _get_numeric_series(frame, price_column)
    rolling_mean = prices.rolling(window=window, min_periods=1).mean()
    rolling_std = prices.rolling(window=window, min_periods=1).std(ddof=0)

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    bandwidth = (upper_band - lower_band) / rolling_mean.replace(0, np.nan)
    percent_b = (prices - lower_band) / (upper_band - lower_band).replace(0, np.nan)

    output = pd.DataFrame(
        {
            "middle": rolling_mean,
            "upper": upper_band,
            "lower": lower_band,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        }
    )
    return IndicatorOutput(name="bollinger", frame=output)


def volume_spike(
    frame: pd.DataFrame,
    window: int = 20,
    threshold: float = 2.0,
    volume_column: str = "volume",
) -> IndicatorOutput:
    """Detect volume spikes relative to a rolling average."""

    volumes = _get_numeric_series(frame, volume_column)
    rolling_mean = volumes.rolling(window=window, min_periods=1).mean()
    ratio = volumes / rolling_mean.replace(0, np.nan)
    spike = ratio >= threshold

    output = pd.DataFrame(
        {
            "volume": volumes,
            "rolling_mean": rolling_mean,
            "ratio": ratio,
            "signal": spike.astype(float),
            "spike": spike,
        }
    )
    return IndicatorOutput(name="volume_spike", frame=output)


INDICATOR_REGISTRY: dict[str, Any] = {
    "ma_cross": moving_average_cross,
    "ema_cross": exponential_moving_average_cross,
    "macd": macd,
    "rsi": rsi,
    "bollinger": bollinger_bands,
    "volume_spike": volume_spike,
}

__all__ = [
    "IndicatorOutput",
    "moving_average_cross",
    "exponential_moving_average_cross",
    "macd",
    "rsi",
    "bollinger_bands",
    "volume_spike",
    "INDICATOR_REGISTRY",
]
