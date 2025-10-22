from __future__ import annotations

import pandas as pd

from rdagent_china.signals import (
    bollinger_bands,
    gap_open,
    limit_moves,
    macd,
    moving_average_cross,
    rsi,
    volume_spike,
    volatility_breakout,
)


def _sample_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "close": [10, 10.5, 11.0, 10.8, 11.5, 12.0, 12.5, 13.0],
            "open": [9.8, 10.2, 10.9, 10.7, 11.3, 11.8, 12.2, 12.7],
            "high": [10.2, 10.7, 11.2, 11.0, 11.7, 12.3, 12.8, 13.2],
            "low": [9.7, 10.0, 10.8, 10.5, 11.1, 11.6, 12.0, 12.5],
            "volume": [1_000_000, 1_200_000, 950_000, 1_100_000, 1_800_000, 2_000_000, 2_500_000, 2_200_000],
        }
    )


def test_moving_average_cross_signal_behavior():
    frame = _sample_price_frame()
    output = moving_average_cross(frame, short_window=2, long_window=4)
    short_ma = frame["close"].rolling(2, min_periods=1).mean()
    long_ma = frame["close"].rolling(4, min_periods=1).mean()

    expected_signal = pd.Series(0.0, index=frame.index)
    expected_signal[short_ma > long_ma] = 1.0
    expected_signal[short_ma < long_ma] = -1.0

    pd.testing.assert_series_equal(output.frame["signal"].reset_index(drop=True), expected_signal, check_names=False)


def test_macd_histogram_alignment():
    frame = _sample_price_frame()
    output = macd(frame, fast_period=3, slow_period=6, signal_period=3)
    prices = frame.set_index("date")["close"].astype(float)
    fast = prices.ewm(span=3, adjust=False).mean()
    slow = prices.ewm(span=6, adjust=False).mean()
    expected_macd = fast - slow
    expected_signal = expected_macd.ewm(span=3, adjust=False).mean()
    expected_histogram = expected_macd - expected_signal

    pd.testing.assert_series_equal(output.frame["macd"], expected_macd, check_names=False)
    pd.testing.assert_series_equal(output.frame["signal"], expected_signal, check_names=False)
    pd.testing.assert_series_equal(output.frame["histogram"], expected_histogram, check_names=False)


def test_rsi_stays_within_bounds():
    frame = _sample_price_frame()
    output = rsi(frame, period=3)
    rsi_values = output.frame["rsi"]
    assert ((0 <= rsi_values) & (rsi_values <= 100)).all()
    assert rsi_values.iloc[-1] > rsi_values.iloc[1]


def test_bollinger_bands_structure():
    frame = _sample_price_frame()
    output = bollinger_bands(frame, window=3, num_std=1.5)
    bands = output.frame.tail(1).iloc[0]
    assert bands["upper"] > bands["middle"] > bands["lower"]
    assert bands["bandwidth"] >= 0
    assert 0 <= bands["percent_b"] <= 1


def test_volume_spike_signal_flags_expected_day():
    frame = _sample_price_frame()
    frame.loc[frame.index[-1], "volume"] = 5_000_000
    output = volume_spike(frame, window=3, threshold=2.0)
    assert bool(output.frame["spike"].iloc[-1])
    assert output.frame["spike"].iloc[:-1].sum() == 0


def test_limit_moves_detects_extremes():
    frame = _sample_price_frame()
    frame.loc[2, "close"] = frame.loc[1, "close"] * 1.12
    frame.loc[5, "close"] = frame.loc[4, "close"] * 0.85
    output = limit_moves(frame, upper_limit=0.1, lower_limit=0.1)
    assert output.frame["limit_up"].iloc[2]
    assert output.frame["limit_down"].iloc[5]


def test_gap_open_identifies_gap_direction():
    frame = _sample_price_frame()
    frame.loc[4, "open"] = frame.loc[3, "close"] * 1.05
    frame.loc[6, "open"] = frame.loc[5, "close"] * 0.95
    output = gap_open(frame, threshold=0.03)
    assert output.frame["gap_up"].iloc[4]
    assert output.frame["gap_down"].iloc[6]


def test_volatility_breakout_flags_large_moves():
    frame = _sample_price_frame()
    frame.loc[7, "close"] = frame.loc[6, "close"] * 1.2
    output = volatility_breakout(frame, window=3, multiplier=1.2)
    assert bool(output.frame["breakout"].iloc[-1])
    assert output.frame["breakout"].iloc[:-1].sum() == 0
