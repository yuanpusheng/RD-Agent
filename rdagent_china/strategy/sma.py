from __future__ import annotations

import pandas as pd

from rdagent_china.strategy.interface import Signal


def generate_signals(df: pd.DataFrame, short: int = 10, long: int = 30) -> pd.Series:
    close = df["close"].astype(float)
    sma_s = close.rolling(short, min_periods=1).mean()
    sma_l = close.rolling(long, min_periods=1).mean()
    sig = pd.Series(Signal.HOLD, index=df.index)
    sig[sma_s > sma_l] = Signal.BUY
    sig[sma_s < sma_l] = Signal.SELL
    return sig
