from __future__ import annotations

import pandas as pd


class Signal:
    BUY = 1
    SELL = -1
    HOLD = 0


def generate_signals(df: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface placeholder
    raise NotImplementedError
