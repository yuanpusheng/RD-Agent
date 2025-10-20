from __future__ import annotations

from datetime import date, datetime
from typing import List

import pandas as pd
from loguru import logger


def get_trading_days(start: str, end: str) -> List[date]:
    try:
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        return pd.to_datetime(df["trade_date"]).dt.date.tolist()
    except Exception as e:
        logger.warning(f"Failed to fetch trading calendar from akshare: {e}; fallback to business days")
        days = pd.bdate_range(start=start, end=end)
        return [d.date() for d in days]


def is_trading_day(d: date) -> bool:
    try:
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        return pd.to_datetime(d) in pd.to_datetime(df["trade_date"]).values
    except Exception:
        # fallback: Mon-Fri only
        return d.weekday() < 5
