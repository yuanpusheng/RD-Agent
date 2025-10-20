from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from rdagent_china.db import Database


def _fetch_akshare_stock_daily(code: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            },
            inplace=True,
        )
        df = df[["date", "open", "high", "low", "close", "volume"]]
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        df["symbol"] = code
        return df
    except Exception as e:
        logger.warning(f"akshare fetch failed for {code}: {e}; generate synthetic data")
        import numpy as np

        dates = pd.date_range(start or "2020-01-01", end or pd.Timestamp.today().normalize(), freq="B")
        prices = np.cumsum(np.random.randn(len(dates))) + 100
        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices + np.random.randn(len(dates)) * 0.5,
                "high": prices + np.random.rand(len(dates)),
                "low": prices - np.random.rand(len(dates)),
                "close": prices,
                "volume": np.random.randint(1000, 100000, size=len(dates)),
                "symbol": code,
            }
        )
        return df


def ingest_prices(symbols: Iterable[str], start: Optional[str], end: Optional[str], db: Database):
    all_rows = []
    for sym in symbols:
        df = _fetch_akshare_stock_daily(sym, start, end)
        all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])  # ensure datetime
    db.write_prices(full)
