from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd
from loguru import logger

from rdagent_china.data.akshare_client import AkshareClient, AkshareClientConfig


@dataclass
class AkshareAdapterConfig:
    requests_per_sec: float = 5.0
    max_retries: int = 3


class AkshareAdapter:
    """
    Thin adapter wrapping AkshareClient and exposing a vectorized symbols API.

    Returns canonical schema:
      columns: [symbol, date, open, high, low, close, volume]
      dtypes: date -> datetime64, numerics -> float
    """

    def __init__(self, config: Optional[AkshareAdapterConfig] = None):
        cfg = config or AkshareAdapterConfig()
        self._client = AkshareClient(
            AkshareClientConfig(
                requests_per_sec=cfg.requests_per_sec,
                max_retries=cfg.max_retries,
            )
        )

    def price_daily(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for sym in symbols:
            try:
                df = self._client.price_daily(sym, start=start, end=end)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"AkshareAdapter.price_daily failed for {sym}: {e}")
                df = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"]).assign(symbol=sym)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])  # empty
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])  # ensure datetime
        return out
