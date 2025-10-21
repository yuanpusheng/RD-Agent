from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from loguru import logger


@dataclass
class TushareAdapterConfig:
    token: Optional[str] = None


class TushareAdapter:
    """
    Minimal Tushare adapter. Fetches daily bars via pro_bar when tushare is installed and token is provided.
    If tushare is not available at runtime, an ImportError will be raised on first use, which allows the
    provider facade to fallback to the next data source.
    """

    def __init__(self, config: Optional[TushareAdapterConfig] = None):
        cfg = config or TushareAdapterConfig()
        self._token = cfg.token or self._get_env_token()
        self._ts = None
        self._pro = None
        self._init_client()

    @staticmethod
    def _get_env_token() -> Optional[str]:
        import os

        return os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")

    def _init_client(self):  # pragma: no cover - optional dep path
        try:
            import tushare as ts  # type: ignore

            self._ts = ts
            if self._token:
                ts.set_token(self._token)
            self._pro = ts.pro_api(self._token) if self._token else None
        except Exception as e:  # Satisfy provider fallback logic by raising
            raise ImportError(f"tushare not available: {e}")

    def price_daily(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:  # pragma: no cover - optional dep path
        if self._pro is None:
            raise RuntimeError("Tushare token not configured")
        # Tushare uses ts_code like '000001.SZ' or '600000.SH'. If naked 6-digit is provided, try both suffixes.
        all_frames = []
        for sym in symbols:
            codes_to_try = [sym] if "." in sym else [f"{sym}.SZ", f"{sym}.SH"]
            df_ok = None
            for code in codes_to_try:
                try:
                    df = self._pro.daily(ts_code=code, start_date=(start or "19900101").replace("-", ""), end_date=(end or "21000101").replace("-", ""))
                    if df is not None and not df.empty:
                        df_ok = df
                        break
                except Exception:
                    continue
            if df_ok is None:
                logger.warning(f"TushareAdapter: no data returned for {sym}")
                continue
            # Normalize
            df_ok = df_ok.rename(columns={
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
            })
            keep = ["date", "open", "high", "low", "close", "volume"]
            df_ok = df_ok[keep]
            df_ok["date"] = pd.to_datetime(df_ok["date"].astype(str))
            df_ok["symbol"] = sym
            all_frames.append(df_ok)
        if not all_frames:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])  # empty
        out = pd.concat(all_frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])  # ensure datetime
        return out
