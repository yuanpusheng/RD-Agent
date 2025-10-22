from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

from rdagent_china.config import settings
from rdagent_china.data.adapters.akshare_adapter import AkshareAdapter
try:  # optional
    from rdagent_china.data.adapters.tushare_adapter import TushareAdapter
except Exception:  # pragma: no cover - import optional, resolved at runtime
    TushareAdapter = None  # type: ignore
from rdagent_china.db import Database, get_db
from rdagent_china.data.trading_calendar import get_default_calendar


@dataclass
class ProviderConfig:
    source_order: Sequence[str]
    cache_dir: Optional[Path] = None


class UnifiedDataProvider:
    """
    Facade that provides a unified interface to multiple data sources (Tushare, Akshare, etc.)
    with QLib-compatible outputs and simple incremental sync.

    Public methods return canonical schema:
      - get_price_daily(symbols, start, end): DataFrame columns [symbol, date, open, high, low, close, volume]
      - get_price_daily_qlib(...): MultiIndex (date, symbol) with requested fields to emulate QLib reader
    """

    def __init__(self, db: Optional[Database] = None, config: Optional[ProviderConfig] = None):
        self.db = db or get_db()
        src_order = settings.data_source_order
        cfg = config or ProviderConfig(source_order=src_order, cache_dir=settings.data_cache_dir)
        self.config = cfg
        # build adapters in priority order
        self._adapters: List[tuple[str, object]] = []
        for name in cfg.source_order:
            if name.lower() == "akshare":
                self._adapters.append(("akshare", AkshareAdapter()))
            elif name.lower() == "tushare" and TushareAdapter is not None:
                try:
                    self._adapters.append(("tushare", TushareAdapter()))
                except Exception as e:  # optional dep not configured
                    logger.warning(f"Tushare adapter unavailable: {e}")
            # skip unknowns silently
        if not self._adapters:
            # always include Akshare as last resort to enable smoke tests/synthetic fallback
            self._adapters.append(("akshare", AkshareAdapter()))
        self.calendar = get_default_calendar()

    # ---------------- Public API ----------------
    def get_price_daily(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        symbols = list(symbols)
        remaining = set(symbols)
        all_frames: List[pd.DataFrame] = []
        for src_name, adapter in self._adapters:
            if not remaining:
                break
            try:
                df = adapter.price_daily(sorted(remaining), start=start, end=end)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"{src_name} adapter failed: {e}; fallback to next source")
                continue
            if df is None or df.empty:
                continue
            # Keep only remaining symbols
            df = df[df["symbol"].isin(remaining)]
            got_syms = set(df["symbol"].unique().tolist())
            if not df.empty:
                df["source"] = src_name
                all_frames.append(df)
                remaining -= got_syms
        if not all_frames:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])  # empty
        out = pd.concat(all_frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])  # ensure datetime
        # cache to daily table if available
        try:
            self.db.init()
            # we only persist canonical columns to price_daily to keep backward compatibility
            self.db.write_price_daily(out[["symbol", "date", "open", "high", "low", "close", "volume"]])
            # update sync metadata per symbol
            self._upsert_sync_meta(dataset="price_daily", df=out)
        except Exception as e:  # pragma: no cover - persistence best effort
            logger.warning(f"Failed to persist get_price_daily results: {e}")
        return out

    def get_price_daily_qlib(self, symbols: Iterable[str], start: Optional[str] = None, end: Optional[str] = None,
                              fields: Optional[Sequence[str]] = None) -> pd.DataFrame:
        base = self.get_price_daily(symbols, start=start, end=end)
        if base.empty:
            cols = fields or ["open", "high", "low", "close", "volume"]
            mi = pd.MultiIndex.from_tuples([], names=["date", "symbol"])  # empty
            return pd.DataFrame(columns=cols, index=mi)
        keep_fields = fields or ["open", "high", "low", "close", "volume"]
        base = base[["date", "symbol", *keep_fields]].copy()
        base["date"] = pd.to_datetime(base["date"])  # ensure
        base = base.set_index(["date", "symbol"]).sort_index()
        return base

    def latest_synced_date(self, symbol: str, dataset: str = "price_daily") -> Optional[pd.Timestamp]:
        try:
            df = self.db.read_sync_meta(dataset=dataset, symbols=[symbol])
            if df.empty:
                return None
            return pd.to_datetime(df.iloc[0]["last_date"])
        except Exception:
            return None

    # ---------------- Internals ----------------
    def _upsert_sync_meta(self, dataset: str, df: pd.DataFrame):
        if df.empty:
            return
        # Determine last date per symbol
        grp = df.groupby("symbol")["date"].max().reset_index().rename(columns={"date": "last_date"})
        grp["dataset"] = dataset
        # If mixed sources present, pick the max-date row source per symbol
        src = df.sort_values(["symbol", "date"]).groupby("symbol").tail(1)[["symbol", "source"]]
        merged = pd.merge(grp, src, on="symbol", how="left")
        merged["updated_at"] = pd.Timestamp.utcnow()
        self.db.upsert_sync_meta(merged)
