from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from loguru import logger

from rdagent_china.alerts import AlertDispatcher
from rdagent_china.config import settings
from rdagent_china.data.provider import UnifiedDataProvider
from rdagent_china.data.universe import resolve_universe
from rdagent_china.db import Database, get_db
from rdagent_china.signals.base import SignalRecord
from rdagent_china.signals.persistence import persist_signal_records
from rdagent_china.signals.rules import RulesEngine

try:  # pragma: no cover - optional dependency import guard
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except Exception:  # pragma: no cover - handled at runtime if scheduler unavailable
    AsyncIOScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore
    IntervalTrigger = None  # type: ignore

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python <3.9 compatibility
    ZoneInfo = None  # type: ignore


@dataclass(slots=True)
class MonitorRunContext:
    """Parameters controlling a monitoring cycle execution."""

    universe: Optional[str | Sequence[str]] = None
    watchlist: Optional[str | Sequence[str] | Path] = None
    intraday: bool = False
    start: Optional[str] = None
    end: Optional[str] = None


class MonitorLoop:
    """Drive daily and intraday monitoring cycles for RD-Agent China."""

    def __init__(
        self,
        *,
        db: Optional[Database] = None,
        provider: Optional[UnifiedDataProvider] = None,
        rules_engine: Optional[RulesEngine] = None,
        lookback_days: Optional[int] = None,
        backoff_minutes: Optional[int] = None,
        fetch_retries: Optional[int] = None,
        fetch_retry_delay: Optional[float] = None,
        config_version: Optional[str] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None,
    ) -> None:
        self._settings = settings
        self.db = db or get_db()
        self.db.init()
        self.provider = provider or UnifiedDataProvider(db=self.db)
        self.rules_engine = rules_engine or RulesEngine.from_file(self._settings.monitor_rules_path)
        self.lookback_days = lookback_days or self._settings.monitor_lookback_days
        backoff_source = backoff_minutes
        if backoff_source is None:
            backoff_source = self._settings.monitor_alert_backoff_minutes
        self.backoff: Optional[pd.Timedelta]
        if backoff_source and backoff_source > 0:
            self.backoff = pd.to_timedelta(backoff_source, unit="minutes")
        else:
            self.backoff = None
        retries_source = fetch_retries or self._settings.monitor_fetch_retries
        self.fetch_retries = max(1, retries_source)
        self.fetch_retry_delay = fetch_retry_delay or self._settings.monitor_fetch_retry_delay_seconds
        self.config_version = config_version or self._settings.monitor_config_version
        self.alert_dispatcher = alert_dispatcher or self._build_dispatcher()

    @property
    def settings(self):  # pragma: no cover - simple accessor
        return self._settings

    def _build_dispatcher(self) -> AlertDispatcher | None:
        try:
            dispatcher = AlertDispatcher(settings=self._settings, db=self.db)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to initialize alert dispatcher: {}", exc)
            return None
        return dispatcher

    def run_cycle(self, *, context: MonitorRunContext | None = None, **kwargs) -> pd.DataFrame:
        """Execute a monitoring cycle synchronously and persist new alerts.

        Parameters can be supplied either through a :class:`MonitorRunContext` or as
        keyword arguments matching the dataclass fields.
        """

        ctx = context or MonitorRunContext(**kwargs)
        universe_label = self._determine_universe_label(ctx)
        symbols = self._resolve_symbols(ctx)
        if not symbols:
            logger.info("No symbols resolved for monitoring cycle; skipping")
            return SignalRecord.to_frame([])

        start = ctx.start or self._default_start()
        end = ctx.end
        data = self._fetch_with_retry(symbols=symbols, start=start, end=end)
        if data.empty:
            logger.info("No market data returned for monitoring cycle; skipping evaluation")
            return SignalRecord.to_frame([])

        records = self.rules_engine.evaluate(
            data,
            universe=universe_label,
            config_version=self.config_version,
            run_version=self._build_run_version(ctx.intraday),
        )
        filtered = self._apply_backoff(records)
        if not filtered:
            logger.debug("All alerts suppressed by backoff state")
            return SignalRecord.to_frame([])

        persisted = persist_signal_records(self.db, filtered)
        if persisted.empty:
            logger.debug("No alerts persisted in this cycle")
            return persisted

        self._update_state(persisted)
        triggered_records = [record for record in filtered if record.triggered]
        self._dispatch_alerts(triggered_records)
        logger.info(
            "Persisted {} alerts for universe '{}'", len(persisted), universe_label
        )
        return persisted

    async def run_cycle_async(self, *, context: MonitorRunContext | None = None, **kwargs) -> pd.DataFrame:
        """Execute a monitoring cycle off the main event loop."""

        bound_kwargs = {"context": context} if context else {}
        bound_kwargs.update(kwargs)
        loop = asyncio.get_running_loop()
        return await asyncio.to_thread(lambda: self.run_cycle(**bound_kwargs))

    # ------------------------- Internals -------------------------
    def _determine_universe_label(self, ctx: MonitorRunContext) -> str:
        if isinstance(ctx.universe, str) and ctx.universe.strip():
            return ctx.universe.strip()
        if ctx.watchlist:
            return "WATCHLIST"
        return self._settings.monitor_default_universe

    def _resolve_symbols(self, ctx: MonitorRunContext) -> list[str]:
        watchlist = self._normalize_watchlist(ctx.watchlist)
        if watchlist:
            return sorted(set(watchlist))
        universe = ctx.universe if ctx.universe is not None else self._settings.monitor_default_universe
        return resolve_universe(universe)

    def _normalize_watchlist(self, watchlist: Optional[str | Sequence[str] | Path]) -> list[str]:
        if watchlist is None:
            return []
        if isinstance(watchlist, (list, tuple, set)):
            return [str(item).strip() for item in watchlist if str(item).strip()]
        text = str(watchlist).strip()
        if not text:
            return []
        path = Path(text)
        if path.exists():
            content = path.read_text(encoding="utf-8").splitlines()
            return [line.strip() for line in content if line.strip()]
        return [part.strip() for part in text.split(",") if part.strip()]

    def _default_start(self) -> str:
        lookback = max(1, int(self.lookback_days))
        start_dt = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback)
        return start_dt.strftime("%Y-%m-%d")

    def _fetch_with_retry(self, *, symbols: Sequence[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.fetch_retries + 1):
            try:
                frame = self.provider.get_price_daily(symbols, start=start, end=end)
                if frame.empty:
                    logger.debug("No data returned for symbols {}", symbols)
                return frame
            except Exception as exc:  # pragma: no cover - defensive logging
                last_error = exc
                logger.warning(
                    "Data fetch attempt {}/{} failed: {}", attempt, self.fetch_retries, exc
                )
                if attempt < self.fetch_retries:
                    time.sleep(max(0.0, float(self.fetch_retry_delay)))
        if last_error is not None:
            logger.error("Market data fetch failed after retries: {}", last_error)
        return pd.DataFrame()

    def _build_run_version(self, intraday: bool) -> str:
        suffix = "intraday" if intraday else "eod"
        return f"{suffix}-{pd.Timestamp.utcnow().isoformat()}"

    def _apply_backoff(self, records: Sequence[SignalRecord]) -> list[SignalRecord]:
        if not records:
            return []
        if self.backoff is None:
            return list(records)

        universes = {record.universe or "" for record in records}
        state_map: dict[tuple[str, str, str], pd.Timestamp] = {}
        for universe in universes:
            state = self.db.read_monitor_state(universe=universe)
            if state.empty:
                continue
            for _, row in state.iterrows():
                state_map[(row.get("universe") or "", row["symbol"], row["rule"])] = pd.to_datetime(
                    row["last_triggered"]
                )
        filtered: list[SignalRecord] = []
        for record in records:
            key = (record.universe or "", record.symbol, record.rule)
            last_ts = state_map.get(key)
            if last_ts is not None and record.timestamp <= last_ts + self.backoff:
                logger.debug(
                    "Backoff suppressing alert for {}/{} at {} (last {})",
                    record.symbol,
                    record.rule,
                    record.timestamp,
                    last_ts,
                )
                continue
            filtered.append(record)
        return filtered

    def _update_state(self, persisted: pd.DataFrame) -> None:
        payload = persisted[["universe", "symbol", "rule", "timestamp", "value"]].copy()
        payload = payload.rename(columns={"timestamp": "last_triggered", "value": "last_value"})
        self.db.write_monitor_state(payload)

    def _dispatch_alerts(self, records: Sequence[SignalRecord]) -> None:
        if not records:
            return
        if self.alert_dispatcher is None:
            return
        try:
            self.alert_dispatcher.dispatch(records)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Alert dispatch failed: {}", exc)


class MonitorLoopRunner:
    """Async scheduler wrapper powering the monitoring loop."""

    def __init__(self, monitor_loop: MonitorLoop, *, timezone: Optional[str] = None) -> None:
        if AsyncIOScheduler is None or CronTrigger is None or IntervalTrigger is None:
            raise RuntimeError("APScheduler is required to use MonitorLoopRunner")
        self.loop = monitor_loop
        self.timezone_name = timezone or self.loop.settings.monitor_timezone
        self.scheduler: Optional[AsyncIOScheduler] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(
        self,
        *,
        intraday: bool = False,
        universe: Optional[str | Sequence[str]] = None,
        watchlist: Optional[str | Sequence[str] | Path] = None,
    ) -> None:
        tzinfo = self._resolve_timezone(self.timezone_name)
        scheduler = AsyncIOScheduler(timezone=tzinfo)
        self.scheduler = scheduler

        hour, minute = self._parse_eod_time(self.loop.settings.monitor_eod_time)
        scheduler.add_job(
            self._scheduled_job,
            CronTrigger(hour=hour, minute=minute, day_of_week="mon-fri", timezone=tzinfo),
            kwargs={
                "intraday": False,
                "universe": universe,
                "watchlist": watchlist,
            },
            id="monitor-eod",
            replace_existing=True,
        )

        intraday_minutes = self.loop.settings.monitor_intraday_interval_minutes
        if intraday and intraday_minutes and intraday_minutes > 0:
            scheduler.add_job(
                self._scheduled_job,
                IntervalTrigger(minutes=intraday_minutes, timezone=tzinfo),
                kwargs={
                    "intraday": True,
                    "universe": universe,
                    "watchlist": watchlist,
                },
                id="monitor-intraday",
                replace_existing=True,
            )
        scheduler.start()
        logger.info("Monitoring scheduler started (intraday={})", intraday)

        self._stop_event = asyncio.Event()
        try:
            await self._stop_event.wait()
        finally:
            scheduler.shutdown(wait=False)
            logger.info("Monitoring scheduler stopped")

    def stop(self) -> None:
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()

    async def _scheduled_job(
        self,
        *,
        intraday: bool,
        universe: Optional[str | Sequence[str]],
        watchlist: Optional[str | Sequence[str] | Path],
    ) -> None:
        context = MonitorRunContext(universe=universe, watchlist=watchlist, intraday=intraday)
        try:
            persisted = await self.loop.run_cycle_async(context=context)
            if not persisted.empty:
                logger.info("Scheduled monitoring run persisted {} alerts", len(persisted))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Scheduled monitoring run failed: {}", exc)

    @staticmethod
    def _parse_eod_time(value: str) -> tuple[int, int]:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid monitor_eod_time format: {value}")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid monitor_eod_time value: {value}")
        return hour, minute

    @staticmethod
    def _resolve_timezone(name: Optional[str]):
        if name and ZoneInfo:
            try:
                return ZoneInfo(name)
            except Exception:  # pragma: no cover - fallback to naive timezone
                logger.warning("Failed to resolve timezone '{}'; using system default", name)
        return None


__all__ = ["MonitorLoop", "MonitorLoopRunner", "MonitorRunContext"]
