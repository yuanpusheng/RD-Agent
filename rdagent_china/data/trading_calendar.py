from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

try:  # optional at runtime
    from zoneinfo import ZoneInfo  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - Python <3.9 fallback
    from pytz import timezone as ZoneInfo  # type: ignore

DateLike = Union[str, date, datetime]


CN_TZ = ZoneInfo("Asia/Shanghai")


def _to_date(d: DateLike) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    # string like 'YYYY-MM-DD'
    return pd.to_datetime(d).date()


def _daterange(start: date, end: date) -> List[date]:
    # inclusive endpoints
    return [d.date() for d in pd.date_range(start, end, freq="D")]


def _as_date_set(dates: Optional[Iterable[DateLike]]) -> Set[date]:
    if not dates:
        return set()
    return { _to_date(d) for d in dates }


def _default_cn_holidays_between(start: date, end: date) -> Set[date]:
    # minimal fixed-date holidays we know are non-trading: New Year (Jan 1), Labour Day (May 1), National Day (Oct 1)
    # Note: actual A-share holidays include multi-day periods and special make-up workdays; callers can override via parameters.
    hols: Set[date] = set()
    for y in range(start.year, end.year + 1):
        for md in [(1, 1), (5, 1), (10, 1)]:
            hols.add(date(y, md[0], md[1]))
    return {d for d in hols if start <= d <= end}


@dataclass
class ChinaAShareCalendarConfig:
    cache_dir: Optional[Path] = None
    prefer: str = "auto"  # auto | exchange_calendars | akshare | builtin


class ChinaAShareCalendar:
    """
    Lightweight A-share trading calendar.

    Features:
    - Weekend exclusion baseline
    - Optional integration with exchange_calendars or Akshare when available
    - Holiday overrides (add/remove)
    - Simple caching to avoid recomputing ranges
    - Session segmentation (morning/afternoon) and T+N utilities
    """

    open_morning = time(9, 30)
    close_morning = time(11, 30)
    open_afternoon = time(13, 0)
    close_afternoon = time(15, 0)

    def __init__(
        self,
        config: Optional[ChinaAShareCalendarConfig] = None,
        holiday_overrides: Optional[Iterable[DateLike]] = None,
        workday_overrides: Optional[Iterable[DateLike]] = None,
    ):
        self.config = config or ChinaAShareCalendarConfig()
        self._holiday_overrides: Set[date] = _as_date_set(holiday_overrides)
        self._workday_overrides: Set[date] = _as_date_set(workday_overrides)
        # simple in-memory cache: key = (start,end,hol_hash,work_hash,prefer)
        self._cache: dict[Tuple[date, date, int, int, str], List[date]] = {}
        # disk cache for akshare/exchange results
        self._disk_cache_path = None
        if self.config.cache_dir:
            self._disk_cache_path = Path(self.config.cache_dir) / "cn_trading_days.csv"
            self._disk_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- public API ----------
    def sessions(self, start: DateLike, end: DateLike) -> List[date]:
        s = _to_date(start)
        e = _to_date(end)
        if s > e:
            s, e = e, s
        hol_hash = hash(tuple(sorted(self._holiday_overrides)))
        work_hash = hash(tuple(sorted(self._workday_overrides)))
        key = (s, e, hol_hash, work_hash, self.config.prefer)
        if key in self._cache:
            return self._cache[key][:]

        days = self._sessions_impl(s, e)
        self._cache[key] = days
        return days[:]

    def is_trading_day(self, d: DateLike) -> bool:
        dd = _to_date(d)
        return dd in set(self.sessions(dd, dd))

    def next_session(self, d: DateLike) -> date:
        dd = _to_date(d)
        rng = self.sessions(dd, dd + timedelta(days=10))
        for x in rng:
            if x > dd:
                return x
        # extend search if needed
        return self.sessions(dd + timedelta(days=1), dd + timedelta(days=30))[0]

    def prev_session(self, d: DateLike) -> date:
        dd = _to_date(d)
        rng = self.sessions(dd - timedelta(days=10), dd)
        prev = dd
        for x in rng:
            if x < dd:
                prev = x
        if prev == dd:
            # need earlier
            return self.sessions(dd - timedelta(days=30), dd - timedelta(days=1))[-1]
        return prev

    def t_plus(self, d: DateLike, n: int = 1) -> date:
        if n == 0:
            return _to_date(d)
        dd = _to_date(d)
        if n > 0:
            days = self.sessions(dd, dd + timedelta(days=365))
            idx = days.index(dd) if dd in days else None
            if idx is None:
                # find first > dd
                days = [x for x in days if x > dd]
                if not days:
                    # extremely unlikely
                    return self.sessions(dd + timedelta(days=1), dd + timedelta(days=400))[n - 1]
                return days[n - 1]
            return days[idx + n]
        # n < 0
        days = self.sessions(dd - timedelta(days=365), dd)
        idx = days.index(dd) if dd in days else len(days)
        return days[idx + n]

    def session_segments(self, d: DateLike) -> List[Tuple[datetime, datetime]]:
        dd = _to_date(d)
        # Raise if not trading day? Keep permissive and compute for date anyway
        seg0_start = datetime.combine(dd, self.open_morning).replace(tzinfo=CN_TZ)
        seg0_end = datetime.combine(dd, self.close_morning).replace(tzinfo=CN_TZ)
        seg1_start = datetime.combine(dd, self.open_afternoon).replace(tzinfo=CN_TZ)
        seg1_end = datetime.combine(dd, self.close_afternoon).replace(tzinfo=CN_TZ)
        return [
            (seg0_start, seg0_end),
            (seg1_start, seg1_end),
        ]

    def session_open_close(self, d: DateLike) -> Tuple[datetime, datetime]:
        segs = self.session_segments(d)
        return segs[0][0], segs[-1][1]

    # ---------- internals ----------
    def _sessions_impl(self, start: date, end: date) -> List[date]:
        # Try preferred providers first
        prefer_order = self._resolve_prefer_order()
        for provider in prefer_order:
            try:
                if provider == "exchange_calendars":
                    days = self._sessions_exchange_calendars(start, end)
                elif provider == "akshare":
                    days = self._sessions_akshare(start, end)
                else:
                    days = self._sessions_builtin(start, end)
                # Apply overrides
                days = self._apply_overrides(days)
                return days
            except Exception:
                # fall through to next provider
                continue
        # Fallback
        days = self._sessions_builtin(start, end)
        days = self._apply_overrides(days)
        return days

    def _resolve_prefer_order(self) -> Sequence[str]:
        if self.config.prefer == "auto":
            return ["exchange_calendars", "akshare", "builtin"]
        if self.config.prefer in ("exchange_calendars", "akshare", "builtin"):
            return [self.config.prefer, "builtin"]
        return ["builtin"]

    def _apply_overrides(self, days: List[date]) -> List[date]:
        s, e = days[0], days[-1]
        hols = _default_cn_holidays_between(s, e) | self._holiday_overrides
        works = self._workday_overrides
        base = [d for d in days if d not in hols]
        if works:
            base = sorted(set(base) | {d for d in works if s <= d <= e})
        return base

    def _sessions_builtin(self, start: date, end: date) -> List[date]:
        bdays = pd.bdate_range(start=start, end=end)
        days = [d.date() for d in bdays]
        # builtin already excludes weekends; leave holidays to overrides
        return days

    def _sessions_exchange_calendars(self, start: date, end: date) -> List[date]:  # pragma: no cover - optional dep
        import exchange_calendars as ec  # type: ignore

        # XSHG for Shanghai, which aligns with A-share
        cal = ec.get_calendar("XSHG")
        # sessions_in_range returns pandas.DatetimeIndex (UTC)
        sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        days = [ts.tz_localize("UTC").tz_convert(CN_TZ).date() for ts in sessions]
        # persist minimal disk cache if requested
        self._append_disk_cache(days)
        return days

    def _sessions_akshare(self, start: date, end: date) -> List[date]:  # pragma: no cover - optional dep and network
        import akshare as ak  # type: ignore

        df = ak.tool_trade_date_hist_sina()
        df = df[(df["trade_date"] >= str(start)) & (df["trade_date"] <= str(end))]
        days = pd.to_datetime(df["trade_date"]).dt.date.tolist()
        self._append_disk_cache(days)
        return days

    def _append_disk_cache(self, days: List[date]):  # pragma: no cover - best-effort cache
        if not self._disk_cache_path:
            return
        try:
            existing: Set[date] = set()
            if self._disk_cache_path.exists():
                s = pd.read_csv(self._disk_cache_path)
                existing = set(pd.to_datetime(s["date"]).dt.date.tolist())
            merged = sorted(existing | set(days))
            pd.DataFrame({"date": merged}).to_csv(self._disk_cache_path, index=False)
        except Exception:
            # ignore cache errors
            pass


@lru_cache(maxsize=16)
def get_default_calendar() -> ChinaAShareCalendar:
    return ChinaAShareCalendar()
