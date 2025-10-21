from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal, Tuple

from . import dates as legacy_dates
from rdagent_china.data.trading_calendar import ChinaAShareCalendar, CN_TZ


@dataclass
class SessionAlignment:
    session_date: date
    open: datetime
    close: datetime
    in_session: bool


def to_cn_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=CN_TZ)
    return dt.astimezone(CN_TZ)


def get_session_bounds(dt: datetime, cal: ChinaAShareCalendar) -> SessionAlignment:
    dt_cn = to_cn_tz(dt)
    d = dt_cn.date()
    if not cal.is_trading_day(d):
        # map to nearest previous trading day
        d = cal.prev_session(d)
    o, c = cal.session_open_close(d)
    in_sess = o <= dt_cn <= c and not _is_lunch_break(dt_cn, cal)
    return SessionAlignment(session_date=d, open=o, close=c, in_session=in_sess)


def align_to_session(
    dt: datetime,
    cal: ChinaAShareCalendar,
    boundary: Literal["open", "close"] = "open",
    direction: Literal["previous", "next", "auto"] = "auto",
) -> datetime:
    """
    Align timestamp to nearest session open/close.

    - If direction=="auto":
        - If inside a session: return dt rounded to boundary within same session (no change if already past boundary)
        - If outside sessions: snap to next session boundary for 'next', previous for 'previous' based on time.
    """
    dt_cn = to_cn_tz(dt)
    sess = get_session_bounds(dt_cn, cal)

    # If during lunch break, treat as in-session but snap forward/back according to boundary
    if direction == "previous":
        target_day = sess.session_date if dt_cn >= sess.open else cal.prev_session(sess.session_date)
    elif direction == "next":
        if dt_cn <= sess.close:
            target_day = sess.session_date
        else:
            target_day = cal.next_session(sess.session_date)
    else:  # auto
        if dt_cn < sess.open:
            target_day = sess.session_date
        elif dt_cn > sess.close:
            target_day = cal.next_session(sess.session_date)
        else:
            target_day = sess.session_date

    o, c = cal.session_open_close(target_day)
    return o if boundary == "open" else c


def is_trading_time(dt: datetime, cal: ChinaAShareCalendar) -> bool:
    dt_cn = to_cn_tz(dt)
    if not cal.is_trading_day(dt_cn.date()):
        return False
    # in either segment
    segs = cal.session_segments(dt_cn)
    return (segs[0][0] <= dt_cn <= segs[0][1]) or (segs[1][0] <= dt_cn <= segs[1][1])


def _is_lunch_break(dt_cn: datetime, cal: ChinaAShareCalendar) -> bool:
    segs = cal.session_segments(dt_cn)
    return segs[0][1] < dt_cn < segs[1][0]
