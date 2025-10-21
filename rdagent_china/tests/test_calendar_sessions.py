from datetime import date

import pandas as pd

from rdagent_china.data.trading_calendar import ChinaAShareCalendar, ChinaAShareCalendarConfig


def test_sessions_across_month_boundary_with_may_day_holiday():
    cal = ChinaAShareCalendar(ChinaAShareCalendarConfig(prefer="builtin"))
    # Range spanning May 1st which should be a holiday by default override set
    sessions = cal.sessions("2024-04-29", "2024-05-03")
    # Exclude weekend (May 4-5) and May 1 holiday
    assert sessions == [
        date(2024, 4, 29),
        date(2024, 4, 30),
        date(2024, 5, 2),
        date(2024, 5, 3),
    ]


def test_sessions_across_year_boundary_with_new_year_holiday():
    cal = ChinaAShareCalendar(ChinaAShareCalendarConfig(prefer="builtin"))
    sessions = cal.sessions("2024-12-30", "2025-01-03")
    # Jan 1 should be holiday, skip weekend if any
    assert sessions == [
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]


def test_holiday_override_custom_exclusion():
    # Mark 2024-04-30 as an extra holiday via override
    cal = ChinaAShareCalendar(
        ChinaAShareCalendarConfig(prefer="builtin"),
        holiday_overrides=["2024-04-30"],
    )
    sessions = cal.sessions("2024-04-29", "2024-05-03")
    assert sessions == [
        date(2024, 4, 29),
        date(2024, 5, 2),
        date(2024, 5, 3),
    ]


def test_t_plus_offsets_weekend():
    cal = ChinaAShareCalendar(ChinaAShareCalendarConfig(prefer="builtin"))
    # Friday May 3, next trading day should be Monday May 6 (May 4-5 are weekend)
    t = cal.t_plus(date(2024, 5, 3), 1)
    assert t == date(2024, 5, 6)
