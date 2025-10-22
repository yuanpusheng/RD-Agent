from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdagent_china.db import Database
from rdagent_china.monitor import MonitorLoop, MonitorRunContext
from rdagent_china.signals.rules import RulesEngine


class DummyProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_price_daily(self, symbols, start=None, end=None):
        return self._frame[self._frame["symbol"].isin(symbols)].copy()


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * len(dates),
            "date": dates,
            "open": [10.0, 10.2, 10.3, 10.4, 10.5],
            "high": [10.2, 10.4, 10.5, 10.6, 11.0],
            "low": [9.9, 10.0, 10.2, 10.3, 10.4],
            "close": [10.1, 10.3, 10.35, 10.4, 11.5],
            "volume": [1_000_000, 1_100_000, 1_050_000, 1_200_000, 4_200_000],
        }
    )


def _rules_config() -> dict:
    return {
        "signals": {
            "volume_watch": {
                "indicator": "volume_spike",
                "params": {"window": 3, "threshold": 3.0},
            }
        },
        "rules": [
            {
                "name": "volume_spike_alert",
                "any": [
                    {"signal": "volume_watch", "field": "spike", "operator": "is_true"},
                ],
                "emit": {"label": "VOLUME_SPIKE_ALERT", "severity": "medium"},
                "value": {"signal": "volume_watch", "field": "ratio"},
            }
        ],
    }


def test_monitor_loop_persists_and_deduplicates(tmp_path: Path):
    frame = _sample_frame()
    provider = DummyProvider(frame)
    rules_engine = RulesEngine.from_dict(_rules_config())

    db_path = tmp_path / "monitor.duckdb"
    db = Database(db_path)
    db.init()

    loop = MonitorLoop(
        db=db,
        provider=provider,
        rules_engine=rules_engine,
        backoff_minutes=1_440,
        lookback_days=10,
    )
    context = MonitorRunContext(
        universe="CUSTOM",
        watchlist=["AAA"],
        intraday=False,
        start="2024-01-01",
        end="2024-01-10",
    )

    first = loop.run_cycle(context=context)
    assert not first.empty
    assert set(first["rule"]) == {"volume_spike_alert"}
    assert first.iloc[0]["universe"] == "CUSTOM"
    assert first.iloc[0]["triggered"]

    second = loop.run_cycle(context=context)
    assert second.empty, "Backoff should suppress repeated alerts"

    state = db.read_monitor_state(universe="CUSTOM")
    assert not state.empty
    assert state.iloc[0]["rule"] == "volume_spike_alert"
    assert pd.to_datetime(state.iloc[0]["last_triggered"]) == pd.to_datetime(first.iloc[0]["timestamp"])
    assert state.iloc[0]["last_value"] is not None

    persisted = db.read_signals(universe="CUSTOM")
    assert len(persisted) == len(first)
    assert "volume_watch" in persisted.iloc[0]["signals"]
