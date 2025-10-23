from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from rdagent_china.alerts import AlertDispatcher
from rdagent_china.config import settings as global_settings
from rdagent_china.db import Database
from rdagent_china.signals.base import SignalRecord


@dataclass
class MockResponse:
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error: {self.status_code}")


class MockHTTPClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        json: Any,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> MockResponse:
        self.calls.append({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return MockResponse()


def _sample_record(symbol: str, *, timestamp: str) -> SignalRecord:
    return SignalRecord(
        universe="CUSTOM",
        symbol=symbol,
        timestamp=pd.Timestamp(timestamp),
        rule="volume_spike_alert",
        label="VOLUME_SPIKE",
        severity="medium",
        triggered=True,
        value=3.2,
        config_version="monitor-test",
        run_version="run-123",
        signals={"ratio": 3.2},
    )


def test_alert_dispatcher_rate_limits(tmp_path):
    db_path = tmp_path / "monitor.duckdb"
    db = Database(db_path)
    db.init()

    initial_state = pd.DataFrame(
        [
            {
                "universe": "CUSTOM",
                "symbol": "AAA",
                "rule": "volume_spike_alert",
                "last_triggered": pd.Timestamp("2024-01-01T09:30:00Z"),
                "last_value": 2.5,
            }
        ]
    )
    db.write_monitor_state(initial_state)

    settings = global_settings.model_copy(
        update={
            "monitor_alert_channels_enabled": ["feishu"],
            "monitor_alert_feishu_webhook": "https://example.com/feishu",
            "monitor_alert_notification_cooldown_minutes": 60,
            "monitor_alert_subscriptions": [],
        }
    )

    client = MockHTTPClient()
    dispatcher = AlertDispatcher(settings=settings, db=db, http_client=client)

    record = _sample_record("AAA", timestamp="2024-01-02T09:30:00Z")
    dispatcher.dispatch([record])
    assert len(client.calls) == 1

    state = db.read_monitor_state(universe="CUSTOM", symbols=["AAA"], rules=["volume_spike_alert"])
    assert not state.empty
    assert pd.to_datetime(state.iloc[0]["last_notified"]) == record.timestamp

    dispatcher.dispatch([record])
    assert len(client.calls) == 1, "Cooldown should suppress duplicate notifications"


def test_alert_dispatcher_respects_subscriptions(tmp_path):
    db_path = tmp_path / "monitor.duckdb"
    db = Database(db_path)
    db.init()

    seed_state = pd.DataFrame(
        [
            {
                "universe": "CUSTOM",
                "symbol": "AAA",
                "rule": "volume_spike_alert",
                "last_triggered": pd.Timestamp("2024-01-01T09:30:00Z"),
                "last_value": None,
            },
            {
                "universe": "CUSTOM",
                "symbol": "BBB",
                "rule": "volume_spike_alert",
                "last_triggered": pd.Timestamp("2024-01-01T09:45:00Z"),
                "last_value": None,
            },
        ]
    )
    db.write_monitor_state(seed_state)

    settings = global_settings.model_copy(
        update={
            "monitor_alert_channels_enabled": ["feishu"],
            "monitor_alert_feishu_webhook": "https://example.com/feishu",
            "monitor_alert_notification_cooldown_minutes": 0,
            "monitor_alert_subscriptions": [
                {"rule": "volume_spike_alert", "symbols": ["AAA"], "channels": ["feishu"]},
            ],
        }
    )

    client = MockHTTPClient()
    dispatcher = AlertDispatcher(settings=settings, db=db, http_client=client)

    matching = _sample_record("AAA", timestamp="2024-01-02T09:30:00Z")
    non_matching = _sample_record("BBB", timestamp="2024-01-02T09:30:00Z")

    dispatcher.dispatch([matching, non_matching])
    assert len(client.calls) == 1
    payload = client.calls[0]["json"]
    assert "AAA" in payload["content"]["text"]

    state = db.read_monitor_state(universe="CUSTOM", symbols=["AAA"], rules=["volume_spike_alert"])
    assert not state.empty
    assert pd.to_datetime(state.iloc[0]["last_notified"]) == matching.timestamp

    other_state = db.read_monitor_state(universe="CUSTOM", symbols=["BBB"], rules=["volume_spike_alert"])
    assert pd.isna(other_state.iloc[0]["last_notified"])
