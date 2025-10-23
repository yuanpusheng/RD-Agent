from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rdagent_china.alerts import AlertDispatcher
from rdagent_china.config import settings as global_settings
from rdagent_china.signals.rules import RulesEngine


class DummyDB:
    def read_monitor_state(self, universe: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["universe", "symbol", "rule", "last_notified"])

    def update_monitor_notification_state(self, updates):
        self.updates = list(updates)


class RecordingHTTPClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, json, headers=None, timeout=None):  # type: ignore[override]
        self.calls.append({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })

        class Response:
            status_code = 200

            def raise_for_status(self_inner):  # pragma: no cover - always OK
                return None

        return Response()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_sample_monitor_rules_and_subscriptions_route_alerts():
    root = _project_root()
    price_path = Path(__file__).parent / "fixtures" / "price_daily_sample.csv"
    frame = pd.read_csv(price_path)
    frame["date"] = pd.to_datetime(frame["date"])

    engine = RulesEngine.from_file(root / "examples" / "ashare_monitoring" / "monitor_rules.yaml")
    records = engine.evaluate(frame, universe="CSI300", config_version="sample", run_version="tests")
    triggered = [record for record in records if record.triggered]
    assert triggered, "sample configuration should emit at least one alert"

    settings = global_settings.model_copy(
        update={
            "monitor_alert_channels_enabled": ["feishu", "email", "slack", "wecom"],
            "monitor_alert_feishu_webhook": "https://example.com/feishu",
            "monitor_alert_email_webhook": "https://example.com/email",
            "monitor_alert_slack_webhook": "https://example.com/slack",
            "monitor_alert_wecom_webhook": "https://example.com/wecom",
            "monitor_alert_notification_cooldown_minutes": 0,
            "monitor_alert_subscriptions": [],
            "monitor_alert_subscriptions_path": root
            / "examples"
            / "ashare_monitoring"
            / "alert_subscriptions.yaml",
        }
    )

    client = RecordingHTTPClient()
    dispatcher = AlertDispatcher(settings=settings, db=DummyDB(), http_client=client)
    dispatcher.dispatch(triggered)

    urls = [call["url"] for call in client.calls]
    assert "https://example.com/feishu" in urls
    assert "https://example.com/email" in urls
    # only the breakout alert should trigger, so unrelated channels stay idle
    assert "https://example.com/slack" not in urls


def test_sample_universe_payload_lists_expected_symbols():
    root = _project_root()
    payload = json.loads((root / "examples" / "ashare_monitoring" / "universe_custom.json").read_text(encoding="utf-8"))

    assert payload["base_universe"] == "CSI300"
    assert "000001.SZ" in payload["include"]
    assert "watchlist-growth" in payload["watchlists"]
