from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import httpx
import pandas as pd
import yaml
from loguru import logger

from rdagent_china.alerts.base import (
    AlertChannel,
    AlertMessage,
    EmailWebhookChannel,
    FeishuWebhookChannel,
    SlackWebhookChannel,
    SupportsPost,
    WeComWebhookChannel,
)
from rdagent_china.config import Settings
from rdagent_china.db import Database
from rdagent_china.signals.base import SignalRecord


@dataclass(frozen=True, slots=True)
class SubscriptionSpec:
    """Normalized representation of a subscription rule."""

    rule: str | None
    symbols: frozenset[str] | None
    universes: frozenset[str] | None
    channels: frozenset[str] | None

    def matches(self, record: SignalRecord) -> bool:
        if self.rule and self.rule not in {"*", record.rule}:
            return False
        if self.symbols and record.symbol not in self.symbols:
            return False
        universe = record.universe or ""
        if self.universes and universe not in self.universes:
            return False
        return True


class AlertSubscriptionIndex:
    """Helper that resolves channel subscriptions for a signal record."""

    def __init__(self, specs: Sequence[SubscriptionSpec], available_channels: Iterable[str]) -> None:
        self._specs = tuple(specs)
        self._available = {channel.lower() for channel in available_channels}
        self._has_rules = bool(self._specs)

    def channels_for(self, record: SignalRecord) -> set[str]:
        if not self._specs:
            return set(self._available)
        matched: set[str] = set()
        for spec in self._specs:
            if not spec.matches(record):
                continue
            if spec.channels:
                matched.update({channel for channel in spec.channels if channel in self._available})
            else:
                matched.update(self._available)
        if matched:
            return matched
        if self._has_rules:
            return set()
        return set(self._available)


class AlertDispatcher:
    """Dispatch alert notifications across configured channels with throttling."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        http_client: SupportsPost | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        timeout = float(settings.monitor_alert_http_timeout_seconds)
        self._client = http_client or httpx.Client(timeout=timeout)
        self._channels: dict[str, AlertChannel] = self._build_channels(timeout=timeout)
        self._subscriptions = self._build_subscription_index()
        cooldown = settings.monitor_alert_notification_cooldown_minutes
        self._cooldown = pd.to_timedelta(cooldown, unit="minutes") if cooldown and cooldown > 0 else None
        self._last_notified = self._load_last_notified()

    @property
    def channels(self) -> dict[str, AlertChannel]:  # pragma: no cover - simple accessor
        return self._channels

    def dispatch(self, records: Sequence[SignalRecord]) -> None:
        if not records or not self._channels:
            return
        updates: list[tuple[Optional[str], str, str, pd.Timestamp]] = []
        for record in records:
            if not record.triggered:
                continue
            channels = self._subscriptions.channels_for(record)
            if not channels:
                logger.debug("No alert subscriptions matched for %s/%s", record.symbol, record.rule)
                continue
            if self._is_rate_limited(record):
                logger.debug(
                    "Rate limiter suppressed alert for %s/%s at %s",
                    record.symbol,
                    record.rule,
                    record.timestamp,
                )
                continue
            message = self._build_message(record)
            delivered = False
            for channel_name in channels:
                channel = self._channels.get(channel_name)
                if channel is None:
                    continue
                try:
                    channel.send(message)
                    delivered = True
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Dispatch via channel '%s' failed for %s/%s: %s",
                        channel_name,
                        record.symbol,
                        record.rule,
                        exc,
                    )
            if delivered:
                ts = pd.to_datetime(record.timestamp)
                updates.append((record.universe, record.symbol, record.rule, ts))
                self._last_notified[self._key(record)] = ts
        if updates:
            self._persist_notification_updates(updates)

    def _build_channels(self, *, timeout: float) -> dict[str, AlertChannel]:
        enabled = {channel.lower() for channel in self._settings.monitor_alert_channels_enabled}
        if not enabled:
            enabled = {"feishu"}
        channels: dict[str, AlertChannel] = {}
        client = self._client
        if "feishu" in enabled and self._settings.monitor_alert_feishu_webhook:
            channels["feishu"] = FeishuWebhookChannel(
                webhook_url=self._settings.monitor_alert_feishu_webhook,
                secret=self._settings.monitor_alert_feishu_secret,
                client=client,
                timeout=timeout,
            )
        if "wecom" in enabled and self._settings.monitor_alert_wecom_webhook:
            channels["wecom"] = WeComWebhookChannel(
                webhook_url=self._settings.monitor_alert_wecom_webhook,
                client=client,
                timeout=timeout,
            )
        if "slack" in enabled and self._settings.monitor_alert_slack_webhook:
            channels["slack"] = SlackWebhookChannel(
                webhook_url=self._settings.monitor_alert_slack_webhook,
                client=client,
                timeout=timeout,
            )
        if "email" in enabled and self._settings.monitor_alert_email_webhook:
            channels["email"] = EmailWebhookChannel(
                webhook_url=self._settings.monitor_alert_email_webhook,
                client=client,
                secret=self._settings.monitor_alert_email_secret,
                timeout=timeout,
            )
        return channels

    def _build_subscription_index(self) -> AlertSubscriptionIndex:
        specs = self._load_subscription_specs()
        return AlertSubscriptionIndex(specs, self._channels.keys())

    def _load_subscription_specs(self) -> list[SubscriptionSpec]:
        raw_entries: list[Any] = []
        raw_entries.extend(self._settings.monitor_alert_subscriptions or [])
        path = self._settings.monitor_alert_subscriptions_path
        if path:
            try:
                resolved = Path(path)
                if not resolved.is_absolute():
                    resolved = Path.cwd() / resolved
                if resolved.exists():
                    with resolved.open("r", encoding="utf-8") as handle:
                        loaded = yaml.safe_load(handle) or []
                        raw_entries.extend(self._normalize_loaded_entries(loaded))
                else:
                    logger.warning("Alert subscription file '%s' does not exist", resolved)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to load alert subscription config: %s", exc)
        specs: list[SubscriptionSpec] = []
        for entry in raw_entries:
            spec = self._parse_subscription_entry(entry)
            if spec is not None:
                specs.append(spec)
        return specs

    def _normalize_loaded_entries(self, loaded: Any) -> list[Any]:
        if isinstance(loaded, list):
            return list(loaded)
        if isinstance(loaded, dict):
            if "subscriptions" in loaded and isinstance(loaded["subscriptions"], list):
                return list(loaded["subscriptions"])
            entries: list[Any] = []
            for rule, payload in loaded.items():
                if isinstance(payload, dict):
                    candidate = dict(payload)
                    candidate.setdefault("rule", rule)
                    entries.append(candidate)
            return entries
        return []

    def _parse_subscription_entry(self, entry: Any) -> SubscriptionSpec | None:
        if not isinstance(entry, dict):
            return None
        rule = self._coerce_rule(entry.get("rule"))
        symbols = self._coerce_token_set(entry.get("symbols"))
        universes = self._coerce_token_set(entry.get("universes") or entry.get("universe"))
        channels = self._coerce_token_set(entry.get("channels"), to_lower=True)
        return SubscriptionSpec(rule=rule, symbols=symbols, universes=universes, channels=channels)

    def _coerce_rule(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_token_set(self, value: Any, *, to_lower: bool = False) -> frozenset[str] | None:
        if value is None:
            return None
        tokens: list[str]
        if isinstance(value, str):
            tokens = [item.strip() for item in value.split(",") if item.strip()]
        else:
            try:
                tokens = [str(item).strip() for item in value if str(item).strip()]
            except TypeError:
                tokens = []
        if not tokens:
            return None
        normalized: list[str] = []
        has_wildcard = False
        for token in tokens:
            lower = token.lower()
            if lower in {"*", "all"}:
                has_wildcard = True
                continue
            normalized.append(lower if to_lower else token)
        if has_wildcard or not normalized:
            return None
        return frozenset(normalized)

    def _build_message(self, record: SignalRecord) -> AlertMessage:
        timestamp = pd.to_datetime(record.timestamp).isoformat()
        universe_part = f"Universe: {record.universe}\n" if record.universe else ""
        value_part = f"Value: {record.value}\n" if record.value is not None else ""
        config_part = f"Config: {record.config_version}\n" if record.config_version else ""
        run_part = f"Run: {record.run_version}\n" if record.run_version else ""
        signals_part = ""
        if record.signals:
            try:
                serialized = json.dumps(record.signals, ensure_ascii=False, default=str)
            except Exception:  # pragma: no cover - defensive fallback
                serialized = str(record.signals)
            signals_part = f"Signals: {serialized}\n"
        body = (
            f"Symbol: {record.symbol}\n"
            f"Rule: {record.rule}\n"
            f"Severity: {record.severity}\n"
            f"Timestamp: {timestamp}\n"
            f"Triggered: {record.triggered}\n"
            f"{universe_part}{value_part}{config_part}{run_part}{signals_part}"
        ).strip()
        title = f"{record.label} · {record.symbol}"
        return AlertMessage(record=record, title=title, body=body)

    def _is_rate_limited(self, record: SignalRecord) -> bool:
        if self._cooldown is None:
            return False
        key = self._key(record)
        previous = self._last_notified.get(key)
        if previous is None:
            return False
        current = pd.to_datetime(record.timestamp)
        return current <= previous + self._cooldown

    def _persist_notification_updates(
        self, updates: Sequence[tuple[Optional[str], str, str, pd.Timestamp]]
    ) -> None:
        if not updates:
            return
        try:
            self._db.update_monitor_notification_state(updates)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to persist monitor notification state: %s", exc)

    def _load_last_notified(self) -> dict[tuple[str | None, str, str], pd.Timestamp]:
        state = self._db.read_monitor_state()
        if state.empty or "last_notified" not in state.columns:
            return {}
        mapping: dict[tuple[str | None, str, str], pd.Timestamp] = {}
        for _, row in state.iterrows():
            value = row.get("last_notified")
            if pd.isna(value):
                continue
            universe_raw = row.get("universe")
            universe_key: str | None
            if pd.isna(universe_raw):
                universe_key = None
            else:
                universe_key = str(universe_raw)
            symbol_raw = row.get("symbol")
            rule_raw = row.get("rule")
            if symbol_raw is None or rule_raw is None:
                continue
            mapping[(universe_key, str(symbol_raw), str(rule_raw))] = pd.to_datetime(value)
        return mapping

    def _key(self, record: SignalRecord) -> tuple[str | None, str, str]:
        return (record.universe, record.symbol, record.rule)


__all__ = ["AlertDispatcher"]
