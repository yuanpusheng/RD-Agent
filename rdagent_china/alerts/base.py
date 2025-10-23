from __future__ import annotations

import abc
import base64
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import httpx
from loguru import logger

from rdagent_china.signals.base import SignalRecord


class SupportsPost(Protocol):
    """Protocol describing the HTTP client surface used by alert channels."""

    def post(
        self,
        url: str,
        *,
        json: Any,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:  # pragma: no cover - structural protocol
        ...


@dataclass(slots=True)
class AlertMessage:
    """Materialized payload delivered to downstream alert channels."""

    record: SignalRecord
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertChannel(abc.ABC):
    """Abstract base class for all alert delivery channels."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def send(self, message: AlertMessage) -> None:  # pragma: no cover - interface definition
        """Deliver the supplied :class:`AlertMessage` to the downstream channel."""


class WebhookAlertChannel(AlertChannel):
    """Alert channel implementation that POSTs JSON payloads to a webhook."""

    def __init__(
        self,
        *,
        name: str,
        webhook_url: str,
        client: SupportsPost,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(name=name)
        self.webhook_url = webhook_url
        self.client = client
        self.timeout = timeout

    def _build_headers(self, message: AlertMessage) -> dict[str, str]:  # pragma: no cover - simple default
        return {"Content-Type": "application/json; charset=utf-8"}

    @abc.abstractmethod
    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:  # pragma: no cover - channel-specific
        ...

    def send(self, message: AlertMessage) -> None:
        payload = self._build_payload(message)
        headers = self._build_headers(message)
        logger.debug("Dispatching %s alert to %s", self.name, self.webhook_url)
        response = self.client.post(self.webhook_url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()


class FeishuWebhookChannel(WebhookAlertChannel):
    """Feishu/Lark webhook integration."""

    def __init__(
        self,
        *,
        webhook_url: str,
        client: SupportsPost,
        secret: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(name="feishu", webhook_url=webhook_url, client=client, timeout=timeout)
        self.secret = secret

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        content = {
            "msg_type": "text",
            "content": {"text": f"{message.title}\n{message.body}"},
        }
        if self.secret:
            timestamp = str(int(time.time()))
            string_to_sign = f"{timestamp}\n{self.secret}"
            digest = hmac.new(self.secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).digest()
            content["timestamp"] = timestamp
            content["sign"] = base64.b64encode(digest).decode("utf-8")
        return content


class WeComWebhookChannel(WebhookAlertChannel):
    """Enterprise WeCom (WeChat Work) webhook adapter."""

    def __init__(
        self,
        *,
        webhook_url: str,
        client: SupportsPost,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(name="wecom", webhook_url=webhook_url, client=client, timeout=timeout)

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        return {
            "msgtype": "text",
            "text": {
                "content": f"{message.title}\n{message.body}",
            },
        }


class SlackWebhookChannel(WebhookAlertChannel):
    """Slack incoming webhook adapter."""

    def __init__(
        self,
        *,
        webhook_url: str,
        client: SupportsPost,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(name="slack", webhook_url=webhook_url, client=client, timeout=timeout)

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        return {
            "text": f"*{message.title}*\n{message.body}",
        }


class EmailWebhookChannel(WebhookAlertChannel):
    """Generic email webhook (e.g. serverless mail relays)."""

    def __init__(
        self,
        *,
        webhook_url: str,
        client: SupportsPost,
        secret: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(name="email", webhook_url=webhook_url, client=client, timeout=timeout)
        self.secret = secret

    def _build_payload(self, message: AlertMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subject": message.title,
            "body": message.body,
        }
        if message.metadata:
            payload.update(message.metadata)
        return payload

    def _build_headers(self, message: AlertMessage) -> dict[str, str]:
        headers = super()._build_headers(message)
        if self.secret:
            headers = dict(headers)
            headers["Authorization"] = f"Bearer {self.secret}"
        return headers


__all__ = [
    "AlertChannel",
    "AlertMessage",
    "EmailWebhookChannel",
    "FeishuWebhookChannel",
    "SlackWebhookChannel",
    "SupportsPost",
    "WeComWebhookChannel",
    "WebhookAlertChannel",
]
