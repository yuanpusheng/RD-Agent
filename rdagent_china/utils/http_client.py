from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import httpx
from loguru import logger


@dataclass
class HttpClientConfig:
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    retry_statuses: Sequence[int] = field(default_factory=lambda: (429, 500, 502, 503, 504))
    headers: Optional[Mapping[str, str]] = None
    base_url: Optional[str] = None


class ResilientHTTPClient:
    """
    Simple resilient HTTP client built on top of httpx with manual retry/backoff.

    - Retries on network errors and 5xx/429 statuses
    - Exponential backoff with jitter
    - Shared connection pool via httpx.Client
    """

    def __init__(self, config: Optional[HttpClientConfig] = None):
        self.config = config or HttpClientConfig()
        self._client = httpx.Client(
            base_url=self.config.base_url or "",
            timeout=self.config.timeout,
            headers=self.config.headers,
        )

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self._client.request(method, url, **kwargs)
                if resp.status_code in self.config.retry_statuses:
                    raise httpx.HTTPStatusError(
                        f"retryable status: {resp.status_code}", request=resp.request, response=resp
                    )
                return resp
            except Exception as e:  # pragma: no cover - only triggers on runtime failures
                last_exc = e
                if attempt >= self.config.max_retries:
                    break
                wait = self.config.backoff_factor * (2 ** (attempt - 1))
                # add +/- up to 20% jitter
                jitter = wait * 0.2
                sleep_s = wait + (jitter if (attempt % 2 == 0) else -jitter)
                logger.warning(f"HTTP {method} {url} failed on attempt {attempt}/{self.config.max_retries}: {e}; sleeping {sleep_s:.2f}s")
                time.sleep(max(0.0, sleep_s))
        assert last_exc is not None
        raise last_exc

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def close(self):  # pragma: no cover - simple wrapper
        self._client.close()
