from __future__ import annotations

import pandas as pd

from rdagent_china.data.adapters.akshare_adapter import AkshareAdapter


def _adapter_frame(symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.1, 10.4],
            "volume": [1_000_000, 1_100_000],
        }
    )


class DummyAkshareClient:
    def __init__(self, responses: dict[str, pd.DataFrame | Exception]):
        self._responses = responses
        self.calls: list[tuple[str, str | None, str | None]] = []

    def price_daily(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        payload = self._responses.get(symbol)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            raise KeyError(symbol)
        return payload.copy()


def test_akshare_adapter_vectorizes_price_daily(monkeypatch):
    client = DummyAkshareClient(
        {
            "000001.SZ": _adapter_frame("000001.SZ"),
            "600519.SH": _adapter_frame("600519.SH"),
        }
    )
    monkeypatch.setattr("rdagent_china.data.adapters.akshare_adapter.AkshareClient", lambda *_args, **_kwargs: client)

    adapter = AkshareAdapter()
    frame = adapter.price_daily(["000001.SZ", "600519.SH"], start="2024-01-01", end="2024-01-31")

    assert set(frame["symbol"]) == {"000001.SZ", "600519.SH"}
    assert pd.api.types.is_datetime64_dtype(frame["date"])
    assert client.calls == [
        ("000001.SZ", "2024-01-01", "2024-01-31"),
        ("600519.SH", "2024-01-01", "2024-01-31"),
    ]


def test_akshare_adapter_handles_failed_symbols(monkeypatch):
    client = DummyAkshareClient(
        {
            "000001.SZ": _adapter_frame("000001.SZ"),
            "600519.SH": RuntimeError("boom"),
        }
    )
    monkeypatch.setattr("rdagent_china.data.adapters.akshare_adapter.AkshareClient", lambda *_args, **_kwargs: client)

    adapter = AkshareAdapter()
    frame = adapter.price_daily(["000001.SZ", "600519.SH"])

    assert list(frame["symbol"].unique()) == ["000001.SZ"], "failing symbols should be skipped gracefully"


def test_akshare_adapter_empty_symbols_returns_schema():
    adapter = AkshareAdapter()
    frame = adapter.price_daily([])
    assert list(frame.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]
    assert frame.empty
