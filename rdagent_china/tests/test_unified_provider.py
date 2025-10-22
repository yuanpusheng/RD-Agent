from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from rdagent_china.cli import app
from rdagent_china.config import settings
from rdagent_china.data.provider import UnifiedDataProvider
from rdagent_china.db import get_db


runner = CliRunner()


class DummyAdapterOK:
    def __init__(self, name: str, data_map: dict[str, pd.DataFrame]):
        self._name = name
        self._data_map = data_map

    def price_daily(self, symbols, start=None, end=None):
        frames = []
        for s in symbols:
            if s in self._data_map:
                frames.append(self._data_map[s])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class DummyAdapterFail:
    def price_daily(self, symbols, start=None, end=None):
        raise RuntimeError("adapter failure")


def _df(sym: str, days=2):
    dates = pd.date_range("2024-01-02", periods=days, freq="D")
    return pd.DataFrame({
        "symbol": [sym] * days,
        "date": dates,
        "open": [1.0] * days,
        "high": [2.0] * days,
        "low": [0.5] * days,
        "close": [1.5] * days,
        "volume": [100] * days,
    })


def test_provider_fallback_ordering(monkeypatch, tmp_path):
    db_path = tmp_path / "market.duckdb"
    monkeypatch.setattr(settings, "duckdb_path", db_path, raising=False)
    # Construct provider with dummy adapters: tushare fails, akshare ok
    provider = UnifiedDataProvider(db=get_db())
    provider._adapters = [
        ("tushare", DummyAdapterFail()),
        ("akshare", DummyAdapterOK("akshare", {"000001": _df("000001"), "000002": _df("000002")})),
    ]
    out = provider.get_price_daily(["000001", "000002"], start="2024-01-01", end="2024-01-31")
    assert len(out) == 4
    # ensure source column set to akshare
    assert out["source"].unique().tolist() == ["akshare"]


def test_provider_merges_across_sources(monkeypatch, tmp_path):
    db_path = tmp_path / "market.duckdb"
    monkeypatch.setattr(settings, "duckdb_path", db_path, raising=False)
    provider = UnifiedDataProvider(db=get_db())
    # tushare returns only for 000001, akshare returns for 000002
    provider._adapters = [
        ("tushare", DummyAdapterOK("tushare", {"000001": _df("000001")})),
        ("akshare", DummyAdapterOK("akshare", {"000002": _df("000002")})),
    ]
    out = provider.get_price_daily(["000001", "000002"], start="2024-01-01", end="2024-01-31")
    assert len(out) == 4
    # verify each symbol came from its respective source
    g = out.groupby("symbol")["source"].unique().to_dict()
    assert g["000001"].tolist() == ["tushare"]
    assert g["000002"].tolist() == ["akshare"]
