from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdagent_china.db import Database
from rdagent_china.signals import RulesEngine, load_config, persist_signal_records, records_to_dataframe


def _price_universe() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * len(dates),
            "date": dates,
            "open": [10.0, 10.1, 10.2, 10.3, 10.5, 10.9],
            "close": [10.1, 10.25, 10.4, 10.6, 10.8, 11.6],
            "volume": [1_000_000, 1_050_000, 1_020_000, 1_150_000, 1_300_000, 5_200_000],
        }
    )


def test_rules_engine_evaluates_indicator_and_event_conditions():
    frame = _price_universe()
    config = {
        "signals": {
            "trend": {
                "indicator": "ma_cross",
                "params": {"short_window": 2, "long_window": 4},
            },
            "volume_signal": {
                "indicator": "volume_spike",
                "params": {"window": 3, "threshold": 2.0},
            },
        },
        "rules": [
            {
                "name": "bullish_drive",
                "all": [
                    {"signal": "trend", "field": "signal", "operator": "eq", "value": 1.0},
                    {"signal": "volume_signal", "field": "spike", "operator": "is_true"},
                ],
                "emit": {"label": "BULLISH_DRIVE", "severity": "high"},
                "value": {"signal": "volume_signal", "field": "ratio"},
            }
        ],
    }

    engine = RulesEngine.from_dict(config)
    records = engine.evaluate(frame, universe="ASHARE", config_version="v1", run_version="r1")

    assert len(records) == 1
    record = records[0]
    assert record.rule == "bullish_drive"
    assert record.label == "BULLISH_DRIVE"
    assert record.universe == "ASHARE"
    assert record.config_version == "v1"
    assert record.run_version == "r1"
    assert record.signals["volume_signal"]["spike"] is True
    assert record.value and record.value > 2

    df = records_to_dataframe(records)
    assert set(["universe", "symbol", "timestamp", "rule", "signals"]).issubset(df.columns)


def test_rules_engine_loads_yaml_config(tmp_path: Path):
    config_path = tmp_path / "signals.yaml"
    config_path.write_text(
        """
        signals:
          gap_checker:
            type: event
            event: gap_open
            params:
              threshold: 0.03
        rules:
          - name: gap_up_alert
            any:
              - signal: gap_checker
                field: gap_up
                operator: is_true
            emit:
              label: GAP_UP_ALERT
              severity: medium
            value: "gap"
        """,
        encoding="utf-8",
    )

    engine = RulesEngine.from_file(config_path)
    loaded = load_config(config_path)
    assert "signals" in loaded and "rules" in loaded

    frame = _price_universe()
    frame.loc[4, "open"] = frame.loc[3, "close"] * 1.05  # induce gap-up
    records = engine.evaluate(frame)
    assert any(record.label == "GAP_UP_ALERT" for record in records)


def test_persist_signal_records_roundtrip(tmp_path: Path):
    frame = _price_universe()
    config = {
        "signals": {
            "trend": {
                "indicator": "ma_cross",
                "params": {"short_window": 2, "long_window": 4},
            },
            "volume_signal": {
                "indicator": "volume_spike",
                "params": {"window": 3, "threshold": 2.0},
            },
        },
        "rules": [
            {
                "name": "bullish_drive",
                "all": [
                    {"signal": "trend", "field": "signal", "operator": "eq", "value": 1.0},
                    {"signal": "volume_signal", "field": "spike", "operator": "is_true"},
                ],
                "emit": {"label": "BULLISH_DRIVE", "severity": "high"},
                "value": {"signal": "volume_signal", "field": "ratio"},
            }
        ],
    }
    engine = RulesEngine.from_dict(config)
    records = engine.evaluate(frame, universe="ASHARE", config_version="v1", run_version="r1")

    db_path = tmp_path / "signals.duckdb"
    db = Database(db_path)
    db.init()
    persist_signal_records(db, records)
    persisted = db.read_signals(universe="ASHARE")

    assert not persisted.empty
    assert set(persisted["rule"]) == {"bullish_drive"}
    assert persisted.loc[0, "label"] == "BULLISH_DRIVE"
    assert isinstance(persisted.loc[0, "signals"], dict)
    assert persisted.loc[0, "signals"]["trend"]["signal"] == 1.0
