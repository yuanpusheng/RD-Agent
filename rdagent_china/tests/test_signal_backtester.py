from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rdagent_china.backtest.config import BenchmarkConfig, LabelingConfig, SignalBacktestConfig
from rdagent_china.backtest.signals import SignalBacktester
from rdagent_china.db import Database


@pytest.fixture()
def sample_database(tmp_path: Path) -> Database:
    db_path = tmp_path / "signals.duckdb"
    db = Database(db_path)
    db.init()
    return db


def _price_frame(symbol: str, dates: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000] * len(dates),
        }
    )


def _signals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "universe": ["TEST", "TEST"],
            "symbol": ["AAA", "AAA"],
            "timestamp": [
                pd.Timestamp("2024-01-01T09:30:00Z"),
                pd.Timestamp("2024-01-02T09:30:00Z"),
            ],
            "as_of_date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "rule": ["rule_alpha", "rule_alpha"],
            "label": ["ALERT", "ALERT"],
            "severity": ["medium", "medium"],
            "triggered": [True, True],
            "value": [1.0, 1.0],
            "signals": [{"score": 0.8}, {"score": 0.5}],
            "config_version": ["v1", "v1"],
            "run_version": ["rv1", "rv1"],
        }
    )


def test_signal_backtester_metrics(sample_database: Database, tmp_path: Path):
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    primary = _price_frame("AAA", dates, [100.0, 110.0, 108.0, 113.0, 115.0, 118.0])
    benchmark = _price_frame("BENCH", dates, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    sample_database.write_price_daily(pd.concat([primary, benchmark], ignore_index=True))
    sample_database.write_signals(_signals_frame())

    config = SignalBacktestConfig(
        symbols=["AAA"],
        start="2024-01-01",
        end="2024-01-04",
        evaluation_windows=[2],
        labeling=LabelingConfig(horizon=2, threshold=0.05, price_field="close"),
        benchmark=BenchmarkConfig(symbol="BENCH"),
    )

    backtester = SignalBacktester(db=sample_database, config=config)
    result = backtester.run()

    assert not result.is_empty()
    assert result.window_positive_counts[2] == 2
    assert len(result.trades) == 2

    metric = next(summary for summary in result.metrics if summary.rule == "rule_alpha" and summary.window == 2)
    assert metric.total_signals == 2
    assert metric.true_positives == 1
    assert metric.false_positives == 1
    assert metric.false_negatives == 1
    assert metric.ignored_signals == 0
    assert metric.precision == pytest.approx(0.5)
    assert metric.recall == pytest.approx(0.5)
    assert metric.hit_rate == pytest.approx(0.5)
    assert metric.avg_return == pytest.approx(0.0536, rel=1e-3)
    assert metric.avg_benchmark_return == pytest.approx(0.0199, rel=1e-3)
    assert metric.avg_excess_return == pytest.approx(0.0337, rel=1e-3)
    assert metric.max_drawdown == pytest.approx(0.0)
    assert isinstance(metric.equity_curve, pd.Series)
    assert len(metric.equity_curve) == 2

    output_dir = tmp_path / "reports"
    result.save(output_dir)
    report_file = output_dir / "report.html"
    metrics_file = output_dir / "metrics.json"
    trades_file = output_dir / "trades.csv"
    assert report_file.exists()
    assert metrics_file.exists()
    assert trades_file.exists()

    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    overall = payload["overall"]
    assert overall["2"]["precision"] == pytest.approx(0.5)
    assert "Signal Backtest Summary" in report_file.read_text(encoding="utf-8")
    trades_content = trades_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(trades_content) == 3
