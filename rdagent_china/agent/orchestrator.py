from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rdagent_china.backtest.engine import BacktestSummary, SimpleBacktestEngine
from rdagent_china.dashboard.builder import DashboardShell
from rdagent_china.data.schemas import PriceDailyRecord, SignalRecord
from rdagent_china.reports.generator import ReportGenerator, ReportPayload
from rdagent_china.strategy.baseline import MovingAverageStrategy


@dataclass
class ChinaAgent:
    """Lightweight orchestrator that wires data, strategy, backtest, and reporting components."""

    strategy: MovingAverageStrategy = field(default_factory=MovingAverageStrategy)
    backtest_engine: SimpleBacktestEngine = field(default_factory=SimpleBacktestEngine)
    reporter: ReportGenerator = field(default_factory=ReportGenerator)
    dashboard: DashboardShell = field(default_factory=DashboardShell)

    def generate_signals(self, prices: Sequence[PriceDailyRecord]) -> list[SignalRecord]:
        """Transform raw price records into strategy signals."""

        return self.strategy.generate_signals(prices)

    def run_backtest(self, signals: Sequence[SignalRecord]) -> BacktestSummary:
        """Aggregate signal quality via the configured backtest engine."""

        return self.backtest_engine.run(signals)

    def build_report(self, result: BacktestSummary) -> ReportPayload:
        """Produce a structured report for downstream consumers."""

        return self.reporter.generate(result)

    def build_dashboard_context(self, report: ReportPayload) -> dict[str, object]:
        """Translate the report into a dictionary consumable by UI layers."""

        return self.dashboard.build_context(report)

    def run(self, prices: Sequence[PriceDailyRecord]) -> dict[str, object]:
        """Execute end-to-end flow: data → signals → backtest → report → dashboard."""

        signals = self.generate_signals(prices)
        result = self.run_backtest(signals)
        report = self.build_report(result)
        return self.build_dashboard_context(report)


__all__ = ["ChinaAgent"]
