from __future__ import annotations

from itertools import cycle
from typing import Iterable

from rdagent.core.proposal import Hypothesis, Hypothesis2Experiment, HypothesisGen, Trace
from rdagent.scenarios.a_share_monitor.experiment import AshareMonitorExperiment, AshareMonitorTask


class AshareMonitorHypothesisGen(HypothesisGen):
    """Generate simple monitoring hypotheses cycling through configured symbols."""

    def __init__(self, scen) -> None:  # type: ignore[no-untyped-def]
        super().__init__(scen)
        from rdagent.app.a_share_monitor.conf import ASHARE_MONITOR_PROP_SETTING

        self.settings = ASHARE_MONITOR_PROP_SETTING
        symbols: Iterable[str] = self.settings.symbols or ["000300.SH"]
        self._symbols = cycle(symbols)

    def gen(self, trace: Trace, plan=None) -> Hypothesis:  # type: ignore[override]
        symbol = next(self._symbols)
        lookback_days = self.settings.lookback_days

        hypothesis = Hypothesis(
            hypothesis=f"Monitor {symbol} price action over the past {lookback_days} days.",
            reason="Detect shifts in liquidity and momentum before the next session opens.",
            concise_reason=f"Analyse {symbol} windowed trend.",
            concise_observation="Pending data refresh.",
            concise_justification="Monitoring ensures timely alerts for execution teams.",
            concise_knowledge="Use aggregated OHLCV feeds from the unified provider.",
        )
        setattr(hypothesis, "symbol", symbol)
        setattr(hypothesis, "lookback_days", lookback_days)
        return hypothesis


class AshareMonitorHypothesis2Experiment(Hypothesis2Experiment[AshareMonitorExperiment]):
    """Translate monitoring hypotheses into runnable experiments."""

    def __init__(self) -> None:
        from rdagent.app.a_share_monitor.conf import ASHARE_MONITOR_PROP_SETTING

        self.settings = ASHARE_MONITOR_PROP_SETTING

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> AshareMonitorExperiment:  # type: ignore[override]
        symbol = getattr(hypothesis, "symbol", None) or (
            self.settings.symbols[0] if self.settings.symbols else "000300.SH"
        )
        lookback_days = getattr(hypothesis, "lookback_days", self.settings.lookback_days)
        task = AshareMonitorTask(symbol=symbol, lookback_days=lookback_days)
        experiment = AshareMonitorExperiment(task=task, hypothesis=hypothesis)
        experiment.mode = self.settings.mode  # type: ignore[attr-defined]
        experiment.backtest_window = {  # type: ignore[attr-defined]
            "start": self.settings.backtest_start,
            "end": self.settings.backtest_end,
        }
        return experiment
