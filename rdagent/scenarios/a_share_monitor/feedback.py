from __future__ import annotations

from rdagent.core.proposal import Experiment2Feedback, ExperimentFeedback
from rdagent.scenarios.a_share_monitor.experiment import AshareMonitorExperiment


class AshareMonitorExperiment2Feedback(Experiment2Feedback):
    """Summarise monitoring runs with lightweight feedback."""

    def generate_feedback(self, exp: AshareMonitorExperiment, trace):  # type: ignore[override]
        result = getattr(exp, "result", {}) or {}
        symbol = result.get("symbol", exp.monitor_task.symbol)
        timestamp = result.get("timestamp")
        reason = f"Monitoring run completed for {symbol}."
        if timestamp:
            reason += f" Latest snapshot recorded at {timestamp}."
        return ExperimentFeedback(reason=reason, decision=True)
