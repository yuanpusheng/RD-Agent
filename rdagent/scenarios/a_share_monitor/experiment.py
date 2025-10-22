from __future__ import annotations

from dataclasses import dataclass

from rdagent.core.experiment import Experiment, FBWorkspace, Task
from rdagent.core.proposal import ExperimentFeedback, Hypothesis


@dataclass
class AshareMonitorTask(Task):
    """Light-weight task describing an A-share monitoring target."""

    symbol: str
    lookback_days: int

    def __init__(self, symbol: str, lookback_days: int, name: str = "monitor_signal") -> None:
        description = f"Monitor {symbol} with a {lookback_days}-day lookback window."
        super().__init__(name=name, description=description)
        self.symbol = symbol
        self.lookback_days = lookback_days

    def get_task_information(self) -> str:
        return (
            f"Task Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Symbol: {self.symbol}\n"
            f"Lookback Days: {self.lookback_days}"
        )


class AshareMonitorWorkspace(FBWorkspace[AshareMonitorTask, ExperimentFeedback]):
    """A minimal workspace for monitoring experiments."""

    def __init__(self, task: AshareMonitorTask) -> None:
        super().__init__(target_task=task)


class AshareMonitorExperiment(
    Experiment[AshareMonitorTask, AshareMonitorWorkspace, AshareMonitorWorkspace]
):
    """Experiment wrapping a single monitoring task and workspace."""

    def __init__(self, task: AshareMonitorTask, hypothesis: Hypothesis | None = None) -> None:
        super().__init__(sub_tasks=[task], hypothesis=hypothesis)
        self.experiment_workspace = AshareMonitorWorkspace(task)
        self.sub_workspace_list = [self.experiment_workspace]

    @property
    def monitor_task(self) -> AshareMonitorTask:
        return self.sub_tasks[0]

    def ensure_workspace(self) -> AshareMonitorWorkspace:
        if self.experiment_workspace is None:
            workspace = AshareMonitorWorkspace(self.monitor_task)
            self.experiment_workspace = workspace
            self.sub_workspace_list = [workspace]
        else:
            workspace = self.experiment_workspace
        return workspace
