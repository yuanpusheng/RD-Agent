from __future__ import annotations

import platform
from dataclasses import dataclass

from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario


@dataclass
class RuntimeSnapshot:
    python_version: str
    platform: str

    def as_text(self) -> str:
        return f"Python {self.python_version} on {self.platform}"


class AshareMonitorScenario(Scenario):
    """Scenario description for the A-share monitoring workflow."""

    def __init__(self) -> None:
        super().__init__()
        from rdagent.app.a_share_monitor.conf import ASHARE_MONITOR_PROP_SETTING

        settings = ASHARE_MONITOR_PROP_SETTING
        symbols = ", ".join(settings.symbols) if settings.symbols else "000300.SH"
        self._background = (
            "You operate an autonomous monitoring loop for Mainland China A-share equities. "
            "Generate concise diagnostics for the configured watchlist and surface anomalies early."
        )
        self._source_data = (
            "Primary data feeds originate from unified on-premise collectors (AkShare/Tushare). "
            f"Default watchlist: {symbols}."
        )
        self._rich_style_description = (
            "<b>A-share Monitor</b><br>"
            f"<ul><li><b>Symbols:</b> {symbols}</li>"
            f"<li><b>Lookback:</b> {settings.lookback_days} days</li>"
            f"<li><b>Mode:</b> {settings.mode}</li></ul>"
        )
        self._runtime_snapshot = RuntimeSnapshot(
            python_version=platform.python_version(),
            platform=f"{platform.system()} {platform.release()}",
        )

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        return (
            f"Background:\n{self.background}\n\n"
            f"Data Sources:\n{self.get_source_data_desc(task)}\n"
            f"Runtime Environment:\n{self.get_runtime_environment()}"
        )

    def get_runtime_environment(self) -> str:
        return self._runtime_snapshot.as_text()
