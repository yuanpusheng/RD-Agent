from __future__ import annotations

from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class AshareMonitorPropSetting(BasePropSetting):
    """Settings driving the A-share monitoring RD loop."""

    model_config = SettingsConfigDict(env_prefix="ASHARE_MONITOR_", protected_namespaces=())

    # Workflow wiring
    scen: str = "rdagent.scenarios.a_share_monitor.scenario.AshareMonitorScenario"
    hypothesis_gen: str = "rdagent.scenarios.a_share_monitor.hypothesis.AshareMonitorHypothesisGen"
    hypothesis2experiment: str = (
        "rdagent.scenarios.a_share_monitor.hypothesis.AshareMonitorHypothesis2Experiment"
    )
    coder: str = "rdagent.scenarios.a_share_monitor.developer.AshareMonitorCoder"
    runner: str = "rdagent.scenarios.a_share_monitor.developer.AshareMonitorRunner"
    summarizer: str = "rdagent.scenarios.a_share_monitor.feedback.AshareMonitorExperiment2Feedback"

    # Scenario specific options
    mode: Literal["live", "backtest"] = "live"
    symbols: List[str] = Field(default_factory=lambda: ["000300.SH", "600519.SS"])
    lookback_days: int = 30
    refresh_minutes: int = 15
    summary_window: int = 5
    data_provider: str = "akshare"

    # Backtest window configuration
    backtest_start: str | None = None
    backtest_end: str | None = None

    @field_validator("symbols", mode="before")
    @classmethod
    def _coerce_symbols(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, str):
            return [sym.strip() for sym in value.split(",") if sym.strip()]
        return value


ASHARE_MONITOR_PROP_SETTING = AshareMonitorPropSetting()
