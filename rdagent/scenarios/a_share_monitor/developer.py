from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

from rdagent.core.developer import Developer
from rdagent.core.scenario import Scenario
from rdagent.scenarios.a_share_monitor.experiment import AshareMonitorExperiment


class AshareMonitorCoder(Developer[AshareMonitorExperiment]):
    """Generate lightweight monitoring code artifacts."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        from rdagent.app.a_share_monitor.conf import ASHARE_MONITOR_PROP_SETTING

        self.settings = ASHARE_MONITOR_PROP_SETTING

    def develop(self, exp: AshareMonitorExperiment) -> AshareMonitorExperiment:  # type: ignore[override]
        workspace = exp.ensure_workspace()
        task = exp.monitor_task
        script = dedent(
            f"""
            import datetime as _dt
            from typing import Dict, Any

            def monitor(symbol: str, lookback_days: int, mode: str) -> Dict[str, Any]:
                now = _dt.datetime.now(tz=_dt.timezone.utc)
                return {{
                    "symbol": symbol,
                    "lookback_days": lookback_days,
                    "mode": mode,
                    "generated_at": now.isoformat(),
                }}

            if __name__ == "__main__":
                result = monitor(symbol="{task.symbol}", lookback_days={task.lookback_days}, mode="{self.settings.mode}")
                print("[A-share monitor] {task.symbol} @", result["generated_at"], f"({task.lookback_days}d window)")
            """
        ).strip()
        workspace.inject_files(**{"monitor.py": script})
        exp.experiment_workspace = workspace
        exp.sub_workspace_list[0] = workspace
        return exp


class AshareMonitorRunner(Developer[AshareMonitorExperiment]):
    """Collect synthetic monitoring results without external dependencies."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        from rdagent.app.a_share_monitor.conf import ASHARE_MONITOR_PROP_SETTING

        self.settings = ASHARE_MONITOR_PROP_SETTING

    def develop(self, exp: AshareMonitorExperiment) -> AshareMonitorExperiment:  # type: ignore[override]
        task = exp.monitor_task
        exp.result = {
            "symbol": task.symbol,
            "lookback_days": task.lookback_days,
            "mode": getattr(exp, "mode", self.settings.mode),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "backtest_window": getattr(
                exp,
                "backtest_window",
                {
                    "start": self.settings.backtest_start,
                    "end": self.settings.backtest_end,
                },
            ),
        }
        return exp
