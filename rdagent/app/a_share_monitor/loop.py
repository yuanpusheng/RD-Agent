from __future__ import annotations

import asyncio
from pathlib import Path

from rdagent.app.a_share_monitor.conf import AshareMonitorPropSetting
from rdagent.components.workflow.rd_loop import RDLoop


class AshareMonitorRDLoop(RDLoop):
    """RDLoop thin wrapper for the monitoring scenario."""

    def __init__(self, settings: AshareMonitorPropSetting) -> None:
        super().__init__(settings)


def launch(
    settings: AshareMonitorPropSetting,
    *,
    resume_path: str | Path | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
) -> None:
    """Instantiate the RD loop and execute it."""

    if resume_path is not None:
        loop_instance = AshareMonitorRDLoop.load(resume_path, checkout=checkout)
    else:
        loop_instance = AshareMonitorRDLoop(settings)
    asyncio.run(loop_instance.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))
