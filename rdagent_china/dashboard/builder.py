from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdagent_china.reports.generator import ReportPayload


@dataclass
class DashboardShell:
    """Streamlit-free dashboard placeholder used for smoke tests."""

    name: str = "rdagent-china-dashboard"

    def build_context(self, report: ReportPayload) -> dict[str, Any]:
        return {
            "dashboard": self.name,
            "title": report.title,
            "summary": dict(report.summary),
            "report": report,
        }


__all__ = ["DashboardShell"]
