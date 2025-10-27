from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ChinaAgentConfig(BaseModel):
    """Minimal configuration surface for RD-Agent China scaffolding."""

    duckdb_path: Path = Field(default=Path("rdagent_china/data/market.duckdb"))
    schema: str = Field(default="analytics")
    default_universe: str = Field(default="CSI300")
    strategy_short_window: int = Field(default=5, ge=1)
    strategy_long_window: int = Field(default=20, ge=1)

    def ensure_database_path(self) -> Path:
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        return self.duckdb_path


__all__ = ["ChinaAgentConfig"]
