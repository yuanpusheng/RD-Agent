from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "rdagent_china/.env"), env_prefix="RDC_", case_sensitive=False)

    # Database
    duckdb_path: Path = Path("rdagent_china/data/market.duckdb")
    clickhouse_url: Optional[str] = None

    # Data provider preferences
    data_source_order: List[str] = ["tushare", "akshare"]
    data_cache_dir: Path = Path("rdagent_china/cache")
    provider_yaml: Optional[Path] = None

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Monitoring loop configuration
    monitor_rules_path: Path = Path("rdagent_china/configs/monitor_signals.yaml")
    monitor_default_universe: str = "CSI300"
    monitor_lookback_days: int = 60
    monitor_timezone: str = "Asia/Shanghai"
    monitor_eod_time: str = "15:30"
    monitor_intraday_interval_minutes: Optional[int] = None
    monitor_alert_backoff_minutes: int = 120
    monitor_fetch_retries: int = 3
    monitor_fetch_retry_delay_seconds: float = 2.0
    monitor_config_version: str = "monitor-v1"

    @property
    def db_url(self) -> str:
        if self.clickhouse_url:
            return self.clickhouse_url
        # use duckdb engine format so that SQLAlchemy can read if needed
        return f"duckdb:///{self.duckdb_path}"


settings = Settings()
