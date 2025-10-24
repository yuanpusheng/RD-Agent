from pathlib import Path
import os
from typing import Any, List, Optional

from pydantic import field_validator
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

    monitor_alert_channels_enabled: List[str] = ["feishu"]
    monitor_alert_http_timeout_seconds: float = 5.0
    monitor_alert_notification_cooldown_minutes: int = 60
    monitor_alert_feishu_webhook: Optional[str] = None
    monitor_alert_feishu_secret: Optional[str] = None
    monitor_alert_wecom_webhook: Optional[str] = None
    monitor_alert_slack_webhook: Optional[str] = None
    monitor_alert_email_webhook: Optional[str] = None
    monitor_alert_email_secret: Optional[str] = None
    monitor_alert_subscriptions_path: Optional[Path] = None
    monitor_alert_subscriptions: list[dict[str, Any]] = []

    @field_validator("monitor_alert_wecom_webhook", mode="before")
    @classmethod
    def _fallback_wechat_webhook(cls, value: Optional[str]) -> Optional[str]:
        if value:
            return value
        return os.getenv("WECHAT_WEBHOOK_URL") or os.getenv("WECHAT_WEBHOOK")

    @property
    def db_url(self) -> str:
        if self.clickhouse_url:
            return self.clickhouse_url
        # use duckdb engine format so that SQLAlchemy can read if needed
        return f"duckdb:///{self.duckdb_path}"


settings = Settings()
