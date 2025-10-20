# isort: skip_file
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from rdagent.core.conf import ExtendedBaseSettings


DEFAULT_CONFIG_PATH = Path.cwd() / "configs" / "config.toml"
CONFIG_ENV_VARS = ("APP_CONFIG_FILE", "RDAGENT_CONFIG_FILE", "CONFIG_FILE")


def _parse_toml_minimal(text: str) -> dict[str, Any]:  # pragma: no cover - simple fallback
    data: dict[str, Any] = {}
    current: dict[str, Any] = data
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current = data.setdefault(section, {}) if isinstance(data, dict) else {}
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Strip comments at end of line
            if "#" in val:
                val = val.split("#", 1)[0].strip()
            # Parse quoted strings
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                parsed: Any = val[1:-1]
            elif val.lower() in ("true", "false"):
                parsed = val.lower() == "true"
            else:
                try:
                    parsed = int(val)
                except ValueError:
                    parsed = val
            current[key] = parsed
    return data


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Load TOML data from a file using tomllib/tomli if available, otherwise a minimal parser.

    This keeps imports local to avoid isort import-order issues.
    """
    try:  # Python 3.11+
        import tomllib as _tomllib  # type: ignore[attr-defined]

        with path.open("rb") as f:
            return _tomllib.load(f)  # type: ignore[operator]
    except Exception:
        try:
            import tomli as _tomli  # type: ignore[import-not-found]

            with path.open("rb") as f:
                return _tomli.load(f)
        except Exception:
            with path.open("r", encoding="utf-8") as f2:
                return _parse_toml_minimal(f2.read())


class TomlSettingsSource(PydanticBaseSettingsSource):
    """
    Read configuration values from a TOML file. Values in this source have
    lower priority than environment variables by design (we insert this
    source after env sources in settings_customise_sources).
    """

    def __init__(self, settings_cls: type[BaseSettings], section: Optional[str] = None):
        super().__init__(settings_cls)
        self.section = section

    def __call__(self) -> dict[str, Any]:
        file_path = self._resolve_path()
        if file_path is None or not file_path.exists():
            return {}
        try:
            data = _load_toml_file(file_path)
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}

        if self.section:
            section_data = data.get(self.section, {})
            return section_data if isinstance(section_data, dict) else {}
        return data

    @staticmethod
    def _resolve_path() -> Optional[Path]:
        for env_name in CONFIG_ENV_VARS:
            env_value = os.getenv(env_name)
            if env_value:
                p = Path(env_value)
                # If environment variable points to a directory, append config.toml
                if p.is_dir():
                    p = p / "config.toml"
                return p
        return DEFAULT_CONFIG_PATH


class ConfigFileBaseSettings(ExtendedBaseSettings):
    """
    Base settings that add a TOML configuration source with a specific section name.
    Environment variables always override the values in the TOML file.
    """

    # Subclasses can set this to the TOML section they want to read
    config_section: Optional[str] = None

    # Use common nested delimiter; keep protected_namespaces empty so attributes are not rejected
    model_config = SettingsConfigDict(env_nested_delimiter="__", protected_namespaces=())

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Get the default order first (includes parent env sources via ExtendedBaseSettings)
        default_sources = ExtendedBaseSettings.settings_customise_sources(
            settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        )
        # Insert our TOML source after env + dotenv sources, but before file secrets
        sources: list[PydanticBaseSettingsSource] = []
        inserted = False
        for src in default_sources:
            # Place the TOML source just before file secrets
            if src is file_secret_settings and not inserted:
                sources.append(TomlSettingsSource(settings_cls, getattr(cls, "config_section", None)))
                inserted = True
            sources.append(src)
        if not inserted:
            # If for some reason file_secret_settings wasn't in default sources, append at end
            sources.append(TomlSettingsSource(settings_cls, getattr(cls, "config_section", None)))
        return tuple(sources)


def mask_secrets(data: dict[str, Any]) -> dict[str, Any]:
    def _mask(v: Any, key: str) -> Any:
        if isinstance(v, SecretStr):
            return v.__repr__()
        if isinstance(v, str) and any(tok in key.lower() for tok in ("key", "token", "secret", "password")):
            if not v:
                return v
            # Keep only first and last 2 characters for minimal context
            if len(v) <= 6:
                return "*" * len(v)
            return f"{v[:2]}{'*' * (len(v) - 4)}{v[-2:]}"
        if isinstance(v, dict):
            return {k: _mask(val, k) for k, val in v.items()}
        return v

    return {k: _mask(v, k) for k, v in data.items()}


class SecretsSettings(ConfigFileBaseSettings):
    """Provider/API secrets and tokens.

    Env variables without a prefix are supported, e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY.
    Values can also be provided in [api] of configs/config.toml.
    """

    config_section: Optional[str] = "api"

    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    azure_openai_api_key: Optional[SecretStr] = None
    huggingface_api_token: Optional[SecretStr] = None


class DatabaseSettings(ConfigFileBaseSettings):
    """Database-related configuration."""

    config_section: Optional[str] = "database"
    model_config = SettingsConfigDict(env_prefix="DB_", env_nested_delimiter="__", protected_namespaces=())

    url: str = "sqlite:///" + str(Path.cwd() / "git_ignore_folder" / "rdagent.db")
    path: Path = Path.cwd() / "git_ignore_folder" / "rdagent.db"


class ScheduleSettings(ConfigFileBaseSettings):
    """Scheduling options for periodic jobs."""

    config_section: Optional[str] = "schedule"
    model_config = SettingsConfigDict(env_prefix="SCHED_", env_nested_delimiter="__", protected_namespaces=())

    enabled: bool = False
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None


class LLMProviderSettings(ConfigFileBaseSettings):
    """LLM provider and model configuration, separate from actual secret keys."""

    config_section: Optional[str] = "llm"
    model_config = SettingsConfigDict(env_prefix="LLM_", env_nested_delimiter="__", protected_namespaces=())

    provider: str = Field(default="openai", description="LLM provider: openai|anthropic|azure_openai|huggingface")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


class AppConfig(BaseModel):
    secrets: SecretsSettings
    database: DatabaseSettings
    llm: LLMProviderSettings
    schedule: ScheduleSettings

    @model_validator(mode="after")
    def _validate_secrets(self) -> "AppConfig":
        provider = (self.llm.provider or "").lower()
        missing: list[str] = []
        if provider == "openai" and (self.secrets.openai_api_key is None or not self.secrets.openai_api_key.get_secret_value()):
            missing.append("OPENAI_API_KEY")
        elif provider == "anthropic" and (
            self.secrets.anthropic_api_key is None or not self.secrets.anthropic_api_key.get_secret_value()
        ):
            missing.append("ANTHROPIC_API_KEY")
        elif provider == "azure_openai" and (
            self.secrets.azure_openai_api_key is None
            or not self.secrets.azure_openai_api_key.get_secret_value()
        ):
            missing.append("AZURE_OPENAI_API_KEY")
        elif provider == "huggingface" and (
            self.secrets.huggingface_api_token is None
            or not self.secrets.huggingface_api_token.get_secret_value()
        ):
            missing.append("HUGGINGFACE_API_TOKEN")

        if missing:
            raise ValueError(
                f"Missing required secret(s) for provider '{provider}': {', '.join(missing)}. "
                "Set as environment variables or provide them under the [api] section of configs/config.toml."
            )
        return self

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a dict with secret-like fields masked."""
        # Convert to dict first, preserving SecretStr values
        as_dict: dict[str, Any] = {
            "secrets": self.secrets.model_dump(),
            "database": self.database.model_dump(),
            "llm": self.llm.model_dump(),
            "schedule": self.schedule.model_dump(),
        }
        return mask_secrets(as_dict)


def get_app_config() -> AppConfig:
    """
    Load application configuration with proper precedence:
    - Environment variables (including .env) override
    - TOML config file (configs/config.toml by default)
    - Default values
    """

    secrets = SecretsSettings()
    database = DatabaseSettings()
    llm = LLMProviderSettings()
    schedule = ScheduleSettings()
    cfg = AppConfig(secrets=secrets, database=database, llm=llm, schedule=schedule)
    return cfg
