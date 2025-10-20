import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

from rdagent.config.settings import get_app_config


def _clear_env(keys: list[str]) -> None:
    for k in keys:
        if k in os.environ:
            os.environ.pop(k)


def test_loading_order_env_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure a clean env for this test
    _clear_env([
        "OPENAI_API_KEY",
        "LLM_CHAT_MODEL",
        "APP_CONFIG_FILE",
    ])

    # Point to the repo default config.toml
    cfg_path = Path.cwd() / "configs" / "config.toml"
    assert cfg_path.exists(), "configs/config.toml should exist"

    # Set env to override a value from config file
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_CHAT_MODEL", "env-model")

    app_cfg = get_app_config()
    assert app_cfg.llm.chat_model == "env-model"
    assert app_cfg.llm.provider.lower() == "openai"


def test_secret_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set a secret, then verify it's masked in safe dict
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdef123456")
    # Make sure provider is openai
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    app_cfg = get_app_config()
    safe = app_cfg.to_safe_dict()

    # Should not expose the raw secret
    val = safe["secrets"]["openai_api_key"]
    assert isinstance(val, str)
    assert "abcdef" not in val
    assert "*" in val


def test_missing_required_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no relevant env is set
    for k in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "HUGGINGFACE_API_TOKEN",
    ]:
        monkeypatch.delenv(k, raising=False)

    # Provider is openai by default; with no OPENAI_API_KEY, should raise
    with pytest.raises(ValueError) as ei:
        _ = get_app_config()
    assert "OPENAI_API_KEY" in str(ei.value)
