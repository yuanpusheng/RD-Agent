from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, field_validator

from rdagent.core.conf import ExtendedBaseSettings


class LLMSettings(ExtendedBaseSettings):
    # backend
    backend: str = "rdagent.oai.backend.LiteLLMAPIBackend"

    chat_model: str = "gpt-4-turbo"
    embedding_model: str = "text-embedding-3-small"

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    enable_response_schema: bool = True
    # Whether to enable response_schema in chat models. may not work for models that do not support it.

    # Handling format
    reasoning_think_rm: bool = False
    """
    Some LLMs include <think>...</think> tags in their responses, which can interfere with the main output.
    Set reasoning_think_rm to True to remove any <think>...</think> content from responses.
    """

    # TODO: most of the settings are only used on deprec.DeprecBackend.
    # So they should move the settings to that folder.

    log_llm_chat_content: bool = True

    use_azure: bool = Field(default=False, deprecated=True)
    chat_use_azure: bool = False
    embedding_use_azure: bool = False

    chat_use_azure_token_provider: bool = False
    embedding_use_azure_token_provider: bool = False
    managed_identity_client_id: str | None = None
    max_retry: int = 10
    retry_wait_seconds: int = 1
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    max_past_message_include: int = 10
    timeout_fail_limit: int = 10
    violation_fail_limit: int = 1

    # Behavior of returning answers to the same question when caching is enabled
    use_auto_chat_cache_seed_gen: bool = False
    """
    `_create_chat_completion_inner_function` provides a feature to pass in a seed to affect the cache hash key
    We want to enable a auto seed generator to get different default seed for `_create_chat_completion_inner_function`
    if seed is not given.
    So the cache will only not miss you ask the same question on same round.
    """
    init_chat_cache_seed: int = 42

    # Chat configs
    openai_api_key: str = ""  # TODO: simplify the key design.
    chat_openai_api_key: str | None = None
    chat_openai_base_url: str | None = None  #
    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""
    chat_max_tokens: int | None = None
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    chat_token_limit: int = (
        100000  # 100000 is the maximum limit of gpt4, which might increase in the future version of gpt
    )
    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions."
    system_prompt_role: str = "system"
    """Some models (like o1) do not support the 'system' role.
    Therefore, we make the system_prompt_role customizable to ensure successful calls."""

    # Embedding configs
    embedding_openai_api_key: str = ""
    embedding_openai_base_url: str = ""
    embedding_azure_api_base: str = ""
    embedding_azure_api_version: str = ""
    embedding_max_str_num: int = 50
    embedding_max_length: int = 8192

    # offline llama2 related config
    use_llama2: bool = False
    llama2_ckpt_dir: str = "Llama-2-7b-chat"
    llama2_tokenizer_path: str = "Llama-2-7b-chat/tokenizer.model"
    llams2_max_batch_size: int = 8

    # server served endpoints
    use_gcr_endpoint: bool = False
    gcr_endpoint_type: str = "llama2_70b"  # or "llama3_70b", "phi2", "phi3_4k", "phi3_128k"

    llama2_70b_endpoint: str = ""
    llama2_70b_endpoint_key: str = ""
    llama2_70b_endpoint_deployment: str = ""

    llama3_70b_endpoint: str = ""
    llama3_70b_endpoint_key: str = ""
    llama3_70b_endpoint_deployment: str = ""

    phi2_endpoint: str = ""
    phi2_endpoint_key: str = ""
    phi2_endpoint_deployment: str = ""

    phi3_4k_endpoint: str = ""
    phi3_4k_endpoint_key: str = ""
    phi3_4k_endpoint_deployment: str = ""

    phi3_128k_endpoint: str = ""
    phi3_128k_endpoint_key: str = ""
    phi3_128k_endpoint_deployment: str = ""

    gcr_endpoint_temperature: float = 0.7
    gcr_endpoint_top_p: float = 0.9
    gcr_endpoint_do_sample: bool = False
    gcr_endpoint_max_token: int = 100

    chat_use_azure_deepseek: bool = False
    chat_azure_deepseek_endpoint: str = ""
    chat_azure_deepseek_key: str = ""

    # DeepSeek configs
    deepseek_api_key: str | None = None
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek/deepseek-chat"
    deepseek_chat_temperature: float = 0.4
    deepseek_chat_max_tokens: int | None = None
    deepseek_coder_model: str | None = "deepseek/deepseek-coder"
    deepseek_coder_temperature: float = 0.15
    deepseek_coder_max_tokens: int | None = 4096
    deepseek_coder_json_mode: bool = False
    deepseek_summarizer_model: str | None = None
    deepseek_summarizer_temperature: float = 0.35
    deepseek_summarizer_max_tokens: int | None = 2048
    deepseek_summarizer_json_mode: bool = False
    deepseek_interactor_model: str | None = None
    deepseek_interactor_temperature: float = 0.35
    deepseek_interactor_max_tokens: int | None = 2048
    deepseek_interactor_json_mode: bool = False
    deepseek_reasoner_model: str | None = "deepseek/deepseek-reasoner"
    deepseek_reasoner_temperature: float = 0.4
    deepseek_use_for: set[str] = Field(
        default_factory=set,
        description="Roles (general, coder, summarizer, interactor, reasoner) routed through DeepSeek.",
    )

    chat_model_map: dict[str, dict[str, Any]] = {}

    @field_validator("deepseek_use_for", mode="before")
    @classmethod
    def _parse_deepseek_use_for(cls, value: Any) -> set[str]:
        if value in (None, "", []):
            return set()
        if isinstance(value, str):
            return {item.strip().lower() for item in value.split(",") if item.strip()}
        if isinstance(value, Iterable):
            parsed = {str(item).strip().lower() for item in value if str(item).strip()}
            return parsed
        raise TypeError("deepseek_use_for must be a comma-separated string or an iterable of strings")

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.__dict__["_deepseek_applied_keys"] = set()
        self.apply_provider_overrides()

    def apply_provider_overrides(self) -> None:
        """Apply provider-specific dynamic overrides such as DeepSeek routing."""
        self._apply_deepseek_defaults()

    def _apply_deepseek_defaults(self) -> None:
        applied_keys: set[str] = self.__dict__.get("_deepseek_applied_keys", set())
        if applied_keys:
            updated_map = dict(self.chat_model_map)
            for key in applied_keys:
                updated_map.pop(key, None)
            self.chat_model_map = updated_map
        applied_keys = set()
        self.__dict__["_deepseek_applied_keys"] = applied_keys

        api_key = (self.deepseek_api_key or "").strip()
        if not api_key:
            return

        roles = {role.lower() for role in self.deepseek_use_for if role}
        if not roles:
            roles = {"general"}

        os.environ["DEEPSEEK_API_KEY"] = api_key
        if self.deepseek_api_base:
            os.environ["DEEPSEEK_API_BASE"] = self.deepseek_api_base

        if "general" in roles:
            # DeepSeek currently does not support response schema/json mode in the OpenAI sense.
            self.enable_response_schema = False
            if self.deepseek_chat_model:
                self.chat_model = self.deepseek_chat_model
            self.chat_temperature = self.deepseek_chat_temperature
            if self.deepseek_chat_max_tokens is not None:
                self.chat_max_tokens = self.deepseek_chat_max_tokens

        if "coder" in roles:
            self._register_deepseek_route(
                "coding",
                model=self.deepseek_coder_model or self.deepseek_chat_model,
                temperature=self.deepseek_coder_temperature,
                max_tokens=self.deepseek_coder_max_tokens,
                json_mode=self.deepseek_coder_json_mode,
            )

        if "summarizer" in roles:
            summarizer_model = self.deepseek_summarizer_model or self.deepseek_chat_model
            self._register_deepseek_route(
                "feedback",
                model=summarizer_model,
                temperature=self.deepseek_summarizer_temperature,
                max_tokens=self.deepseek_summarizer_max_tokens,
                json_mode=self.deepseek_summarizer_json_mode,
            )

        if "interactor" in roles:
            interactor_model = self.deepseek_interactor_model or self.deepseek_chat_model
            self._register_deepseek_route(
                "direct_exp_gen",
                model=interactor_model,
                temperature=self.deepseek_interactor_temperature,
                max_tokens=self.deepseek_interactor_max_tokens,
                json_mode=self.deepseek_interactor_json_mode,
            )

        if "reasoner" in roles:
            if self.deepseek_reasoner_model:
                self._register_deepseek_route(
                    "reasoner",
                    model=self.deepseek_reasoner_model,
                    temperature=self.deepseek_reasoner_temperature,
                    json_mode=False,
                )
            self.reasoning_think_rm = True

    def _register_deepseek_route(
        self,
        tag: str,
        *,
        model: str | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool | None = None,
    ) -> None:
        if not model:
            return
        updated_map = dict(self.chat_model_map)
        config: dict[str, Any] = dict(updated_map.get(tag, {}))
        config["model"] = model
        if temperature is not None:
            config["temperature"] = float(temperature)
        if max_tokens is not None:
            config["max_tokens"] = int(max_tokens)
        if json_mode is not None:
            config["json_mode"] = bool(json_mode)
        updated_map[tag] = config
        applied_keys: set[str] = self.__dict__.setdefault("_deepseek_applied_keys", set())
        applied_keys.add(tag)
        self.chat_model_map = updated_map


LLM_SETTINGS = LLMSettings()
