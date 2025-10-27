from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Generator
from unittest import TestCase
from unittest.mock import patch

from rdagent.log import rdagent_logger as logger

LLM_CONF_MODULE = "rdagent.oai.llm_conf"
LITELLM_MODULE = "rdagent.oai.backend.litellm"


@contextmanager
def configure_deepseek_env(use_for: str) -> Generator[tuple[object, object], None, None]:
    env_updates = {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_USE_FOR": use_for,
    }
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env_updates}
    for key, value in env_updates.items():
        os.environ[key] = value
    try:
        llm_conf_mod = importlib.reload(importlib.import_module(LLM_CONF_MODULE))
        litellm_mod = importlib.reload(importlib.import_module(LITELLM_MODULE))
        yield llm_conf_mod, litellm_mod
    finally:
        for key, original in previous.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        importlib.reload(importlib.import_module(LLM_CONF_MODULE))
        importlib.reload(importlib.import_module(LITELLM_MODULE))


class TestDeepSeekIntegration(TestCase):

    def test_deepseek_coder_routing(self) -> None:
        with configure_deepseek_env("general,coder") as (_, litellm_backend):
            litellm_backend.LITELLM_SETTINGS.chat_stream = False
            backend = litellm_backend.LiteLLMAPIBackend()

            with patch.object(litellm_backend, "completion") as mock_completion, patch.object(
                litellm_backend, "completion_cost", return_value=0
            ), patch.object(litellm_backend, "token_counter", return_value=0), patch.object(
                litellm_backend, "supports_response_schema", return_value=False
            ):
                mock_completion.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")]
                )

                with logger.tag("Loop_0.coding"):
                    result = backend.build_messages_and_create_chat_completion(
                        user_prompt="{}", system_prompt="system", json_mode=True
                    )

                self.assertIsNotNone(mock_completion.call_args)
                called_kwargs = mock_completion.call_args.kwargs
                self.assertIn("coding", litellm_backend.LITELLM_SETTINGS.chat_model_map)
                self.assertFalse(litellm_backend.LITELLM_SETTINGS.chat_model_map["coding"].get("json_mode", True))
                self.assertEqual(
                    called_kwargs["model"], litellm_backend.LITELLM_SETTINGS.deepseek_coder_model
                )
                self.assertEqual(
                    called_kwargs["temperature"], litellm_backend.LITELLM_SETTINGS.deepseek_coder_temperature
                )
                self.assertEqual(
                    called_kwargs["max_tokens"], litellm_backend.LITELLM_SETTINGS.deepseek_coder_max_tokens
                )
                self.assertNotIn("response_format", called_kwargs)
                self.assertEqual(result, "{}")

    def test_deepseek_payload_shape(self) -> None:
        with configure_deepseek_env("general") as (llm_conf_mod, litellm_backend):
            litellm_backend.LITELLM_SETTINGS.chat_stream = False
            backend = litellm_backend.LiteLLMAPIBackend()

            with patch.object(litellm_backend, "completion") as mock_completion, patch.object(
                litellm_backend, "completion_cost", return_value=0
            ), patch.object(litellm_backend, "token_counter", return_value=0), patch.object(
                litellm_backend, "supports_response_schema", return_value=False
            ):
                mock_completion.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="pong"), finish_reason="stop")]
                )

                with logger.tag("Loop_0.direct_exp_gen"):
                    backend.build_messages_and_create_chat_completion(
                        user_prompt="hello", system_prompt="system"
                    )

                called_args = mock_completion.call_args
                self.assertIsNotNone(called_args)
                self.assertEqual(
                    litellm_backend.LITELLM_SETTINGS.chat_model,
                    litellm_backend.LITELLM_SETTINGS.deepseek_chat_model,
                )
                payload_messages = called_args.kwargs["messages"]
                self.assertEqual(
                    payload_messages,
                    [
                        {"role": llm_conf_mod.LLM_SETTINGS.system_prompt_role, "content": "system"},
                        {"role": "user", "content": "hello"},
                    ],
                )
                self.assertEqual(
                    called_args.kwargs["model"], litellm_backend.LITELLM_SETTINGS.deepseek_chat_model
                )
                self.assertEqual(
                    called_args.kwargs["temperature"], litellm_backend.LITELLM_SETTINGS.deepseek_chat_temperature
                )

    def test_deepseek_fallback_without_key(self) -> None:
        saved_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        saved_use_for = os.environ.pop("DEEPSEEK_USE_FOR", None)
        try:
            llm_conf_mod = importlib.reload(importlib.import_module(LLM_CONF_MODULE))
            self.assertNotEqual(
                llm_conf_mod.LLM_SETTINGS.chat_model, llm_conf_mod.LLM_SETTINGS.deepseek_chat_model
            )
        finally:
            if saved_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved_key
            if saved_use_for is not None:
                os.environ["DEEPSEEK_USE_FOR"] = saved_use_for
            importlib.reload(importlib.import_module(LLM_CONF_MODULE))
            importlib.reload(importlib.import_module(LITELLM_MODULE))
