import copyreg
from typing import Any, Literal, Optional, Type, TypedDict, Union, cast

import numpy as np
from litellm import (
    BadRequestError,
    completion,
    completion_cost,
    embedding,
    get_model_info,
    supports_function_calling,
    supports_response_schema,
    token_counter,
)
from pydantic import BaseModel

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLMSettings


# NOTE: Patching! Otherwise, the exception will call the constructor and with following error:
# `BadRequestError.__init__() missing 2 required positional arguments: 'model' and 'llm_provider'`
def _reduce_no_init(exc: Exception) -> tuple:
    cls = exc.__class__
    return (cls.__new__, (cls,), exc.__dict__)


# suppose you want to apply this to MyError
copyreg.pickle(BadRequestError, _reduce_no_init)


class LiteLLMSettings(LLMSettings):

    class Config:
        env_prefix = "LITELLM_"
        """Use `LITELLM_` as prefix for environment variables"""

    # Placeholder for LiteLLM specific settings, so far it's empty


LITELLM_SETTINGS = LiteLLMSettings()
ACC_COST = 0.0


class LiteLLMAPIBackend(APIBackend):
    """LiteLLM implementation of APIBackend interface"""

    _has_logged_settings: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        LITELLM_SETTINGS.apply_provider_overrides()
        self._active_route_config: dict[str, Any] = {}
        self._last_complete_kwargs: LiteLLMAPIBackend.CompleteKwargs | None = None
        if not self.__class__._has_logged_settings:
            logger.info(f"{LITELLM_SETTINGS}")
            logger.log_object(LITELLM_SETTINGS.model_dump(), tag="LITELLM_SETTINGS")
            self.__class__._has_logged_settings = True
        super().__init__(*args, **kwargs)

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages
        """
        num_tokens = token_counter(
            model=LITELLM_SETTINGS.chat_model,
            messages=messages,
        )
        logger.info(f"{LogColors.CYAN}Token count: {LogColors.END} {num_tokens}", tag="debug_litellm_token")
        return num_tokens

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        Call the embedding function
        """
        model_name = LITELLM_SETTINGS.embedding_model
        logger.info(f"{LogColors.GREEN}Using emb model{LogColors.END} {model_name}", tag="debug_litellm_emb")
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(
                f"{LogColors.MAGENTA}Creating embedding{LogColors.END} for: {input_content_list}",
                tag="debug_litellm_emb",
            )
        response = embedding(
            model=model_name,
            input=input_content_list,
        )
        response_list = [data["embedding"] for data in response.data]
        return response_list

    class CompleteKwargs(TypedDict):
        model: str
        temperature: float
        max_tokens: int | None
        reasoning_effort: Literal["low", "medium", "high"] | None

    def _select_model_config(self) -> tuple["LiteLLMAPIBackend.CompleteKwargs", dict[str, Any]]:
        model = LITELLM_SETTINGS.chat_model
        temperature = LITELLM_SETTINGS.chat_temperature
        max_tokens = LITELLM_SETTINGS.chat_max_tokens
        reasoning_effort = LITELLM_SETTINGS.reasoning_effort
        route_config: dict[str, Any] = {}

        if LITELLM_SETTINGS.chat_model_map:
            for tag, config in LITELLM_SETTINGS.chat_model_map.items():
                if tag in logger._tag:
                    route_config = dict(config)
                    if "model" in config:
                        model = config["model"]
                    if "temperature" in config:
                        try:
                            temperature = float(config["temperature"])
                        except (TypeError, ValueError):
                            pass
                    if "max_tokens" in config:
                        try:
                            max_tokens = int(config["max_tokens"])
                        except (TypeError, ValueError):
                            max_tokens = None
                    if "reasoning_effort" in config:
                        effort = config["reasoning_effort"]
                        if effort in ["low", "medium", "high"]:
                            reasoning_effort = cast(Literal["low", "medium", "high"], effort)
                        else:
                            reasoning_effort = None
                    break

        complete_kwargs = self.CompleteKwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return complete_kwargs, route_config

    def get_complete_kwargs(self) -> CompleteKwargs:
        if self._last_complete_kwargs is not None:
            complete_kwargs = self._last_complete_kwargs
            self._last_complete_kwargs = None
            return complete_kwargs

        complete_kwargs, route_config = self._select_model_config()
        self._active_route_config = route_config
        return complete_kwargs

    def get_route_config(self) -> dict[str, Any]:
        complete_kwargs, route_config = self._select_model_config()
        self._last_complete_kwargs = complete_kwargs
        self._active_route_config = route_config
        return dict(route_config)

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function
        """

        if response_format and not supports_response_schema(model=LITELLM_SETTINGS.chat_model):
            # Deepseek will enter this branch
            logger.warning(
                f"{LogColors.YELLOW}Model {LITELLM_SETTINGS.chat_model} does not support response schema, ignoring response_format argument.{LogColors.END}",
                tag="llm_messages",
            )
            response_format = None

        if response_format:
            kwargs["response_format"] = response_format

        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        complete_kwargs = self.get_complete_kwargs()
        model = complete_kwargs["model"]

        response = completion(
            messages=messages,
            stream=LITELLM_SETTINGS.chat_stream,
            max_retries=0,
            **complete_kwargs,
            **kwargs,
        )
        logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {model}", tag="llm_messages")

        if LITELLM_SETTINGS.chat_stream:
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.BLUE}assistant:{LogColors.END}", tag="llm_messages")
            content = ""
            finish_reason = None
            for message in response:
                if message["choices"][0]["finish_reason"]:
                    finish_reason = message["choices"][0]["finish_reason"]
                if "content" in message["choices"][0]["delta"]:
                    chunk = (
                        message["choices"][0]["delta"]["content"] or ""
                    )  # when finish_reason is "stop", content is None
                    content += chunk
                    if LITELLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}Finish reason: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}assistant:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages"
                )

        global ACC_COST
        try:
            cost = completion_cost(model=model, messages=messages, completion=content)
        except Exception as e:
            logger.warning(f"Cost calculation failed for model {model}: {e}. Skip cost statistics.")
            cost = np.nan
        else:
            ACC_COST += cost
            logger.info(
                f"Current Cost: ${float(cost):.10f}; Accumulated Cost: ${float(ACC_COST):.10f}; {finish_reason=}",
            )

        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(model=model, text=content)
        logger.log_object(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "accumulated_cost": ACC_COST,
            },
            tag="token_cost",
        )
        self._active_route_config = {}
        self._last_complete_kwargs = None
        return content, finish_reason

    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling
        """
        model = self._active_route_config.get("model") if self._active_route_config else None
        if model is None:
            complete_kwargs, _ = self._select_model_config()
            model = complete_kwargs["model"]
        return supports_response_schema(model=model) and LITELLM_SETTINGS.enable_response_schema

    @property
    def chat_token_limit(self) -> int:
        """Suggest an input token limit, ensuring enough space in the context window for the maximum output tokens."""
        try:
            model_info = get_model_info(LITELLM_SETTINGS.chat_model)
            if model_info is None:
                return super().chat_token_limit

            max_input = model_info.get("max_input_tokens")
            max_output = model_info.get("max_output_tokens")

            if max_input is None or max_output is None:
                return super().chat_token_limit

            max_input_tokens = max_input - max_output
            return max_input_tokens
        except Exception as e:
            return super().chat_token_limit
