from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger


def get_llm(model: Optional[str] = None, provider: Optional[str] = None) -> Dict[str, Any]:
    # Placeholder for LLM integration: default to OpenAI via litellm in main repo
    try:
        from litellm import completion
    except Exception as e:  # pragma: no cover - optional dep
        logger.warning(f"LiteLLM not available: {e}")
        completion = None
    return {"completion": completion, "model": model or "gpt-4o-mini", "provider": provider or "openai"}
