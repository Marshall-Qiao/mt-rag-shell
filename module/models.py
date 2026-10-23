"""Lazily reuse model clients without making calls during module import."""

from functools import lru_cache

from langchain.chat_models import init_chat_model

from module.settings import get_settings


@lru_cache
def get_chat_model(role: str, streaming: bool = False):
    settings = get_settings()
    return init_chat_model(
        getattr(settings, f"{role}_model"),
        model_provider="openai",
        api_key=settings.openai_api_key,
        temperature=0,
        streaming=streaming,
        stream_usage=True,
    )
