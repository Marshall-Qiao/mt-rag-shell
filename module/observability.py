"""Request correlation, structured logs and opt-in LangSmith export."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langsmith import Client, get_current_run_tree, tracing_context

from module.settings import get_settings

request_fields = ContextVar("request_fields", default={})


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging():
    logger = logging.getLogger("mt_rag")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(get_settings().log_level.upper())
    logger.propagate = False


@lru_cache
def get_langsmith_client() -> Client:
    settings = get_settings()
    return Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
        workspace_id=settings.langsmith_workspace_id,
        hide_inputs=settings.langsmith_hide_inputs,
        hide_outputs=settings.langsmith_hide_outputs,
        tracing_sampling_rate=settings.langsmith_tracing_sampling_rate,
        tracing_error_callback=trace_export_failed,
    )


def trace_export_failed(error):
    logging.getLogger("mt_rag.tracing").error(
        "trace_export_failed", extra={"fields": {"error_type": type(error).__name__}}
    )


@contextmanager
def trace_request(metadata=None):
    settings = get_settings()
    # Explicit False also disables tracing enabled by legacy ambient variables.
    token = request_fields.set(metadata or {})
    try:
        with tracing_context(
            enabled=settings.langsmith_tracing,
            client=get_langsmith_client() if settings.langsmith_tracing else None,
            project_name=settings.langsmith_project,
            metadata=metadata,
        ):
            yield
    finally:
        request_fields.reset(token)


def request_config(thread_id: str, request_id: UUID) -> RunnableConfig:
    settings = get_settings()
    return {
        "configurable": {"thread_id": thread_id},
        "run_id": request_id,
        "run_name": "financial_analyst",
        "tags": ["mt-rag", settings.app_env],
        "metadata": {
            "thread_id": thread_id,
            "request_id": str(request_id),
            "environment": settings.app_env,
            "revision": settings.app_revision,
        },
    }


def record_metrics(**metrics):
    """Only pass counts, distances and hashed document IDs, never document bodies."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(metrics)
    logging.getLogger("mt_rag.retrieval").info(
        "retrieval_metrics",
        extra={"fields": {**request_fields.get(), **metrics}},
    )
