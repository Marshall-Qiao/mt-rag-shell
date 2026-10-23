import json
import logging
from uuid import uuid4

import pytest
from langsmith import Client, tracing_context
from pydantic import ValidationError

from AIAgentPoC.agents.financialAnalyst.graph import build_graph
from module import observability
from module.settings import Settings, get_settings


class RecordingClient(Client):
    """Capture synthetic SDK runs in memory; no transport is used."""

    def __init__(self):
        super().__init__(api_key="test-key", api_url="https://example.invalid", auto_batch_tracing=False)
        self.created = []
        self.updated = []

    def create_run(self, name, inputs, run_type, **kwargs):
        self.created.append({"name": name, "inputs": inputs, "run_type": run_type, **kwargs})

    def update_run(self, run_id, **kwargs):
        self.updated.append({"id": run_id, **kwargs})


def test_tracing_defaults_and_required_key():
    settings = Settings(_env_file=None)
    assert not settings.langsmith_tracing
    assert settings.langsmith_hide_inputs and settings.langsmith_hide_outputs
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        Settings(_env_file=None, langsmith_tracing=True)


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_sampling_range(rate):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, langsmith_tracing_sampling_rate=rate)


def test_env_file_and_process_override(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.env"
    path.write_text("APP_ENV=synthetic\nANSWER_MODEL=file-model\nLANGSMITH_TRACING=false\n")
    monkeypatch.setenv("MT_RAG_ENV_FILE", str(path))
    monkeypatch.setenv("ANSWER_MODEL", "process-model")
    settings = get_settings()
    assert settings.app_env == "synthetic"
    assert settings.answer_model == "process-model"


async def test_disabled_tracing_overrides_ambient_context(fake_rag, monkeypatch):
    client = RecordingClient()
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    from langchain_core.messages import HumanMessage
    with tracing_context(enabled=True, client=client):
        with observability.trace_request():
            await build_graph().ainvoke({"messages": [HumanMessage(content="synthetic")]})
    assert not client.created


async def test_trace_tree_and_request_metadata(fake_rag, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    client = RecordingClient()
    monkeypatch.setattr(observability, "get_langsmith_client", lambda: client)
    from langchain_core.messages import HumanMessage
    request_id = uuid4()
    config = observability.request_config("synthetic-thread", request_id)
    with observability.trace_request(config["metadata"]):
        await build_graph().ainvoke({"messages": [HumanMessage(content="synthetic")]}, config)

    by_id = {str(run["id"]): run for run in client.created}
    names = {run["name"] for run in client.created}
    assert {"financial_analyst", "retrieve_docs", "timeliness_grader", "filter_and_sort", "grade_documents", "answer_node", "vector_search"} <= names
    root = by_id[str(request_id)]
    assert not root.get("parent_run_id")
    for run in client.created:
        metadata = run.get("extra", {}).get("metadata", {})
        assert metadata["thread_id"] == "synthetic-thread"
        assert metadata["request_id"] == str(request_id)
        if str(run["id"]) != str(request_id):
            assert str(run["parent_run_id"]) in by_id
    assert len([run for run in client.created if run["name"] == "vector_search"]) == 2


def test_explicit_privacy_client_configuration(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING_SAMPLING_RATE", "0.25")
    captured = {}
    monkeypatch.setattr(observability, "Client", lambda **kwargs: captured.update(kwargs) or object())
    observability.get_langsmith_client()
    assert captured["hide_inputs"] is True
    assert captured["hide_outputs"] is True
    assert captured["tracing_sampling_rate"] == 0.25
    assert captured["tracing_error_callback"] is observability.trace_export_failed


def test_sdk_export_hides_bodies_but_keeps_usage_metadata(monkeypatch):
    exported = []
    monkeypatch.setattr(Client, "_create_run", lambda self, run, **kwargs: exported.append(run))
    client = Client(
        api_key="test-key", api_url="https://example.invalid", auto_batch_tracing=False,
        hide_inputs=True, hide_outputs=True,
    )
    client.create_run(
        "synthetic_llm", {"question": "private question", "context": "private document"}, "llm",
        outputs={"answer": "private answer"},
        extra={"metadata": {"thread_id": "synthetic", "usage_metadata": {"input_tokens": 10, "output_tokens": 5}}},
    )
    assert exported[0]["inputs"] == {}
    assert exported[0]["outputs"] == {}
    assert "private" not in str(exported)
    assert exported[0]["extra"]["metadata"]["usage_metadata"]["input_tokens"] == 10


def test_json_logs_and_offline_request_correlation(caplog):
    request_id = uuid4()
    config = observability.request_config("synthetic-thread", request_id)
    with caplog.at_level(logging.INFO, logger="mt_rag.retrieval"):
        with observability.trace_request(config["metadata"]):
            observability.record_metrics(retrieved_count=2)
    payload = json.loads(observability.JsonFormatter().format(caplog.records[-1]))
    assert payload["request_id"] == str(request_id)
    assert payload["thread_id"] == "synthetic-thread"
    assert payload["retrieved_count"] == 2
