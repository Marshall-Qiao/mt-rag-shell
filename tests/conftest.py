import importlib.util
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from AIAgentPoC.agents.financialAnalyst import node_and_edge as nodes
from AIAgentPoC.agents.timelinessGrader import node_and_edge as timeliness
from AIAgentPoC.agents.timelinessGrader.agent_state import TimelinessScore
from AIAgentPoC.tools import financial_doc_retriever as retriever
from module import observability
from module.models import get_chat_model
from module.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Never load real credentials or contact models, telemetry or databases.
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setenv("MT_RAG_ENV_FILE", "/dev/null")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    get_chat_model.cache_clear()
    observability.get_langsmith_client.cache_clear()
    nodes.get_entry_chain.cache_clear()
    nodes.get_relevance_chain.cache_clear()
    nodes.get_answer_chain.cache_clear()
    timeliness.get_timeliness_chain.cache_clear()
    retriever.get_store_managers.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_rag(monkeypatch):
    fixture = SimpleNamespace(
        score=0,
        relevant=True,
        docs=[(Document(page_content="synthetic document", metadata={"source": "synthetic.txt"}), 0.2)],
        entry_inputs=[],
        timeliness_inputs=[],
        relevance_inputs=[],
        answer_inputs=[],
        searches=[],
        direct=False,
    )

    def entry(payload):
        fixture.entry_inputs.append(list(payload["messages"]))
        question = payload["messages"][-1].text
        if fixture.direct:
            return AIMessage(content="直接回答:" + question)
        return AIMessage(content="", tool_calls=[{
            "name": "financial_documents_retriever",
            "args": {"rephrased_question": "改写:" + question},
            "id": "synthetic-call-" + str(len(fixture.entry_inputs)),
            "type": "tool_call",
        }])

    def score(payload):
        fixture.timeliness_inputs.append(payload)
        return TimelinessScore(value=fixture.score, reason="synthetic score")

    def relevance(payload):
        fixture.relevance_inputs.append(payload)
        return nodes.RelevanceGrade(binary_score="yes" if fixture.relevant else "no")

    def answer(no_docs):
        def run(payload):
            fixture.answer_inputs.append((no_docs, payload))
            # No finish_reason metadata: output delivery must not depend on it.
            return AIMessage(content=("常识:" if no_docs else "有据:") + payload["input"])
        return RunnableLambda(run, name="synthetic_answer")

    def store(name):
        def search(question, k):
            fixture.searches.append((name, question, k))
            return list(fixture.docs)
        return SimpleNamespace(collection_name=name, store=SimpleNamespace(similarity_search_with_score=search))

    monkeypatch.setattr(nodes, "get_entry_chain", lambda: RunnableLambda(entry, name="synthetic_rewrite"))
    monkeypatch.setattr(nodes, "get_relevance_chain", lambda: RunnableLambda(relevance, name="synthetic_relevance"))
    monkeypatch.setattr(nodes, "get_answer_chain", answer)
    monkeypatch.setattr(timeliness, "get_timeliness_chain", lambda: RunnableLambda(score, name="synthetic_timeliness"))
    monkeypatch.setattr(retriever, "get_store_managers", lambda: (store("default"), store("reference")))
    return fixture


@pytest.fixture
def server():
    spec = importlib.util.spec_from_file_location("chat_server", Path(__file__).parents[1] / "chat-server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
