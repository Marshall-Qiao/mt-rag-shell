import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from AIAgentPoC.agents.financialAnalyst.graph import build_graph
from AIAgentPoC.agents.financialAnalyst.node_and_edge import filter_and_sort
from AIAgentPoC.agents.timelinessGrader import node_and_edge as timeliness
from AIAgentPoC.agents.timelinessGrader.agent_state import TimelinessScore
from module.observability import request_config, trace_request


async def invoke(graph, text="金融问题", override=None, thread="test"):
    message = HumanMessage(content=text, additional_kwargs={"timeliness": override} if override is not None else {})
    config = request_config(thread, uuid4())
    with trace_request(config["metadata"]):
        return await graph.ainvoke({"messages": [message]}, config)


async def test_zero_timeliness_and_both_collections(fake_rag):
    result = await invoke(build_graph())
    assert result["messages"][-1].text == "有据:改写:金融问题"
    assert result["retrievalStats"]["within_days"] is None
    assert result["retrievalStats"]["original_count"] == 2
    assert [item[0] for item in fake_rag.searches] == ["default", "reference"]
    assert all(item[2] == 20 for item in fake_rag.searches)
    assert fake_rag.timeliness_inputs == [{"question": "改写:金融问题"}]


@pytest.mark.parametrize("score", range(5))
async def test_user_override_skips_timeliness_and_filters_once(fake_rag, score):
    now = datetime.now()
    fake_rag.docs = [
        (Document(page_content="recent", metadata={"created_at": now.timestamp()}), 0.1),
        (Document(page_content="old", metadata={"created_at": (now - timedelta(days=400)).timestamp()}), 0.2),
    ]
    result = await invoke(build_graph(), override=score)
    assert not fake_rag.timeliness_inputs
    assert result["questionTimelinessScore"].reason == "user override"
    assert result["retrievalStats"]["selected_count"] == (4 if score == 0 else 2)
    assert len(fake_rag.answer_inputs) == 1


async def test_nested_grader_finishes_before_filtering(fake_rag, monkeypatch):
    async def delayed(payload):
        await asyncio.sleep(0.02)
        return TimelinessScore(value=4, reason="delayed grader")
    fake_rag.docs = [(Document(page_content="old", metadata={"created_at": 0}), 0.1)]
    monkeypatch.setattr(timeliness, "get_timeliness_chain", lambda: RunnableLambda(delayed))
    result = await invoke(build_graph())
    assert result["retrievalStats"]["timeliness"] == 4
    assert result["retrievalStats"]["selected_count"] == 0
    assert len(fake_rag.answer_inputs) == 1


async def test_empty_retrieval_skips_relevance_model(fake_rag):
    fake_rag.docs = []
    result = await invoke(build_graph())
    assert result["messages"][-1].text.startswith("常识:")
    assert not fake_rag.relevance_inputs
    assert fake_rag.answer_inputs[0] == (True, {"context": "", "input": "改写:金融问题"})


async def test_irrelevant_docs_use_general_knowledge_branch(fake_rag):
    fake_rag.relevant = False
    result = await invoke(build_graph())
    assert result["documentsRelevant"] is False
    assert result["messages"][-1].text.startswith("常识:")
    assert fake_rag.answer_inputs[0][1]["context"] == ""


async def test_direct_answer_does_not_retrieve(fake_rag):
    fake_rag.direct = True
    result = await invoke(build_graph(), text="你好")
    assert result["messages"][-1].text == "直接回答:你好"
    assert not fake_rag.searches
    assert not fake_rag.timeliness_inputs


async def test_thread_history_is_isolated(fake_rag):
    graph = build_graph(InMemorySaver())
    await invoke(graph, "A1", thread="A")
    await invoke(graph, "B1", thread="B")
    await invoke(graph, "A2", thread="A")
    histories = [[m.text for m in history if isinstance(m, HumanMessage)] for history in fake_rag.entry_inputs]
    assert histories == [["A1"], ["B1"], ["A1", "A2"]]


def test_top_five_distances_and_safe_document_metrics():
    docs = [(Document(page_content=f"private-body-{i}", metadata={"source": "private-source.txt"}), float(i)) for i in reversed(range(8))]
    result = filter_and_sort({"originalDocs": docs, "questionTimelinessScore": TimelinessScore(value=0, reason="override"), "retrievalToolCallID": "test"})
    selected = result["retrievalStats"]["selected_documents"]
    assert [item["distance"] for item in selected] == [0, 1, 2, 3, 4]
    assert "private" not in str(result["retrievalStats"])
    assert "private-body-0" in result["context"]
