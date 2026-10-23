"""Gated live check: export/read one synthetic trace, with no model or vector DB calls."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from AIAgentPoC.agents.financialAnalyst import node_and_edge as nodes
from AIAgentPoC.agents.financialAnalyst.graph import build_graph
from AIAgentPoC.agents.timelinessGrader import node_and_edge as timeliness
from AIAgentPoC.agents.timelinessGrader.agent_state import TimelinessScore
from AIAgentPoC.tools import financial_doc_retriever as retriever
from module.observability import get_langsmith_client, request_config, trace_request
from module.settings import get_settings


def synthetic_store(name):
    return SimpleNamespace(
        collection_name=name,
        store=SimpleNamespace(similarity_search_with_score=lambda question, k: [
            (Document(page_content="Synthetic financial document", metadata={"source": "synthetic.txt"}), 0.1)
        ]),
    )


async def main():
    settings = get_settings()
    if not settings.langsmith_tracing or settings.langsmith_tracing_sampling_rate != 1:
        raise ValueError("Configure LANGSMITH_TRACING=true and sampling rate 1 for this approved live check")
    if not settings.langsmith_hide_inputs or not settings.langsmith_hide_outputs:
        raise ValueError("This check requires both LangSmith input/output hide flags to be true")

    run_id = uuid4()
    config = request_config("synthetic-validation-" + str(uuid4()), run_id)
    config["tags"].append("synthetic-validation")
    config["metadata"]["synthetic"] = True
    entry = RunnableLambda(lambda payload: AIMessage(content="", tool_calls=[{
        "name": "financial_documents_retriever", "id": "synthetic-call", "type": "tool_call",
        "args": {"rephrased_question": "Synthetic question"},
    }]))
    grade = RunnableLambda(lambda payload: nodes.RelevanceGrade(binary_score="yes"))
    score = RunnableLambda(lambda payload: TimelinessScore(value=0, reason="Synthetic score"))
    answer = RunnableLambda(lambda payload: AIMessage(content="Synthetic answer"))

    with (
        patch.object(nodes, "get_entry_chain", return_value=entry),
        patch.object(nodes, "get_relevance_chain", return_value=grade),
        patch.object(nodes, "get_answer_chain", return_value=answer),
        patch.object(timeliness, "get_timeliness_chain", return_value=score),
        patch.object(retriever, "get_store_managers", return_value=(synthetic_store("synthetic-default"), synthetic_store("synthetic-reference"))),
        trace_request(config["metadata"]),
    ):
        await build_graph().ainvoke({"messages": [HumanMessage(content="Synthetic question")]}, config)

    client = get_langsmith_client()
    await asyncio.to_thread(client.flush, timeout=10)
    # Read only this newly generated trace and its children, never an existing project scan.
    root = await asyncio.to_thread(client.read_run, run_id, load_child_runs=True)
    pending, runs = [root], []
    while pending:
        run = pending.pop()
        runs.append(run)
        pending.extend(run.child_runs or [])
    required = {"financial_analyst", "retrieve_docs", "timeliness_grader", "filter_and_sort", "grade_documents", "answer_node", "vector_search"}
    if not required <= {run.name for run in runs}:
        raise AssertionError("Required graph spans are missing from the uploaded trace")
    if len([run for run in runs if run.name == "vector_search"]) != 2:
        raise AssertionError("Expected two independent vector-search spans")
    for run in runs:
        if run.error or run.inputs or run.outputs:
            raise AssertionError("Uploaded trace has an error or unhidden input/output bodies")
        metadata = (run.extra or {}).get("metadata", {})
        if metadata.get("request_id") != str(run_id) or metadata.get("thread_id") != config["metadata"]["thread_id"]:
            raise AssertionError("Request correlation did not propagate to every span")
    print(f"Verified synthetic trace {run_id}: {len(runs)} runs, hidden bodies, no model/DB calls")


if __name__ == "__main__":
    asyncio.run(main())
