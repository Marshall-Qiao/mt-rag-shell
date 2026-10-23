"""Financial RAG nodes. Each model/tool call inherits the graph's trace context."""

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from AIAgentPoC.agents.create_agent import create_agent
from AIAgentPoC.agents.financialAnalyst.agent_state import AgentState
from AIAgentPoC.agents.timelinessGrader.agent_state import TimelinessScore
from AIAgentPoC.agents.timelinessGrader.graph import graph as timeliness_graph
from AIAgentPoC.prompts.prompts import prompt_doc_grader, prompt_no_docs_found, prompt_q_and_a
from AIAgentPoC.tools.financial_doc_retriever import document_summary, search_documents
from module.models import get_chat_model
from module.observability import record_metrics
from LongTermMemory.metadata import sourceWeights

REASON_UNSET = "unset"


class RelevanceGrade(BaseModel):
    binary_score: Literal["yes", "no"]


@tool
async def financial_documents_retriever(rephrased_question: str):
    """Search and return financial documents for a clearly rephrased question."""
    return await search_documents(rephrased_question)


@lru_cache
def get_entry_chain():
    return create_agent(
        get_chat_model("entry", streaming=True),
        tools=[financial_documents_retriever],
        system_message="""
        You are an expert financial analyst. You want to answer questions related to financial statements and investments.
        Before answering the question, you need to rephrase the latest question in a clearer manner and search for relevant financial documents.
        """,
    ).with_config(run_name="question_rewrite")


@lru_cache
def get_relevance_chain():
    llm = get_chat_model("relevance", streaming=True).with_structured_output(
        RelevanceGrade, method="json_schema", strict=True
    )
    return (prompt_doc_grader | llm).with_config(run_name="document_relevance")


@lru_cache
def get_answer_chain(no_docs: bool = False):
    prompt = prompt_no_docs_found if no_docs else prompt_q_and_a
    return (prompt | get_chat_model("answer", streaming=True)).with_config(
        run_name="general_knowledge_answer" if no_docs else "grounded_answer"
    )


async def entry_node(state: AgentState, config: RunnableConfig):
    latest = state["messages"][-1]
    score = TimelinessScore(value=0, reason=REASON_UNSET)
    if "timeliness" in latest.additional_kwargs:
        score = TimelinessScore(
            value=latest.additional_kwargs["timeliness"], reason="user override"
        )
    response = await get_entry_chain().ainvoke({"messages": state["messages"]}, config)
    tool_call = response.tool_calls[0] if response.tool_calls else None
    return {
        "messages": [response],
        "questionTimelinessScore": score,
        "originalDocs": [],
        "context": "",
        "retrievalStats": {},
        "documentsRelevant": False,
        "rephrasedQuestion": tool_call["args"]["rephrased_question"] if tool_call else "",
        "retrievalToolCallID": tool_call["id"] if tool_call else "",
    }


async def evaluate_timeliness(state: AgentState, config: RunnableConfig):
    output = await timeliness_graph.ainvoke(
        {"messages": [HumanMessage(content=state["rephrasedQuestion"])]}, config
    )
    return {"questionTimelinessScore": output["final_response"]}


async def retrieve_docs(state: AgentState, config: RunnableConfig):
    docs = await financial_documents_retriever.ainvoke(
        {"rephrased_question": state["rephrasedQuestion"]}, config
    )
    return {"originalDocs": docs}


def filter_and_sort(state: AgentState):
    score = state["questionTimelinessScore"]
    original = state["originalDocs"]
    within_days = {0: None, 1: 365, 2: 90, 3: 14, 4: 2}[score.value]
    filtered = original
    if within_days is not None:
        minimum = datetime.now() - timedelta(days=within_days)
        filtered = [
            (doc, distance)
            for doc, distance in original
            if datetime.fromtimestamp(doc.metadata["created_at"]) > minimum
        ]
    # Chroma returns distances: smaller is closer. Preserve the existing ordering.
    selected = sorted(filtered, key=lambda item: item[1])[:5]
    context = "\n\n".join(
        f"Metadata:{doc.metadata}\n Content:{doc.page_content}\n "
        for doc, _ in selected
    )
    stats = {
        "timeliness": score.value,
        "within_days": within_days,
        "original_count": len(original),
        "filtered_count": len(filtered),
        "selected_count": len(selected),
        "selected_documents": document_summary(selected),
    }
    record_metrics(**stats)
    return {
        "context": context,
        "retrievalStats": stats,
        "messages": [
            ToolMessage(content=context, tool_call_id=state["retrievalToolCallID"])
        ],
    }


async def grade_documents(state: AgentState, config: RunnableConfig):
    result = await get_relevance_chain().ainvoke(
        {"question": state["rephrasedQuestion"], "context": state["context"]}, config
    )
    relevant = result.binary_score == "yes"
    record_metrics(documents_relevant=relevant)
    return {"documentsRelevant": relevant}


async def finalize_answer(state: AgentState, config: RunnableConfig, no_docs=False):
    response = await get_answer_chain(no_docs).ainvoke(
        {
            "context": state["context"] if not no_docs else "",
            "input": state["rephrasedQuestion"],
        },
        config,
    )
    score = state["questionTimelinessScore"]
    return {
        "messages": [
            AIMessage(
                content="这是一个需要认真回答的金融相关问题，本问题时效性打分为: "
                + str(score.value)
                + "\n\n 原因如下:\n   "
                + score.reason,
                additional_kwargs={"debug": True},
            ),
            response,
        ]
    }


async def answer_node(state: AgentState, config: RunnableConfig):
    return await finalize_answer(state, config)


async def no_related_doc_found_node(state: AgentState, config: RunnableConfig):
    return await finalize_answer(state, config, no_docs=True)


def weighted_sort(doc):
    return doc.relevance * sourceWeights.get(doc.metadata["source"], 1.0)
