from typing import Literal

from langgraph.graph import END, START, StateGraph

from AIAgentPoC.agents.financialAnalyst.agent_state import AgentState
from AIAgentPoC.agents.financialAnalyst.node_and_edge import (
    REASON_UNSET,
    answer_node,
    entry_node,
    evaluate_timeliness,
    filter_and_sort,
    grade_documents,
    no_related_doc_found_node,
    retrieve_docs,
)


def router(state: AgentState):
    if not state["messages"][-1].tool_calls:
        return END
    if state["questionTimelinessScore"].reason != REASON_UNSET:
        return "retrieve_docs"
    return ["retrieve_docs", "timeliness_grader"]


def route_filtered_docs(state: AgentState) -> Literal["grade_documents", "no_related_doc_found_node"]:
    return "grade_documents" if state["retrievalStats"]["selected_count"] else "no_related_doc_found_node"


def route_relevance(state: AgentState) -> Literal["answer_node", "no_related_doc_found_node"]:
    return "answer_node" if state["documentsRelevant"] else "no_related_doc_found_node"


def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("entry_node", entry_node)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("timeliness_grader", evaluate_timeliness)
    # Wait for all scheduled branches, including the nested timeliness graph.
    # A fixed two-node join would hang when a user override skips the grader.
    workflow.add_node("filter_and_sort", filter_and_sort, defer=True)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("answer_node", answer_node)
    workflow.add_node("no_related_doc_found_node", no_related_doc_found_node)
    workflow.add_edge(START, "entry_node")
    workflow.add_conditional_edges(
        "entry_node", router,
        {"retrieve_docs": "retrieve_docs", "timeliness_grader": "timeliness_grader", END: END},
    )
    workflow.add_edge("retrieve_docs", "filter_and_sort")
    workflow.add_edge("timeliness_grader", "filter_and_sort")
    workflow.add_conditional_edges("filter_and_sort", route_filtered_docs)
    workflow.add_conditional_edges("grade_documents", route_relevance)
    workflow.add_edge("answer_node", END)
    workflow.add_edge("no_related_doc_found_node", END)
    return workflow.compile(checkpointer=checkpointer, name="financial_analyst")


# Studio supplies its own persistence. The WebSocket server supplies InMemorySaver.
graph = build_graph()
