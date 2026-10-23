from langgraph.graph import MessagesState
from AIAgentPoC.agents.timelinessGrader.agent_state import TimelinessScore


class AgentState(MessagesState):
    # docs
    originalDocs: list
    questionTimelinessScore: TimelinessScore

    # rephrased question for this Q & A Round
    rephrasedQuestion: str
    retrievalToolCallID: str
    context: str
    retrievalStats: dict
    documentsRelevant: bool
