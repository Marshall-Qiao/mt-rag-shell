from langgraph.graph import END, StateGraph, START
from AIAgentPoC.agents.timelinessGrader.node_and_edge import get_timeliness_score
from AIAgentPoC.agents.timelinessGrader.agent_state import AgentState

# Define a new graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("entry_node", get_timeliness_score)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "entry_node")

# Add edges - build up the relationship between nodes
workflow.add_edge("entry_node", END)

# Compile the graph
graph = workflow.compile(name="timeliness_grader")
