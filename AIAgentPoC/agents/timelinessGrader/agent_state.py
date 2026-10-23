from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import Literal


class TimelinessScore(BaseModel):
    value: Literal[0, 1, 2, 3, 4]
    reason: str = Field(description="Explain why the score was given in CHINESE")


# Inherit 'messages' key from MessagesState, which is a list of chat messages
class AgentState(MessagesState):
    # The add_messages function defines how an update should be processed
    # Final structured response from the agent
    final_response: TimelinessScore
