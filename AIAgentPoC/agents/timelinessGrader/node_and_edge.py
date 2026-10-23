from functools import lru_cache

from langchain_core.runnables import RunnableConfig

from AIAgentPoC.agents.timelinessGrader.agent_state import AgentState, TimelinessScore
from AIAgentPoC.prompts.prompts import prompt_question_timeliness
from module.models import get_chat_model


@lru_cache
def get_timeliness_chain():
    model = get_chat_model("timeliness").with_structured_output(
        TimelinessScore, method="json_schema", strict=True
    )
    return (prompt_question_timeliness | model).with_config(run_name="question_timeliness")


async def get_timeliness_score(state: AgentState, config: RunnableConfig):
    result = await get_timeliness_chain().ainvoke(
        {"question": state["messages"][-1].text}, config
    )
    return {"final_response": result}
