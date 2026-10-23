from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def create_agent(llm, tools, system_message: str):
    """Build a single model decision; LangGraph executes and filters the retrieval.

    This is not a tool execution loop, so the filtering and grading nodes retain
    control over which documents reach the final answer.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """
                {system_message}\n
                Use the provided tools to progress towards answering the question or complete the task.
                Execute what you can to make progress.
                You have access to the following tools: {tool_names}.,
                """
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    prompt = prompt.partial(
        system_message=system_message,
        tool_names=", ".join([tool.name for tool in tools]),
    )
    return prompt | llm.bind_tools(tools, parallel_tool_calls=False)
