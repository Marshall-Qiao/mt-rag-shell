from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, chat

prompt_q_and_a = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are an expert financial analyst. Use you knowledge base to answer questions about audited financial statements. Please note: 
        1. Always answer questions in CHINESE 
        2. If you can't find a relevant answer from the given context, please state that you don't know.
        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              Context: {context}
              Question: {input}
            """
        ),
    ]
)

prompt_text_completion = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are an assistant that helps users to write articles based on given context.
        And based on current input, you are going to complete the sentence or paragraph.
        This article should be in Chinese.
        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              context: {context}
              input: {input}
            """
        ),
    ]
)


def get_prompt(prompt: str):
    if prompt == "Q_A":
        return prompt_q_and_a
    elif prompt == "TEXT_COMPLETION":
        return prompt_text_completion
