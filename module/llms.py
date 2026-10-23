from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from module.settings import get_settings


def get_llm(llmModel="gpt-4o"):
    if llmModel == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", api_key=get_settings().openai_api_key)
    elif llmModel == "llama3.1":
        return ChatOllama(model="llama3.1")
    raise ValueError(f"Unsupported model: {llmModel}")
