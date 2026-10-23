from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import get_prompt
from .load_vector_store import load_vector_store
from .print_runner import with_print_listener
from .llms import get_llm
from .settings import get_settings


class ChainManager:
    chain = None

    def loadChain(self, llmModel, prompt):
        settings = get_settings()
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
        vector_store = load_vector_store(settings.vector_db_path, embeddings)
        llm = get_llm(llmModel)
        prompt_in_chain = with_print_listener(get_prompt(prompt))
        self.chain = (
            {
                "context": with_print_listener(vector_store.as_retriever()),
                "input": RunnablePassthrough(),
            }
            | prompt_in_chain
            | llm
            | StrOutputParser()
        )

    def disableChain(self):
        self.chain = None


chain_manager = ChainManager()
