import asyncio
from functools import lru_cache
from hashlib import sha256
from time import perf_counter

from langsmith import traceable

from module.observability import record_metrics


@lru_cache
def get_store_managers():
    # Importing a graph must not construct clients or open/migrate a persistent DB.
    from LongTermMemory.store_manager import StoreManager, DEFAULT_STORE_PARAMS, REFERENCE_STORE_PARAMS
    from module.settings import get_settings

    settings = get_settings()
    common = {"persist_directory": settings.vector_db_path, "embedding_model": settings.embedding_model}
    return (
        StoreManager(**{**DEFAULT_STORE_PARAMS, **common, "collection_name": settings.default_collection}),
        StoreManager(**{**REFERENCE_STORE_PARAMS, **common, "collection_name": settings.reference_collection}),
    )


def document_summary(docs_and_scores):
    return [
        {
            "document_id": sha256(
                (str(doc.metadata.get("source", "")) + "\0" + doc.page_content).encode()
            ).hexdigest()[:16],
            "distance": float(distance),
        }
        for doc, distance in docs_and_scores
    ]


@traceable(name="vector_search", run_type="retriever", process_outputs=lambda output: {"documents": document_summary(output)})
async def search_collection(manager, question: str):
    start = perf_counter()
    docs = await asyncio.to_thread(manager.store.similarity_search_with_score, question, k=20)
    record_metrics(
        collection=manager.collection_name,
        retrieved_count=len(docs),
        duration_ms=round((perf_counter() - start) * 1000, 2),
        retrieved_documents=document_summary(docs),
    )
    return docs


async def search_documents(question: str):
    default, reference = await asyncio.to_thread(get_store_managers)
    results = await asyncio.gather(
        search_collection(default, question),
        search_collection(reference, question),
    )
    return [item for result in results for item in result]
