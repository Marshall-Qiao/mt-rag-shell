import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from starlette.websockets import WebSocketDisconnect

from AIAgentPoC.tools import financial_doc_retriever as retriever
from LongTermMemory.files import find_non_added_files, get_files_created_at
from LongTermMemory.metadata import update_docs_metadata


def test_same_names_in_subdirectories_have_matching_portable_identities(tmp_path):
    docs = []
    for name in ("a/report.pdf", "b/report.pdf"):
        path = tmp_path / name
        path.parent.mkdir()
        path.write_text("synthetic document")
        docs.append(Document(page_content="synthetic", metadata={"source": str(path)}))
    update_docs_metadata(docs, source_root=tmp_path)
    existing = {"metadatas": [doc.metadata for doc in docs]}
    store = SimpleNamespace(get=lambda **kwargs: existing)
    assert {doc.metadata["source"] for doc in docs} == {"a/report.pdf", "b/report.pdf"}
    assert get_files_created_at(tmp_path) == {
        (doc.metadata["source"], doc.metadata["created_at"]) for doc in docs
    }
    assert not find_non_added_files(store, tmp_path)["new_sources"]


async def test_collections_retrieve_concurrently_and_keep_result_order(monkeypatch):
    entered = set()
    both_entered = asyncio.Event()

    async def search(manager, question):
        entered.add(manager)
        if len(entered) == 2:
            both_entered.set()
        await both_entered.wait()
        return [(manager, question)]

    monkeypatch.setattr(retriever, "get_store_managers", lambda: ("default", "reference"))
    monkeypatch.setattr(retriever, "search_collection", search)
    result = await asyncio.wait_for(retriever.search_documents("synthetic"), timeout=2)
    assert result == [("default", "synthetic"), ("reference", "synthetic")]


def test_legacy_initial_import_adds_documents_once(tmp_path, monkeypatch):
    from module import load_vector_store as legacy

    calls = []
    store = SimpleNamespace(add_documents=lambda docs: calls.append(docs))
    docs = [Document(page_content="synthetic")]
    monkeypatch.setattr(legacy, "Chroma", lambda **kwargs: store)
    monkeypatch.setattr(legacy, "PyPDFDirectoryLoader", lambda *args, **kwargs: SimpleNamespace(load=lambda: docs))
    monkeypatch.setattr(legacy, "SemanticChunker", lambda *args, **kwargs: SimpleNamespace(split_documents=lambda items: items))
    assert legacy.load_vector_store(str(tmp_path / "nonexistent"), object()) is store
    assert calls == [docs]


def test_unchanged_import_does_not_parse_or_embed(tmp_path, monkeypatch):
    from LongTermMemory import store_manager

    path = tmp_path / "report.pdf"
    path.write_text("synthetic")
    manager = store_manager.StoreManager.__new__(store_manager.StoreManager)
    manager.store = SimpleNamespace(get=lambda **kwargs: {"metadatas": [{
        "source": path.name, "created_at": path.stat().st_mtime,
    }]})

    def forbidden(*args, **kwargs):
        pytest.fail("Unchanged documents must not be parsed or embedded")

    monkeypatch.setattr(store_manager, "pdf_loader", forbidden)
    monkeypatch.setattr(manager, "_add_docs_to_vector_store", forbidden)
    manager.add_files(str(tmp_path))


def test_liveness_does_not_initialize_models_or_stores(server, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Liveness must not initialize a model or persistent store")

    monkeypatch.setattr(retriever, "get_store_managers", forbidden)
    with TestClient(server.app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}


def test_websocket_rejects_untrusted_browser_origin(server, fake_rag):
    with TestClient(server.app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect("/chat", headers={"origin": "https://untrusted.invalid"}):
                pass
        assert rejected.value.code == 1008
        assert not fake_rag.entry_inputs
        with client.websocket_connect("/chat", headers={"origin": "http://localhost:3000"}):
            pass
