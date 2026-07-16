"""
Tests for src/framework/rag/retriever.py and
src/framework/database/chroma_store.py.

ChromaStore itself is tested with a fake `chromadb` module injected
via sys.modules, so the suite never needs a real ChromaDB installation
or touches disk.
"""

import sys
import types

import pytest

from src.framework.rag.retriever import Retriever
from src.framework.core.data_models import DocumentChunk, SearchResult

from conftest import MockVectorStore, make_chunk


# Retriever

def test_retriever_delegates_to_vector_store_search(mock_vector_store):
    mock_vector_store.add_documents(
        chunks=[make_chunk("chunk one"), make_chunk("chunk two", chunk_id="test_2")],
        collection_name="agent2_individual_income_tax",
    )

    retriever = Retriever(mock_vector_store)

    result = retriever.retrieve(
        query="how much tax do I pay?",
        collection_name="agent2_individual_income_tax",
        top_k=1,
    )

    assert isinstance(result, SearchResult)
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "chunk one"


def test_retriever_returns_empty_result_for_unknown_collection(mock_vector_store):
    retriever = Retriever(mock_vector_store)

    result = retriever.retrieve(query="anything", collection_name="does_not_exist", top_k=3)

    assert result.chunks == []


def test_retriever_respects_top_k(mock_vector_store):
    mock_vector_store.add_documents(
        chunks=[make_chunk(f"chunk {i}", chunk_id=f"c{i}") for i in range(5)],
        collection_name="agent1_tin_registration",
    )

    retriever = Retriever(mock_vector_store)
    result = retriever.retrieve(query="q", collection_name="agent1_tin_registration", top_k=2)

    assert len(result.chunks) == 2



class _FakeCollection:
    def __init__(self):
        self._ids = []
        self._documents = []
        self._metadatas = []

    def add(self, ids, documents, metadatas):
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

    def query(self, query_texts, n_results):
        return {
            "ids": [self._ids[:n_results]],
            "documents": [self._documents[:n_results]],
            "metadatas": [self._metadatas[:n_results]],
        }


class _FakePersistentClient:
    def __init__(self, path=None):
        self.path = path
        self._collections = {}

    def get_or_create_collection(self, name):
        return self._collections.setdefault(name, _FakeCollection())

    def get_collection(self, name):
        return self._collections[name]


@pytest.fixture
def chroma_store_with_fake_backend(monkeypatch):
    """
    Injects a minimal fake `chromadb` module before importing
    ChromaStore, so we exercise ChromaStore's own translation logic
    (DocumentChunk <-> chromadb's dict format) without needing the
    real chromadb package installed.
    """

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = _FakePersistentClient
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    sys.modules.pop("src.framework.database.chroma_store", None)
    from src.framework.database.chroma_store import ChromaStore

    return ChromaStore(db_path="unused_in_tests")


def test_chroma_store_add_and_search_roundtrip(chroma_store_with_fake_backend):
    store = chroma_store_with_fake_backend

    chunks = [
        DocumentChunk(id="doc_1", text="TIN registration requires an NIC.", source="doc.pdf"),
        DocumentChunk(id="doc_2", text="Applications go through e-Services.", source="doc.pdf"),
    ]

    store.add_documents(chunks=chunks, collection_name="agent1_tin_registration")

    result = store.search(query="how do I register?", collection_name="agent1_tin_registration", k=2)

    assert len(result.chunks) == 2
    assert result.chunks[0].id == "doc_1"
    assert result.chunks[0].source == "doc.pdf"
    assert result.chunks[0].text == "TIN registration requires an NIC."


def test_chroma_store_search_respects_k(chroma_store_with_fake_backend):
    store = chroma_store_with_fake_backend

    chunks = [
        DocumentChunk(id=f"doc_{i}", text=f"text {i}", source="doc.pdf") for i in range(5)
    ]
    store.add_documents(chunks=chunks, collection_name="agent2_individual_income_tax")

    result = store.search(query="q", collection_name="agent2_individual_income_tax", k=3)

    assert len(result.chunks) == 3
