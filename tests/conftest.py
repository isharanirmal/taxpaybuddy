"""
Shared pytest fixtures for the TaxPayBuddy test suite.

Ensures the project root is importable (so `from src...` and
`from evaluation...` work no matter where pytest is invoked from),
and provides mock implementations of ILLMClient / IVectorStore so
tests never make real Gemini API calls or need a real ChromaDB
instance.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.framework.interfaces.interfaces import ILLMClient, IVectorStore
from src.framework.core.data_models import DocumentChunk, SearchResult


class MockLLMClient(ILLMClient):
    """
    Drop-in replacement for GeminiClient that never touches the
    network. Two ways to control what it returns:

    - `responses`: a list of strings popped in call order (FIFO).
      Once exhausted, `default_response` is returned for every
      further call.
    - `router_fn`: an optional callable(prompt, system_instruction)
      -> str, used instead of the list when you need the response to
      depend on the input (e.g. routing tests that need a different
      JSON label per query).

    Every call is recorded in `self.calls` so tests can assert on
    what prompt/system_instruction the agent actually sent.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        default_response: str = "This is a mock answer about Sri Lankan tax.",
        router_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.router_fn = router_fn
        self.calls: List[Dict[str, str]] = []

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        self.calls.append({"prompt": prompt, "system_instruction": system_instruction})

        if self.router_fn is not None:
            return self.router_fn(prompt, system_instruction)

        if self.responses:
            return self.responses.pop(0)

        return self.default_response


class MockVectorStore(IVectorStore):
    """
    In-memory stand-in for ChromaStore. Documents are kept in a plain
    dict keyed by collection name, so `add_documents` followed by
    `search` behaves consistently without any real vector similarity
    (search just returns the first `k` chunks added to that
    collection, which is enough to exercise the retrieval contract).
    """

    def __init__(self):
        self.storage: Dict[str, List[DocumentChunk]] = {}

    def add_documents(self, chunks: List[DocumentChunk], collection_name: str) -> None:
        self.storage.setdefault(collection_name, []).extend(chunks)

    def search(self, query: str, collection_name: str, k: int = 3) -> SearchResult:
        chunks = self.storage.get(collection_name, [])[:k]
        return SearchResult(chunks=chunks)


def make_chunk(text: str, source: str = "test.pdf", chunk_id: str = "test_1") -> DocumentChunk:
    return DocumentChunk(id=chunk_id, text=text, source=source)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


@pytest.fixture
def populated_vector_store() -> MockVectorStore:
    """
    A MockVectorStore pre-loaded with a couple of chunks per agent
    collection, useful for agent-level tests that need `search` to
    return something non-empty.
    """

    store = MockVectorStore()

    collections = {
        "agent1_tin_registration": "To register for a TIN, apply via the IRD e-Services portal with your NIC.",
        "agent2_individual_income_tax": "Individual income tax is charged on employment income under APIT.",
        "agent3_corporate_income_tax": "Companies pay corporate income tax on their taxable business income.",
        "agent4_withholding_tax": "Withholding tax is deducted at source from dividends, interest and rent.",
    }

    for i, (collection_name, text) in enumerate(collections.items()):
        store.add_documents(
            chunks=[make_chunk(text, source=f"{collection_name}.pdf", chunk_id=f"{collection_name}_1")],
            collection_name=collection_name,
        )

    return store
