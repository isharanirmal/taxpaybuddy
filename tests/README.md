# TaxPayBuddy - Test Suite

Unit tests for the `agents`, `ingestion`, `retrieval`, and `evaluation`
modules. All tests use mock LLM and vector-store clients
(`tests/conftest.py: MockLLMClient`, `MockVectorStore`) so the suite
runs fully offline - no Gemini API key and no real ChromaDB required.

## Setup

```bash
pip install -r requirements-dev.txt
```

## Running

From the project root (`TaxPayBuddy/`):

```bash
pytest tests/ -v
```

## What's covered

| File                  | Covers                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `test_agents.py`       | The 4 specialist agents, `BaseAgent`'s RAG flow, `RouterAgent`'s keyword/LLM routing and dispatch, `FallbackAgent` |
| `test_retrieval.py`    | `Retriever`, and `ChromaStore` (with a fake `chromadb` module injected, so no real ChromaDB needed) |
| `test_ingestion.py`    | `Chunker` splitting/overlap logic, `PDFLoader` (fake `pypdf` module), `Ingestion.ingest_folder` |
| `test_evaluation.py`   | `evaluation/scoring.py` helpers and the `evaluation/run_evaluation.py` pipeline (mocked router, no real API calls) |

`conftest.py` also exposes a `populated_vector_store` fixture pre-loaded
with one chunk per agent collection, so agent-level tests have
something realistic to retrieve.
