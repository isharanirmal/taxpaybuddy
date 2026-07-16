"""
Tests for src/framework/rag/chunker.py, src/framework/loaders/pdf_loader.py,
and src/framework/rag/ingestion.py.

PDFLoader's use of pypdf.PdfReader is monkeypatched so this suite never
needs a real PDF file or the pypdf package.
"""

import sys
import types

import pytest

from src.framework.rag.chunker import Chunker
from src.framework.rag.ingestion import Ingestion
from src.framework.core.data_models import DocumentChunk

from conftest import MockVectorStore


# Chunker

def test_chunker_splits_text_into_expected_number_of_chunks():
    chunker = Chunker(chunk_size=10, overlap=0)

    chunks = chunker.split_text(text="a" * 25, source="doc.pdf")

    assert len(chunks) == 3
    assert [len(c.text) for c in chunks] == [10, 10, 5]


def test_chunker_applies_overlap_between_chunks():
    chunker = Chunker(chunk_size=10, overlap=4)

    text = "0123456789ABCDEFGHIJ"  # 20 chars
    chunks = chunker.split_text(text=text, source="doc.pdf")

    assert chunks[0].text == "0123456789"
    assert chunks[1].text.startswith("6789")


def test_chunker_assigns_sequential_ids_and_source():
    chunker = Chunker(chunk_size=5, overlap=0)

    chunks = chunker.split_text(text="0123456789", source="mydoc.pdf")

    assert [c.id for c in chunks] == ["mydoc.pdf_1", "mydoc.pdf_2"]
    assert all(c.source == "mydoc.pdf" for c in chunks)
    assert all(isinstance(c, DocumentChunk) for c in chunks)


def test_chunker_handles_empty_text():
    chunker = Chunker(chunk_size=10, overlap=2)

    chunks = chunker.split_text(text="", source="empty.pdf")

    assert chunks == []



# PDFLoader (pypdf mocked out)


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pages = [_FakePage("Page one text."), _FakePage("Page two text.")]


@pytest.fixture
def pdf_loader_with_fake_backend(monkeypatch):
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = _FakePdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    sys.modules.pop("src.framework.loaders.pdf_loader", None)
    from src.framework.loaders.pdf_loader import PDFLoader

    return PDFLoader()


def test_pdf_loader_concatenates_all_page_text(pdf_loader_with_fake_backend):
    text = pdf_loader_with_fake_backend.load("fake/path.pdf")

    assert text == "Page one text.\nPage two text.\n"


def test_pdf_loader_skips_pages_with_no_extractable_text(monkeypatch):
    fake_pypdf = types.ModuleType("pypdf")

    class ReaderWithBlankPage:
        def __init__(self, file_path):
            self.pages = [_FakePage("Real text."), _FakePage(None), _FakePage("")]

    fake_pypdf.PdfReader = ReaderWithBlankPage
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    sys.modules.pop("src.framework.loaders.pdf_loader", None)
    from src.framework.loaders.pdf_loader import PDFLoader

    text = PDFLoader().load("fake/path.pdf")

    assert text == "Real text.\n"


# Ingestion

class _FakeLoader:
    """Stands in for PDFLoader: returns fixed text regardless of path."""

    def __init__(self, text="Sample tax document content."):
        self.text = text
        self.loaded_paths = []

    def load(self, file_path):
        self.loaded_paths.append(file_path)
        return self.text


def test_ingestion_processes_only_pdf_files(tmp_path, mock_vector_store):
    (tmp_path / "doc1.pdf").write_text("irrelevant, loader is mocked")
    (tmp_path / "doc2.pdf").write_text("irrelevant, loader is mocked")
    (tmp_path / "notes.txt").write_text("should be ignored")

    loader = _FakeLoader()
    chunker = Chunker(chunk_size=1000, overlap=0)
    ingestion = Ingestion(loader=loader, chunker=chunker, vector_store=mock_vector_store)

    ingestion.ingest_folder(str(tmp_path), collection_name="agent1_tin_registration")

    assert len(loader.loaded_paths) == 2
    assert all(path.endswith(".pdf") for path in loader.loaded_paths)


def test_ingestion_stores_chunks_in_the_given_collection(tmp_path, mock_vector_store):
    (tmp_path / "doc1.pdf").write_text("irrelevant, loader is mocked")

    loader = _FakeLoader(text="x" * 25)
    chunker = Chunker(chunk_size=10, overlap=0)
    ingestion = Ingestion(loader=loader, chunker=chunker, vector_store=mock_vector_store)

    ingestion.ingest_folder(str(tmp_path), collection_name="agent3_corporate_income_tax")

    stored = mock_vector_store.storage["agent3_corporate_income_tax"]
    assert len(stored) == 3  # 25 chars / 10 per chunk -> 3 chunks
    assert all(chunk.source == "doc1.pdf" for chunk in stored)


def test_ingestion_handles_empty_folder(tmp_path, mock_vector_store):
    loader = _FakeLoader()
    chunker = Chunker(chunk_size=1000, overlap=0)
    ingestion = Ingestion(loader=loader, chunker=chunker, vector_store=mock_vector_store)

    ingestion.ingest_folder(str(tmp_path), collection_name="agent4_withholding_tax")

    assert loader.loaded_paths == []
    assert mock_vector_store.storage.get("agent4_withholding_tax", []) == []
