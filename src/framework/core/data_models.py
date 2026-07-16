from dataclasses import dataclass
from typing import List


@dataclass
class DocumentChunk:
    """
    Represents one chunk of text extracted from a PDF.
    """
    id: str
    text: str
    source: str


@dataclass
class SearchResult:
    """
    Represents the retrieved chunks from ChromaDB.
    """
    chunks: List[DocumentChunk]


@dataclass
class RAGResponse:
    """
    Represents the final response returned by an agent.
    """
    question: str
    retrieved_chunks: List[DocumentChunk]
    answer: str