from abc import ABC, abstractmethod
from typing import List

from src.framework.core.data_models import (
    DocumentChunk,
    SearchResult,
    RAGResponse,
)


class ILLMClient(ABC):
    """
    Interface for all Large Language Models.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class IVectorStore(ABC):
    """
    Interface for Vector Database.
    """

    @abstractmethod
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: str,
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 3,
    ) -> SearchResult:
        pass


class IDocumentLoader(ABC):
    """
    Interface for loading documents.
    """

    @abstractmethod
    def load(self, file_path: str) -> str:
        pass


class IAgent(ABC):
    """
    Interface for every TaxPayBuddy agent.
    """

    @abstractmethod
    def process_query(self, query: str) -> RAGResponse:
        pass