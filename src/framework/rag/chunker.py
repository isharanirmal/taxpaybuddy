from typing import List

from src.framework.core.data_models import DocumentChunk
from src.framework.utils.logger import Logger


class Chunker:
    """
    Splits text into smaller chunks for RAG.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        """

        Logger.info("Creating text chunks...")

        chunks = []
        start = 0
        chunk_number = 1

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end]

            chunks.append(
                DocumentChunk(
                    id=f"{source}_{chunk_number}",
                    text=chunk_text,
                    source=source,
                )
            )

            start += self.chunk_size - self.overlap
            chunk_number += 1

        Logger.success(f"{len(chunks)} chunks created.")

        return chunks