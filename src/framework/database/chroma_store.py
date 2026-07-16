import chromadb

from src.framework.interfaces.interfaces import IVectorStore
from src.framework.core.data_models import (
    DocumentChunk,
    SearchResult,
)
from src.framework.utils.logger import Logger


class ChromaStore(IVectorStore):
    """
    ChromaDB implementation of the Vector Store interface.
    """

    def __init__(self, db_path: str = "data/chroma_db"):

        self.client = chromadb.PersistentClient(path=db_path)

    def add_documents(
        self,
        chunks,
        collection_name,
    ):

        collection = self.client.get_or_create_collection(
            name=collection_name
        )

        Logger.info("Saving chunks to ChromaDB...")

        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"source": chunk.source}
                for chunk in chunks
            ]
        )

        Logger.success(f"{len(chunks)} chunks saved.")

    def search(
        self,
        query,
        collection_name,
        k=3,
    ):

        collection = self.client.get_collection(
            collection_name
        )

        results = collection.query(
            query_texts=[query],
            n_results=k,
        )

        chunks = []

        for i in range(len(results["documents"][0])):

            chunks.append(

                DocumentChunk(
                    id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    source=results["metadatas"][0][i]["source"],
                )

            )

        return SearchResult(chunks)