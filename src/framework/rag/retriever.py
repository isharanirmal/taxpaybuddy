from src.framework.interfaces.interfaces import IVectorStore


class Retriever:
    """
    Retrieves the most relevant document chunks.
    """

    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
    ):

        return self.vector_store.search(
            query=query,
            collection_name=collection_name,
            k=top_k,
        )