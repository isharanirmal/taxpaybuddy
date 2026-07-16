import os

from src.framework.loaders.pdf_loader import PDFLoader
from src.framework.rag.chunker import Chunker
from src.framework.interfaces.interfaces import IVectorStore
from src.framework.utils.logger import Logger


class Ingestion:
    """
    Reads PDFs, creates chunks, and stores them in the vector database.
    """

    def __init__(
        self,
        loader: PDFLoader,
        chunker: Chunker,
        vector_store: IVectorStore,
    ):
        self.loader = loader
        self.chunker = chunker
        self.vector_store = vector_store

    def ingest_folder(self, folder_path: str, collection_name: str):
        """
        Process every PDF inside a folder.
        """

        Logger.info("Starting document ingestion...")

        for file in os.listdir(folder_path):

            if file.endswith(".pdf"):

                pdf_path = os.path.join(folder_path, file)

                Logger.info(f"Processing: {file}")

                text = self.loader.load(pdf_path)

                chunks = self.chunker.split_text(
                    text=text,
                    source=file,
                )

                self.vector_store.add_documents(
                    chunks=chunks,
                    collection_name=collection_name,
                )

        Logger.success("Knowledge Base Created Successfully.")