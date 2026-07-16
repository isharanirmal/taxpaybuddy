from src.framework.loaders.pdf_loader import PDFLoader
from src.framework.rag.chunker import Chunker
from src.framework.rag.ingestion import Ingestion
from src.framework.database.chroma_store import ChromaStore

from src.agents.agent1_tin_registration.config import COLLECTION_NAME


loader = PDFLoader()
chunker = Chunker()
vector_store = ChromaStore()

ingestion = Ingestion(
    loader=loader,
    chunker=chunker,
    vector_store=vector_store,
)

ingestion.ingest_folder(
    folder_path="data/raw_pdfs/agent1_tin_registration",
    collection_name=COLLECTION_NAME,
)

print("Knowledge Base Created!")