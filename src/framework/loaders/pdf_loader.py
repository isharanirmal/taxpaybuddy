from pypdf import PdfReader

from src.framework.interfaces.interfaces import IDocumentLoader
from src.framework.utils.logger import Logger


class PDFLoader(IDocumentLoader):
    """
    Loads text from PDF files.
    """

    def load(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        """

        Logger.info(f"Loading PDF: {file_path}")

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        Logger.success("PDF Loaded Successfully.")

        return text