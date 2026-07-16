from src.framework.core.base_agent import BaseAgent

from .config import (
    COLLECTION_NAME,
    TOP_K,
    SYSTEM_PROMPT_FILE
)


class WithholdingTaxAgent(BaseAgent):
    """
    Agent responsible for answering
    Withholding Tax questions.
    """

    def __init__(self, llm, vector_store):

        super().__init__(
            llm=llm,
            vector_store=vector_store,
            collection_name=COLLECTION_NAME,
            system_prompt_file=SYSTEM_PROMPT_FILE,
            top_k=TOP_K
        )