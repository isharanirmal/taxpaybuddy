from src.framework.core.data_models import RAGResponse
from src.framework.interfaces.interfaces import IAgent


GENERAL_FALLBACK_MESSAGE = (
    "I am sorry, I am a specialized Sri Lankan Tax Assistant and I can only "
    "assist you with tax-related queries."
)


class FallbackAgent(IAgent):
    """
    Null Object implementation of IAgent.

    Instead of special-casing 'no matching agent' with an if/else branch,
    this class plugs into the exact same registry/dispatch table as every
    other specialist agent and returns a fixed refusal RAGResponse. This
    keeps every entry in the router's registry polymorphic: everything is
    "just an IAgent", so the caller never needs to ask "is this the real
    agent or the fallback?".
    """

    def process_query(self, query: str) -> RAGResponse:
        return RAGResponse(
            question=query,
            retrieved_chunks=[],
            answer=GENERAL_FALLBACK_MESSAGE,
        )