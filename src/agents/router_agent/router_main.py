import json

from src.framework.core.data_models import RAGResponse
from src.framework.interfaces.interfaces import IAgent
from src.framework.database.chroma_store import ChromaStore
from src.framework.llm.gemini_client import GeminiClient
from src.agents.agent1_tin_registration.agent import TINRegistrationAgent
from src.agents.agent2_individual_income_tax.agent import IndividualIncomeTaxAgent
from src.agents.agent3_corporate_income_tax.agent import CorporateIncomeTaxAgent
from src.agents.agent4_withholding_tax.agent import WithholdingTaxAgent

from src.agents.router_agent.fallback_agent import FallbackAgent
from src.agents.router_agent.routing_rules import (
    ROUTER_SYSTEM_PROMPT,
    VALID_LABELS,
    KEYWORD_RULES,
)


class RouterAgent:
    """
    Orchestrator agent. Decides which specialist tax agent should
    handle a given user query, and delegates execution to it.

    Routing is implemented as a Dispatch Table (a form of the Strategy
    pattern): both the keyword pre-check and the final agent selection are
    plain data lookups, so there is no if/elif ladder to extend whenever a
    new agent or keyword is added.
    """

    def __init__(self, llm, vector_store):

        self.llm_client = llm
        self.vector_store = vector_store
        self.last_routed_label = None

        self.agent1 = TINRegistrationAgent(llm=self.llm_client, vector_store=self.vector_store)
        self.agent2 = IndividualIncomeTaxAgent(llm=self.llm_client, vector_store=self.vector_store)
        self.agent3 = CorporateIncomeTaxAgent(llm=self.llm_client, vector_store=self.vector_store)
        self.agent4 = WithholdingTaxAgent(llm=self.llm_client, vector_store=self.vector_store)
        self.fallback_agent = FallbackAgent()

        self.last_routed_label: str | None = None

        self._agent_registry: dict[str, IAgent] = {
            "agent1_tin_registration": self.agent1,
            "agent2_individual_income_tax": self.agent2,
            "agent3_corporate_income_tax": self.agent3,
            "agent4_withholding_tax": self.agent4,
            "general_fallback": self.fallback_agent,
        }

    def _fast_keyword_route(self, query_clean: str):
        """
        Cheap keyword pre-check so obvious queries skip the LLM call.

        Scores every rule by how many of its keywords appear in the query
        and returns the label with the HIGHEST score, instead of the
        first rule that happens to match. This stops a generic word that
        overlaps between two agents from hijacking a query just because
        its rule was checked first.

        Returns a label, or None if nothing matched confidently (no
        keywords found, or a tie between two agents) -- in which case
        route_and_execute() falls back to the LLM router.
        """

        scores = [
            (rule.match_count(query_clean), rule.label)
            for rule in KEYWORD_RULES
        ]

        best_score, best_label = max(scores, key=lambda item: item[0])

        if best_score == 0:
            return None

        top_labels = {label for score, label in scores if score == best_score}
        if len(top_labels) > 1:
            return None

        return best_label

    def _llm_route(self, user_query: str) -> str:
        """
        Falls back to the Gemini LLM to classify the query when the
        keyword pre-check is not confident enough.
        """

        try:
            raw_response = self.llm_client.generate(
                prompt=user_query,
                system_instruction=ROUTER_SYSTEM_PROMPT,
            )

            cleaned = raw_response.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

            decision = json.loads(cleaned)
            label = decision.get("next_agent", "general_fallback")

            return label if label in VALID_LABELS else "general_fallback"

        except Exception:
            return "general_fallback"

    def route_and_execute(self, user_query: str) -> RAGResponse:
        """
        Routes the user's query to the right specialist agent and
        returns its RAGResponse. Out-of-domain queries get a fixed
        refusal message instead of hitting any agent.
        """

        query_clean = user_query.lower()

        selected_label = self._fast_keyword_route(query_clean) or self._llm_route(user_query)

        self.last_routed_label = selected_label
        
        print(f"[ROUTER AGENT LOG] Routing query to: --> {selected_label}")

        selected_agent = self._agent_registry.get(selected_label, self.fallback_agent)

        return selected_agent.process_query(user_query)


def main():

    llm = GeminiClient()

    vector_store = ChromaStore()

    router = RouterAgent(llm=llm, vector_store=vector_store)

    print("=" * 50)
    print("TaxPayBuddy - Router Agent")
    print("Ask about TIN registration, individual/corporate income tax,")
    print("or withholding tax. Type 'exit' to quit.")
    print("=" * 50)

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        response = router.route_and_execute(question)

        print("\nAnswer:")
        print(response.answer)


if __name__ == "__main__":
    main()