"""
Tests for src/agents (specialist agents + router) using mock LLM and
vector store clients so no real Gemini/ChromaDB calls are made.
"""

import json

import pytest

from src.agents.agent1_tin_registration.agent import TINRegistrationAgent
from src.agents.agent2_individual_income_tax.agent import IndividualIncomeTaxAgent
from src.agents.agent3_corporate_income_tax.agent import CorporateIncomeTaxAgent
from src.agents.agent4_withholding_tax.agent import WithholdingTaxAgent
from src.agents.router_agent.router_main import RouterAgent
from src.agents.router_agent.fallback_agent import FallbackAgent, GENERAL_FALLBACK_MESSAGE
from src.framework.core.data_models import RAGResponse

from conftest import MockLLMClient, MockVectorStore


SPECIALIST_AGENT_CLASSES = [
    TINRegistrationAgent,
    IndividualIncomeTaxAgent,
    CorporateIncomeTaxAgent,
    WithholdingTaxAgent,
]


@pytest.mark.parametrize("agent_class", SPECIALIST_AGENT_CLASSES)
def test_specialist_agent_returns_rag_response(agent_class, mock_llm, populated_vector_store):
    """
    Every specialist agent should run the retrieve -> prompt -> generate
    flow and return a well-formed RAGResponse, regardless of which
    specific agent it is (they all share BaseAgent).
    """

    agent = agent_class(llm=mock_llm, vector_store=populated_vector_store)

    response = agent.process_query("What do I need to do?")

    assert isinstance(response, RAGResponse)
    assert response.question == "What do I need to do?"
    assert response.answer == mock_llm.default_response
    assert len(response.retrieved_chunks) == 1


@pytest.mark.parametrize("agent_class", SPECIALIST_AGENT_CLASSES)
def test_specialist_agent_retrieves_from_its_own_collection(agent_class, mock_llm, populated_vector_store):
    """
    Each agent must only ever search its own collection, never another
    agent's, so answers stay on-topic.
    """

    agent = agent_class(llm=mock_llm, vector_store=populated_vector_store)

    response = agent.process_query("test question")

    retrieved_sources = {chunk.source for chunk in response.retrieved_chunks}
    assert retrieved_sources == {f"{agent.collection_name}.pdf"}


def test_base_agent_includes_retrieved_context_in_prompt(mock_llm, populated_vector_store):
    agent = IndividualIncomeTaxAgent(llm=mock_llm, vector_store=populated_vector_store)

    agent.process_query("How is APIT calculated?")

    assert len(mock_llm.calls) == 1
    system_instruction = mock_llm.calls[0]["system_instruction"]
    assert "Context:" in system_instruction
    assert "Reference 1:" in system_instruction
    assert "APIT" in system_instruction


def test_base_agent_handles_no_retrieved_chunks(mock_llm, mock_vector_store):
    """
    An empty vector store (nothing retrieved) shouldn't crash the
    agent - it should still call the LLM with an empty context block.
    """

    agent = TINRegistrationAgent(llm=mock_llm, vector_store=mock_vector_store)

    response = agent.process_query("How do I register?")

    assert response.retrieved_chunks == []
    assert response.answer == mock_llm.default_response


def test_fallback_agent_returns_fixed_refusal():
    agent = FallbackAgent()

    response = agent.process_query("What's the weather today?")

    assert isinstance(response, RAGResponse)
    assert response.answer == GENERAL_FALLBACK_MESSAGE
    assert response.retrieved_chunks == []


# RouterAgent

@pytest.fixture
def router(mock_llm, populated_vector_store):
    return RouterAgent(llm=mock_llm, vector_store=populated_vector_store)


@pytest.mark.parametrize(
    "query,expected_label",
    [
        ("How do I register a new TIN?", "agent1_tin_registration"),
        ("What is my APIT salary tax deduction?", "agent2_individual_income_tax"),
        ("What is the corporate tax rate for my company?", "agent3_corporate_income_tax"),
        ("What withholding tax applies to this dividend?", "agent4_withholding_tax"),
    ],
)
def test_fast_keyword_route_matches_expected_agent(router, query, expected_label):
    label = router._fast_keyword_route(query.lower())
    assert label == expected_label


def test_fast_keyword_route_returns_none_when_no_keyword_matches(router):
    label = router._fast_keyword_route("what is the meaning of life?")
    assert label is None


def test_llm_route_parses_valid_json_label(mock_vector_store):
    llm = MockLLMClient(responses=[json.dumps({"next_agent": "agent2_individual_income_tax"})])
    router = RouterAgent(llm=llm, vector_store=mock_vector_store)

    label = router._llm_route("some ambiguous tax question")

    assert label == "agent2_individual_income_tax"


def test_llm_route_falls_back_on_invalid_label(mock_vector_store):
    llm = MockLLMClient(responses=[json.dumps({"next_agent": "not_a_real_agent"})])
    router = RouterAgent(llm=llm, vector_store=mock_vector_store)

    label = router._llm_route("some ambiguous question")

    assert label == "general_fallback"


def test_llm_route_falls_back_on_malformed_json(mock_vector_store):
    llm = MockLLMClient(responses=["this is not json at all"])
    router = RouterAgent(llm=llm, vector_store=mock_vector_store)

    label = router._llm_route("some ambiguous question")

    assert label == "general_fallback"


def test_route_and_execute_dispatches_to_correct_specialist(router, populated_vector_store):
    response = router.route_and_execute("How do I register for a TIN?")

    assert isinstance(response, RAGResponse)
    assert response.retrieved_chunks[0].source == "agent1_tin_registration.pdf"


def test_route_and_execute_uses_fallback_for_out_of_domain_query(mock_vector_store):
    llm = MockLLMClient(responses=[json.dumps({"next_agent": "general_fallback"})])
    router = RouterAgent(llm=llm, vector_store=mock_vector_store)

    response = router.route_and_execute("What's the best pizza topping?")

    assert response.answer == GENERAL_FALLBACK_MESSAGE
    assert response.retrieved_chunks == []


def test_route_and_execute_never_raises_keyerror_on_unknown_label(mock_vector_store):
    """
    Regression test for the dispatch-table lookup: even if the LLM
    somehow returned a label outside VALID_LABELS, _llm_route already
    coerces it to 'general_fallback', so route_and_execute must never
    raise KeyError.
    """

    llm = MockLLMClient(responses=[json.dumps({"next_agent": "totally_made_up"})])
    router = RouterAgent(llm=llm, vector_store=mock_vector_store)

    response = router.route_and_execute("some unrelated question")

    assert response.answer == GENERAL_FALLBACK_MESSAGE
