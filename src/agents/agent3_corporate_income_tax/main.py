from src.framework.database.chroma_store import ChromaStore
from src.framework.llm.gemini_client import GeminiClient
from src.agents.agent3_corporate_income_tax.agent import CorporateIncomeTaxAgent


def main():

    # Create shared services
    llm = GeminiClient()

    vector_store = ChromaStore()

    # Create Agent 3
    agent = CorporateIncomeTaxAgent(
        llm=llm,
        vector_store=vector_store
    )

    print("=" * 50)
    print("TaxPayBuddy - Agent 3")
    print("Corporate Income Tax Assistant")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        response = agent.process_query(question)

        print("\nAnswer:")
        print(response.answer)


if __name__ == "__main__":
    main()