ROUTER_SYSTEM_PROMPT = """
You are 'RouterAgent' for TaxPayBuddy, a Sri Lankan Tax Assistant.

Choose exactly ONE of the following labels based on the user's query:
1. agent1_tin_registration
- TIN
- TIN Registration
- IRD Registration
- PIN
- SSID
- Taxpayer Registration
- e-Services
- Updating Registration
- Registration Documents

2. agent2_individual_income_tax
- Individual Income Tax
- Personal Income Tax
- APIT
- PAYE
- Salary Tax
- Employment Income
- Individual Tax Return
- Individual Tax Calculation

3. agent3_corporate_income_tax
- Corporate Income Tax
- Company Tax
- Business Tax
- Corporate Tax Return
- Company Tax Calculation

4. agent4_withholding_tax
- Withholding Tax
- WHT
- AIT
- Dividends
- Royalties
- Interest
- Rent
- Commission

5. general_fallback

Choose this only if the question is NOT related to Sri Lankan taxation.

Return ONLY

{
    "next_agent":"agent_name"
}

Never explain.
Never answer.
"""

VALID_LABELS = {
    "agent1_tin_registration",
    "agent2_individual_income_tax",
    "agent3_corporate_income_tax",
    "agent4_withholding_tax",
    "general_fallback",
}


class KeywordRoutingRule:
    """
    A single Strategy for the cheap keyword pre-check.

    Encapsulates "these keywords => this label" as a data object instead of
    a hard-coded if/elif branch, so new rules can be added by appending to
    a list rather than editing branching logic.
    """

    __slots__ = ("keywords", "label")

    def __init__(self, keywords, label: str):
        self.keywords = keywords
        self.label = label

    def match_count(self, query_clean: str) -> int:
        """How many of this rule's keywords appear in the query.
        Used so the MOST specific rule wins, instead of the FIRST rule
        that happens to match."""
        return sum(1 for keyword in self.keywords if keyword in query_clean)


KEYWORD_RULES = (
    KeywordRoutingRule(("tin","tin registration","register","registration","new tin","create tin","apply tin","get tin","obtain tin","tax identification number","tax id","tin number","register tin","open tin","activate tin"), "agent1_tin_registration"),
    KeywordRoutingRule(("individual","personal","person","employee","salary","employment income","paye","apit","monthly tax","resident individual","non-resident individual","relief"), "agent2_individual_income_tax"),
    KeywordRoutingRule(("corporate","company","business","organisation","organization","firm","enterprise","private limited","pvt ltd","limited","ltd","corporation","business income","company tax","corporate tax","entity"), "agent3_corporate_income_tax"),
    KeywordRoutingRule(("withholding","withholding tax","wht","ait","dividend","interest","rent","royalty","service fee","commission","deduction","deduct at source"), "agent4_withholding_tax")
)