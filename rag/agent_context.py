
from rag.hybrid_retriever import hybrid_search


# ==========================================
# AGENT-SPECIFIC RETRIEVAL INSTRUCTIONS
# ==========================================

AGENT_SEARCH_FOCUS = {

    "Finance Analyst": """
    Focus on:
    revenue, net sales, segment performance,
    operating income, net income, profit margins,
    business performance, management discussion,
    financial statements and year-over-year changes.
    """,

    "Quant Analyst": """
    Focus on:
    reported financial figures, numerical data,
    revenue, expenses, net income, assets,
    liabilities, equity, cash flow,
    accounting definitions and historical values.
    """,

    "Risk Analyst": """
    Focus on:
    risk factors, debt, borrowings, liquidity,
    cash flow risks, liabilities, interest expense,
    market risks, operational risks,
    contingencies and financial vulnerabilities.
    """,

    "Research Analyst": """
    Focus on:
    source evidence, reported claims,
    supporting disclosures, contradictory information,
    accounting notes, footnotes, limitations,
    reporting periods and missing evidence.
    """
}


# ==========================================
# BUILD AGENT-SPECIFIC SEARCH QUERY
# ==========================================

def build_agent_query(agent_name, question):

    focus = AGENT_SEARCH_FOCUS.get(agent_name)

    if focus is None:
        raise ValueError(
            f"Unknown agent: {agent_name}"
        )

    return f"""
    Research question:
    {question}

    Agent perspective:
    {focus}
    """


# ==========================================
# FORMAT RETRIEVED EVIDENCE
# ==========================================

def format_evidence(results):

    if not results:
        return (
            "No relevant document evidence was retrieved. "
            "Do not treat missing retrieval results as proof "
            "that the information does not exist."
        )

    formatted_context = ""

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "Unknown source"
        )

        page = metadata.get(
            "page",
            "Unknown page"
        )

        retrieval_methods = ", ".join(
            result["retrieval_sources"]
        )

        formatted_context += f"""

--- EVIDENCE {index} ---

Source: {source}
Page: {page}
Retrieval methods: {retrieval_methods}

Content:
{result["text"]}

"""

    return formatted_context


# ==========================================
# MAIN AGENT CONTEXT FUNCTION
# ==========================================

def get_agent_context(
    agent_name,
    question,
    top_k=4,
    source=None
):

    search_query = build_agent_query(
        agent_name,
        question
    )

    results = hybrid_search(
        query=search_query,
        top_k=top_k,
        source=source
    )

    evidence = format_evidence(results)

    return evidence
