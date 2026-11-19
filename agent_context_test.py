
from rag.agent_context import get_agent_context


question = (
    "Analyse Apple's financial performance "
    "and identify its major financial risks."
)


agents = [
    "Finance Analyst",
    "Quant Analyst",
    "Risk Analyst",
    "Research Analyst"
]


for agent in agents:

    print("\n" + "=" * 60)
    print(f"TESTING: {agent}")
    print("=" * 60)

    context = get_agent_context(
        agent_name=agent,
        question=question,
        top_k=4
    )

    print(context)
