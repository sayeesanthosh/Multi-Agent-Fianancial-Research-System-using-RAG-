
from rag.agent_context import get_agent_context
from risk_agent import run_risk_agent

question = """
Analyse Apple's financial risks for FY2025.
Evaluate debt, liquidity, revenue and profitability.
"""

evidence = get_agent_context(
    agent_name="Risk Analyst",
    question=question,
    top_k=6
)

report = run_risk_agent(
    question,
    evidence
)

print("\n========== RISK ANALYST REPORT ==========\n")
print(report)
