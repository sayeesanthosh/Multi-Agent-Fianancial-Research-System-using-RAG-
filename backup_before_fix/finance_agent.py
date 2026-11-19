from agent_runtime import run_tool_agent
from fact_ledger import (
    list_facts,
    get_fact,
    calc_revenue_growth,
    calc_operating_margin,
    calc_net_margin,
    calc_growth,
    calc_current_ratio,
    calc_return_on_equity,
)

TOOLS = [
    list_facts,
    get_fact,
    calc_revenue_growth,
    calc_operating_margin,
    calc_net_margin,
    calc_growth,
    calc_current_ratio,
    calc_return_on_equity,
]

SYSTEM_PROMPT = """
You are a professional Financial Statement Analyst.

Your role: financial statement performance.

1. Analyse revenue, operating income and net income over time.
2. Interpret margins and liquidity ratios.
3. Identify financial strengths and weaknesses.
4. Identify missing financial information.
5. Distinguish historical performance from forecasts.

Your report must contain:

1. Financial Performance
2. Revenue Analysis
3. Profitability Analysis
4. Financial Strengths
5. Financial Weaknesses
6. Missing Information
7. Overall Assessment
"""

TASK = """Analyse the company's financial statement performance.
Call the calculation tools with a fiscal year or metric name.
Then write the report."""


def run_finance_agent(question, context=""):
    return run_tool_agent(
        "Finance Analyst", SYSTEM_PROMPT, question, context, TASK, TOOLS
    )