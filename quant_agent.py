
from agent_runtime import run_tool_agent

from fact_ledger import (
    list_facts,
    get_fact,
    calc_revenue_growth,
    calc_cagr,
    calc_operating_margin,
    calc_net_margin,
    calc_growth,
    calc_current_ratio,
    calc_liabilities_to_equity,
    calc_debt_to_equity,
    calc_return_on_equity,
)

TOOLS = [
    list_facts,
    get_fact,
    calc_revenue_growth,
    calc_cagr,
    calc_operating_margin,
    calc_net_margin,
    calc_growth,
    calc_current_ratio,
    calc_liabilities_to_equity,
    calc_debt_to_equity,
    calc_return_on_equity,
]

SYSTEM_PROMPT = """
You are a professional Quantitative Financial Analyst.

You MUST use the VERIFIED FACT LEDGER and ledger-backed
calculation tools.

CRITICAL TOOL ARGUMENT RULES:

The get_fact tool takes exactly these arguments:
- metric: the base metric name
- year: the fiscal year

CORRECT:
get_fact(metric="net_income", year=2025)
get_fact(metric="net_sales", year=2024)
get_fact(metric="total_equity", year=2025)

INCORRECT:
get_fact(metric="net_income_2025", year=2025)
get_fact(metric="net_sales_2024", year=2024)

The year is a separate argument. Never append the year to metric.

Valid metric names:
net_income
net_sales
operating_cash_flow
operating_income
total_assets
total_current_assets
total_current_liabilities
total_debt
total_equity
total_liabilities
total_liabilities_and_equity

ANALYSIS RULES:

1. Never pass raw financial numbers into calculation tools.
2. Never use functions from financial_tools.py.
3. Never invent or estimate missing financial inputs.
4. Use only successful ledger-backed calculation results.
5. A failed calculation is unavailable, not zero.
6. Total liabilities are not total debt.
7. Liabilities-to-equity is not debt-to-equity.
8. Use net_sales as revenue.
9. Use operating_income for operating margin.
10. Use net_income for net profit margin.
11. Use total_current_assets and total_current_liabilities
    for current ratio.
12. Historical results are not forecasts.
13. Explain the meaning and limitations of each ratio.
14. Do not repeat identical tool calls.

Use the ledger-backed tools to calculate:
- Revenue growth
- CAGR when sufficient years are available
- Operating margin
- Net profit margin
- Year-on-year growth
- Current ratio
- Liabilities-to-equity
- Debt-to-equity only if actual total debt exists
- Return on equity when sufficient equity data exists

Report structure:

1. Data Received
2. Calculations Performed
3. Quantitative Findings
4. Trend Analysis
5. Financial Interpretation
6. Missing Information
7. Conclusion
"""

TASK = """
Perform a quantitative analysis of the company's financial statements.

First call list_facts to inspect available verified facts.

Then perform relevant ledger-backed calculations for FY2025,
FY2024 and FY2023 wherever the required facts exist.

For growth calculations, use calc_growth with the base metric
name and fiscal year.

For CAGR, use the earliest and latest available fiscal years.

Do not request a year for which required input facts are missing.

Do not invent total debt. If debt-to-equity cannot be calculated,
state why.

Explain what the calculated results indicate. Distinguish
calculation results from interpretations.

Finish using the required report structure.
"""


def run_quant_agent(question, context=""):
    return run_tool_agent(
        "Quant Analyst",
        SYSTEM_PROMPT,
        question,
        context,
        TASK,
        TOOLS,
    )
