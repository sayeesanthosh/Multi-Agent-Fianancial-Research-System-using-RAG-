
from agent_runtime import run_tool_agent

from fact_ledger import (
    list_facts,
    get_fact,
    calc_revenue_growth,
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
    calc_operating_margin,
    calc_net_margin,
    calc_growth,
    calc_current_ratio,
    calc_liabilities_to_equity,
    calc_debt_to_equity,
    calc_return_on_equity,
]

SYSTEM_PROMPT = """
You are a professional Financial Risk Analyst.

Use only verified facts and successful ledger-backed calculations.

CRITICAL TOOL ARGUMENT RULES:

get_fact requires:
- metric: base metric name
- year: fiscal year

Correct examples:
get_fact(metric="net_income", year=2025)
get_fact(metric="net_sales", year=2024)
get_fact(metric="operating_cash_flow", year=2025)
get_fact(metric="total_equity", year=2025)

Incorrect examples:
get_fact(metric="net_income_2025", year=2025)
get_fact(metric="operating_cash_flow_2024", year=2024)

Never append the fiscal year to the metric argument.

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

STRICT RULES:

1. Never invent financial figures.
2. Never pass raw numbers into calculation tools.
3. Use the verified fact ledger for financial inputs.
4. A failed calculation is unavailable.
5. Total liabilities are NOT total debt.
6. Current liabilities are NOT total debt.
7. Never describe liabilities-to-equity as debt-to-equity.
8. Use net_sales as revenue.
9. Use operating_income for operating margin.
10. Use net_income for net margin.
11. Do not predict future outcomes from historical ratios.
12. Do not classify a potential risk as a demonstrated problem
    without supporting evidence.
13. Do not repeat identical tool calls.
14. If evidence is missing, explicitly identify the limitation.

RISK AREAS:

- Debt and leverage
- Liquidity
- Revenue stability and growth
- Profitability
- Operating cash flow
- Market and operational risks
- Major vulnerabilities

For every material risk explain:
- Evidence
- Fiscal year
- Calculation or fact supporting it
- Why it matters
- Limitations
- Additional information required

Accuracy is more important than completeness.
"""

TASK = """
Perform a detailed financial risk assessment.

STEP 1:
Call list_facts to inspect the available verified financial facts.

STEP 2:
Use the appropriate calculation tools:

- Revenue growth: FY2025 and FY2024.
- Operating margin: FY2025, FY2024 and FY2023.
- Net margin: FY2025, FY2024 and FY2023.
- Current ratio: FY2025 and FY2024.
- Liabilities-to-equity: FY2025 and FY2024.
- Debt-to-equity: attempt only if actual total debt is verified.
- Return on equity: where the required equity inputs are available.
- Operating cash flow growth: use calc_growth with
  metric="operating_cash_flow" for FY2025 and FY2024.

Use the base metric name and pass the fiscal year separately.
Do not create metric names with years appended.

Do not repeatedly call tools with identical arguments.
Do not calculate ratios using guessed or substituted inputs.

STEP 3:
Interpret the evidence.

In particular:
- A current ratio below 1 indicates current liabilities exceed
  current assets, but does not alone prove insolvency.
- A decline in operating cash flow deserves investigation,
  but does not alone establish financial distress.
- High ROE should be interpreted in the context of the equity base.
- Missing total debt means debt-to-equity cannot be established.

Finish with:

1. Debt Risk
2. Liquidity Risk
3. Revenue Risk
4. Profitability Risk
5. Cash Flow Risk
6. Market and Operational Risks
7. Major Vulnerabilities
8. Missing Information
9. Overall Risk Assessment
"""

def run_risk_agent(question, context=""):
    return run_tool_agent(
        "Risk Analyst",
        SYSTEM_PROMPT,
        question,
        context,
        TASK,
        TOOLS,
    )
