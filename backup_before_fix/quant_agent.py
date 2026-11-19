
import ollama

from financial_tools import (
    revenue_growth,
    calculate_cagr,
    profit_margin,
    operating_margin,
    debt_to_equity,
    current_ratio,
    return_on_equity,
    year_on_year_growth
)

MODEL = "llama3.2"

# ==========================================
# AVAILABLE PYTHON TOOLS
# ==========================================

TOOLS = [
    revenue_growth,
    calculate_cagr,
    profit_margin,
    operating_margin,
    debt_to_equity,
    current_ratio,
    return_on_equity,
    year_on_year_growth
]

AVAILABLE_FUNCTIONS = {
    function.__name__: function
    for function in TOOLS
}


# ==========================================
# QUANT ANALYST INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """
You are a professional Quantitative Financial Analyst.

You have access to Python calculation tools.

Use these tools whenever a financial calculation
is required and the necessary inputs are available.

Available calculations include:
- Revenue growth
- CAGR
- Net profit margin
- Operating margin
- Debt-to-equity ratio
- Current ratio
- Return on equity
- Year-on-year growth

IMPORTANT RULES:

1. Never invent financial figures.
2. Never calculate important ratios by guessing.
3. Use Python tools for numerical calculations.
4. Explain the results returned by Python.
5. Identify missing inputs.
6. Distinguish actual figures from assumptions.
7. Explain the financial meaning of each result.

Present your final report with:

1. Data Received
2. Calculations Performed
3. Quantitative Findings
4. Financial Interpretation
5. Missing Information
6. Conclusion
"""


# ==========================================
# TOOL EXECUTION LOOP
# ==========================================

def run_quant_agent(question, context=""):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
Research question:
{question}

Available financial information:
{context}

Analyse the data.
Use the Python tools when calculations are needed.
Explain your results.
"""
        }
    ]

    # Limit tool iterations to prevent endless loops.
    for iteration in range(5):

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )

        # Preserve the model's response and tool calls.
        messages.append(response.message)

        tool_calls = response.message.tool_calls or []

        # No more tool calls: model has finished.
        if not tool_calls:
            return response.message.content

        # Execute each requested Python function.
        for tool_call in tool_calls:

            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"\n[TOOL REQUEST] {function_name}")
            print("[ARGUMENTS]", arguments)

            function = AVAILABLE_FUNCTIONS.get(function_name)

            if function is None:

                result = "Error: Unknown function."

            else:

                try:
                    result = function(**arguments)

                except (TypeError, ValueError) as error:
                    result = f"Calculation error: {error}"

                except Exception as error:
                    result = f"Unexpected tool error: {error}"

            print("[PYTHON RESULT]", result)

            # Return the result to the model.
            messages.append({
                "role": "tool",
                "tool_name": function_name,
                "content": str(result)
            })

    return (
        "Maximum tool iterations reached. "
        "The analysis may be incomplete."
    )


# ==========================================
# STANDALONE EXECUTION
# ==========================================

if __name__ == "__main__":

    print("QUANTITATIVE ANALYST WITH PYTHON TOOLS")

    question = input("\nEnter your research question: ")

    context = input("\nEnter financial data: ")

    report = run_quant_agent(question, context)

    print("\n========== QUANT ANALYST REPORT ==========\n")
    print(report)
