
import ollama

from evidence_tools import assess_source_quality

MODEL = "llama3.2"

# ==========================================
# RESEARCH ANALYST TOOLS
# ==========================================

TOOLS = [
    assess_source_quality
]

AVAILABLE_FUNCTIONS = {
    function.__name__: function
    for function in TOOLS
}


# ==========================================
# RESEARCH ANALYST INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """
You are a professional Financial Research
and Evidence Verification Analyst.

Your responsibilities:

1. Examine financial claims.
2. Evaluate supplied evidence.
3. Distinguish primary and secondary sources.
4. Identify missing citations.
5. Detect contradictions between supplied sources.
6. Identify outdated information.
7. Highlight unsupported claims.

You have access to a Python source metadata
assessment tool.

Use it when source metadata is provided.

IMPORTANT:

You do not have independent web browsing
or document retrieval at this stage.

Never claim that you independently verified
a source or financial claim.

Use these evidence classifications:

- Supported by supplied evidence
- Supported but not independently verified
- Contradicted by supplied evidence
- Unsupported
- Unknown

For each important claim, explain:

1. The claim
2. Supporting evidence
3. Source quality
4. Contradicting evidence
5. Verification status

Never invent citations, URLs or sources.

Your final report must contain:

1. Claims Examined
2. Evidence Assessment
3. Source Quality
4. Contradictions
5. Unsupported Claims
6. Missing Evidence
7. Research Limitations
"""


# ==========================================
# RESEARCH ANALYST EXECUTION
# ==========================================

def run_research_agent(question, context):

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

Available financial information and sources:
{context}

Evaluate the evidence carefully.
Use the source assessment tool when appropriate.
Do not invent verification.
"""
        }
    ]

    for iteration in range(5):

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )

        messages.append(response.message)

        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            return response.message.content

        for tool_call in tool_calls:

            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"\n[RESEARCH TOOL] {function_name}")

            function = AVAILABLE_FUNCTIONS.get(function_name)

            if function is None:
                result = "Error: Unknown function."

            else:

                try:
                    result = function(**arguments)

                except (TypeError, ValueError) as error:
                    result = f"Tool error: {error}"

                except Exception as error:
                    result = f"Unexpected tool error: {error}"

            print("[RESULT]", result)

            messages.append({
                "role": "tool",
                "tool_name": function_name,
                "content": str(result)
            })

    return "Maximum tool iterations reached."
