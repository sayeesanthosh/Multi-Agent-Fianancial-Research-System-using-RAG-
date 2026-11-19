
import ollama

SYSTEM_PROMPT = """
You are the Research Director of a multi-agent
financial research organisation.

You receive reports from four specialist agents:

1. Finance Analyst
2. Quantitative Analyst
3. Risk Analyst
4. Research and Verification Analyst

YOUR RESPONSIBILITIES:

1. Understand the original research question.
2. Examine every specialist report.
3. Identify conclusions shared by multiple agents.
4. Identify disagreements between agents.
5. Examine whether disagreements are supported by evidence.
6. Challenge unsupported conclusions.
7. Give appropriate weight to reliable evidence.
8. Never force agreement between agents.
9. Preserve legitimate minority opinions.
10. Identify information that is still missing.

IMPORTANT:

Do not automatically trust a conclusion just because
multiple agents agree.

Do not invent facts or sources.

If the agents disagree, explain:
- What they disagree about.
- Why they disagree.
- Which argument has stronger evidence.
- What additional information is needed.

OUTPUT FORMAT:

1. Executive Summary
2. Financial Analysis
3. Quantitative Findings
4. Risk Assessment
5. Evidence and Verification
6. Agent Agreements
7. Agent Disagreements
8. Minority Opinions
9. Unresolved Questions
10. Final Research Conclusion

End with a confidence assessment:
Low / Moderate / High.

Explain why that confidence level is appropriate.

You are a research coordinator, not a stock
promotion or investment recommendation machine.
"""

def create_final_report(question, reports):

    combined_reports = ""

    for agent_name, report in reports.items():

        combined_reports += f"""

--- {agent_name} REPORT ---

{report}

"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
ORIGINAL RESEARCH QUESTION:

{question}

SPECIALIST AGENT REPORTS:

{combined_reports}

Analyse all reports and produce the final
multi-agent research report.
"""
            }
        ]
    )

    return response.message.content


if __name__ == "__main__":

    question = input(
        "Enter the main research question: "
    )

    reports = {}

    print("\nPaste each agent's report below.")
    print("Type END on a new line when finished.\n")

    agents = [
        "Finance Analyst",
        "Quant Analyst",
        "Risk Analyst",
        "Research Analyst"
    ]

    for agent in agents:

        print(f"\n--- {agent} ---")

        lines = []

        while True:

            line = input()

            if line.strip() == "END":
                break

            lines.append(line)

        reports[agent] = "\n".join(lines)

    print("\nGenerating final research report...\n")

    final_report = create_final_report(
        question,
        reports
    )

    print("\n========== FINAL RESEARCH REPORT ==========\n")
    print(final_report)
