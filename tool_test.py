
import ollama



def calculate_growth(previous, current):

    # Convert inputs to strings first
    previous = str(previous).lower()
    current = str(current).lower()

    # Remove common financial units
    previous = previous.replace("crore", "").replace(" ", "").replace(",", "")
    current = current.replace("crore", "").replace(" ", "").replace(",", "")

    # Convert to numbers
    previous = float(previous)
    current = float(current)

    if previous == 0:
        raise ValueError("Previous revenue cannot be zero.")

    return round(
        ((current - previous) / previous) * 100,
        2
    )


messages = [
    {
        "role": "user",
        "content": """
        Calculate revenue growth from 100 crore
        to 150 crore.

        Use the calculate_growth tool.
        Explain the result.
        """
    }
]

tools = [calculate_growth]

# First model call
response = ollama.chat(
    model="llama3.2",
    messages=messages,
    tools=tools
)

messages.append(response.message)

# Execute requested tools
for tool_call in response.message.tool_calls or []:

    function_name = tool_call.function.name
    arguments = tool_call.function.arguments

    if function_name == "calculate_growth":

        result = calculate_growth(**arguments)

        print("PYTHON CALCULATION:", result)

        messages.append({
            "role": "tool",
            "tool_name": function_name,
            "content": str(result)
        })

# Ask model to interpret Python result
final_response = ollama.chat(
    model="llama3.2",
    messages=messages,
    tools=tools
)

print("\nFINAL ANALYSIS:")
print(final_response.message.content)
