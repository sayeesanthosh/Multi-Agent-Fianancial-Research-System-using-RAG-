
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "You are a financial analyst. Explain what a P/E ratio means."
        }
    ]
)

print(response["message"]["content"])
