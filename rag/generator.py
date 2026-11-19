
import ollama


def generate_answer(question, retrieved_chunks):

    # Prepare the retrieved information
    context = ""

    for i, chunk in enumerate(retrieved_chunks):

        context += f"""
        Document {i + 1}

        Source: {chunk['metadata']['source']}
        Page: {chunk['metadata']['page']}

        Content:
        {chunk['text']}

        """

    # Instructions for Llama
    prompt = f"""
    You are a professional financial research analyst.

    Answer the user's question using the provided context.

    Rules:
    1. Use the supplied context as your primary evidence.
    2. Do not invent financial facts or figures.
    3. If the context does not contain enough information,
       clearly say so.
    4. Explain your reasoning clearly.
    5. Mention the source document and page numbers.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Provide a clear and structured answer.
    """

    # Generate answer using Llama 3.2
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
