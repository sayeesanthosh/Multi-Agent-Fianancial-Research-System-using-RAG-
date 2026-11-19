
from rag.retriever import retrieve_documents
from rag.generator import generate_answer


def ask_question(question, top_k=3):

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_documents(
        question,
        top_k=top_k
    )

    # Step 2: Generate an answer
    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer
