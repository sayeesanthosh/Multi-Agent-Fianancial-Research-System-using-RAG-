
from rag.retriever import retrieve_documents
from rag.generator import generate_answer

question = "What were the main reasons for Apple's revenue growth?"

# STEP 1: Retrieve relevant chunks
retrieved_chunks = retrieve_documents(
    question,
    top_k=5
)

print("\n========== RETRIEVAL RESULTS ==========")

for i, chunk in enumerate(retrieved_chunks):

    print("\nResult:", i + 1)
    print("Source:", chunk["metadata"]["source"])
    print("Page:", chunk["metadata"]["page"])
    print("Distance:", chunk["distance"])
    print("Retrieved text:")
    print(chunk["text"][:700])

# STEP 2: Generate answer
answer = generate_answer(
    question,
    retrieved_chunks
)

print("\n========== FINAL ANSWER ==========")
print(answer)
