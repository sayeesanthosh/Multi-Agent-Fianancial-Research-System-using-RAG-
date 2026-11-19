
from rag.retriever import retrieve_documents

question = "What were the main reasons for Apple's revenue growth?"

results = retrieve_documents(question, top_k=3)

print("\nUSER QUESTION:")
print(question)

print("\nRETRIEVED DOCUMENTS:")

for i, result in enumerate(results):

    print("\n-----------------------------")
    print("Result:", i + 1)
    print("Page:", result["metadata"]["page"])
    print("Source:", result["metadata"]["source"])
    print("Distance:", result["distance"])
    print("Text:")
    print(result["text"][:500])
