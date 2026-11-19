
from rag.retriever import retrieve_documents

question = "What were the main reasons for revenue growth?"

results = retrieve_documents(
    query=question,
    top_k=3,
    source="data/documents/apple_annual_report.pdf"
)

print("\nMETADATA FILTERING RESULTS")

for i, result in enumerate(results):

    print("\n----------------------")
    print("Result:", i + 1)
    print("Source:", result["metadata"]["source"])
    print("Page:", result["metadata"]["page"])
    print("Distance:", result["distance"])
    print("Text:", result["text"][:400])
