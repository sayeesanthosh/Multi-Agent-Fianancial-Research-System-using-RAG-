
from rag.hybrid_retriever import hybrid_search

question = "What were Apple's main revenue growth drivers in 2025?"

results = hybrid_search(
    query=question,
    top_k=5,
    source="data/documents/apple_annual_report.pdf"
)

print("\nHYBRID SEARCH RESULTS")

for i, result in enumerate(results):

    print("\n---------------------------")

    print("Rank:", i + 1)
    print("Page:", result["metadata"]["page"])

    print("RRF Score:", result["rrf_score"])

    print(
        "Retrieved by:",
        result["retrieval_sources"]
    )

    print("Text:")
    print(result["text"][:500])
