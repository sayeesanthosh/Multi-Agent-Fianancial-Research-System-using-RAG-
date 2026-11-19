
from rag.embeddings import generate_embedding
from rag.vector_store import collection


def retrieve_documents(query, top_k=3):

    # Convert the user question into an embedding
    query_embedding = generate_embedding(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    # Extract the matching documents and metadata
    for i in range(len(results["documents"][0])):

        chunk = {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }

        retrieved_chunks.append(chunk)

    return retrieved_chunks
