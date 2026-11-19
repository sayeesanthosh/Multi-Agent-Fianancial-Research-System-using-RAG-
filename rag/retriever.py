
from rag.embeddings import generate_embedding
from rag.vector_store import collection


def retrieve_documents(query, top_k=3, source=None):

    # Convert question into embedding
    query_embedding = generate_embedding(query)

    # Prepare search parameters
    search_parameters = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }

    # Apply metadata filter if provided
    if source is not None:

        search_parameters["where"] = {
            "source": source
        }

    # Search ChromaDB
    results = collection.query(**search_parameters)

    retrieved_chunks = []

    for i in range(len(results["documents"][0])):

        chunk = {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }

        retrieved_chunks.append(chunk)

    return retrieved_chunks
