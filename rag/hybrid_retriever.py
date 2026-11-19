
import re

from rank_bm25 import BM25Okapi

from rag.embeddings import generate_embedding
from rag.vector_store import collection


def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9.]+\b",
        text.lower()
    )


def hybrid_search(query, top_k=3, source=None):

    # STEP 1: Get documents eligible for keyword search
    where = None

    if source is not None:
        where = {"source": source}

    stored = collection.get(
        where=where,
        include=["documents", "metadatas"]
    )

    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]

    if not ids:
        return []

    # STEP 2: BM25 keyword search
    tokenized_documents = [
        tokenize(doc) for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    keyword_ranking = sorted(
        range(len(ids)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    # Ignore documents with no keyword match
    keyword_ranking = [
        i for i in keyword_ranking
        if bm25_scores[i] > 0
    ][:top_k * 3]

    # STEP 3: Vector search
    query_embedding = generate_embedding(query)

    dense_count = min(top_k * 3, len(ids))

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=dense_count,
        where=where
    )

    vector_ids = vector_results["ids"][0]

    # STEP 4: Reciprocal Rank Fusion
    rrf_scores = {}
    retrieval_sources = {}

    k = 60

    for rank, chunk_id in enumerate(
        vector_ids, start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        retrieval_sources.setdefault(
            chunk_id, set()
        ).add("semantic")

    for rank, index in enumerate(
        keyword_ranking, start=1
    ):

        chunk_id = ids[index]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        retrieval_sources.setdefault(
            chunk_id, set()
        ).add("keyword")

    # STEP 5: Sort by combined score
    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    document_lookup = dict(zip(ids, documents))
    metadata_lookup = dict(zip(ids, metadatas))

    final_results = []

    for chunk_id in ranked_ids[:top_k]:

        final_results.append({
            "text": document_lookup[chunk_id],
            "metadata": metadata_lookup[chunk_id],
            "rrf_score": rrf_scores[chunk_id],
            "retrieval_sources": sorted(
                retrieval_sources[chunk_id]
            )
        })

    return final_results
