
import chromadb
import hashlib

from rag.embeddings import generate_embedding

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="financial_documents",
    metadata={"hnsw:space": "cosine"}
)


def generate_chunk_id(chunk):

    unique_text = (
        chunk["source"]
        + str(chunk["page"])
        + str(chunk["chunk_index"])
        + chunk["text"]
    )

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


def store_chunks(chunks):

    print("Chunks received:", len(chunks))

    for chunk in chunks:

        chunk_id = generate_chunk_id(chunk)
        embedding = generate_embedding(chunk["text"])

        collection.upsert(
            ids=[chunk_id],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"]
            }]
        )

    print("Successfully processed", len(chunks), "chunks!")
