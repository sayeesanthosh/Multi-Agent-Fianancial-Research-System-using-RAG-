
from rag.vector_store import client

existing_collections = [
    c.name for c in client.list_collections()
]

if "financial_documents" in existing_collections:

    client.delete_collection(
        name="financial_documents"
    )

    print("Old collection deleted.")

print("Database reset completed.")
