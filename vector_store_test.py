
from rag.document_loader import load_pdf
from rag.chunking import chunk_documents
from rag.vector_store import store_chunks, collection

# Load PDF
pdf_path = "data/documents/apple_annual_report.pdf"

documents = load_pdf(pdf_path)

# Generate chunks
chunks = chunk_documents(documents)

print("Extracted pages:", len(documents))
print("Generated chunks:", len(chunks))

# IMPORTANT: Actually store the chunks!
store_chunks(chunks)

# Verify database
print("\nDatabase verification:")
print("Total stored records:", collection.count())
