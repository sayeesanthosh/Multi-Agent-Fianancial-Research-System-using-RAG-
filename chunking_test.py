
from rag.document_loader import load_pdf
from rag.chunking import chunk_documents

pdf_path = "data/documents/apple_annual_report.pdf"

documents = load_pdf(pdf_path)

chunks = chunk_documents(documents)

print("Total PDF pages:", len(documents))
print("Total chunks generated:", len(chunks))

for chunk in chunks[:3]:

    print("\n-------------------------")
    print("Page:", chunk["page"])
    print("Chunk:", chunk["chunk_index"])
    print("Source:", chunk["source"])
    print("Text:", chunk["text"][:300])
