
from rag.document_loader import load_pdf


pdf_path = "data/documents/apple_annual_report.pdf"

documents = load_pdf(pdf_path)

print("Total extracted pages:", len(documents))

for document in documents[:3]:

    print("\nPage:", document["page"])

    print(document["text"][:500])
