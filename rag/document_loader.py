
from pypdf import PdfReader


def load_pdf(file_path):

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            document = {
                "text": text,
                "page": page_number + 1,
                "source": file_path
            }

            documents.append(document)

    return documents
