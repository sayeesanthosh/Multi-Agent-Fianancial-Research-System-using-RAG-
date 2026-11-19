
def chunk_documents(documents, chunk_size=500, overlap=100):

    all_chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0
        chunk_index = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunk = {
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"],
                "chunk_index": chunk_index
            }

            all_chunks.append(chunk)

            chunk_index += 1

            start += chunk_size - overlap

    return all_chunks
