
from rag.embeddings import generate_embedding


text = "Apple revenue increased due to strong iPhone sales."

embedding = generate_embedding(text)

print("Original text:", text)

print("Embedding dimensions:", len(embedding))

print("First 10 embedding values:", embedding[:10])
