
from rag.embeddings import generate_embedding
from rag.similarity import cosine_similarity


sentences = [
    "Apple revenue increased due to strong iPhone sales.",

    "Apple's sales growth improved its financial performance.",

    "The weather is pleasant today."
]

# Generate embeddings
embeddings = []

for sentence in sentences:
    vector = generate_embedding(sentence)
    embeddings.append(vector)


# Compare sentence 1 with sentence 2
similarity_1 = cosine_similarity(
    embeddings[0],
    embeddings[1]
)

# Compare sentence 1 with sentence 3
similarity_2 = cosine_similarity(
    embeddings[0],
    embeddings[2]
)


print("\nSemantic Similarity Results")

print("Sentence 1:", sentences[0])
print("Sentence 2:", sentences[1])
print("Similarity:", similarity_1)

print("\nSentence 1:", sentences[0])
print("Sentence 3:", sentences[2])
print("Similarity:", similarity_2)
