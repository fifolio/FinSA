from app.embeddings import embed_texts

texts = [
    "Revenue declined 4% year over year due to weaker demand in China.",
    "Sales dropped in the Asia-Pacific region this fiscal year.",
    "The company issued a new dividend policy for shareholders.",
]

vectors = embed_texts(texts)

print(f"Number of vectors: {len(vectors)}")
print(f"Dimensions per vector: {len(vectors[0])}")
print(f"First 5 values of vector 0: {vectors[0][:5]}")