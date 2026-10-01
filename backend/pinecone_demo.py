from app.ingestion import ingest_pdf
from app.vector_store import upsert_chunks, search

chunks = ingest_pdf("data/apple_2025.pdf", company="Apple", year=2025)
print(f"Upserted: {upsert_chunks(chunks)} chunks")

for result in search("What are the main risks from supply chain disruption?", top_k=3):
    print("-" * 60)
    print(f"score={result.score:.3f}  page={result.metadata['page']}")
    print(result.metadata["text"][:300])